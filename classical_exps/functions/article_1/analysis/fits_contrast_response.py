############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF
from .shared import neuron_key


## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
from datetime import datetime
import json
import pandas as pd
import os
import openpyxl
from matplotlib.ticker import ScalarFormatter
import time
from tqdm import tqdm
from scipy.special import erf
from scipy.optimize import least_squares
from time import perf_counter


def naka_rushton(c, Rmax, c50, n):
    c = np.asarray(c, float)
    c50 = max(float(c50), 1e-12)
    n = max(float(n), 1e-6)
    return Rmax * (c**n) / (c**n + c50**n)


def fit_naka_rushton_single(
    contrasts, y,
    *,
    # bounds
    R0_bounds=(-np.inf, np.inf),
    Rmax_bounds=(0.0, np.inf),
    c50_bounds=(1e-4, 10.0),     # contrasts are usually in [0..1], allow wider
    n_bounds=(0.2, 8.0),

    # multi-start
    n_starts=10,
    n_random_starts=5,
    random_seed=0,

    # robustness / weighting
    robust_loss="soft_l1",       # "linear" | "soft_l1" | "huber" | ...
    use_weights=True,
    weight_eps=0.05,
    f_scale_mode="adaptive",     # "adaptive" or float
):
    """
    Fits Naka-Rushton curve to a single 1D contrast-response curve.

    Returns dict:
      params: {R0, Rmax, c50, n}
      y_fit
      success
      cost
      nfev
      best_start
    """
    c = np.asarray(contrasts, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if c.shape != y.shape:
        raise ValueError(f"contrasts and y must have same shape, got {c.shape} vs {y.shape}")

    # filter finite
    m = np.isfinite(c) & np.isfinite(y)
    c = c[m]
    y = y[m]

    if len(c) < 4:
        return {
            "params": {"R0": float(np.nan), "Rmax": float(np.nan), "c50": float(np.nan), "n": float(np.nan)},
            "y_fit": np.full_like(c, np.nan, dtype=float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    # NR expects non-negative contrasts; if you have exactly 0, it's fine.
    # But c50 must stay >0.
    c = np.maximum(c, 0.0)

    # weights: mild down-weight large responses
    if use_weights:
        y_pos = np.maximum(y, 0.0)
        w = 1.0 / np.sqrt(y_pos + float(weight_eps))
    else:
        w = np.ones_like(y, dtype=float)

    # adaptive scale for robust loss
    if f_scale_mode == "adaptive":
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med))) + 1e-6
        f_scale = mad
    else:
        f_scale = float(f_scale_mode)

    def nr_response(c_, R0, Rmax, c50, n):
        c50 = max(float(c50), 1e-12)
        n = float(n)
        # safe power
        cn = np.power(c_, n)
        c50n = (c50 ** n)
        frac = cn / (cn + c50n)
        return float(R0) + float(Rmax) * frac

    # bounds
    lb = np.array([R0_bounds[0], Rmax_bounds[0], c50_bounds[0], n_bounds[0]], dtype=float)
    ub = np.array([R0_bounds[1], Rmax_bounds[1], c50_bounds[1], n_bounds[1]], dtype=float)

    # init heuristics
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    amp = max(y_max - y_min, 1e-6)

    # baseline near low-contrast response (use smallest contrast)
    i0 = int(np.argmin(c))
    R0_base = float(y[i0])

    # Rmax roughly amplitude
    Rmax_base = float(max(amp, 1e-6))

    # c50: contrast where response reaches half-amp
    target = R0_base + 0.5 * amp
    # sort by contrast for this heuristic
    order = np.argsort(c)
    c_s = c[order]
    y_s = y[order]
    idx = np.searchsorted(y_s, target, side="left")
    if 0 < idx < len(c_s):
        c50_base = float(np.clip(c_s[idx], c50_bounds[0], c50_bounds[1]))
    else:
        c50_base = float(np.clip(np.median(c_s[c_s > 0]) if np.any(c_s > 0) else 0.1, c50_bounds[0], c50_bounds[1]))

    n_base = 2.0

    # build starts
    starts = []
    for n0 in (1.0, 2.0, 3.5):
        for c50_mul in (0.5, 1.0, 2.0):
            c50_0 = float(np.clip(c50_base * c50_mul, c50_bounds[0], c50_bounds[1]))
            for R0_shift in (0.0, -0.25*amp, 0.25*amp):
                R0_0 = float(R0_base + R0_shift)
                Rmax_0 = float(Rmax_base)
                starts.append(np.array([R0_0, Rmax_0, c50_0, float(n0)], dtype=float))

    rng = np.random.default_rng(int(random_seed))
    for _ in range(int(n_random_starts)):
        R0_0 = float(R0_base + rng.normal(0.0, 0.3*amp))
        Rmax_0 = float(max(1e-6, Rmax_base * np.exp(rng.normal(0.0, 0.6))))
        c50_0 = float(np.clip(c50_base * np.exp(rng.normal(0.0, 0.7)), c50_bounds[0], c50_bounds[1]))
        n0 = float(np.clip(np.exp(rng.normal(np.log(n_base), 0.5)), n_bounds[0], n_bounds[1]))
        starts.append(np.array([R0_0, Rmax_0, c50_0, n0], dtype=float))

    # trim to n_starts
    if len(starts) > int(n_starts):
        # keep first few deterministic + last few random
        det_keep = max(0, int(n_starts) - int(n_random_starts))
        starts = starts[:det_keep] + (starts[-int(n_random_starts):] if int(n_random_starts) > 0 else [])
    else:
        starts = starts[:]

    def residuals(p):
        R0, Rmax, c50, n = p
        y_hat = nr_response(c, R0, Rmax, c50, n)
        r = (y_hat - y) * w
        return r

    best = None
    best_start = None
    total_nfev = 0

    for si, p0 in enumerate(starts):
        p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)
        try:
            res = least_squares(
                residuals,
                p0,
                bounds=(lb, ub),
                method="trf",
                loss=robust_loss,
                f_scale=f_scale,
                max_nfev=3000,
            )
        except Exception:
            continue

        total_nfev += int(getattr(res, "nfev", 0))

        if best is None or (res.cost < best.cost):
            best = res
            best_start = {"idx": si, "p0": p0.copy(), "cost": float(res.cost), "success": bool(res.success)}

    if best is None:
        return {
            "params": {"R0": float(np.nan), "Rmax": float(np.nan), "c50": float(np.nan), "n": float(np.nan)},
            "y_fit": np.full_like(c, np.nan, dtype=float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    R0, Rmax, c50, n = best.x
    y_hat = nr_response(c, R0, Rmax, c50, n)

    return {
        "params": {"R0": float(R0), "Rmax": float(Rmax), "c50": float(c50), "n": float(n)},
        "y_fit": np.asarray(y_hat, dtype=float),
        "success": bool(best.success),
        "cost": float(best.cost),
        "nfev": int(total_nfev),
        "best_start": best_start,
    }


def save_contrast_response_fits_h5(h5_file, nid, *, y_fit, group="/contrast_response/fits", attrs=None):
    k = neuron_key(nid)
    with h5py.File(h5_file, "a") as f:
        grp = f.require_group(group)
        ng = grp.require_group(k)
        for name in list(ng.keys()):
            del ng[name]
        ng.create_dataset("y_fit", data=np.asarray(y_fit, float))
        if attrs:
            for kk, vv in attrs.items():
                ng.attrs[kk] = vv


def load_contrast_response_fits_h5(h5_file, neuron_ids, group="/contrast_response/fits", strict=True):
    present, missing, by_id = [], [], {}
    with h5py.File(h5_file, "r") as f:
        if group not in f:
            return {"present_ids": np.array([], int), "missing_ids": np.asarray(neuron_ids, int), "by_id": {}}
        grp = f[group]
        for nid in np.asarray(neuron_ids, int):
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid)); continue
            ng = grp[k]
            try:
                by_id[int(nid)] = {"y_fit": np.asarray(ng["y_fit"][:], float), "attrs": dict(ng.attrs)}
            except Exception:
                if strict: raise
                missing.append(int(nid)); continue
            present.append(int(nid))
    return {"present_ids": np.asarray(present, int), "missing_ids": np.asarray(missing, int), "by_id": by_id}


def fit_contrast_response(
    loaded: dict,
    *,
    h5_file: str,
    run: dict,
    strict: bool = True,
):
    """
    Requires loaded["contrast_response"] already present:
      - loaded["contrast_response"]["x"] = center contrasts (C,)
      - loaded["contrast_response"]["by_id"][nid] = raw mat (S,C)

    Adds:
      loaded["contrast_response_fits"]
      loaded["contrast_response_active"]  (raw or fit)
    """
    if "contrast_response" not in loaded:
        raise KeyError("fit_contrast_response requires loaded['contrast_response'] present.")

    do_fit   = bool(run.get("fit_contrast_response", False))
    use_fits = bool(run.get("use_contrast_response_fits", False))
    force    = bool(run.get("fit_force", False))
    fit_strict = bool(run.get("fit_strict", True))

    cr = loaded["contrast_response"]
    c = np.asarray(cr["x"], float)
    present_ids = np.asarray(cr.get("present_ids", []), int)

    loaded["contrast_response_raw"] = cr
    loaded.setdefault("contrast_response_fits", {"present_ids": np.array([], int), "missing_ids": present_ids.copy(), "by_id": {}})

    if do_fit:
        existing = load_contrast_response_fits_h5(h5_file, present_ids, strict=False)
        existing_set = set(existing.get("present_ids", []))

        fits_by_id = {}
        t0 = perf_counter()
        n_fit = n_loaded = 0

        for nid in tqdm(present_ids, desc="Fitting contrast response (NR)", unit="neuron"):
            nid_i = int(nid)

            if (not force) and (nid_i in existing_set):
                fits_by_id[nid_i] = existing["by_id"][nid_i]
                n_loaded += 1
                continue

            y = np.asarray(cr["by_id"][nid_i], float)  # (S,C)
            S, Cn = y.shape

            y_fit = np.zeros_like(y, dtype=float)
            attrs = {}

            # Fit each surround curve separately
            for si in range(S):
                fit = fit_naka_rushton_single(c, y[si])
                y_fit[si] = fit["y_fit"]
                p = fit["params"]
                attrs[f"s{si}_Rmax"] = p["Rmax"]
                attrs[f"s{si}_c50"]  = p["c50"]
                attrs[f"s{si}_n"]    = p["n"]
                attrs[f"s{si}_R0"]   = p["R0"]
                attrs[f"s{si}_ok"]   = int(bool(fit["success"]))


            save_contrast_response_fits_h5(h5_file, nid_i, y_fit=y_fit, attrs=attrs)
            fits_by_id[nid_i] = {"y_fit": y_fit, "attrs": attrs}
            n_fit += 1

        dt = perf_counter() - t0
        rate = (len(present_ids) / dt) if dt > 0 else float("inf")
        print(f"[contrast_response fit] total={len(present_ids)} fit={n_fit} loaded={n_loaded} "
              f"time={dt:.2f}s ({rate:.1f} neurons/s)")

        loaded["contrast_response_fits"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": fits_by_id,
        }

    elif use_fits:
        fits = load_contrast_response_fits_h5(h5_file, present_ids, strict=fit_strict and strict)
        loaded["contrast_response_fits"] = fits
        if fit_strict and strict and len(fits.get("missing_ids", [])) > 0:
            raise ValueError(
                f"Missing contrast_response fits for {len(fits['missing_ids'])} neurons. "
                f"Run fit_contrast_response first or set fit_strict=False."
            )

    # active view
    if use_fits and len(loaded["contrast_response_fits"].get("present_ids", [])) > 0:
        active_by_id = {}
        fits_by = loaded["contrast_response_fits"]["by_id"]
        for nid in present_ids:
            nid_i = int(nid)
            if nid_i in fits_by:
                active_by_id[nid_i] = fits_by[nid_i]["y_fit"]
            else:
                active_by_id[nid_i] = cr["by_id"][nid_i]

        loaded["contrast_response_active"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": active_by_id,
            "x": c,
            "s": np.asarray(cr["s"], float),
            "using_fits": True,
        }
    else:
        loaded["contrast_response_active"] = {**cr, "using_fits": False}

    return loaded
