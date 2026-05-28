############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.core.tools.filtering_functions import *
from classical_exps.core.tools.utils import *
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF
from .shared import neuron_key


## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
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
    robust_loss="linear",       # "linear" | "soft_l1" | "huber" | ...
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

def _mm(cc, K, gamma, n):
    cc = np.asarray(cc, float)
    gamma = max(float(gamma), 1e-12)
    n = max(float(n), 1e-6)
    cc_n = np.power(cc, n)
    g_n = gamma ** n
    return float(K) * (cc_n / (cc_n + g_n))

def fit_contrast_family_joint(
    cc, Y, *,
    model="response_gain",   # "response_gain" | "contrast_gain" | "subtractive"
    # bounds
    n_bounds=(0.2, 8.0),
    gamma_bounds=(1e-4, 10.0),
    K_bounds=(0.0, np.inf),
    k0_bounds=(0.0, np.inf),
    # loss / weighting
    loss="soft_l1",          # "linear" | "soft_l1" | "huber" ...
    f_scale_mode="adaptive",
    use_weights=False,
    weight_eps=0.05,
    # multistart
    n_starts=16,
    seed=0,
    max_nfev=6000,
    var_floor_mode = "fraction_of_max",
    var_floor_frac = 0.01,
    alpha = 1.0
):
    """
    Joint fit of contrast-response family Y of shape (S,C) to one of 3 paper models.
    No paper chi^2 weights; robust loss instead.
    Returns dict: params, Y_fit, success, cost, nfev
    """
    cc = np.asarray(cc, float).reshape(-1)
    Y = np.asarray(Y, float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be (S,C), got {Y.shape}")
    S, C = Y.shape
    if cc.shape[0] != C:
        raise ValueError(f"cc length {cc.shape[0]} != Y.shape[1] {C}")

    cc = np.maximum(cc, 0.0)

    # finite mask
    m = np.isfinite(Y) & np.isfinite(cc[None, :])
    if np.sum(m) < max(10, 2*S):
        return {"params": {}, "Y_fit": np.full_like(Y, np.nan), "success": False, "cost": float("inf"), "nfev": 0}

    # weights: optional, mild (NOT "paper variance")
    if use_weights:
        o = np.maximum(Y, 0.0)
        W = 1.0 / np.sqrt(o + float(weight_eps))
    else:
        W = np.ones_like(Y, float)

    # robust scale
    if f_scale_mode == "adaptive":
        med = float(np.nanmedian(Y))
        mad = float(np.nanmedian(np.abs(Y - med))) + 1e-6
        f_scale = mad
    else:
        f_scale = float(f_scale_mode)

    model = str(model).lower().strip()
    if model not in {"response_gain", "contrast_gain", "subtractive"}:
        raise ValueError(f"model must be response_gain|contrast_gain|subtractive, got {model}")

    # --- helpers to pack/unpack ---
    def unpack(p):
        if model == "response_gain":
            gamma, n = float(p[0]), float(p[1])
            Kvec = np.asarray(p[2:2+S], float)
            return {"gamma": gamma, "n": n, "K": Kvec}
        if model == "contrast_gain":
            K, n = float(p[0]), float(p[1])
            gvec = np.asarray(p[2:2+S], float)
            return {"K": K, "n": n, "gamma": gvec}
        # subtractive
        K, gamma, n = float(p[0]), float(p[1]), float(p[2])
        k0 = np.asarray(p[3:3+S], float)
        return {"K": K, "gamma": gamma, "n": n, "k0": k0}

    def predict(par):
        if model == "response_gain":
            gamma, n, Kvec = par["gamma"], par["n"], par["K"]
            Yhat = np.zeros((S, C), float)
            for s in range(S):
                Yhat[s] = _mm(cc, Kvec[s], gamma, n)
            return Yhat
        if model == "contrast_gain":
            K, n, gvec = par["K"], par["n"], par["gamma"]
            Yhat = np.zeros((S, C), float)
            for s in range(S):
                Yhat[s] = _mm(cc, K, gvec[s], n)
            return Yhat
        # subtractive
        K, gamma, n, k0 = par["K"], par["gamma"], par["n"], par["k0"]
        base = _mm(cc, K, gamma, n)
        Yhat = np.zeros((S, C), float)
        for s in range(S):
            Yhat[s] = np.maximum(0.0, base - k0[s])
        return Yhat

    def residuals(p):
        par = unpack(p)
        Yhat = predict(par)
        r = (Yhat - Y) * W
        return r[m].reshape(-1)

    # --- init heuristics ---
    rng = np.random.default_rng(int(seed))
    Ypos = np.maximum(Y, 0.0)
    K0_vec = np.maximum(np.nanmax(Ypos, axis=1), 1e-6)
    K0 = float(np.nanmax(Ypos))
    n0s = [1.0, 2.0, 3.5]

    # gamma guess from half-max point on the least-suppressed curve
    s0 = int(np.nanargmax(np.nanmax(Ypos, axis=1)))  # curve with biggest peak
    order = np.argsort(cc)
    cc_s, y_s = cc[order], Ypos[s0][order]
    target = 0.5 * float(np.nanmax(y_s))
    idx = np.searchsorted(y_s, target, side="left")
    gamma_base = float(cc_s[idx]) if 0 < idx < len(cc_s) else float(np.median(cc_s[cc_s > 0]) if np.any(cc_s > 0) else 0.1)
    gamma_base = float(np.clip(gamma_base, gamma_bounds[0], gamma_bounds[1]))

    # --- bounds + starts ---
    starts = []
    if model == "response_gain":
        lb = np.concatenate([[gamma_bounds[0], n_bounds[0]], np.full(S, K_bounds[0])])
        ub = np.concatenate([[gamma_bounds[1], n_bounds[1]], np.full(S, K_bounds[1])])

        for n0 in n0s:
            for gmul in (0.5, 1.0, 2.0):
                g0 = float(np.clip(gamma_base * gmul, gamma_bounds[0], gamma_bounds[1]))
                starts.append(np.concatenate([[g0, float(n0)], np.clip(K0_vec, K_bounds[0], K_bounds[1])]))

        while len(starts) < n_starts:
            g0 = float(np.clip(gamma_base * np.exp(rng.normal(0, 0.6)), gamma_bounds[0], gamma_bounds[1]))
            n0 = float(np.clip(np.exp(rng.normal(np.log(2.0), 0.5)), n_bounds[0], n_bounds[1]))
            Kvec = np.clip(K0_vec * np.exp(rng.normal(0, 0.5, size=S)), K_bounds[0], K_bounds[1])
            starts.append(np.concatenate([[g0, n0], Kvec]))

    elif model == "contrast_gain":
        lb = np.concatenate([[K_bounds[0], n_bounds[0]], np.full(S, gamma_bounds[0])])
        ub = np.concatenate([[K_bounds[1], n_bounds[1]], np.full(S, gamma_bounds[1])])

        K_base = float(np.clip(K0, K_bounds[0], K_bounds[1]))

        for n0 in n0s:
            for gmul in (0.5, 1.0, 2.0):
                g0 = float(np.clip(gamma_base * gmul, gamma_bounds[0], gamma_bounds[1]))
                starts.append(np.concatenate([[K_base, float(n0)], np.full(S, g0)]))

        while len(starts) < n_starts:
            n0 = float(np.clip(np.exp(rng.normal(np.log(2.0), 0.5)), n_bounds[0], n_bounds[1]))
            K_ = float(np.clip(K_base * np.exp(rng.normal(0, 0.4)), K_bounds[0], K_bounds[1]))
            gvec = np.clip(gamma_base * np.exp(rng.normal(0, 0.6, size=S)), gamma_bounds[0], gamma_bounds[1])
            starts.append(np.concatenate([[K_, n0], gvec]))

    else:  # subtractive
        lb = np.concatenate([[K_bounds[0], gamma_bounds[0], n_bounds[0]], np.full(S, k0_bounds[0])])
        ub = np.concatenate([[K_bounds[1], gamma_bounds[1], n_bounds[1]], np.full(S, k0_bounds[1])])

        K_base = float(np.clip(K0, K_bounds[0], K_bounds[1]))
        k0_base = float(np.maximum(0.0, np.nanmax(Ypos[s0]) - np.nanmin(Ypos[s0])))
        k0_vec0 = np.full(S, k0_base, float)

        for n0 in n0s:
            for gmul in (0.5, 1.0, 2.0):
                g0 = float(np.clip(gamma_base * gmul, gamma_bounds[0], gamma_bounds[1]))
                starts.append(np.concatenate([[K_base, g0, float(n0)], np.clip(k0_vec0, k0_bounds[0], k0_bounds[1])]))

        while len(starts) < n_starts:
            n0 = float(np.clip(np.exp(rng.normal(np.log(2.0), 0.5)), n_bounds[0], n_bounds[1]))
            K_ = float(np.clip(K_base * np.exp(rng.normal(0, 0.4)), K_bounds[0], K_bounds[1]))
            g0 = float(np.clip(gamma_base * np.exp(rng.normal(0, 0.6)), gamma_bounds[0], gamma_bounds[1]))
            k0 = np.clip(k0_vec0 * np.exp(rng.normal(0, 0.7, size=S)), k0_bounds[0], k0_bounds[1])
            starts.append(np.concatenate([[K_, g0, n0], k0]))

    # --- multistart solve ---
    best = None
    total_nfev = 0
    for p0 in starts[:n_starts]:
        p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)
        try:
            res = least_squares(
                residuals, p0,
                bounds=(lb, ub),
                method="trf",
                loss=loss,
                f_scale=f_scale,
                max_nfev=int(max_nfev),
            )
        except Exception:
            continue
        total_nfev += int(getattr(res, "nfev", 0))
        if (best is None) or (res.cost < best.cost):
            best = res

    if best is None:
        return {"params": {}, "Y_fit": np.full_like(Y, np.nan), "success": False, "cost": float("inf"), "nfev": 0}

    par = unpack(best.x)
    Y_fit = predict(par)

    # standardize params
    out_params = {"model": model}
    for k, v in par.items():
        out_params[k] = (float(v) if np.isscalar(v) else np.asarray(v, float))

    return {
        "params": out_params,
        "Y_fit": Y_fit.astype(float),
        "success": bool(best.success),
        "cost": float(best.cost),
        "nfev": int(total_nfev),
    }


def fit_contrast_response_joint(
    loaded: dict,
    *,
    h5_file: str,
    run: dict,
    strict: bool = True,
):
    """
    Post-loader hook for contrast response *families* (Naka-Rushton family models).
    Expects loaded["contrast_response"] already present with:
      - loaded["contrast_response"]["x"] = center contrasts (C,)
      - loaded["contrast_response"]["s"] = surround contrasts (S,)   (name "s" from your current code)
      - loaded["contrast_response"]["by_id"][nid] = raw mat (S,C)

    Adds:
      loaded["contrast_response_fits"]
      loaded["contrast_response_active"] (raw or joint fit)
    """
    if "contrast_response" not in loaded:
        raise KeyError("fit_contrast_response_joint requires loaded['contrast_response'] present.")

    do_fit   = bool(run.get("fit_contrast_response", False))  # reuse your flag
    use_fits = bool(run.get("use_contrast_response_fits", False))
    force    = bool(run.get("fit_force", False))
    fit_strict = bool(run.get("fit_strict", True))

    # which model to fit (or pick best)
    model = str(run.get("contrast_response_fit_model", "response_gain"))
    pick_best = bool(run.get("contrast_response_fit_pick_best", True))
    models_all = ["response_gain", "contrast_gain", "subtractive"]

    # fitter knobs (defaults tuned for your deterministic-float world)
    loss = str(run.get("nr_loss", "linear"))
    f_scale_mode = run.get("nr_f_scale_mode", "adaptive")
    use_weights = bool(run.get("nr_use_weights", False))
    weight_eps = float(run.get("nr_weight_eps", 0.02))
    n_starts = int(run.get("nr_n_starts", 72))
    seed = int(run.get("nr_seed", 0))
    max_nfev = int(run.get("nr_max_nfev", 20000))

    cr = loaded["contrast_response"]
    cc = np.asarray(cr["center_contrasts"], float)                # (C,)
    cs = np.asarray(cr.get("s", cr.get("surround_contrasts")), float)  # (S,)
    present_ids = np.asarray(cr.get("present_ids", []), int)

    # expected output shape
    expected_shape = (len(cs), len(cc))

    loaded["contrast_response_raw"] = cr
    loaded.setdefault("contrast_response_fits", {"present_ids": np.array([], int), "missing_ids": present_ids.copy(), "by_id": {}})

    if do_fit:
        existing = load_contrast_response_joint_fits_h5(
            h5_file, present_ids, strict=False, expected_shape=expected_shape
        )
        existing_set = set(existing.get("present_ids", []))

        fits_by_id = {}
        t0 = perf_counter()
        n_fit = n_loaded = 0

        for nid in tqdm(present_ids, desc="Fitting contrast response (NR joint)", unit="neuron"):
            nid_i = int(nid)

            if (not force) and (nid_i in existing_set):
                fits_by_id[nid_i] = existing["by_id"][nid_i]
                n_loaded += 1
                continue

            Y = np.asarray(cr["by_id"][nid_i], float)  # (S,C)
            if tuple(Y.shape) != tuple(expected_shape):
                if strict:
                    raise ValueError(f"contrast_response/{nid_i} has shape {Y.shape}, expected {expected_shape}")
                fits_by_id[nid_i] = {"y_fit": np.full(expected_shape, np.nan), "params": {"model": "bad_shape"}}
                continue

            # fit one model or pick best
            if pick_best:
                best = None
                costs = {}
                params_by_model = {}
                yfit_by_model = {}
                for m in models_all:
                    out = fit_contrast_family_joint(
                        cc, Y,
                        model=m,
                        loss=loss,
                        f_scale_mode=f_scale_mode,
                        use_weights=use_weights,
                        weight_eps=weight_eps,
                        n_starts=n_starts,
                        seed=seed,
                        max_nfev=max_nfev,
                    )
                    cost = float(out.get("cost", np.inf))
                    costs[m] = cost
                    params_by_model[m] = out.get("params", {})
                    yfit_by_model[m] = out.get("Y_fit", np.full_like(Y, np.nan))
                    if (best is None) or (cost < best["cost"]):
                        best = {"model": m, "cost": cost, "out": out}
                out = best["out"]
                # stash costs in params for H5 (scalars only)
                out_params = dict(out.get("params", {}))
                out_params["picked_model"] = best["model"]
                for m, cst in costs.items():
                    out_params[f"cost_{m}"] = float(cst)
                out = {**out, "params": out_params}
            else:
                out = fit_contrast_family_joint(
                    cc, Y,
                    model=model,
                    loss=loss,
                    f_scale_mode=f_scale_mode,
                    use_weights=use_weights,
                    weight_eps=weight_eps,
                    n_starts=n_starts,
                    seed=seed,
                    max_nfev=max_nfev,
                )

            y_fit = np.asarray(out["Y_fit"], float)
            params = dict(out.get("params", {}))
            params["model"] = str(params.get("model", model))

            save_contrast_response_joint_fits_h5(
                h5_file, nid_i,
                y_fit=y_fit,
                params=params,
            )

            fits_by_id[nid_i] = {"y_fit": y_fit, "params": params}
            n_fit += 1

        dt = perf_counter() - t0
        rate = (len(present_ids) / dt) if dt > 0 else float("inf")
        print(f"[contrast_response joint fit] total={len(present_ids)} fit={n_fit} loaded={n_loaded} time={dt:.2f}s ({rate:.1f} neurons/s)")

        loaded["contrast_response_fits"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": fits_by_id,
        }

    elif use_fits:
        fits = load_contrast_response_joint_fits_h5(
            h5_file, present_ids, strict=fit_strict and strict, expected_shape=expected_shape
        )
        loaded["contrast_response_fits"] = fits
        if fit_strict and strict and len(fits.get("missing_ids", [])) > 0:
            raise ValueError(
                f"Missing contrast_response joint fits for {len(fits['missing_ids'])} neurons. "
                f"Run fit_contrast_response first or set fit_strict=False."
            )

    # ---- choose active ----
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
            "x": cc,
            "s": cs,
            "using_fits": True,
        }
    else:
        loaded["contrast_response_active"] = {**cr, "using_fits": False}

    return loaded



def save_contrast_response_joint_fits_h5(
    h5_file, nid, *,
    y_fit, params: dict,
    group="/contrast_response/fits_joint",
):
    k = neuron_key(nid)
    with h5py.File(h5_file, "a") as f:
        grp = f.require_group(group)
        ng = grp.require_group(k)

        # overwrite cleanly
        for name in list(ng.keys()):
            del ng[name]
        for a in list(ng.attrs.keys()):
            del ng.attrs[a]

        ng.create_dataset("y_fit", data=np.asarray(y_fit, float))  # (S,C)

        # attrs + param_* datasets
        ng.attrs["model"] = str(params.get("model", ""))
        for kk, vv in params.items():
            if kk == "model":
                continue
            vv = np.asarray(vv)
            if vv.ndim == 0:
                ng.attrs[kk] = float(vv)
            else:
                dname = f"param_{kk}"
                if dname in ng:
                    del ng[dname]
                ng.create_dataset(dname, data=vv.astype(float))


def load_contrast_response_joint_fits_h5(
    h5_file, neuron_ids, *,
    group="/contrast_response/fits_joint",
    strict=False,
    expected_shape=None,  # (S,C)
):
    neuron_ids = np.asarray(neuron_ids, int)
    present, missing, by_id = [], [], {}

    with h5py.File(h5_file, "r") as f:
        if group not in f:
            return {"present_ids": np.array([], int), "missing_ids": neuron_ids.copy(), "by_id": {}}
        grp = f[group]

        for nid in neuron_ids:
            k = neuron_key(int(nid))
            if k not in grp:
                missing.append(int(nid)); continue
            ng = grp[k]
            try:
                y_fit = np.asarray(ng["y_fit"][:], float)
                if expected_shape is not None and tuple(y_fit.shape) != tuple(expected_shape):
                    if strict:
                        raise ValueError(f"{group}/{k} shape {y_fit.shape} != expected {expected_shape}")
                params = dict(ng.attrs)
                for dname in ng.keys():
                    if dname.startswith("param_"):
                        params[dname.replace("param_", "")] = np.asarray(ng[dname][:], float)

                by_id[int(nid)] = {"y_fit": y_fit, "params": params}
                present.append(int(nid))
            except Exception:
                if strict:
                    raise
                missing.append(int(nid))

    return {"present_ids": np.asarray(present, int), "missing_ids": np.asarray(missing, int), "by_id": by_id}

def add_fitted_contrast_response_to_scraped_config(
    nature_and_interactions_scraped_data: dict,
    *,
    run: dict,
    strict: bool = True,
):
    """
    Fit scraped contrast_tunning data using the same joint-fit settings as model data,
    then write fitted curves back into the scraped config.

    Adds:
        nature_and_interactions_scraped_data["contrast_tunning_fit"]
        nature_and_interactions_scraped_data["contrast_tunning_fit_summary"]

    Expected input:
        nature_and_interactions_scraped_data["contrast_tunning"][cell_id][surround_key] = {
            "center_contrast": [...],
            "response": [...]
        }
    """
    if nature_and_interactions_scraped_data is None:
        return nature_and_interactions_scraped_data
    if "contrast_tunning" not in nature_and_interactions_scraped_data:
        return nature_and_interactions_scraped_data

    scraped = nature_and_interactions_scraped_data["contrast_tunning"]

    # Build common matrix representation once for fitting
    loaded_scraped = build_loaded_contrast_response_from_scraped(scraped)

    cc = np.asarray(loaded_scraped["contrast_response"]["center_contrasts"], dtype=float)
    cs = np.asarray(loaded_scraped["contrast_response"]["surround_contrasts"], dtype=float)
    by_id = loaded_scraped["contrast_response"]["by_id"]

    model = str(run.get("contrast_response_fit_model", "response_gain"))
    pick_best = bool(run.get("contrast_response_fit_pick_best", False))
    models_all = ["response_gain", "contrast_gain", "subtractive"]

    loss = str(run.get("nr_loss", "soft_l1"))
    f_scale_mode = run.get("nr_f_scale_mode", "adaptive")
    use_weights = bool(run.get("nr_use_weights", False))
    weight_eps = float(run.get("nr_weight_eps", 0.05))
    n_starts = int(run.get("nr_n_starts", 72))
    seed = int(run.get("nr_seed", 0))
    max_nfev = int(run.get("nr_max_nfev", 20000))

    fitted_cfg = {}
    fit_summary = {}

    for cell_id, Y in by_id.items():
        Y = np.asarray(Y, dtype=float)
        expected_shape = (len(cs), len(cc))

        if tuple(Y.shape) != tuple(expected_shape):
            if strict:
                raise ValueError(
                    f"Scraped contrast_tunning cell {cell_id} has shape {Y.shape}, expected {expected_shape}"
                )
            continue

        if pick_best:
            best = None
            costs = {}

            for m in models_all:
                out = fit_contrast_family_joint(
                    cc,
                    Y,
                    model=m,
                    loss=loss,
                    f_scale_mode=f_scale_mode,
                    use_weights=use_weights,
                    weight_eps=weight_eps,
                    n_starts=n_starts,
                    seed=seed,
                    max_nfev=max_nfev,
                )
                cost = float(out.get("cost", np.inf))
                costs[m] = cost
                if (best is None) or (cost < best["cost"]):
                    best = {"model": m, "cost": cost, "out": out}

            out = best["out"]
            params = dict(out.get("params", {}))
            params["picked_model"] = best["model"]
            for m, cst in costs.items():
                params[f"cost_{m}"] = float(cst)
        else:
            out = fit_contrast_family_joint(
                cc,
                Y,
                model=model,
                loss=loss,
                f_scale_mode=f_scale_mode,
                use_weights=use_weights,
                weight_eps=weight_eps,
                n_starts=n_starts,
                seed=seed,
                max_nfev=max_nfev,
            )
            params = dict(out.get("params", {}))

        Y_fit = np.asarray(out["Y_fit"], dtype=float)
        params["model"] = str(params.get("model", model))

        # Write back in the same human-readable structure as the scrape,
        # but sampled on each raw curve's original x positions.
        cell_fit = {}

        raw_cell = scraped[cell_id]
        raw_surround_keys_sorted = sorted(raw_cell.keys(), key=lambda k: float(k))

        if len(raw_surround_keys_sorted) != len(cs):
            raise ValueError(
                f"Cell {cell_id}: raw surround-key count {len(raw_surround_keys_sorted)} "
                f"!= fitted surround count {len(cs)}"
            )

        for i, raw_surr_key in enumerate(raw_surround_keys_sorted):
            raw_entry = raw_cell[raw_surr_key]
            x_raw = np.asarray(raw_entry["center_contrast"], dtype=float)

            valid = np.isfinite(Y_fit[i]) & np.isfinite(cc)
            x_fit_grid = cc[valid].astype(float)
            y_fit_grid = Y_fit[i, valid].astype(float)

            if len(x_fit_grid) == 0:
                y_fit_on_raw = np.full_like(x_raw, np.nan, dtype=float)
            elif len(x_fit_grid) == 1:
                y_fit_on_raw = np.full_like(x_raw, y_fit_grid[0], dtype=float)
            else:
                order = np.argsort(x_fit_grid)
                x_fit_grid = x_fit_grid[order]
                y_fit_grid = y_fit_grid[order]

                y_fit_on_raw = np.interp(
                    x_raw,
                    x_fit_grid,
                    y_fit_grid,
                    left=np.nan,
                    right=np.nan,
                )

            cell_fit[raw_surr_key] = {
                "center_contrast": x_raw.astype(float).tolist(),
                "response": y_fit_on_raw.astype(float).tolist(),
            }

        fitted_cfg[cell_id] = cell_fit
        fit_summary[cell_id] = {
            "success": bool(out.get("success", False)),
            "cost": float(out.get("cost", np.inf)),
            "nfev": int(out.get("nfev", 0)),
            "params": params,
        }

    nature_and_interactions_scraped_data["contrast_tunning_fit"] = fitted_cfg
    nature_and_interactions_scraped_data["contrast_tunning_fit_summary"] = fit_summary

    return nature_and_interactions_scraped_data

def build_loaded_contrast_response_from_scraped(
    scraped_contrast_tunning,
    *,
    neuron_id_parser=None,
    sort_surround_numeric=True,
    round_center_contrast_decimals=6,
    atol_center=1e-6,
):
    """
    Convert scraped contrast_tunning dict into the internal loaded format.

    Input format:
        scraped_contrast_tunning = {
            "cell1": {
                "0":  {"center_contrast": [...], "response": [...]},
                "3":  {"center_contrast": [...], "response": [...]},
                ...
            },
            ...
        }

    Output:
        loaded = {
            "contrast_response": {
                "center_contrasts": np.ndarray,         # shared x-grid
                "surround_contrasts": np.ndarray,       # sorted numeric if possible
                "by_id": {
                    neuron_id: np.ndarray shape (n_surround, n_center)
                }
            },
            "contrast_response_results": {
                neuron_id: float
            }
        }

    Notes:
    - surround keys are interpreted numerically, e.g. "0","3","6","12","25","50" -> 0,3,6,12,25,50
    - center contrast values are merged across the cell, then each curve is placed on the union grid
    - missing entries become np.nan
    """

    if neuron_id_parser is None:
        neuron_id_parser = lambda x: x

    by_id = {}
    spread_results = {}
    global_surrounds = None
    global_centers = None

    # first pass: determine if there is a common global grid
    per_cell_surrounds = {}
    per_cell_centers = {}

    for raw_neuron_id, sdict in scraped_contrast_tunning.items():
        neuron_id = neuron_id_parser(raw_neuron_id)

        # parse surround keys
        try:
            surround_vals = [float(k) for k in sdict.keys()]
        except Exception as e:
            raise ValueError(f"Could not parse surround keys for {raw_neuron_id}: {list(sdict.keys())}") from e

        if sort_surround_numeric:
            surround_vals = sorted(surround_vals)

        all_centers = []
        for skey, entry in sdict.items():
            x = np.asarray(entry["center_contrast"], dtype=float)
            all_centers.extend(list(x))

        # stabilize tiny scrape jitter
        all_centers = np.asarray(all_centers, dtype=float)
        all_centers = np.round(all_centers, round_center_contrast_decimals)
        center_union = np.unique(all_centers)

        per_cell_surrounds[neuron_id] = np.asarray(surround_vals, dtype=float)
        per_cell_centers[neuron_id] = center_union

        if global_surrounds is None:
            global_surrounds = np.asarray(surround_vals, dtype=float)
        else:
            if len(global_surrounds) != len(surround_vals) or not np.allclose(global_surrounds, surround_vals):
                raise ValueError(
                    f"Scraped surround contrast grid differs across cells. "
                    f"Expected {global_surrounds}, got {surround_vals} for {raw_neuron_id}."
                )

        if global_centers is None:
            global_centers = center_union
        else:
            merged = np.unique(np.concatenate([global_centers, center_union]))
            global_centers = np.asarray(np.sort(merged), dtype=float)

    # second pass: build matrices on common global grid
    for raw_neuron_id, sdict in scraped_contrast_tunning.items():
        neuron_id = neuron_id_parser(raw_neuron_id)
        surrounds = per_cell_surrounds[neuron_id]

        mat = np.full((len(global_surrounds), len(global_centers)), np.nan, dtype=float)

        for i, surr in enumerate(global_surrounds):
            skey_candidates = [str(int(surr)), str(surr)]
            entry = None
            for cand in skey_candidates:
                if cand in sdict:
                    entry = sdict[cand]
                    break

            if entry is None:
                # very annoying but possible with ugly keys
                for k, v in sdict.items():
                    if np.isclose(float(k), surr):
                        entry = v
                        break

            if entry is None:
                continue

            x = np.round(np.asarray(entry["center_contrast"], dtype=float), round_center_contrast_decimals)
            y = np.asarray(entry["response"], dtype=float)

            if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
                raise ValueError(
                    f"Bad scraped curve for neuron {raw_neuron_id}, surround {surr}: "
                    f"x shape {x.shape}, y shape {y.shape}"
                )

            for xx, yy in zip(x, y):
                idx = np.where(np.isclose(global_centers, xx, atol=atol_center))[0]
                if len(idx) != 1:
                    raise RuntimeError(
                        f"Could not map center contrast {xx} for neuron {raw_neuron_id}, surround {surr}"
                    )
                mat[i, idx[0]] = yy

        by_id[neuron_id] = mat

        # simple spread metric fallback: ratio of max/min of last valid points across surround curves
        last_vals = []
        for i in range(mat.shape[0]):
            row = mat[i]
            valid = row[np.isfinite(row)]
            if len(valid) > 0:
                last_vals.append(valid[-1])

        last_vals = np.asarray(last_vals, dtype=float)
        if len(last_vals) >= 2:
            min_pos = np.nanmin(last_vals[last_vals > 0]) if np.any(last_vals > 0) else np.nan
            max_val = np.nanmax(last_vals)
            spread = (max_val / min_pos) if np.isfinite(min_pos) and min_pos > 0 else np.nan
        else:
            spread = np.nan

        spread_results[neuron_id] = spread

    loaded = {
        "contrast_response": {
            "center_contrasts": global_centers,
            "surround_contrasts": global_surrounds,
            "by_id": by_id,
        },
        "contrast_response_results": spread_results,
    }

    return loaded

def build_loaded_contrast_response_from_scraped_config(
    nature_and_interactions_scraped_data,
    *,
    raw_key="contrast_tunning",
    fit_key="contrast_tunning_fit",
    neuron_id_parser=None,
    sort_surround_numeric=True,
    round_center_contrast_decimals=6,
    atol_center=1e-6,
):
    """
    Build a loaded-like contrast_response object from scraped config.

    Returns a hybrid structure:
      - matrix representation in loaded_scraped["contrast_response"] for compatibility / fitting summaries
      - original per-curve scraped representation in loaded_scraped["contrast_response_scraped_raw"]
      - optional fitted per-curve scraped representation in loaded_scraped["contrast_response_scraped_fit"]
    """
    if nature_and_interactions_scraped_data is None:
        return None
    if raw_key not in nature_and_interactions_scraped_data:
        return None

    raw_src = nature_and_interactions_scraped_data[raw_key]

    # matrix representation, still useful for spread sorting / compatibility
    loaded_raw = build_loaded_contrast_response_from_scraped(
        raw_src,
        neuron_id_parser=neuron_id_parser,
        sort_surround_numeric=sort_surround_numeric,
        round_center_contrast_decimals=round_center_contrast_decimals,
        atol_center=atol_center,
    )

    loaded_scraped = {
        "contrast_response": loaded_raw["contrast_response"],
        "contrast_response_results": loaded_raw.get("contrast_response_results", {}),
        "contrast_response_scraped_raw": raw_src,
    }

    # optional fits
    if fit_key in nature_and_interactions_scraped_data:
        fit_src = nature_and_interactions_scraped_data[fit_key]
        loaded_scraped["contrast_response_scraped_fit"] = fit_src

        # optional matrix fit compatibility, but only if axes really match
        loaded_fit = build_loaded_contrast_response_from_scraped(
            fit_src,
            neuron_id_parser=neuron_id_parser,
            sort_surround_numeric=sort_surround_numeric,
            round_center_contrast_decimals=round_center_contrast_decimals,
            atol_center=atol_center,
        )

        raw_cr = loaded_scraped["contrast_response"]
        fit_cr = loaded_fit["contrast_response"]

        raw_ids = set(raw_cr["by_id"].keys())
        fit_ids = set(fit_cr["by_id"].keys())
        common_ids = raw_ids & fit_ids

        fits_by_id = {}
        for nid in common_ids:
            y_raw = np.asarray(raw_cr["by_id"][nid], dtype=float)
            y_fit = np.asarray(fit_cr["by_id"][nid], dtype=float)

            if y_raw.shape != y_fit.shape:
                raise ValueError(
                    f"Scraped fit shape mismatch for {nid}: raw {y_raw.shape} vs fit {y_fit.shape}"
                )

            if not np.allclose(
                np.asarray(raw_cr["center_contrasts"], dtype=float),
                np.asarray(fit_cr["center_contrasts"], dtype=float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Scraped fit center grid mismatch for {nid}: "
                    f"raw {raw_cr['center_contrasts']} vs fit {fit_cr['center_contrasts']}"
                )

            if not np.allclose(
                np.asarray(raw_cr["surround_contrasts"], dtype=float),
                np.asarray(fit_cr["surround_contrasts"], dtype=float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Scraped fit surround grid mismatch for {nid}: "
                    f"raw {raw_cr['surround_contrasts']} vs fit {fit_cr['surround_contrasts']}"
                )

            fits_by_id[nid] = {"y_fit": y_fit}

        loaded_scraped["contrast_response_fits"] = {"by_id": fits_by_id}

    # optional summaries
    summary_key = f"{fit_key}_summary"
    if summary_key in nature_and_interactions_scraped_data:
        summary_src = nature_and_interactions_scraped_data[summary_key]
        summary_out = {}
        for nid, val in summary_src.items():
            parsed_id = neuron_id_parser(nid) if neuron_id_parser is not None else nid
            summary_out[parsed_id] = val
        loaded_scraped["contrast_response_fit_summary"] = summary_out

    return loaded_scraped