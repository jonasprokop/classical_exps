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


def fit_size_tuning(
    loaded: dict,
    *,
    h5_file: str,
    run: dict,
    strict: bool = True,
):
    """
    Post-loader hook.
    Expects loaded["size_tuning_curves"] already present.

    Adds:
      loaded["size_tuning_fits"]
      loaded["size_tuning_curves_raw"]
      loaded["size_tuning_curves_active"]  (raw or fits)
    """
    if "size_tuning_curves" not in loaded:
        raise KeyError("fit_size_tuning requires loaded['size_tuning_curves'] already present.")

    do_fit   = bool(run.get("fit_size_tuning", False))
    use_fits = bool(run.get("use_size_tuning_fits", False))
    force    = bool(run.get("fit_force", False))
    fit_strict = bool(run.get("fit_strict", True))

    st = loaded["size_tuning_curves"]
    radii = np.asarray(st.get("radii"), dtype=float)
    present_ids = np.asarray(st.get("present_ids", []), dtype=int)

    loaded["size_tuning_curves_raw"] = st
    loaded.setdefault("size_tuning_fits", {"present_ids": np.array([], int), "missing_ids": present_ids.copy(), "by_id": {}})

    # --- case 1: fit now ---
    if do_fit:
        fits_by_id = {}

        existing = load_size_tuning_fits_h5(h5_file, present_ids, strict=False)
        existing_set = set(existing.get("present_ids", []))

        t0 = time.perf_counter()

        # knobs
        fit_circ = bool(run.get("rog_fit_circular", True))
        fit_ann  = bool(run.get("rog_fit_annular", True))   # you can default False if ann fits are messy
        # forward these into fit_rog_single_curve
        n_starts = int(run.get("rog_n_starts", 12))
        n_rand   = int(run.get("rog_n_random_starts", 6))
        seed     = int(run.get("rog_seed", 0))
        loss     = run.get("rog_loss", "soft_l1")
        use_w    = bool(run.get("rog_use_weights", True))
        w_eps    = float(run.get("rog_weight_eps", 0.05))
        f_scale_mode = run.get("rog_f_scale_mode", "adaptive")

        for nid in tqdm(present_ids, desc="Fitting size tuning", unit="neuron"):
            nid_i = int(nid)

            if (not force) and (nid_i in existing_set):
                fits_by_id[nid_i] = existing["by_id"][nid_i]
                continue

            raw = st["by_id"][nid_i]
            circ_raw = np.asarray(raw["circular"], dtype=float)
            ann_raw  = np.asarray(raw.get("annular"), dtype=float) if "annular" in raw else None

            # --- fit circular independently ---
            if fit_circ:
                circ_fit = fit_rog_single_curve(
                    radii, circ_raw,
                    n_starts=n_starts,
                    n_random_starts=n_rand,
                    random_seed=seed,
                    robust_loss=loss,
                    use_weights=use_w,
                    weight_eps=w_eps,
                    f_scale_mode=f_scale_mode,
                )
            else:
                circ_fit = {
                    "params": {"kc": float("nan"), "ks": float("nan"), "wc": float("nan"), "ws": float("nan")},
                    "y_fit": np.full_like(2.0 * np.asarray(radii, float), np.nan, dtype=float),
                    "success": False, "cost": float("inf"), "nfev": 0, "best_start": None,
                }

            # --- fit annular independently (optional) ---
            if fit_ann and (ann_raw is not None):
                ann_fit = fit_rog_single_curve(
                    radii, ann_raw,
                    n_starts=n_starts,
                    n_random_starts=n_rand,
                    random_seed=seed + 1,   # tiny decorrelation
                    robust_loss=loss,
                    use_weights=use_w,
                    weight_eps=w_eps,
                    f_scale_mode=f_scale_mode,
                )
            else:
                ann_fit = {
                    "params": {"kc": float("nan"), "ks": float("nan"), "wc": float("nan"), "ws": float("nan")},
                    "y_fit": np.full_like(2.0 * np.asarray(radii, float), np.nan, dtype=float),
                    "success": False, "cost": float("inf"), "nfev": 0, "best_start": None,
                }

            # Save to H5 (NEW schema: separate circ/ann params)
            save_size_tuning_fits_h5(
                h5_file,
                nid_i,
                radii=radii,
                circ_fit={"y_fit": circ_fit["y_fit"], "params": circ_fit["params"], "meta": circ_fit},
                ann_fit={"y_fit": ann_fit["y_fit"], "params": ann_fit["params"], "meta": ann_fit},
            )

            # Keep in memory (NEW keys + per-curve attrs)
            fits_by_id[nid_i] = {
                "radii": radii,
                "circular_yfit": np.asarray(circ_fit["y_fit"], float),
                "annular_yfit":  np.asarray(ann_fit["y_fit"],  float),
                "params": {
                    "circular": dict(circ_fit["params"]),
                    "annular":  dict(ann_fit["params"]),
                },
                "attrs": {
                    # circular attrs
                    **{f"rog_circ_{k}": float(v) for k, v in circ_fit["params"].items()},
                    "success_circ": int(bool(circ_fit.get("success", False))),
                    "cost_circ": float(circ_fit.get("cost", np.nan)),
                    "nfev_circ": int(circ_fit.get("nfev", 0)),
                    # annular attrs
                    **{f"rog_ann_{k}": float(v) for k, v in ann_fit["params"].items()},
                    "success_ann": int(bool(ann_fit.get("success", False))),
                    "cost_ann": float(ann_fit.get("cost", np.nan)),
                    "nfev_ann": int(ann_fit.get("nfev", 0)),
                }
            }

        elapsed = time.perf_counter() - t0
        print(f">> Size tuning INDEPENDENT fits finished in {elapsed/60:.2f} min ({elapsed:.1f} s)")

        loaded["size_tuning_fits"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": fits_by_id,
        }

    # --- case 2: load fits (no fitting) ---
    elif use_fits:
        fits = load_size_tuning_fits_h5(h5_file, present_ids, strict=fit_strict and strict)
        loaded["size_tuning_fits"] = fits
        if fit_strict and strict and len(fits.get("missing_ids", [])) > 0:
            raise ValueError(
                f"Missing size_tuning fits for {len(fits['missing_ids'])} neurons. "
                f"Either run fit_size_tuning first or set fit_strict=False."
            )

    # --- choose active curves for downstream usage ---
    if use_fits and ("size_tuning_fits" in loaded) and len(loaded["size_tuning_fits"].get("present_ids", [])) > 0:
        active_by_id = {}
        fits_by = loaded["size_tuning_fits"]["by_id"]
        for nid in present_ids:
            nid_i = int(nid)
            if nid_i in fits_by:
                active_by_id[nid_i] = {
                    "circular": fits_by[nid_i]["circular_yfit"],
                    "annular":  fits_by[nid_i]["annular_yfit"],
                }
            else:
                active_by_id[nid_i] = st["by_id"][nid_i]

        loaded["size_tuning_curves_active"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": active_by_id,
            "radii": radii,
            "using_fits": True,
        }
    else:
        loaded["size_tuning_curves_active"] = {**st, "using_fits": False}

    return loaded


def rog_L(x_diam, w):
    """
    L(x) = [ (2/sqrt(pi)) * ∫_0^x exp(-(y/w)^2) dy ]^2
         = [ w * erf(x/w) ]^2  (constants absorbed into kc/ks anyway)
    Using the exact integral form via erf is stable.
    """
    x = np.asarray(x_diam, dtype=float)
    w = float(w)
    return (w * erf(x / max(w, 1e-12)))**2


def rog_response(x_diam, kc, ks, wc, ws):
    """
    R(x) = kc * Lc(x) / (1 + ks * Ls(x))
    with constraint wc <= ws, ks >= 0
    """
    Lc = rog_L(x_diam, wc)
    Ls = rog_L(x_diam, ws)
    return (kc * Lc) / (1.0 + ks * Ls)


def rog_residuals(params, x_diam, y_obs):
    kc, ks, wc, ws = params
    y_hat = rog_response(x_diam, kc, ks, wc, ws)
    return (y_hat - y_obs)



def fit_rog_single_curve(
    radii, y,
    *,
    # bounds in diameter space (deg). Keep wide, but not insane.
    wc_bounds=(0.005, 80.0),
    ws_bounds=(0.005, 120.0),
    kc_bounds=(0.0, np.inf),
    ks_bounds=(0.0, np.inf),

    # multi-start controls
    n_starts=12,
    n_random_starts=6,
    random_seed=0,

    # robustness / weighting
    robust_loss="soft_l1",   # "linear" | "soft_l1" | "huber" | ...
    use_weights=True,        # mild heteroskedastic down-weighting
    weight_eps=0.05,         # larger => gentler weighting near 0
    f_scale_mode="adaptive", # "adaptive" | float
):
    """
    Fit a Ratio-of-Gaussians-like size tuning curve:
        R(x) = kc * L(x,wc) / (1 + ks * L(x,ws))
    with constraint wc <= ws enforced by parametrization:
        ws = wc + d,  d >= 0

    Inputs:
      radii: array (R,) in deg (radius)
      y:     array (R,) response
    Returns dict with:
      params: kc, ks, wc, ws
      y_fit
      success, cost, nfev
      best_start (debug)
    """
    radii = np.asarray(radii, dtype=float)
    y = np.asarray(y, dtype=float)

    x = 2.0 * radii  # diameter in deg

    # basic sanity
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"radii and y must be 1D with same length, got {x.shape} and {y.shape}")
    if len(x) < 4:
        # fitting with 3 points is astrology, not science
        return {
            "params": {"kc": float(np.nan), "ks": float(np.nan), "wc": float(np.nan), "ws": float(np.nan)},
            "y_fit": np.full_like(x, np.nan, dtype=float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    # ensure finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return {
            "params": {"kc": float(np.nan), "ks": float(np.nan), "wc": float(np.nan), "ws": float(np.nan)},
            "y_fit": np.full_like(x, np.nan, dtype=float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    # optional mild weights: down-weight very large y (heteroskedastic)
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

    # helper: model
    def rog_L(x_diam, w_):
        w_ = float(max(w_, 1e-12))
        return (w_ * erf(x_diam / w_))**2

    def rog_response(x_diam, kc, ks, wc, ws):
        Lc = rog_L(x_diam, wc)
        Ls = rog_L(x_diam, ws)
        return (kc * Lc) / (1.0 + ks * Ls)

    # params are (kc, ks, wc, d) with ws = wc + d
    # bounds for d derived from ws_bounds
    d_min = 0.0
    d_max = float(ws_bounds[1] - wc_bounds[0])
    if d_max <= 0:
        d_max = float(ws_bounds[1])

    lb = np.array([kc_bounds[0], ks_bounds[0], wc_bounds[0], d_min], dtype=float)
    ub = np.array([kc_bounds[1], ks_bounds[1], wc_bounds[1], d_max], dtype=float)

    # --------
    # Initializations (deterministic + random)
    # --------
    # robust y scale
    y_pos = np.maximum(y, 0.0)
    ymax = float(np.max(y_pos)) if np.any(np.isfinite(y_pos)) else 1.0
    half = 0.5 * ymax if np.isfinite(ymax) else 0.0

    # wc0: where curve crosses half-max (or fallback)
    if np.any(y_pos >= half) and (half > 0):
        idx = int(np.argmax(y_pos >= half))
        wc_base = float(np.clip(x[idx], wc_bounds[0], wc_bounds[1]))
    else:
        wc_base = float(np.clip(np.median(x), wc_bounds[0], wc_bounds[1]))

    # ks/kc base
    # kc scales the numerator; start so that max roughly matches ymax
    Lc_base = rog_L(x, max(wc_base, 1e-3))
    kc_base = float(ymax / max(float(np.max(Lc_base)), 1e-9))
    ks_base = 0.3

    # deterministic starts (good coverage)
    starts = []
    for wc_mul in (0.6, 1.0, 1.6):
        wc0 = float(np.clip(wc_base * wc_mul, wc_bounds[0], wc_bounds[1]))
        for ws_mul in (1.3, 2.0, 3.0):
            ws0 = float(np.clip(wc0 * ws_mul, ws_bounds[0], ws_bounds[1]))
            d0 = max(ws0 - wc0, 1e-3)
            for ks0 in (0.0, ks_base, 1.0):
                starts.append(np.array([kc_base, ks0, wc0, d0], dtype=float))

    # random starts
    rng = np.random.default_rng(int(random_seed))
    for _ in range(int(n_random_starts)):
        wc0 = float(rng.uniform(wc_bounds[0], wc_bounds[1]))
        ws0 = float(rng.uniform(max(ws_bounds[0], wc0), ws_bounds[1]))
        d0 = max(ws0 - wc0, 1e-3)
        # kc: keep positive, scale around kc_base with log jitter
        kc0 = float(kc_base * np.exp(rng.normal(0.0, 0.6)))
        ks0 = float(np.exp(rng.normal(np.log(0.3 + 1e-6), 0.8)) - 1e-6)
        ks0 = max(0.0, ks0)
        starts.append(np.array([kc0, ks0, wc0, d0], dtype=float))

    # trim / pad to n_starts (keep best coverage)
    if len(starts) > int(n_starts):
        # keep some deterministic + some random
        det = starts[:min(len(starts), int(n_starts) - int(n_random_starts))]
        rnd = starts[-int(n_random_starts):] if int(n_random_starts) > 0 else []
        starts = det + rnd
    else:
        starts = starts[:]

    # residual function
    def residuals(p):
        kc, ks, wc, d = p
        ws = wc + d
        y_hat = rog_response(x, kc, ks, wc, ws)
        r = (y_hat - y) * w
        return r

    best = None

    # run multi-start
    total_nfev = 0
    for si, p0 in enumerate(starts):
        # clip p0 into bounds to avoid least_squares complaining
        p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)

        try:
            res = least_squares(
                residuals,
                p0,
                bounds=(lb, ub),
                method="trf",
                loss=robust_loss,
                f_scale=f_scale,
                max_nfev=4000,
            )
        except Exception:
            continue

        total_nfev += int(getattr(res, "nfev", 0))

        if best is None or (res.cost < best.cost):
            best = res
            best_start = {"idx": si, "p0": p0.copy(), "cost": float(res.cost), "success": bool(res.success)}

    if best is None:
        return {
            "params": {"kc": float(np.nan), "ks": float(np.nan), "wc": float(np.nan), "ws": float(np.nan)},
            "y_fit": np.full_like(x, np.nan, dtype=float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    kc, ks, wc, d = best.x
    ws = wc + d
    y_hat = rog_response(x, kc, ks, wc, ws)

    return {
        "params": {"kc": float(kc), "ks": float(ks), "wc": float(wc), "ws": float(ws)},
        "y_fit": y_hat.astype(float),
        "success": bool(best.success),
        "cost": float(best.cost),
        "nfev": int(total_nfev),
        "best_start": best_start,
    }

def _auto_width_bounds_from_x(x):
    x = np.asarray(x, float)
    x_max = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 1.0

    xu = np.unique(x[np.isfinite(x)])
    dx = np.diff(np.sort(xu))
    dx_pos = dx[dx > 0]
    x_step = float(np.nanmin(dx_pos)) if dx_pos.size else (0.01 * x_max if x_max > 0 else 1e-3)

    wc_bounds = (max(1e-6, 0.5 * x_step), 2.0 * x_max)
    ws_bounds = (max(1e-6, 0.5 * x_step), 6.0 * x_max)
    return wc_bounds, ws_bounds


def _kc_init(x, y_pos, wc0):
    base_Lc = rog_L(x, wc0)
    denom = max(float(np.max(base_Lc)), 1e-9)
    return np.array([float(np.max(y_pos[c]) / denom) for c in range(y_pos.shape[0])], float)


def _make_model_spec(model, C, wc0, ws0, ks0, kc0s, wc_bounds, ws_bounds):
    if model == "uniform":
        if model == "uniform":
            # p = [wc, u, ks, kc0..kcC-1],  ws = wc + exp(u)
            def unpack(p):
                wc = float(p[0])
                u  = float(p[1])
                ws = wc + np.exp(u)
                ks = float(p[2])
                kc = np.asarray(p[3:3+C], float)
                return {"wc": wc, "ws": float(ws), "ks": ks, "kc": kc}

            def predict(par, x):
                wc, ws, ks, kc = par["wc"], par["ws"], par["ks"], par["kc"]
                return np.stack([rog_response(x, kc[c], ks, wc, ws) for c in range(C)], axis=0)

            def penalty(par):
                return np.zeros(0, dtype=float)  # constraint is built-in now

            u0 = np.log(max(ws0 - wc0, 1e-6))
            p0 = np.concatenate([[wc0, u0, ks0], kc0s])

            # bounds
            # wc stays as before
            # u: ensure ws stays within ws_bounds over plausible wc range
            u_lb = np.log(max(ws_bounds[0] - wc_bounds[1], 1e-6))
            u_ub = np.log(max(ws_bounds[1] - wc_bounds[0], 1e-6))

            lb = np.concatenate([[wc_bounds[0], u_lb, 0.0], np.zeros(C)])
            ub = np.concatenate([[wc_bounds[1], u_ub, np.inf], np.full(C, np.inf)])

    elif model == "gain":
            # p = [wc, ws, kc[0..], ks[0..]]
        def unpack(p):
            wc = float(p[0])
            u  = float(p[1])
            ws = wc + np.exp(u)
            kc = np.asarray(p[2:2+C], float)
            ks = np.asarray(p[2+C:2+2*C], float)
            return {"wc": wc, "ws": float(ws), "kc": kc, "ks": ks}

        def predict(par, x):
            wc, ws, kc, ks = par["wc"], par["ws"], par["kc"], par["ks"]
            return np.stack([rog_response(x, kc[c], ks[c], wc, ws) for c in range(C)], axis=0)

        def penalty(par):
            return np.zeros(0, dtype=float)

        u0 = np.log(max(ws0 - wc0, 1e-6))
        p0 = np.concatenate([[wc0, u0], kc0s, np.full(C, ks0)])

        u_lb = np.log(max(ws_bounds[0] - wc_bounds[1], 1e-6))
        u_ub = np.log(max(ws_bounds[1] - wc_bounds[0], 1e-6))

        lb = np.concatenate([[wc_bounds[0], u_lb], np.zeros(C), np.zeros(C)])
        ub = np.concatenate([[wc_bounds[1], u_ub], np.full(C, np.inf), np.full(C, np.inf)])

    elif model == "size":
        # p = [ws, kc[0..], ks[0..], wc[0..]]
        def unpack(p):
            ws = p[0]
            kc = p[1:1+C]
            ks = p[1+C:1+2*C]
            wc = p[1+2*C:1+3*C]
            return {"ws": ws, "wc": wc, "kc": kc, "ks": ks}

        def predict(par, x):
            ws, wc, kc, ks = par["ws"], par["wc"], par["kc"], par["ks"]
            return np.stack([rog_response(x, kc[c], ks[c], wc[c], ws) for c in range(C)], axis=0)

        def penalty(par):
            # vector penalty is fine, keep scalar too
            return np.array([10.0 * float(np.sum(np.maximum(0.0, par["wc"] - par["ws"])))], float)

        p0 = np.concatenate([[ws0], kc0s, np.full(C, ks0), np.full(C, wc0)])
        lb = np.concatenate([[ws_bounds[0]], np.zeros(C), np.zeros(C), np.full(C, wc_bounds[0])])
        ub = np.concatenate([[ws_bounds[1]], np.full(C, np.inf), np.full(C, np.inf), np.full(C, wc_bounds[1])])

    else:
        raise ValueError("model must be one of: uniform, gain, size")

    return {"p0": p0, "lb": lb, "ub": ub, "unpack": unpack, "predict": predict, "penalty": penalty}


def _safe_jitter(rng, p, lb, ub, frac=0.8):
    p = np.asarray(p, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    # log-jitter where positive, additive where not
    p_try = p.copy()
    pos = p > 0

    # multiplicative jitter: p * exp(N(0, frac))
    p_try[pos] = p[pos] * np.exp(rng.normal(0.0, frac, size=np.sum(pos)))

    # tiny additive jitter for zero-ish
    p_try[~pos] = p[~pos] + rng.normal(0.0, 1e-3, size=np.sum(~pos))

    return np.clip(p_try, lb, ub)


def sample_global(rng, p0, lb, ub, *, frac=1.0,
                  inf_log_hi=12.0,  # exp(12) ~ 1.6e5 range
                  inf_log_lo=6.0):  # exp(6)  ~ 403 range below anchor
    """
    Global-ish sampler that never samples uniform over infinite ranges.

    - Finite positive bounds: log-uniform in [lb, ub]
    - Finite nonpositive bounds: uniform in [lb, ub]
    - Positive with ub=+inf: log-uniform over a wide range anchored to max(p0, lb, 1e-3)
    - Anything else: normal around p0, then clip
    """
    p0 = np.asarray(p0, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    p = p0.copy()

    m = rng.random(p.shape) < frac

    finite = m & np.isfinite(lb) & np.isfinite(ub) & (ub > lb)
    pos_finite = finite & (lb > 0) & (ub > 0)
    other_finite = finite & ~pos_finite

    # 1) finite + positive -> log-uniform
    if np.any(pos_finite):
        p[pos_finite] = np.exp(rng.uniform(np.log(lb[pos_finite]), np.log(ub[pos_finite])))

    # 2) finite + other -> uniform
    if np.any(other_finite):
        p[other_finite] = rng.uniform(lb[other_finite], ub[other_finite])

    # 3) positive with ub=inf -> wide log-uniform around anchor
    pos_inf = m & (ub == np.inf) & (lb >= 0) & (p0 >= 0)
    if np.any(pos_inf):
        anchor = np.maximum.reduce([p0[pos_inf], lb[pos_inf], np.full(np.sum(pos_inf), 1e-3)])
        lo = np.maximum(anchor / np.exp(inf_log_lo), 1e-6)
        hi = anchor * np.exp(inf_log_hi)
        p[pos_inf] = np.exp(rng.uniform(np.log(lo), np.log(hi)))

    # 4) remaining dims (degenerate / weird) -> additive jitter
    rest = m & ~(finite | pos_inf)
    if np.any(rest):
        scale = np.maximum(np.abs(p0[rest]), 1e-3)
        p[rest] = p0[rest] + rng.normal(0.0, 1.0, size=np.sum(rest)) * scale

    return np.clip(p, lb, ub)


def _rmse_metrics_from_params(spec, x, Y, p, W):
    par = spec["unpack"](p)
    Yhat = spec["predict"](par, x)

    err2 = (Yhat - Y) ** 2

    rmse_c = np.sqrt(np.nanmean(err2, axis=1))          # (C,)
    mean_rmse = float(np.nanmean(rmse_c))
    peak_rmse = float(np.sqrt(np.nanmean(W * err2)))

    return mean_rmse, peak_rmse, rmse_c, Yhat


def multistart_screen_refine(
    residuals_fun, spec, x, Y, *, W,
    n_starts=40, keep=5,
    jitter=0.5, seed=0,
    screen_nfev=800, refine_nfev=10000,
    loss="linear",
    verbose=True,
):
    """
    Screen+refine multistart with printed scores.
    Prints mean RMSE for refined candidates and the winner.
    """
    import numpy as np
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)

    # accept either p0 or x0, but prefer p0 if present
    if "p0" in spec:
        p0 = np.asarray(spec["p0"], float)
    elif "x0" in spec:
        p0 = np.asarray(spec["x0"], float)
    else:
        raise KeyError("spec must contain 'p0' or 'x0'")

    lb = np.asarray(spec["lb"], float)
    ub = np.asarray(spec["ub"], float)

    if p0.shape != lb.shape or p0.shape != ub.shape:
        raise ValueError(
            f"Shape mismatch: p0 {p0.shape}, lb {lb.shape}, ub {ub.shape}"
        )

    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Non-finite values in bounds")
    if not np.all(lb < ub):
        bad = np.where(~(lb < ub))[0].tolist()
        raise ValueError(f"Invalid bounds at indices {bad}")

    span = np.maximum(ub - lb, 1e-8)
    eps = np.maximum(1e-8, 1e-6 * span)
    mid = 0.5 * (lb + ub)

    def _sanitize_start(p):
        p = np.asarray(p, float)
        if p.shape != lb.shape:
            raise ValueError(f"Bad start shape {p.shape}, expected {lb.shape}")
        p = np.where(np.isfinite(p), p, mid)
        p = np.clip(p, lb + eps, ub - eps)
        return p

    # sanitize base start first
    p0 = _sanitize_start(p0)

    starts = [p0]

    n_global = int(0.5 * (n_starts - 1))
    for _ in range(n_global):
        starts.append(_sanitize_start(sample_global(rng, p0, lb, ub, frac=1.0)))
    for _ in range((n_starts - 1) - n_global):
        starts.append(_sanitize_start(_safe_jitter(rng, p0, lb, ub, frac=jitter)))

    P = np.stack(starts, axis=0)

    if verbose:
        print("unique starts (rounded 1e-6):", len({tuple(np.round(p, 6)) for p in starts}), "/", len(starts))
        print("start std (first 12 params):", np.std(P[:, :min(12, P.shape[1])], axis=0))
        print("lb/ub hit rate:", np.mean((P <= lb + eps) | (P >= ub - eps)))

    # --- screen ---
    screened = []
    for i, p_init in enumerate(starts, 1):
        p_init = _sanitize_start(p_init)

        try:
            r = least_squares(
                residuals_fun, p_init,
                bounds=(lb, ub),
                method="trf",
                loss=loss,
                max_nfev=screen_nfev,
            )
        except ValueError as e:
            if "infeasible" in str(e).lower():
                # should not happen anymore, but keep it robust
                p_init = _sanitize_start(p_init)
                r = least_squares(
                    residuals_fun, p_init,
                    bounds=(lb, ub),
                    method="trf",
                    loss=loss,
                    max_nfev=screen_nfev,
                )
            else:
                raise

        screened.append(r)

        if verbose and (i % max(1, n_starts // 10) == 0 or i == 1 or i == n_starts):
            mean_rmse, peak_rmse, _, _ = _rmse_metrics_from_params(spec, x, Y, r.x, W)
            print(
                f"[screen {i:>3}/{n_starts}] "
                f"peak_RMSE={peak_rmse:.4g}  "
                f"mean_RMSE={mean_rmse:.4g}  "
                f"nfev={r.nfev}"
            )

    if len(screened) == 0:
        raise RuntimeError("No valid starts survived screening")

    screened.sort(key=lambda r: _rmse_metrics_from_params(spec, x, Y, r.x, W)[1])
    candidates = screened[:max(1, min(keep, len(screened)))]

    # --- refine ---
    refined = []
    for j, r0 in enumerate(candidates, 1):
        p_refine = _sanitize_start(r0.x)

        r = least_squares(
            residuals_fun, p_refine,
            bounds=(lb, ub),
            method="trf",
            loss=loss,
            max_nfev=refine_nfev,
        )

        mean_rmse, peak_rmse, rmse_c, _ = _rmse_metrics_from_params(spec, x, Y, r.x, W)
        refined.append((r, mean_rmse, peak_rmse, rmse_c))

        if verbose:
            rmse_str = ", ".join(f"{v:.3g}" for v in rmse_c)
            print(
                f"[refine {j:>2}/{len(candidates)}] cost={r.cost:.4g}  "
                f"peak_RMSE={peak_rmse:.4g}  mean_RMSE={mean_rmse:.4g}  "
                f"rmse_by_contrast=[{rmse_str}]  nfev={r.nfev}"
            )

    if len(refined) == 0:
        raise RuntimeError("No candidates survived refinement")

    refined.sort(key=lambda t: (t[2], t[1]))  # (peak_rmse, mean_rmse)
    best_res, best_mean_rmse, best_peak_rmse, best_rmse_c = refined[0]

    if verbose:
        print(
            f"[WINNER] cost={best_res.cost:.4g}  "
            f"peak_RMSE={best_peak_rmse:.4g}  mean_RMSE={best_mean_rmse:.4g}"
        )

    return best_res

def fit_rog_family(radii, Y, model="gain"):
    import numpy as np

    radii = np.asarray(radii, float).reshape(-1)
    Y = np.asarray(Y, float)

    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (C,R), got shape {Y.shape}")
    C, R = Y.shape

    if len(radii) != R:
        raise ValueError(f"len(radii)={len(radii)} must match Y.shape[1]={R}")

    if not np.all(np.isfinite(radii)):
        raise ValueError("radii contains non-finite values")

    x = 2.0 * radii
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values after diameter conversion")

    # ------------------------------------------------------------------
    # sanitize Y only for init / weighting
    # keep original Y for residuals, but avoid all-NaN init explosions
    # ------------------------------------------------------------------
    Y_init = np.asarray(Y, float).copy()
    Ypos = np.maximum(Y_init, 0.0)

    # if a whole row is non-finite, fitter cannot work
    row_has_finite = np.any(np.isfinite(Y_init), axis=1)
    if not np.all(row_has_finite):
        bad_rows = np.where(~row_has_finite)[0].tolist()
        raise ValueError(f"fit_rog_family: rows with no finite data: {bad_rows}")

    # replace NaNs in init-only copy with row median, fallback 0
    for i in range(C):
        row = Y_init[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            Y_init[i] = 0.0
            Ypos[i] = 0.0
            continue

        med = float(np.nanmedian(row[finite]))
        if not np.isfinite(med):
            med = 0.0
        Y_init[i, ~finite] = med

    Ypos = np.maximum(Y_init, 0.0)

    # ------------------------------------------------------------------
    # peak-emphasis weights in raw response space
    # ------------------------------------------------------------------
    W_x_floor = 0.6
    W_y_floor = 0.6

    ymax_c = np.nanmax(Ypos, axis=1)
    ymax_c = np.where(np.isfinite(ymax_c), ymax_c, 0.0)
    ymax_c = np.maximum(ymax_c, 1e-9)

    t_c = 0.95 * ymax_c
    sigma_frac = 0.05
    sigma_c = np.maximum(sigma_frac * ymax_c, 1e-9)

    d = np.maximum(0.0, t_c[:, None] - Ypos)
    W_y = np.exp(-0.5 * (d / sigma_c[:, None]) ** 2)
    W_y = W_y_floor + (1.0 - W_y_floor) * W_y

    # robust peak index
    idx_peak = np.nanargmax(Ypos, axis=1)
    idx_peak = np.clip(idx_peak, 0, len(x) - 1)
    x_peak = x[idx_peak]

    dx = np.diff(np.unique(np.sort(x)))
    dx_pos = dx[dx > 0]
    x_step = float(np.min(dx_pos)) if len(dx_pos) > 0 else 0.05

    sigma_x = 1.5 * x_step + 0.15 * x_peak
    sigma_x = np.maximum(sigma_x, 2.0 * x_step)
    sigma_x = np.where(np.isfinite(sigma_x), sigma_x, 2.0 * x_step)

    W_x = np.exp(-0.5 * ((x[None, :] - x_peak[:, None]) / (sigma_x[:, None] + 1e-9)) ** 2)
    W_x = W_x_floor + (1.0 - W_x_floor) * W_x

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------
    wc_bounds, ws_bounds = _auto_width_bounds_from_x(x)

    wc_bounds = tuple(map(float, wc_bounds))
    ws_bounds = tuple(map(float, ws_bounds))

    if not (np.isfinite(wc_bounds[0]) and np.isfinite(wc_bounds[1]) and wc_bounds[0] < wc_bounds[1]):
        raise ValueError(f"Invalid wc_bounds: {wc_bounds}")
    if not (np.isfinite(ws_bounds[0]) and np.isfinite(ws_bounds[1]) and ws_bounds[0] < ws_bounds[1]):
        raise ValueError(f"Invalid ws_bounds: {ws_bounds}")

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    x_med = float(np.median(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 1.0
    if not np.isfinite(x_med) or x_med <= 0:
        x_med = 1.0

    wc0 = float(np.clip(x_med * 0.3, wc_bounds[0], wc_bounds[1]))
    ws0 = float(np.clip(x_med * 1.2, ws_bounds[0], ws_bounds[1]))

    if not np.isfinite(wc0):
        wc0 = float(0.5 * (wc_bounds[0] + wc_bounds[1]))
    if not np.isfinite(ws0):
        ws0 = float(0.5 * (ws_bounds[0] + ws_bounds[1]))

    if ws0 <= wc0:
        ws0 = min(ws_bounds[1], max(ws_bounds[0], wc0 * 1.5))
        if ws0 <= wc0:
            ws0 = min(ws_bounds[1], wc0 + max(1e-3, 0.1 * (ws_bounds[1] - ws_bounds[0])))

    ks0 = 0.5
    if not np.isfinite(ks0):
        ks0 = 0.5

    kc0s = _kc_init(x, Ypos, wc0)
    kc0s = np.asarray(kc0s, float)

    if kc0s.ndim == 0:
        kc0s = np.full(C, float(kc0s), dtype=float)
    if len(kc0s) != C:
        kc0s = np.resize(kc0s, C)

    # replace non-finite kc init with row peaks / fallback
    row_peaks = np.nanmax(Ypos, axis=1)
    row_peaks = np.where(np.isfinite(row_peaks), row_peaks, 1.0)
    fallback_kc = np.maximum(row_peaks, 1e-3)

    bad_kc = ~np.isfinite(kc0s)
    kc0s[bad_kc] = fallback_kc[bad_kc]
    kc0s = np.maximum(kc0s, 1e-6)

    # ------------------------------------------------------------------
    # spec
    # ------------------------------------------------------------------
    spec = _make_model_spec(model, C, wc0, ws0, ks0, kc0s, wc_bounds, ws_bounds)

    lb = np.asarray(spec["lb"], float)
    ub = np.asarray(spec["ub"], float)

    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        # print("----- BAD SPEC BOUNDS -----")
        # print("model:", model)
        # print("C:", C)
        # print("wc0:", wc0, "ws0:", ws0, "ks0:", ks0)
        # print("kc0s:", kc0s)
        # print("wc_bounds:", wc_bounds)
        # print("ws_bounds:", ws_bounds)
        # print("lb:", lb)
        # print("ub:", ub)
        # print("---------------------------")

        # hard repair for non-finite bounds
        finite_x = x[np.isfinite(x)]
        x_max = float(np.max(finite_x)) if len(finite_x) else 10.0
        y_max = float(np.nanmax(np.maximum(Y_init, 0.0))) if np.any(np.isfinite(Y_init)) else 1.0
        y_max = max(y_max, 1.0)

        lb = np.where(np.isfinite(lb), lb, 0.0)
        ub = np.where(np.isfinite(ub), ub, np.nan)

        # replace bad upper bounds with sane finite caps
        bad_ub = ~np.isfinite(ub)
        ub[bad_ub] = np.maximum(10.0 * y_max, 10.0 * x_max)

        # if still invalid ordering, widen conservatively
        bad_order = ~(lb < ub)
        if np.any(bad_order):
            ub[bad_order] = lb[bad_order] + np.maximum(1.0, 0.1 * np.maximum(np.abs(lb[bad_order]), 1.0))

        spec["lb"] = lb
        spec["ub"] = ub

    if "p0" in spec:
        p0 = np.asarray(spec["p0"], float)
    elif "x0" in spec:
        p0 = np.asarray(spec["x0"], float)
    else:
        raise KeyError("Model spec must contain 'p0' or 'x0'")

    if lb.shape != ub.shape or p0.shape != lb.shape:
        raise ValueError(
            f"Spec bounds/init shape mismatch: p0 {p0.shape}, lb {lb.shape}, ub {ub.shape}"
        )

    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Spec bounds contain non-finite values")
    if not np.all(lb < ub):
        bad = np.where(~(lb < ub))[0].tolist()
        raise ValueError(f"Spec has invalid bounds at indices {bad}")

    # force p0 strictly inside bounds
    mid = 0.5 * (lb + ub)
    p0 = np.where(np.isfinite(p0), p0, mid)

    span = np.maximum(ub - lb, 1e-8)
    eps = np.maximum(1e-8, 1e-6 * span)
    p0 = np.clip(p0, lb + eps, ub - eps)

    spec["p0"] = p0
    spec["lb"] = lb
    spec["ub"] = ub

    # ------------------------------------------------------------------
    # residuals
    # ------------------------------------------------------------------
    def residuals(p):
        p = np.asarray(p, float)
        if not np.all(np.isfinite(p)):
            return np.full(C * R + 1, 1e6, dtype=float)

        par = spec["unpack"](p)
        Yhat = spec["predict"](par, x)  # (C,R)

        alpha = 0.5
        scales = (np.nanstd(Y_init, axis=1) + 1e-6) ** alpha
        scales = np.where(np.isfinite(scales), scales, 1.0)

        W = W_y * W_x
        finite_mask = np.isfinite(Y)

        base = np.zeros_like(Yhat, dtype=float)
        scaled_res = np.sqrt(W) * (Yhat - Y_init) / scales[:, None]
        base[finite_mask] = scaled_res[finite_mask]

        pen = spec["penalty"](par)
        pen_w = 50.0 * np.sqrt(max(np.sum(finite_mask), 1))
        return np.concatenate([base.ravel(), pen_w * np.atleast_1d(pen).ravel()])

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    res = multistart_screen_refine(
        residuals,
        spec,
        x,
        Y_init,         
        W=W_y * W_x,
        n_starts=25,
        keep=5,
        jitter=0.5,
        seed=0,
        screen_nfev=500,
        refine_nfev=10000,
        loss="linear",
        verbose=True,
    )

    par = spec["unpack"](res.x)
    Y_fit = spec["predict"](par, x)

    params = {"model": model}
    params.update({
        k: (float(v) if np.isscalar(v) else np.asarray(v, float))
        for k, v in par.items()
    })

    return {
        "params": params,
        "Y_fit": Y_fit.astype(float),
        "success": bool(res.success),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
    }


def save_size_tuning_fits_h5(h5_file, nid, *, radii, circ_fit, ann_fit, group="/size_tuning/fits"):
    """
    Stores:
      radii
      circular_yfit, annular_yfit
      attrs:
        rog_circ_kc/ks/wc/ws, success_circ, cost_circ, nfev_circ
        rog_ann_kc/ks/wc/ws,  success_ann,  cost_ann,  nfev_ann
    """
    k = neuron_key(nid)
    with h5py.File(h5_file, "a") as f:
        grp = f.require_group(group)
        ng = grp.require_group(k)

        for name in list(ng.keys()):
            del ng[name]

        ng.create_dataset("radii", data=np.asarray(radii, dtype=float))
        ng.create_dataset("circular_yfit", data=np.asarray(circ_fit["y_fit"], dtype=float))
        ng.create_dataset("annular_yfit",  data=np.asarray(ann_fit["y_fit"],  dtype=float))

        # clear old attrs (optional but keeps schema clean)
        for a in list(ng.attrs.keys()):
            del ng.attrs[a]

        # circ params
        for p, v in (circ_fit.get("params") or {}).items():
            ng.attrs[f"rog_circ_{p}"] = float(v)
        meta_c = circ_fit.get("meta", {})
        ng.attrs["success_circ"] = int(bool(meta_c.get("success", False)))
        ng.attrs["cost_circ"]    = float(meta_c.get("cost", np.nan))
        ng.attrs["nfev_circ"]    = int(meta_c.get("nfev", 0))

        # ann params
        for p, v in (ann_fit.get("params") or {}).items():
            ng.attrs[f"rog_ann_{p}"] = float(v)
        meta_a = ann_fit.get("meta", {})
        ng.attrs["success_ann"] = int(bool(meta_a.get("success", False)))
        ng.attrs["cost_ann"]    = float(meta_a.get("cost", np.nan))
        ng.attrs["nfev_ann"]    = int(meta_a.get("nfev", 0))


def load_size_tuning_fits_h5(h5_file, neuron_ids, group="/size_tuning/fits", strict=True):
    present, missing, by_id = [], [], {}

    def _read_params(attrs, prefix):
        out = {}
        for k, v in attrs.items():
            k = str(k)
            if k.startswith(prefix):
                out[k[len(prefix):]] = float(v)
        return out

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
                attrs = dict(ng.attrs)

                by_id[int(nid)] = {
                    "radii": np.asarray(ng["radii"][:], float),
                    "circular_yfit": np.asarray(ng["circular_yfit"][:], float),
                    "annular_yfit":  np.asarray(ng["annular_yfit"][:], float),
                    "params": {
                        "circular": _read_params(attrs, "rog_circ_"),
                        "annular":  _read_params(attrs, "rog_ann_"),
                    },
                    "attrs": attrs,
                }
            except Exception:
                if strict:
                    raise
                missing.append(int(nid)); continue

            present.append(int(nid))

    return {"present_ids": np.asarray(present, int), "missing_ids": np.asarray(missing, int), "by_id": by_id}


def rog_L(x_diam, w):
    x = np.asarray(x_diam, dtype=float)
    w = float(max(w, 1e-12))
    return (w * erf(x / w))**2

def rog_response_patch(x_diam, kc, ks, wc, ws):
    Lc = rog_L(x_diam, wc)
    Ls = rog_L(x_diam, ws)
    return (kc * Lc) / (1.0 + ks * Ls)

def rog_asymptote(kc, ks, wc, ws):
    # L(∞, w) = w^2
    Lc_inf = wc**2
    Ls_inf = ws**2
    return (kc * Lc_inf) / (1.0 + ks * Ls_inf)

def fit_rog_joint_circ_ann(
    radii,
    y_circ,
    y_ann,
    *,
    # bounds in diameter space
    wc_bounds=(0.005, 80.0),
    ws_bounds=(0.005, 120.0),
    kc_bounds=(0.0, float("inf")),
    ks_bounds=(0.0, float("inf")),
    b_bounds=(float("-inf"), float("inf")),

    # multistart / two-stage controls
    n_starts=30,
    keep=6,
    random_seed=0,
    screen_nfev=1200,
    refine_nfev=12000,

    # robustness
    loss="soft_l1",          # "linear" | "soft_l1" | "huber" ...
    f_scale_mode="adaptive", # "adaptive" or float

    # weighting
    use_weights=True,
    weight_eps=0.05,
    weight_clip=(0.25, 4.0),

    # peak emphasis (circular curve only)
    peak_weight=3.0,          # extra weight multiplier near the peak
    peak_sigma_steps=2.0,     # sigma in units of diameter grid step
):
    """    Two-stage joint fit of a shared-parameter ROG to circular patch + annulus data.

    Shared parameters:
      kc, ks, wc, ws   (with constraint wc <= ws)

    Model:
      patch(x) = rog_response_patch(x; kc, ks, wc, ws)
      full     = rog_asymptote(kc, ks, wc, ws)

      circ_hat = bc + patch(x)
      ann_hat  = ba + (full - patch(x))

    Notes:
      - Stage 1 fits (kc,ks,wc,ws,bc) on circular data only.
      - Stage 2 fits (kc,ks,wc,ws,bc,ba) jointly, seeded around stage 1.
      - Peak weighting is applied to circular residuals to stop the fit from
        "buying" tail error by sacrificing the peak.
    """
    import numpy as np
    from scipy.optimize import least_squares

    radii = np.asarray(radii, float)
    y_circ = np.asarray(y_circ, float)
    y_ann  = np.asarray(y_ann, float)

    x = 2.0 * radii  # diameter
    m = np.isfinite(x) & np.isfinite(y_circ) & np.isfinite(y_ann)
    x = x[m]
    y_circ = y_circ[m]
    y_ann  = y_ann[m]

    if x.size < 4:
        return {
            "params": {k: float("nan") for k in ["kc","ks","wc","ws","bc","ba"]},
            "y_fit_circ": np.full_like(x, np.nan, float),
            "y_fit_ann":  np.full_like(x, np.nan, float),
            "success": False,
            "cost": float("inf"),
            "nfev": 0,
            "best_start": None,
        }

    # --- base weights (simple, safe) ---
    if use_weights:
        w_c = 1.0 / np.sqrt(np.abs(y_circ) + weight_eps)
        w_a = 1.0 / np.sqrt(np.abs(y_ann)  + weight_eps)
        w_c = np.clip(w_c, weight_clip[0], weight_clip[1])
        w_a = np.clip(w_a, weight_clip[0], weight_clip[1])
    else:
        w_c = np.ones_like(y_circ, float)
        w_a = np.ones_like(y_ann,  float)

    # --- peak emphasis on circular curve ---
    # (works even if the curve is noisy; only uses argmax)
    y_pos = np.maximum(y_circ, 0.0)
    idx_peak = int(np.nanargmax(y_pos))
    x_peak = float(x[idx_peak])

    xu = np.unique(np.sort(x[np.isfinite(x)]))
    dx = np.diff(xu)
    dx_pos = dx[dx > 0]
    x_step = float(np.min(dx_pos)) if dx_pos.size else max(1e-3, 0.01 * float(np.nanmax(x)))

    sigma = float(max(1e-6, peak_sigma_steps * x_step))
    bump = np.exp(-0.5 * ((x - x_peak) / sigma) ** 2)
    w_c = w_c * (1.0 + float(peak_weight) * bump)

    # --- robust loss scale ---
    if f_scale_mode == "adaptive":
        yy = np.concatenate([y_circ, y_ann])
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med))) + 1e-6
        f_scale = mad
    else:
        f_scale = float(f_scale_mode)

    # ---- init heuristics (from circular curve) ----
    ymax = float(np.max(y_pos)) if np.any(np.isfinite(y_pos)) else 1.0
    half = 0.5 * ymax

    if (half > 0) and np.any(y_pos >= half):
        idx = int(np.argmax(y_pos >= half))
        wc_base = float(np.clip(x[idx], wc_bounds[0], wc_bounds[1]))
    else:
        wc_base = float(np.clip(np.median(x), wc_bounds[0], wc_bounds[1]))

    Lc_base = rog_L(x, max(wc_base, 1e-3))
    kc_base = float(ymax / max(float(np.max(Lc_base)), 1e-9))
    ks_base = 0.3

    # baselines: your data is already Δ-from-gray, so default to ~0
    bc_base = 0.0
    ba_base = 0.0

    # bounds for d keep ws within ws_bounds
    d_min = 0.0
    d_max = float(ws_bounds[1] - wc_bounds[0])
    d_max = max(d_max, 1e-3)

    # -----------------------------
    # Stage 1: circular-only fit
    # -----------------------------
    lb1 = np.array([kc_bounds[0], ks_bounds[0], wc_bounds[0], d_min, b_bounds[0]], float)
    ub1 = np.array([kc_bounds[1], ks_bounds[1], wc_bounds[1], d_max, b_bounds[1]], float)

    def residuals_circ(p):
        kc, ks, wc, d, bc = p
        ws = wc + d
        patch = rog_response_patch(x, kc, ks, wc, ws)
        yhat = bc + patch
        return (yhat - y_circ) * w_c

    rng = np.random.default_rng(int(random_seed))

    starts1 = []
    for wc_mul in (0.6, 1.0, 1.6):
        wc0 = float(np.clip(wc_base * wc_mul, wc_bounds[0], wc_bounds[1]))
        for ws_mul in (1.3, 2.0, 3.0):
            ws0 = float(np.clip(wc0 * ws_mul, ws_bounds[0], ws_bounds[1]))
            d0 = max(ws0 - wc0, 1e-3)
            for ks0 in (0.0, ks_base, 1.0):
                starts1.append(np.array([kc_base, ks0, wc0, d0, bc_base], float))

    while len(starts1) < max(8, int(0.5 * n_starts)):
        wc0 = float(rng.uniform(wc_bounds[0], wc_bounds[1]))
        ws0 = float(rng.uniform(max(ws_bounds[0], wc0), ws_bounds[1]))
        d0  = max(ws0 - wc0, 1e-3)
        kc0 = float(kc_base * np.exp(rng.normal(0.0, 0.6)))
        ks0 = float(np.exp(rng.normal(np.log(0.3 + 1e-6), 0.8)) - 1e-6)
        ks0 = max(0.0, ks0)
        bc0 = float(rng.normal(0.0, 0.02 * (np.std(y_circ) + 1e-6)))
        starts1.append(np.array([kc0, ks0, wc0, d0, bc0], float))

    best1 = None
    best1_start = None
    nfev1 = 0

    # quick screen
    screened1 = []
    for si, p0 in enumerate(starts1[:max(8, int(0.5 * n_starts))]):
        p0 = np.minimum(np.maximum(p0, lb1 + 1e-12), ub1 - 1e-12)
        try:
            r = least_squares(
                residuals_circ, p0,
                bounds=(lb1, ub1),
                method="trf",
                loss=loss,
                f_scale=f_scale,
                max_nfev=max(300, int(0.25 * screen_nfev)),
            )
        except Exception:
            continue
        screened1.append(r)

    if screened1:
        screened1.sort(key=lambda r: r.cost)
        cand1 = screened1[:max(2, min(4, len(screened1)))]
    else:
        cand1 = []

    for si, r0 in enumerate(cand1):
        try:
            r = least_squares(
                residuals_circ, r0.x,
                bounds=(lb1, ub1),
                method="trf",
                loss=loss,
                f_scale=f_scale,
                max_nfev=max(1200, int(0.6 * screen_nfev)),
            )
        except Exception:
            continue
        nfev1 += int(getattr(r, "nfev", 0))
        if (best1 is None) or (r.cost < best1.cost):
            best1 = r
            best1_start = {"idx": si, "p0": r0.x.copy(), "cost": float(r.cost), "success": bool(r.success)}

    if best1 is None:
        # fall back to something sane
        kc1, ks1, wc1, d1, bc1 = kc_base, ks_base, wc_base, max(0.5 * wc_base, 1e-3), bc_base
    else:
        kc1, ks1, wc1, d1, bc1 = map(float, best1.x)

    # -----------------------------
    # Stage 2: joint fit seeded from stage 1
    # -----------------------------
    lb2 = np.array([kc_bounds[0], ks_bounds[0], wc_bounds[0], d_min, b_bounds[0], b_bounds[0]], float)
    ub2 = np.array([kc_bounds[1], ks_bounds[1], wc_bounds[1], d_max, b_bounds[1], b_bounds[1]], float)

    def residuals_joint(p):
        kc, ks, wc, d, bc, ba = p
        ws = wc + d

        patch = rog_response_patch(x, kc, ks, wc, ws)
        full  = rog_asymptote(kc, ks, wc, ws)

        yhat_c = bc + patch
        yhat_a = ba + (full - patch)

        r_c = (yhat_c - y_circ) * w_c
        r_a = (yhat_a - y_ann)  * w_a
        return np.concatenate([r_c, r_a], axis=0)

    p_seed = np.array([kc1, ks1, wc1, d1, bc1, ba_base], float)

    # build starts: seed+jitter + some global
    starts2 = [p_seed]

    def jitter(p, frac=0.6):
        p = np.asarray(p, float)
        q = p.copy()
        # log-jitter positive params
        for idx in (0, 1, 2, 3):
            if q[idx] > 0:
                q[idx] = q[idx] * float(np.exp(rng.normal(0.0, frac)))
        # additive jitter baselines
        q[4] = q[4] + float(rng.normal(0.0, 0.02 * (np.std(y_circ) + 1e-6)))
        q[5] = q[5] + float(rng.normal(0.0, 0.02 * (np.std(y_ann)  + 1e-6)))
        return np.clip(q, lb2, ub2)

    for _ in range(max(8, int(0.6 * n_starts)) - 1):
        starts2.append(jitter(p_seed, frac=0.6))

    while len(starts2) < int(n_starts):
        wc0 = float(rng.uniform(wc_bounds[0], wc_bounds[1]))
        ws0 = float(rng.uniform(max(ws_bounds[0], wc0), ws_bounds[1]))
        d0  = max(ws0 - wc0, 1e-3)
        kc0 = float(kc_base * np.exp(rng.normal(0.0, 0.8)))
        ks0 = float(np.exp(rng.normal(np.log(0.3 + 1e-6), 1.0)) - 1e-6)
        ks0 = max(0.0, ks0)
        bc0 = float(rng.normal(0.0, 0.05 * (np.std(y_circ) + 1e-6)))
        ba0 = float(rng.normal(0.0, 0.05 * (np.std(y_ann)  + 1e-6)))
        starts2.append(np.clip(np.array([kc0, ks0, wc0, d0, bc0, ba0], float), lb2, ub2))

    # --- screen ---
    screened = []
    total_nfev = nfev1
    for p0 in starts2:
        try:
            r = least_squares(
                residuals_joint, p0,
                bounds=(lb2, ub2),
                method="trf",
                loss=loss,
                f_scale=f_scale,
                max_nfev=int(screen_nfev),
            )
        except Exception:
            continue
        screened.append(r)

    if not screened:
        return {
            "params": {k: float("nan") for k in ["kc","ks","wc","ws","bc","ba"]},
            "y_fit_circ": np.full_like(x, np.nan, float),
            "y_fit_ann":  np.full_like(x, np.nan, float),
            "success": False,
            "cost": float("inf"),
            "nfev": int(total_nfev),
            "best_start": None,
        }

    screened.sort(key=lambda r: r.cost)
    candidates = screened[:min(int(keep), len(screened))]

    # --- refine ---
    best = None
    best_start = None
    for j, r0 in enumerate(candidates):
        try:
            r = least_squares(
                residuals_joint, r0.x,
                bounds=(lb2, ub2),
                method="trf",
                loss=loss,
                f_scale=f_scale,
                max_nfev=int(refine_nfev),
            )
        except Exception:
            continue

        total_nfev += int(getattr(r, "nfev", 0))
        if (best is None) or (r.cost < best.cost):
            best = r
            best_start = {"idx": j, "p0": r0.x.copy(), "cost": float(r.cost), "success": bool(r.success)}

    if best is None:
        return {
            "params": {k: float("nan") for k in ["kc","ks","wc","ws","bc","ba"]},
            "y_fit_circ": np.full_like(x, np.nan, float),
            "y_fit_ann":  np.full_like(x, np.nan, float),
            "success": False,
            "cost": float("inf"),
            "nfev": int(total_nfev),
            "best_start": None,
        }

    kc, ks, wc, d, bc, ba = map(float, best.x)
    ws = wc + d

    patch = rog_response_patch(x, kc, ks, wc, ws)
    full  = rog_asymptote(kc, ks, wc, ws)

    yfit_c = bc + patch
    yfit_a = ba + (full - patch)

    return {
        "params": {"kc": float(kc), "ks": float(ks), "wc": float(wc), "ws": float(ws), "bc": float(bc), "ba": float(ba)},
        "y_fit_circ": yfit_c.astype(float),
        "y_fit_ann":  yfit_a.astype(float),
        "success": bool(best.success),
        "cost": float(best.cost),
        "nfev": int(total_nfev),
        "best_start": best_start,
    }
