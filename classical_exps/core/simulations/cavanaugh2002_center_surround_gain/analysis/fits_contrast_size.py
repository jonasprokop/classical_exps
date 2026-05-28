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
from .fits_size_tuning import fit_rog_family

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



def save_contrast_size_tuning_fits_h5(
    h5_file, nid, *,
    radii, contrasts,
    Y_fit,
    params: dict,
    group="/contrast_size_tuning/fits",
):
    k = neuron_key(nid)
    with h5py.File(h5_file, "a") as f:
        grp = f.require_group(group)
        ng = grp.require_group(k)

        # overwrite cleanly
        for name in list(ng.keys()):
            del ng[name]

        ng.create_dataset("radii", data=np.asarray(radii, float))
        ng.create_dataset("contrasts", data=np.asarray(contrasts, float))
        ng.create_dataset("Y_fit", data=np.asarray(Y_fit, float))  # shape (C,R)

        # params into attrs (small scalars only)
        # arrays go as datasets if needed, but attrs OK if short
        ng.attrs["model"] = str(params.get("model", ""))
        for kk, vv in params.items():
            if kk == "model":
                continue
            vv = np.asarray(vv)
            if vv.ndim == 0:
                ng.attrs[kk] = float(vv)
            else:
                # store vectors as datasets to avoid attr weirdness
                dname = f"param_{kk}"
                if dname in ng:
                    del ng[dname]
                ng.create_dataset(dname, data=vv.astype(float))


def load_contrast_size_tuning_fits_h5(
    h5_file, neuron_ids, *,
    group="/contrast_size_tuning/fits",
    strict=False,
    expected_shape=None,  # (C,R)
):
    present, missing, by_id = [], [], {}
    neuron_ids = np.asarray(neuron_ids, int)

    with h5py.File(h5_file, "r") as f:
        if group not in f:
            return {"present_ids": np.array([], int), "missing_ids": neuron_ids.copy(), "by_id": {}}
        grp = f[group]

        for nid in neuron_ids:
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid)); continue
            ng = grp[k]
            try:
                Y_fit = np.asarray(ng["Y_fit"][:], float)
                if expected_shape is not None and tuple(Y_fit.shape) != tuple(expected_shape):
                    if strict:
                        raise ValueError(f"{group}/{k} shape {Y_fit.shape} != expected {expected_shape}")
                # params
                params = dict(ng.attrs)
                # also read any param_* datasets
                for dname in ng.keys():
                    if dname.startswith("param_"):
                        params[dname.replace("param_", "")] = np.asarray(ng[dname][:], float)

                by_id[int(nid)] = {
                    "radii": np.asarray(ng["radii"][:], float),
                    "contrasts": np.asarray(ng["contrasts"][:], float),
                    "Y_fit": Y_fit,
                    "params": params,
                }
                present.append(int(nid))
            except Exception:
                if strict:
                    raise
                missing.append(int(nid))

    return {"present_ids": np.asarray(present, int), "missing_ids": np.asarray(missing, int), "by_id": by_id}


def fit_contrast_size_tuning(
    loaded: dict,
    *,
    h5_file: str,
    run: dict,
    strict: bool = True,
):
    """
    Post-loader hook for contrast-size tuning.
    Expects loaded["contrast_size_tuning"] already present.
    Adds:
      loaded["contrast_size_tuning_fits"]
      loaded["contrast_size_tuning_active"] (raw or fit)
    """
    if "contrast_size_tuning" not in loaded:
        raise KeyError("fit_contrast_size_tuning requires loaded['contrast_size_tuning'] already present.")

    do_fit   = bool(run.get("fit_contrast_size_tuning", False))
    use_fits = bool(run.get("use_contrast_size_tuning_fits", False))
    force    = bool(run.get("fit_force", False))
    fit_strict = bool(run.get("fit_strict", True))
    model = str(run.get("contrast_size_tuning_fit_model", "gain"))  # "uniform" | "gain" | "size"

    cst = loaded["contrast_size_tuning"]
    radii = np.asarray(cst["radii"], float)
    contrasts = np.asarray(cst["contrasts"], float)
    present_ids = np.asarray(cst.get("present_ids", []), int)

    expected_shape = (len(contrasts), len(radii))

    loaded.setdefault("contrast_size_tuning_fits", {"present_ids": np.array([], int), "missing_ids": present_ids.copy(), "by_id": {}})

    # ---- case 1: fit now ----
    if do_fit:
        fits_by_id = {}

        existing = load_contrast_size_tuning_fits_h5(
            h5_file, present_ids, strict=False, expected_shape=expected_shape
        )
        existing_set = set(existing.get("present_ids", []))

        t0 = time.time()
        n_fit = 0
        n_reuse = 0

        for nid in tqdm(present_ids, desc=f"Fitting contrast-size tuning ({model})", unit="neuron"):
            nid_i = int(nid)

            if (not force) and (nid_i in existing_set):
                fits_by_id[nid_i] = existing["by_id"][nid_i]
                n_reuse += 1
                continue

            Y_raw = np.asarray(cst["by_id"][nid_i], float)  # (C,R)
            fit = fit_rog_family(radii, Y_raw, model=model)  # uses your RoG family fitter

            save_contrast_size_tuning_fits_h5(
                h5_file, nid_i,
                radii=radii, contrasts=contrasts,
                Y_fit=fit["Y_fit"],
                params=fit["params"],
            )

            fits_by_id[nid_i] = {
                "radii": radii,
                "contrasts": contrasts,
                "Y_fit": np.asarray(fit["Y_fit"], float),
                "params": fit["params"],
                "success": int(bool(fit.get("success", True))),
                "cost": float(fit.get("cost", np.nan)),
            }
            n_fit += 1

        dt = time.time() - t0
        rate = (n_fit / dt) if dt > 0 else float("nan")
        print(f">> contrast_size_tuning fits: fit={n_fit}, reused={n_reuse}, total={len(present_ids)} | {dt:.2f}s | {rate:.2f} fit/s")

        loaded["contrast_size_tuning_fits"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": fits_by_id,
        }

    # ---- case 2: load fits only ----
    elif use_fits:
        fits = load_contrast_size_tuning_fits_h5(
            h5_file, present_ids, strict=fit_strict and strict, expected_shape=expected_shape
        )
        loaded["contrast_size_tuning_fits"] = fits
        if fit_strict and strict and len(fits.get("missing_ids", [])) > 0:
            raise ValueError(
                f"Missing contrast_size_tuning fits for {len(fits['missing_ids'])} neurons. "
                f"Run fit_contrast_size_tuning first or set fit_strict=False."
            )

    # ---- choose active ----
    if use_fits and len(loaded["contrast_size_tuning_fits"].get("present_ids", [])) > 0:
        active_by_id = {}
        fits_by = loaded["contrast_size_tuning_fits"]["by_id"]
        for nid in present_ids:
            nid_i = int(nid)
            if nid_i in fits_by:
                active_by_id[nid_i] = fits_by[nid_i]["Y_fit"]
            else:
                active_by_id[nid_i] = cst["by_id"][nid_i]
        loaded["contrast_size_tuning_active"] = {
            "present_ids": present_ids.copy(),
            "missing_ids": np.array([], int),
            "by_id": active_by_id,
            "radii": radii,
            "contrasts": contrasts,
            "using_fits": True,
        }
    else:
        loaded["contrast_size_tuning_active"] = {**cst, "using_fits": False}

    return loaded


def build_loaded_contrast_size_tuning_from_scraped(
    scraped_contrast_size_tuning,
    *,
    neuron_id_parser=None,
    sort_contrast_numeric=True,
    round_radius_decimals=6,
    atol_radius=1e-6,
):
    """
    Convert scraped contrast_size_tuning dict into the internal loaded format.

    Input format:
        scraped_contrast_size_tuning = {
            "cell1": {
                "1":    {"radius": [...], "response": [...]},
                "0.50": {"radius": [...], "response": [...]},
                ...
            },
            ...
        }

    Output:
        loaded = {
            "contrast_size_tuning": {
                "radii": np.ndarray,          # shared union radius grid
                "contrasts": np.ndarray,      # sorted numeric if possible
                "by_id": {
                    neuron_id: np.ndarray shape (C,R)
                }
            },
            "contrast_size_tuning_results": {
                "by_id": {
                    neuron_id: {"shift": ...}
                }
            }
        }

    Notes:
    - contrast keys are interpreted numerically
    - radius values are merged across the cell, then each curve is placed on the union grid
    - missing entries become np.nan
    """
    if neuron_id_parser is None:
        neuron_id_parser = lambda x: x

    by_id = {}
    results_by_id = {}

    global_contrasts = None
    global_radii = None

    per_cell_contrasts = {}
    per_cell_radii = {}

    for raw_neuron_id, cdict in scraped_contrast_size_tuning.items():
        neuron_id = neuron_id_parser(raw_neuron_id)

        contrast_keys = list(cdict.keys())
        contrast_vals = np.asarray([float(k) for k in contrast_keys], dtype=float)

        if sort_contrast_numeric:
            order_c = np.argsort(contrast_vals)
            contrast_vals = contrast_vals[order_c]
            contrast_keys = [contrast_keys[i] for i in order_c]

        per_cell_contrasts[neuron_id] = contrast_vals

        merged_radii = []
        for ck in contrast_keys:
            rr = np.asarray(cdict[ck]["radius"], dtype=float)
            merged_radii.extend(rr[np.isfinite(rr)].tolist())

        if len(merged_radii) == 0:
            per_cell_radii[neuron_id] = np.array([], dtype=float)
        else:
            merged_radii = np.asarray(merged_radii, dtype=float)
            merged_radii = np.round(merged_radii, int(round_radius_decimals))
            merged_radii = np.unique(merged_radii)
            merged_radii = np.sort(merged_radii)
            per_cell_radii[neuron_id] = merged_radii

    # check if global common contrast grid exists
    all_contrast_arrays = list(per_cell_contrasts.values())
    if len(all_contrast_arrays) > 0:
        first_c = all_contrast_arrays[0]
        same_contrasts = all(
            len(cc) == len(first_c) and np.allclose(cc, first_c, atol=0.0, rtol=0.0)
            for cc in all_contrast_arrays[1:]
        )
        if same_contrasts:
            global_contrasts = first_c.copy()

    # check if global common radius grid exists
    all_radius_arrays = list(per_cell_radii.values())
    if len(all_radius_arrays) > 0:
        first_r = all_radius_arrays[0]
        same_radii = all(
            len(rr) == len(first_r) and np.allclose(rr, first_r, atol=0.0, rtol=0.0)
            for rr in all_radius_arrays[1:]
        )
        if same_radii:
            global_radii = first_r.copy()

    for raw_neuron_id, cdict in scraped_contrast_size_tuning.items():
        neuron_id = neuron_id_parser(raw_neuron_id)

        contrast_keys = list(cdict.keys())
        contrast_vals = np.asarray([float(k) for k in contrast_keys], dtype=float)

        if sort_contrast_numeric:
            order_c = np.argsort(contrast_vals)
            contrast_vals = contrast_vals[order_c]
            contrast_keys = [contrast_keys[i] for i in order_c]

        if global_contrasts is None:
            cell_contrasts = contrast_vals
        else:
            cell_contrasts = global_contrasts

        if global_radii is None:
            cell_radii = per_cell_radii[neuron_id]
        else:
            cell_radii = global_radii

        Y = np.full((len(cell_contrasts), len(cell_radii)), np.nan, dtype=float)

        contrast_to_row = {float(c): i for i, c in enumerate(cell_contrasts)}

        for ck, cv in zip(contrast_keys, contrast_vals):
            row = contrast_to_row[float(cv)]

            rr = np.asarray(cdict[ck]["radius"], dtype=float)
            yy = np.asarray(cdict[ck]["response"], dtype=float)

            if rr.ndim != 1 or yy.ndim != 1 or len(rr) != len(yy):
                raise ValueError(
                    f"Bad scraped contrast-size curve for neuron {neuron_id}, contrast {ck}: "
                    f"radius shape {rr.shape}, response shape {yy.shape}"
                )

            rr_round = np.round(rr, int(round_radius_decimals))

            for r_val, y_val in zip(rr_round, yy):
                idx = np.where(np.isclose(cell_radii, r_val, atol=atol_radius, rtol=0.0))[0]
                if len(idx) == 0:
                    continue
                Y[row, int(idx[0])] = float(y_val)

        by_id[neuron_id] = Y

        # optional shift estimate from raw matrix at lowest/highest contrast
        shift = np.nan
        try:
            low_curve = Y[0]
            high_curve = Y[-1]

            GSF_low, *_ = get_GSF_surround_AMRF(
                radii=cell_radii,
                circular_tuning_curve=low_curve,
                annular_tuning_curve=None,
            )
            GSF_high, *_ = get_GSF_surround_AMRF(
                radii=cell_radii,
                circular_tuning_curve=high_curve,
                annular_tuning_curve=None,
            )

            if (
                GSF_low is not None and GSF_high is not None and
                np.isfinite(GSF_low) and np.isfinite(GSF_high) and
                GSF_high > 0
            ):
                shift = float(GSF_low / GSF_high)
        except Exception:
            shift = np.nan

        results_by_id[neuron_id] = {"shift": shift}

    if global_contrasts is None:
        merged_contrasts = []
        for cc in per_cell_contrasts.values():
            merged_contrasts.extend(cc.tolist())
        contrasts_out = np.unique(np.asarray(merged_contrasts, dtype=float))
        contrasts_out = np.sort(contrasts_out)
    else:
        contrasts_out = global_contrasts

    if global_radii is None:
        merged_radii = []
        for rr in per_cell_radii.values():
            merged_radii.extend(rr.tolist())
        radii_out = np.unique(np.asarray(merged_radii, dtype=float))
        radii_out = np.sort(radii_out)
    else:
        radii_out = global_radii

    return {
        "contrast_size_tuning": {
            "radii": np.asarray(radii_out, dtype=float),
            "contrasts": np.asarray(contrasts_out, dtype=float),
            "by_id": by_id,
        },
        "contrast_size_tuning_results": {
            "by_id": results_by_id,
        },
    }


def build_loaded_contrast_size_tuning_from_scraped_config(
    nature_and_interactions_scraped_data,
    *,
    raw_key="contrast_size_tuning",
    fit_key="contrast_size_tuning_fit",
    neuron_id_parser=None,
    sort_contrast_numeric=True,
    round_radius_decimals=6,
    atol_radius=1e-6,
):
    """
    Build a loaded-like contrast_size_tuning object from scraped config.

    Returns a hybrid structure:
      - matrix representation in loaded_scraped["contrast_size_tuning"] for compatibility / sorting / metrics
      - original per-contrast scraped representation in loaded_scraped["contrast_size_tuning_scraped_raw"]
      - optional fitted per-contrast scraped representation in loaded_scraped["contrast_size_tuning_scraped_fit"]
    """
    if nature_and_interactions_scraped_data is None:
        return None
    if raw_key not in nature_and_interactions_scraped_data:
        return None

    raw_src = nature_and_interactions_scraped_data[raw_key]

    loaded_raw = build_loaded_contrast_size_tuning_from_scraped(
        raw_src,
        neuron_id_parser=neuron_id_parser,
        sort_contrast_numeric=sort_contrast_numeric,
        round_radius_decimals=round_radius_decimals,
        atol_radius=atol_radius,
    )

    loaded_scraped = {
        "contrast_size_tuning": loaded_raw["contrast_size_tuning"],
        "contrast_size_tuning_results": loaded_raw.get("contrast_size_tuning_results", {"by_id": {}}),
        "contrast_size_tuning_scraped_raw": raw_src,
    }

    if fit_key in nature_and_interactions_scraped_data:
        fit_src = nature_and_interactions_scraped_data[fit_key]
        loaded_scraped["contrast_size_tuning_scraped_fit"] = fit_src

        # optional matrix fit compatibility path
        loaded_fit = build_loaded_contrast_size_tuning_from_scraped(
            fit_src,
            neuron_id_parser=neuron_id_parser,
            sort_contrast_numeric=sort_contrast_numeric,
            round_radius_decimals=round_radius_decimals,
            atol_radius=atol_radius,
        )

        raw_cst = loaded_scraped["contrast_size_tuning"]
        fit_cst = loaded_fit["contrast_size_tuning"]

        raw_ids = set(raw_cst["by_id"].keys())
        fit_ids = set(fit_cst["by_id"].keys())
        common_ids = raw_ids & fit_ids

        fits_by_id = {}
        for nid in common_ids:
            Y_raw = np.asarray(raw_cst["by_id"][nid], dtype=float)
            Y_fit = np.asarray(fit_cst["by_id"][nid], dtype=float)

            if Y_raw.shape != Y_fit.shape:
                raise ValueError(
                    f"Scraped contrast-size fit shape mismatch for {nid}: "
                    f"raw {Y_raw.shape} vs fit {Y_fit.shape}"
                )

            fits_by_id[nid] = {"Y_fit": Y_fit}

        loaded_scraped["contrast_size_tuning_fits"] = {"by_id": fits_by_id}

    summary_key = f"{fit_key}_summary"
    if summary_key in nature_and_interactions_scraped_data:
        summary_src = nature_and_interactions_scraped_data[summary_key]
        summary_out = {}
        for nid, val in summary_src.items():
            parsed_id = neuron_id_parser(nid) if neuron_id_parser is not None else nid
            summary_out[parsed_id] = val
        loaded_scraped["contrast_size_tuning_fit_summary"] = summary_out

    return loaded_scraped

def add_fitted_contrast_size_tuning_to_scraped_config(
    nature_and_interactions_scraped_data: dict,
    *,
    run: dict,
    strict: bool = True,
):
    """
    Fit scraped contrast_size_tuning data using the same RoG family settings as model data,
    then write fitted curves back into the scraped config on each raw curve's original radius grid.

    Adds:
        nature_and_interactions_scraped_data["contrast_size_tuning_fit"]
        nature_and_interactions_scraped_data["contrast_size_tuning_fit_summary"]
    """
    if nature_and_interactions_scraped_data is None:
        return nature_and_interactions_scraped_data
    if "contrast_size_tuning" not in nature_and_interactions_scraped_data:
        return nature_and_interactions_scraped_data

    scraped = nature_and_interactions_scraped_data["contrast_size_tuning"]

    loaded_scraped = build_loaded_contrast_size_tuning_from_scraped(scraped)

    radii = np.asarray(loaded_scraped["contrast_size_tuning"]["radii"], dtype=float)
    contrasts = np.asarray(loaded_scraped["contrast_size_tuning"]["contrasts"], dtype=float)
    by_id = loaded_scraped["contrast_size_tuning"]["by_id"]

    model = str(run.get("contrast_size_tuning_fit_model", "gain"))

    fitted_cfg = {}
    fit_summary = {}

    for cell_id, Y in by_id.items():
        Y = np.asarray(Y, dtype=float)
        expected_shape = (len(contrasts), len(radii))

        if tuple(Y.shape) != tuple(expected_shape):
            if strict:
                raise ValueError(
                    f"Scraped contrast_size_tuning cell {cell_id} has shape {Y.shape}, expected {expected_shape}"
                )
            continue

        # Build a dense matrix for fitting:
        # for each contrast row, interpolate observed scraped points onto the union radius grid.
        Y_fit_input = np.full_like(Y, np.nan, dtype=float)

        raw_cell = scraped[cell_id]
        contrast_keys_sorted = sorted(raw_cell.keys(), key=lambda k: float(k))
        contrast_to_row = {float(c): i for i, c in enumerate(contrasts)}

        for c_key in contrast_keys_sorted:
            c_val = float(c_key)
            if c_val not in contrast_to_row:
                raise ValueError(
                    f"Cell {cell_id}: contrast {c_val} not found in fitted contrast grid {contrasts}"
                )
            i = contrast_to_row[c_val]

            raw_entry = raw_cell[c_key]
            r_obs = np.asarray(raw_entry["radius"], dtype=float)
            y_obs = np.asarray(raw_entry["response"], dtype=float)

            # collapse duplicate radii safely
            r_unique = []
            y_unique = []
            for rv in np.unique(r_obs):
                m = np.isclose(r_obs, rv, atol=1e-12, rtol=0.0)
                r_unique.append(float(rv))
                y_unique.append(float(np.mean(y_obs[m])))

            r_unique = np.asarray(r_unique, dtype=float)
            y_unique = np.asarray(y_unique, dtype=float)

            if len(r_unique) == 1:
                Y_fit_input[i] = np.full_like(radii, y_unique[0], dtype=float)
            else:
                inside = (radii >= np.min(r_unique)) & (radii <= np.max(r_unique))
                Y_fit_input[i, inside] = np.interp(radii[inside], r_unique, y_unique)

        try:
            fit = fit_rog_family(radii, Y_fit_input, model=model)
        except Exception as e:
            if strict:
                raise
            fit_summary[cell_id] = {
                "success": 0,
                "cost": float("nan"),
                "params": {"model": model, "error": repr(e)},
            }
            continue

        Y_fit = np.asarray(fit["Y_fit"], dtype=float)
        params = dict(fit.get("params", {}))
        params["model"] = str(params.get("model", model))

        raw_cell = scraped[cell_id]
        contrast_keys_sorted = sorted(raw_cell.keys(), key=lambda k: float(k))

        if len(contrast_keys_sorted) != len(contrasts):
            raise ValueError(
                f"Cell {cell_id}: raw contrast-key count {len(contrast_keys_sorted)} "
                f"!= fitted contrast count {len(contrasts)}"
            )

        cell_fit = {}

        for c_key in contrast_keys_sorted:
            c_val = float(c_key)
            if c_val not in contrast_to_row:
                raise ValueError(
                    f"Cell {cell_id}: contrast {c_val} not found in fitted contrast grid {contrasts}"
                )
            i = contrast_to_row[c_val]

            raw_entry = raw_cell[c_key]
            r_raw = np.asarray(raw_entry["radius"], dtype=float)

            valid = np.isfinite(Y_fit[i]) & np.isfinite(radii)
            r_fit_grid = radii[valid].astype(float)
            y_fit_grid = Y_fit[i, valid].astype(float)

            y_fit_on_raw = np.interp(
                r_raw,
                r_fit_grid,
                y_fit_grid,
                left=np.nan,
                right=np.nan,
            )

            cell_fit[c_key] = {
                "radius": r_raw.astype(float).tolist(),
                "response": y_fit_on_raw.astype(float).tolist(),
            }

        fitted_cfg[cell_id] = cell_fit
        fit_summary[cell_id] = {
            "success": int(bool(fit.get("success", True))),
            "cost": float(fit.get("cost", np.nan)),
            "params": params,
        }

    nature_and_interactions_scraped_data["contrast_size_tuning_fit"] = fitted_cfg
    nature_and_interactions_scraped_data["contrast_size_tuning_fit_summary"] = fit_summary

    return nature_and_interactions_scraped_data


