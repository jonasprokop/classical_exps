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
from .fits_size_tuning import fit_rog_family

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
