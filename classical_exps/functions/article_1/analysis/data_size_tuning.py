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

def load_size_tuning_results_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/size_tuning/results",
    strict: bool = True,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    by_id = {}

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")
        grp = f[group_path]

        for nid in neuron_ids:
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid))
                continue

            arr = np.asarray(grp[k][:], dtype=float)
            try:
                arr = np.asarray(grp[k][:], dtype=float).reshape(-1)

                GSF, surr_ext, AMRF, SI = map(float, arr)

            except Exception as e:
                if strict:
                    raise ValueError(f"Bad array for {group_path}/{k}, shape={arr.shape}") from e
                missing.append(int(nid))
                continue

            present.append(int(nid))
            by_id[int(nid)] = {"GSF": GSF, "surround_extent": surr_ext, "AMRF": AMRF, "SI": SI}

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": by_id,
    }


def load_size_tuning_curves_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/size_tuning/curves",
    strict: bool = True,
    expected_len: int = None,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    circular_by_id, annular_by_id = {}, {}
    by_id = {}

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")
        grp = f[group_path]

        for nid in neuron_ids:
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid))
                continue

            arr = np.asarray(grp[k][:], dtype=float)
            if arr.ndim != 2 or arr.shape[0] != 2:
                if strict:
                    raise ValueError(f"{group_path}/{k} expected shape (2,R), got {arr.shape}")
                missing.append(int(nid))
                continue

            if expected_len is not None and arr.shape[1] != expected_len:
                if strict:
                    raise ValueError(
                        f"{group_path}/{k} length mismatch: got R={arr.shape[1]}, expected {expected_len}"
                    )
                missing.append(int(nid))
                continue

            present.append(int(nid))
            c = arr[0].copy()
            a = arr[1].copy()
            circular_by_id[int(nid)] = c
            annular_by_id[int(nid)]  = a
            by_id[int(nid)] = {"circular": c, "annular": a}

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "circular_by_id": circular_by_id,
        "annular_by_id": annular_by_id,   
        "by_id": by_id,                   
        "group_path": group_path,
    }


def plot_size_tuning_curves_bulk(
    loaded,
    *,
    save_dir="/project/results/nature_and_interactions/size_tuning/plots/",
    max_neurons=None,
    use_diameter=False,
    h5_file=None,          # <-- add this
    load_if_missing=True,  # <-- and this
):
    os.makedirs(save_dir, exist_ok=True)

    raw = loaded["size_tuning_curves_raw"]
    radii = np.asarray(raw["radii"], dtype=float)

    # IMPORTANT: keep x consistent with your marker units (radii stored, x plotted as diameter optionally)
    x = (2.0 * radii) if use_diameter else radii
    xlab = "Diameter (deg)" if use_diameter else "Radius (deg)"

    fits = loaded.get("size_tuning_fits", {"by_id": {}})["by_id"]
    active = loaded.get("size_tuning_curves_active", raw)
    active_by = active.get("by_id", raw["by_id"])

    # ---- markers source ----
    if "size_results" not in loaded and load_if_missing:
        if h5_file is None:
            raise ValueError("plot_size_tuning_curves_bulk: need h5_file to load size_results if missing.")
        neuron_ids = np.asarray(list(raw["by_id"].keys()), dtype=int)
        loaded["size_results"] = load_size_tuning_results_bulk(h5_file, neuron_ids, strict=False)

    res_by = loaded.get("size_results", {}).get("by_id", {})

    ids = list(raw["by_id"].keys())
    if max_neurons is not None:
        ids = ids[:int(max_neurons)]

    for nid in ids:
        nid = int(nid)
        neuron = neuron_key(nid)

        circ_raw = np.asarray(raw["by_id"][nid]["circular"], dtype=float)
        ann_raw  = np.asarray(raw["by_id"][nid]["annular"], dtype=float)

        has_fit = nid in fits
        circ_fit = np.asarray(fits[nid]["circular_yfit"], dtype=float) if has_fit else None
        ann_fit  = np.asarray(fits[nid]["annular_yfit"], dtype=float) if has_fit else None

        # markers (stored in radii units in H5)
        rr = res_by.get(nid, {})
        GSF  = rr.get("GSF", None)
        AMRF = rr.get("AMRF", None)
        surr = rr.get("surround_extent", rr.get("surround", None))
        SI   = rr.get("SI", None)

        fig, ax = plt.subplots(figsize=(8.6, 4.6))

        ax.plot(x, circ_raw, ls="None", marker="o", ms=3.5, alpha=0.9, label="Center raw")
        ax.plot(x, ann_raw,  ls="None", marker="o", ms=3.5, alpha=0.9, label="Annular raw")

        if circ_fit is not None:
            ax.plot(x, circ_fit, lw=2.2, label="Center fit")
        if ann_fit is not None:
            ax.plot(x, ann_fit, lw=2.2, ls="--", label="Annular fit")

        def vline(rad, label, color):
            if rad is None:
                return
            rad = float(rad)
            if not np.isfinite(rad) or rad <= 0:
                return
            xv = (2.0 * rad) if use_diameter else rad
            ax.axvline(xv, ls=":", lw=2, label=label, color=color)

        vline(GSF,  "GSF",            color="red")
        vline(AMRF, "AMRF",           color="green")
        vline(surr, "Surround extent",color="blue")

        title = f"{neuron}"
        if SI is not None and np.isfinite(SI):
            title += f" | SI={float(SI):.2f}"
        title += " | markers: FIT" if active.get("using_fits", False) else " | markers: RAW"
        ax.set_title(title)

        ax.set_xlabel(xlab)
        ax.set_ylabel("Response (Δ from gray)")
        ax.legend(frameon=False, ncols=2)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        out = os.path.join(save_dir, f"{neuron}_size_tuning_raw_points_fit_lines.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)