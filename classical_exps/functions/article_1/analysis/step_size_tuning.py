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
from classical_exps.functions.article_1.common.metrics_size import (
    get_GSF_surround_AMRF,
    get_GSF_surround_AMRF_from_fit,
)
from .shared import plot_scatter_hist, neuron_key, ensure_dir
from .data_size_tuning import load_size_tuning_results_bulk
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


def size_tuning_results_1(
    *,
    loaded,
    h5_file,
    neuron_ids,
    filtered_neuron_ids,
    overwrite=False,
    run_mode="both",     # "raw", "fit", "both"
    plot_mode="both",    # "none", "raw", "fit", "both"
    plot_dir="/project/results/nature_and_interactions/size_tuning/plots/",
):
    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    # Require curves
    if "size_tuning_curves_active" not in loaded:
        raise KeyError("loaded missing 'size_tuning_curves_active' (run load_size_tuning_curves).")

    st = loaded["size_tuning_curves_active"]
    radii = np.asarray(st["radii"], float)
    by_curve = st["by_id"]

    # Fits optional (only used if needed)
    fits_by = loaded.get("size_tuning_fits", {}).get("by_id", {})

    group_path_size_tuning = "/size_tuning"
    subgroup_raw = group_path_size_tuning + "/results_raw"
    subgroup_fit = group_path_size_tuning + "/results_fit"

    if overwrite:
        if run_mode in ("raw", "both"):
            clear_group(h5_file, subgroup_raw)
        if run_mode in ("fit", "both"):
            clear_group(h5_file, subgroup_fit)

    if run_mode in ("raw", "both"):
        group_init(h5_file=h5_file, group_path=subgroup_raw, group_args_str="")
    if run_mode in ("fit", "both"):
        group_init(h5_file=h5_file, group_path=subgroup_fit, group_args_str="")

    # For global scatter summaries (do separately for raw and fit)
    all_raw_GSF, all_raw_surr = [], []
    all_fit_GSF, all_fit_surr = [], []

    with h5py.File(h5_file, "a") as f:
        grp_raw = f.require_group(subgroup_raw) if run_mode in ("raw", "both") else None
        grp_fit = f.require_group(subgroup_fit) if run_mode in ("fit", "both") else None

        for nid in filtered_neuron_ids:
            nid = int(nid)
            if nid not in by_curve:
                continue

            circ = np.asarray(by_curve[nid]["circular"], float)
            ann  = np.asarray(by_curve[nid].get("annular"), float) if "annular" in by_curve[nid] else None

            raw_markers = None
            fit_markers = None

            # ---------- RAW metrics ----------
            if run_mode in ("raw", "both"):
                if ann is not None:
                    GSF_r, surr_r, AMRF_r, SI_r, Ropt_r, Rsupp_r = get_GSF_surround_AMRF(
                        radii=radii,
                        circular_tuning_curve=circ,
                        annular_tuning_curve=ann,
                    )
                else:
                    GSF_r, surr_r, SI_r, Ropt_r, Rsupp_r = get_GSF_surround_AMRF(
                        radii=radii,
                        circular_tuning_curve=circ,
                        annular_tuning_curve=None,
                    )
                    AMRF_r = float("nan")

                raw_markers = {
                    "GSF": float(GSF_r),
                    "surround_extent": float(surr_r),
                    "AMRF": float(AMRF_r) if np.isfinite(AMRF_r) else float("nan"),
                    "SI": float(SI_r),
                }

                k = neuron_key(nid)
                if k in grp_raw:
                    del grp_raw[k]
                grp_raw.create_dataset(k, data=np.asarray([GSF_r, surr_r, AMRF_r, SI_r], float))

                if np.isfinite(GSF_r) and np.isfinite(surr_r):
                    all_raw_GSF.append(float(GSF_r))
                    all_raw_surr.append(float(surr_r))

            # ---------- FIT metrics ----------
            circular_fit = None
            annular_fit = None
            if run_mode in ("fit", "both"):
                if nid in fits_by:
                    fit_blob = fits_by[nid]

                    # Prefer sampled fit curves, else fallback to raw
                    circular_fit = np.asarray(fit_blob.get("circular_fit", circ), float)
                    if ann is not None:
                        annular_fit = np.asarray(fit_blob.get("annular_fit", ann), float)

                    params_raw = fit_blob.get("params", fit_blob.get("attrs", {}))

                    fit_params = {}
                    for kk in ("kc", "ks", "wc", "ws"):
                        if kk in params_raw:
                            fit_params[kk] = float(params_raw[kk])
                        elif f"rog_{kk}" in params_raw:
                            fit_params[f"rog_{kk}"] = float(params_raw[f"rog_{kk}"])
                        else:
                            # If your from_fit truly doesn't need params, replace this with `fit_params=None`
                            raise KeyError(f"Missing {kk}/rog_{kk} in size_tuning_fits params for neuron {nid}")

                    GSF_f, surr_f, AMRF_f, SI_f, Ropt_f, Rsupp_f = get_GSF_surround_AMRF_from_fit(
                        radii=radii,
                        circular_fit=circular_fit,
                        annular_fit=annular_fit,
                        fit_params=fit_params,
                    )

                    fit_markers = {
                        "GSF": float(GSF_f),
                        "surround_extent": float(surr_f),
                        "AMRF": float(AMRF_f),
                        "SI": float(SI_f),
                    }

                    k = neuron_key(nid)
                    if k in grp_fit:
                        del grp_fit[k]
                    grp_fit.create_dataset(k, data=np.asarray([GSF_f, surr_f, AMRF_f, SI_f], float))

                    if np.isfinite(GSF_f) and np.isfinite(surr_f):
                        all_fit_GSF.append(float(GSF_f))
                        all_fit_surr.append(float(surr_f))
                else:
                    # no fit for this neuron: just leave fit_markers None
                    pass

            # ---------- Plotting ----------
            if plot_mode != "none":
                do_raw = plot_mode in ("raw", "both") and raw_markers is not None
                do_fit = plot_mode in ("fit", "both") and fit_markers is not None

                if do_raw or do_fit:
                    plot_size_tuning_curve_with_markers_both(
                        neuron=str(nid),
                        radii=radii,
                        circular_curve_raw=circ,
                        annular_curve_raw=ann,
                        circular_curve_fit=circular_fit if do_fit else None,
                        annular_curve_fit=annular_fit if do_fit else None,
                        raw_markers=raw_markers if do_raw else None,
                        fit_markers=fit_markers if do_fit else None,
                        save_dir=plot_dir,
                        use_diameter=True,
                    )

    # ---------- Summary scatter(s) ----------
    def _scatter_summary(all_GSF, all_surr, label):
        all_GSF = np.asarray(all_GSF, float)
        all_surr = np.asarray(all_surr, float)
        all_GSF_d = 2.0 * all_GSF
        all_surr_d = 2.0 * all_surr
        n_new = len(all_GSF_d)

        mean_GSF = float(np.nanmean(all_GSF_d)) if n_new else float("nan")
        mean_surr = float(np.nanmean(all_surr_d)) if n_new else float("nan")
        mean_ratio = float(np.nanmean(all_surr_d / all_GSF_d)) if n_new else float("nan")

        print("--------------------------------------")
        print(f"GSF vs surround extent ({label})")
        print(f"    > Analysis made on {len(neuron_ids)} neurons")
        print(f"    > {n_new} neurons with finite GSF+surround extent")
        print(f"    > mean(GSF_diam)={mean_GSF:.3f}  mean(surr_diam)={mean_surr:.3f}  mean(ratio)={mean_ratio:.3f}")
        print("    > Plot :")

        plot_scatter_hist(
            x=all_GSF_d,
            y=all_surr_d,
            title=f"[{label}] GSF diameter vs surround extent diameter for {n_new} neurons",
            x_label="GSF diameter (deg)",
            y_label="surround diameter (deg)",
        )
        print("--------------------------------------\n")

    if run_mode in ("raw", "both"):
        _scatter_summary(all_raw_GSF, all_raw_surr, "raw")

    if run_mode in ("fit", "both"):
        _scatter_summary(all_fit_GSF, all_fit_surr, "fit")

    # attach fresh results in memory for downstream
    loaded["size_results_raw"] = load_size_tuning_results_bulk(h5_file, neuron_ids, strict=False)
    # If your bulk loader reads only one group, you may want a second loader call for /results_fit
    # For now, keep raw as the default downstream payload:
    loaded["size_results"] = loaded["size_results_raw"]

    return loaded


def size_tuning_results_2(
    neuron_ids,
    filtered_neuron_ids=None,   
    size_results=None,          
):
    
    ''' This function aims to visualize the distribution of the suppression index in the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean SI ...)
            - 3) It plots an histogram of the neuron SI values

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - see 'size_tuning_results_1'
    '''
    
    if filtered_neuron_ids is None:
        raise ValueError("size_tuning_results_2 requires filtered_neuron_ids (computed once in relay).")
    if size_results is None or "by_id" not in size_results:
        raise ValueError("size_tuning_results_2 requires size_results from bulk loader.")

    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)
    by_id = size_results["by_id"]

    # For each neuron, get SI (legacy index [-1], now via loader)
    all_SI = []
    for neuron_id in filtered_neuron_ids:
        d = by_id.get(int(neuron_id))
        if d is None:
            raise RuntimeError(
                f"Neuron {neuron_id} missing in size_results loader. "
                "Loader/filter mismatch — fix upstream."
            )
        all_SI.append(d["SI"])

    all_SI = np.array(all_SI)

    n = len(neuron_ids)
    n_new = len(all_SI)
    mean_SI = round(np.mean(all_SI), 2)
    max_SI = round(max(all_SI), 1)
    bins = np.linspace(0, max(max_SI, 1), 6)

    # Show the results (unchanged)
    print("--------------------------------------")
    print("Distribution of the Suppression Index :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean value of the SI is {mean_SI}")
    print(f"    > Plot :")

    weights = np.ones(all_SI.shape) / len(all_SI)
    plt.hist(all_SI, bins=bins, edgecolor='black', density=False, weights=weights)
    plt.xlabel("Suppression Index (SI)")
    plt.ylabel('Distribution')
    plt.xticks(bins)
    plt.title(f"Distribution of the SI for the {len(all_SI)} neurons")

    directory = f"/project/results/nature_and_interactions" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"distribution_of_si.png")
    plt.close()

    print("--------------------------------------")
    print()


def plot_size_tuning_curve_with_markers(
    neuron: str,
    radii,
    circular_curve_raw,
    annular_curve_raw=None,
    circular_curve_fit=None,
    annular_curve_fit=None,
    *,
    raw_markers=None,   # dict: {"GSF":..., "surround_extent":..., "AMRF":..., "SI":...}
    fit_markers=None,   # same keys
    save_dir="/project/results/nature_and_interactions/size_tuning/plots/",
    use_diameter=True,
):
    """
    Plot raw size tuning curves and (optionally) fit curves on the same axes,
    and overlay BOTH marker sets (raw + fit).

    raw_markers / fit_markers: dicts with keys:
      GSF, surround_extent, AMRF, SI
    """
    ensure_dir(save_dir)

    radii = np.asarray(radii, dtype=float)
    x = 2 * radii if use_diameter else radii
    xlab = "Diameter (deg)" if use_diameter else "Radius (deg)"

    # curves to numpy
    circ_raw = np.asarray(circular_curve_raw, dtype=float)
    ann_raw  = None if annular_curve_raw is None else np.asarray(annular_curve_raw, dtype=float)

    circ_fit = None if circular_curve_fit is None else np.asarray(circular_curve_fit, dtype=float)
    ann_fit  = None if annular_curve_fit is None else np.asarray(annular_curve_fit, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # --- raw curves
    ax.plot(x, circ_raw, lw=2, label="Center raw")
    if ann_raw is not None:
        ax.plot(x, ann_raw, lw=2, ls="--", label="Surround raw")

    # --- fit curves (if present)
    if circ_fit is not None:
        ax.plot(x, circ_fit, lw=2, ls="-.", label="Center fit")
    if ann_fit is not None:
        ax.plot(x, ann_fit, lw=2, ls=":", label="Surround fit")

    def _mark(ax, val, label, ls, lw=2):
        if val is None:
            return
        try:
            v = float(val)
        except Exception:
            return
        if not np.isfinite(v) or v < 0:
            return
        xv = 2 * v if use_diameter else v
        ax.axvline(xv, lw=lw, ls=ls, label=label)

    # raw markers
    if raw_markers is not None:
        _mark(ax, raw_markers.get("GSF"),            "GSF raw",            ls=":")
        _mark(ax, raw_markers.get("surround_extent"),"Surround raw",       ls=":")
        _mark(ax, raw_markers.get("AMRF"),           "AMRF raw",           ls=":")

    # fit markers
    if fit_markers is not None:
        _mark(ax, fit_markers.get("GSF"),            "GSF fit",            ls="--")
        _mark(ax, fit_markers.get("surround_extent"),"Surround fit",       ls="--")
        _mark(ax, fit_markers.get("AMRF"),           "AMRF fit",           ls="--")

    ax.set_xlabel(xlab)
    ax.set_ylabel("Response (Δ from gray)")
    ax.set_title(f"Size tuning – {neuron}")

    ax.legend(frameon=False, ncols=2)
    ax.grid(True, alpha=0.25)

    # Print summary line
    msg = f"[{neuron}]"
    if raw_markers is not None and raw_markers.get("GSF") is not None:
        msg += f" raw(GSF={2*raw_markers['GSF']:.3f}°, Surr={2*raw_markers['surround_extent']:.3f}°, AMRF={2*raw_markers.get('AMRF', np.nan):.3f}°, SI={raw_markers.get('SI', np.nan):.3f})"
    if fit_markers is not None and fit_markers.get("GSF") is not None:
        msg += f" fit(GSF={2*fit_markers['GSF']:.3f}°, Surr={2*fit_markers['surround_extent']:.3f}°, AMRF={2*fit_markers.get('AMRF', np.nan):.3f}°, SI={fit_markers.get('SI', np.nan):.3f})"
    print(msg)

    out_path = os.path.join(save_dir, f"{neuron}_size_tuning_curve_both.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path