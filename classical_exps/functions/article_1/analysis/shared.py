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

def plot_scatter_hist(
    x,
    y,
    si_values=None,
    title="Surround Extent vs GSF Diameter",
    x_label="GSF diameter (deg)",
    y_label="Surround diameter (deg)",
    log_axes=True,
    save_dir="./nature_and_interactions"
):
    '''
    Extended plotting function to replicate and expand `plot_scatter_hist`.

    Generates:
        1. Scatter plot with marginal histograms (x vs y)
        2. Histogram comparison of x and y distributions
        3. Suppression Index (SI) histogram (if `si_values` is provided)

    Arguments:
        - x, y         : Arrays to compare (e.g. GSF and Surround diameters)
        - si_values    : Array of suppression index values (optional)
        - title        : Title of the scatter plot
        - x_label, y_label : Axis labels
        - log_axes     : Whether to apply log scaling
        - save_dir     : Directory to save plots
    '''

    os.makedirs(save_dir, exist_ok=True)

    # Scatter + Hist
    fig = plt.figure(figsize=(6, 6))
    plt.suptitle(title)
    from matplotlib.gridspec import GridSpec
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(3.6, 1),
        height_ratios=(1, 3.6),
        left=0.08, right=0.92, bottom=0.08, top=0.92,
        wspace=0.06, hspace=0.06
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    xymin = min(np.min(np.abs(x)), np.min(np.abs(y)))
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))

    minedge = max(0.06, xymin - 0.5)
    maxedge = xymax + 1

    ax.set_xlim(minedge, maxedge)
    ax.set_ylim(minedge, maxedge)
    ax.plot([minedge, maxedge], [minedge, maxedge], 'k--')

    nbins = 15
    weights_x = np.ones(x.shape) / len(x)
    weights_y = np.ones(y.shape) / len(y)
    if log_axes:
        ax_histx.hist(x, weights=weights_x, bins=np.logspace(np.log10(minedge), np.log10(maxedge), nbins))
        ax_histy.hist(y, weights=weights_y, orientation='horizontal', bins=np.logspace(np.log10(minedge), np.log10(maxedge), nbins))
    else:
        ax_histx.hist(x, weights=weights_x, bins=np.linspace(minedge, maxedge, nbins))
        ax_histy.hist(y, weights=weights_y, orientation='horizontal', bins=np.linspace(minedge, maxedge, nbins))
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.set_ylabel("Frac.")
    ax_histy.set_xlabel("Frac.")
    plt.savefig(os.path.join(save_dir, "scatter_histogram.png"))
    plt.close()

    # Histogram of Distributions
    plt.figure(figsize=(7, 4))
    bins = np.logspace(np.log10(0.05), np.log10(2.5), 20)
    plt.hist(x, bins=bins, alpha=0.6, label='GSF', density=True)
    plt.hist(y, bins=bins, alpha=0.6, label='Surround', density=True)
    plt.xscale('log')
    plt.xlabel("Diameter (deg)")
    plt.ylabel("Proportion")
    plt.title("Distribution of GSF and Surround Diameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diameter_histograms.png"))
    plt.close()

    # SI Histogram (Optional)
    if si_values is not None:
        plt.figure(figsize=(6, 4))
        bins = np.linspace(0, 1, 21)
        plt.hist(si_values, bins=bins, color='orange', edgecolor='black')
        plt.axvline(np.mean(si_values), color='red', linestyle='--', label=f"Mean = {np.mean(si_values):.2f}")
        plt.xlabel("Suppression Index (SI)")
        plt.ylabel("Count")
        plt.title(f"Distribution of Suppression Index (n={len(si_values)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "si_distribution.png"))
        plt.close()


def sort_by_spread(
    neuron_ids,
    spread_results,     
    sort_by_std=False,
):
    ''' This function sorts the neurons from the ones with the lowest spread values to the ones with the highest.  
        Spread is assessed thanks to the mean_std or maxmin_ratio values computed in the 'contrast_response_experiment' function

        Prerequisite :
            
            - function 'contrast_response_experiment' executed for the required neurons

        Arguments :

            - sort_by_std   : If set to true, sort with the mean_std value, if set to false, sort with max/min value.

        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_spread        : The sorted spread values (either std or maxmin)
    '''
    if spread_results is None or "by_id" not in spread_results:
        raise ValueError("sort_by_spread requires spread_results loader (with 'by_id').")

    neuron_ids = np.asarray(neuron_ids, dtype=int)
    by_id = spread_results["by_id"]

    vals = []
    for nid in neuron_ids:
        d = by_id.get(int(nid))
        if d is None:
            raise RuntimeError(f"Neuron {nid} missing in spread_results loader.")
        vals.append(d["mean_std"] if sort_by_std else d["maxmin"])

    vals = np.asarray(vals, dtype=float)
    order = np.argsort(vals)

    return neuron_ids[order], vals[order]


def sort_by_shift(
    neuron_ids,
    cst_results,
    *,
    cst_curves=None,
    radii=None,
    low_contrast_id=None,
    high_contrast_id=None,
):
    ''' This function sorts the neurons from the ones with the lowest shift value to the ones with the highest.  
        shift is assessed thanks to the GSF at low contrast divided by the GSF at high contrast. Computed in the 'contrast_size_tuning_experiment_all_phases' function

        Prerequisite :
            
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            
        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_shift         : The sorted shift values
    '''
    if cst_results is None or "by_id" not in cst_results:
        raise ValueError("sort_by_shift requires cst_results loader (with 'by_id').")

    neuron_ids = np.asarray(neuron_ids, dtype=int)
    by_id = cst_results["by_id"]

    recompute = (low_contrast_id is not None) or (high_contrast_id is not None)
    if recompute:
        if cst_curves is None or "by_id" not in cst_curves:
            raise ValueError("Recompute shift requires cst_curves loader (with 'by_id').")
        if radii is None:
            raise ValueError("Recompute shift requires radii.")
        radii = np.asarray(radii, dtype=float)

        if low_contrast_id is None:
            low_contrast_id = 0
        if high_contrast_id is None:
            high_contrast_id = -1

    shifts = []
    for nid in neuron_ids:
        nid = int(nid)

        if not recompute:
            d = by_id.get(nid)
            if d is None:
                raise RuntimeError(f"Neuron {nid} missing in contrast_size_tuning_results loader.")
            shifts.append(float(d["shift"]))
        else:
            mat = cst_curves["by_id"].get(nid)
            if mat is None:
                raise RuntimeError(f"Neuron {nid} missing in contrast_size_tuning curves loader.")

            low_curve = np.asarray(mat[low_contrast_id], dtype=float)
            high_curve = np.asarray(mat[high_contrast_id], dtype=float)

            GSF_low, _, _, _, _  = get_GSF_surround_AMRF(
                radii=radii, circular_tuning_curve=low_curve, annular_tuning_curve=None
            )
            GSF_high, _, _, _, _ = get_GSF_surround_AMRF(
                radii=radii, circular_tuning_curve=high_curve, annular_tuning_curve=None
            )

            if GSF_high is None or not np.isfinite(GSF_high) or GSF_high <= 0:
                shift = np.nan
            else:
                shift = float(GSF_low / GSF_high)

            shifts.append(shift)

    shifts = np.asarray(shifts, dtype=float)

    # NaNs to the end, deterministic
    order = np.argsort(np.where(np.isfinite(shifts), shifts, np.inf))

    return neuron_ids[order], shifts[order]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

    
def neuron_key(neuron_id: int) -> str:
    return f"neuron_{int(neuron_id)}"
