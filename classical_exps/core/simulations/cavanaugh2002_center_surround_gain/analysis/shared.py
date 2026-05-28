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


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "font.size": 10,
    "legend.frameon": False,
})

BLACK = "black"
DARK = "0.35"
MID = "0.55"
LIGHT = "0.85"
WHITE = "white"

def add_vline_with_label(
    ax,
    x,
    label,
    *,
    y_frac=0.90,
    x_mul=1.04,
    x_add_frac=0.015,
    ha="left",
    va="bottom",
    log_x=False,
    fontsize=None,
    clip_on=True,
):
    """
    Draw a vertical dashed line and place its label slightly to the side.
    """
    if x is None or not np.isfinite(x):
        return

    ax.axvline(x, color="black", linestyle="--", linewidth=1.5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if log_x:
        x_text = x * x_mul
        # keep label inside axes
        x_text = min(x_text, xmax / 1.01)
    else:
        x_text = x + x_add_frac * (xmax - xmin)
        x_text = min(x_text, xmax - 0.01 * (xmax - xmin))

    y_text = ymin + y_frac * (ymax - ymin)

    ax.text(
        x_text,
        y_text,
        label,
        ha=ha,
        va=va,
        fontsize=fontsize,
        clip_on=clip_on,
    )

def add_hline_with_label(
    ax,
    y,
    label,
    *,
    x_frac=0.90,
    y_mul=1.04,
    y_add_frac=0.015,
    ha="right",
    va="bottom",
    log_y=False,
    fontsize=None,
    clip_on=True,
):
    """
    Draw a horizontal dashed line and place its label slightly above it.
    """
    if y is None or not np.isfinite(y):
        return

    ax.axhline(y, color="black", linestyle="--", linewidth=1.5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_text = xmin + x_frac * (xmax - xmin)

    if log_y:
        y_text = y * y_mul
        y_text = min(y_text, ymax / 1.01)
    else:
        y_text = y + y_add_frac * (ymax - ymin)
        y_text = min(y_text, ymax - 0.01 * (ymax - ymin))

    ax.text(
        x_text,
        y_text,
        label,
        ha=ha,
        va=va,
        fontsize=fontsize,
        clip_on=clip_on,
    )

    
def plot_scatter_hist(
    x,
    y,
    *,
    si_values=None,
    si_hist_bins=None,
    si_hist_weights=None,
    x_gmean=None,
    y_gmean=None,
    si_mean=None,
    ratio_gmean=None,
    title="",  # kept for compatibility; intentionally unused
    x_label="GSF diameter (deg)",
    y_label="Surround diameter (deg)",
    log_axes=True,
    scatter_save_path=None,
    si_save_path=None,
    ratio_save_path=None,
    show=False,
    close=True,
    jitter_duplicates=True,
    jitter_scale=0.04,
    jitter_seed=42,
    si_ymax=None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}.")
    if len(x) == 0:
        raise ValueError("x and y are empty.")

    valid_xy = np.isfinite(x) & np.isfinite(y)
    x = x[valid_xy]
    y = y[valid_xy]

    if log_axes:
        positive_xy = (x > 0) & (y > 0)
        x = x[positive_xy]
        y = y[positive_xy]
        if len(x) == 0:
            raise ValueError("log_axes=True requires strictly positive x and y values.")

    # -------------------------
    # poster sizing
    # -------------------------
    SCATTER_FIGSIZE = (8.2, 8.2)
    HIST_FIGSIZE = (7.0, 5.2)
    RATIO_FIGSIZE = (6.4, 4.8)

    LABELSIZE = 20
    TICKSIZE = 15
    TITLESIZE = 20
    ANNOTSIZE = 18

    SCATTER_SIZE = 28
    DIAG_LW = 1.8
    HIST_LW = 1.6
    TICKLEN = 6
    TICKWIDTH = 1.4

    # scatter display coordinates: jitter only duplicated points
    x_scatter = x.copy()
    y_scatter = y.copy()

    if jitter_duplicates:
        if not log_axes:
            raise ValueError(
                "jitter_duplicates currently assumes positive data on log axes. "
                "Disable jitter or implement additive jitter for linear axes."
            )
        x_scatter, y_scatter = jitter_log_duplicates_only(
            x,
            y,
            scale=jitter_scale,
            rng=np.random.default_rng(jitter_seed),
        )

    def _apply_article_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            direction="out",
            length=TICKLEN,
            width=TICKWIDTH,
            labelsize=TICKSIZE,
        )
        ax.grid(False)

    # -------------------------
    # 1) Scatter + marginals
    # -------------------------
    fig = plt.figure(figsize=SCATTER_FIGSIZE)

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(3.8, 1.0),
        height_ratios=(1.0, 3.8),
        left=0.12, right=0.88, bottom=0.12, top=0.94,
        wspace=0.06, hspace=0.06,
    )

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_scatter.scatter(x_scatter, y_scatter, s=SCATTER_SIZE, c="black", alpha=0.7, linewidths=0)
    ax_scatter.set_xlabel(x_label, fontsize=LABELSIZE)
    ax_scatter.set_ylabel(y_label, fontsize=LABELSIZE)

    if log_axes:
        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")

    ax_scatter.xaxis.set_major_formatter(ScalarFormatter())
    ax_scatter.yaxis.set_major_formatter(ScalarFormatter())
    ax_scatter.tick_params(which="minor", length=0)

    xy_min = min(np.min(x), np.min(y))
    xy_max = max(np.max(x), np.max(y))

    if log_axes:
        axis_lo = 10 ** np.floor(np.log10(xy_min))
        axis_hi = 10 ** np.ceil(np.log10(xy_max))
    else:
        pad = 0.05 * (xy_max - xy_min + 1e-12)
        axis_lo = xy_min - pad
        axis_hi = xy_max + pad

    ax_scatter.set_xlim(axis_lo, axis_hi)
    ax_scatter.set_ylim(axis_lo, axis_hi)
    ax_scatter.plot([axis_lo, axis_hi], [axis_lo, axis_hi], color=BLACK, linestyle="-", linewidth=DIAG_LW)

    n_marginal_bins = 12
    if log_axes:
        marginal_bins = np.logspace(np.log10(axis_lo), np.log10(axis_hi), n_marginal_bins)
    else:
        marginal_bins = np.linspace(axis_lo, axis_hi, n_marginal_bins)

    x_weights = np.ones_like(x) / len(x)
    y_weights = np.ones_like(y) / len(y)

    ax_hist_x.hist(
        x,
        bins=marginal_bins,
        weights=x_weights,
        facecolor="white",
        edgecolor="black",
        linewidth=HIST_LW,
    )
    ax_hist_y.hist(
        y,
        bins=marginal_bins,
        weights=y_weights,
        orientation="horizontal",
        facecolor="white",
        edgecolor="black",
        linewidth=HIST_LW,
    )

    if log_axes:
        ax_hist_x.set_xscale("log")
        ax_hist_y.set_yscale("log")
        ax_hist_x.xaxis.set_major_formatter(ScalarFormatter())
        ax_hist_y.yaxis.set_major_formatter(ScalarFormatter())
        ax_hist_x.tick_params(which="minor", length=0)
        ax_hist_y.tick_params(which="minor", length=0)

    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)
    ax_hist_x.set_ylabel("Proportion of cells", fontsize=LABELSIZE)
    ax_hist_y.set_xlabel("Proportion of cells", fontsize=LABELSIZE)

    # headroom for labels
    ymax_hist_x = ax_hist_x.get_ylim()[1]
    ax_hist_x.set_ylim(0, ymax_hist_x * 1.08)

    xmax_hist_y = ax_hist_y.get_xlim()[1]
    ax_hist_y.set_xlim(0, xmax_hist_y * 1.08)

    if x_gmean is not None and np.isfinite(x_gmean) and x_gmean > 0:
        add_vline_with_label(
            ax_hist_x,
            x_gmean,
            f"{x_gmean:.2f}",
            y_frac=0.88,
            x_mul=1.05,
            x_add_frac=0.015,
            log_x=log_axes,
            ha="left",
            va="bottom",
            fontsize=ANNOTSIZE,
        )

    if y_gmean is not None and np.isfinite(y_gmean) and y_gmean > 0:
        add_hline_with_label(
            ax_hist_y,
            y_gmean,
            f"{y_gmean:.2f}",
            x_frac=0.88,
            y_mul=1.05,
            y_add_frac=0.015,
            log_y=log_axes,
            ha="right",
            va="bottom",
            fontsize=ANNOTSIZE,
        )

    _apply_article_axes(ax_scatter)
    _apply_article_axes(ax_hist_x)
    _apply_article_axes(ax_hist_y)

    if scatter_save_path is not None:
        os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
        fig.savefig(scatter_save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    # -------------------------
    # 2) SI histogram
    # -------------------------

    if si_save_path is not None and (
        si_values is not None or (si_hist_bins is not None and si_hist_weights is not None)
    ):
        fig, ax_si = plt.subplots(figsize=HIST_FIGSIZE, constrained_layout=True)

        plotted_si = False
        common_si_bins = np.linspace(0.0, 1.2, 7)
        si_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

        # Case A: raw SI samples
        if si_values is not None:
            si_values = np.asarray(si_values, dtype=float)
            si_values = si_values[np.isfinite(si_values)]

            if len(si_values) > 0:
                si_weights = np.ones_like(si_values, dtype=float) / len(si_values)
                ax_si.hist(
                    si_values,
                    bins=common_si_bins,
                    weights=si_weights,
                    facecolor=LIGHT,
                    edgecolor=BLACK,
                    linewidth=HIST_LW,
                )
                ax_si.set_xlim(common_si_bins[0], common_si_bins[-1])
                ax_si.set_xticks(si_ticks)
                ax_si.set_ylim(0, 0.5)
                ax_si.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
                plotted_si = True

        # Case B: pre-binned scraped histogram -> re-bin
        elif si_hist_bins is not None and si_hist_weights is not None:
            bins_arr = np.asarray(si_hist_bins, dtype=float)
            weights_arr = np.asarray(si_hist_weights, dtype=float)

            if bins_arr.ndim != 2 or bins_arr.shape[1] != 2:
                raise ValueError(
                    f"si_hist_bins must have shape (n_bins, 2), got {bins_arr.shape}"
                )
            if len(weights_arr) != len(bins_arr):
                raise ValueError(
                    f"si_hist_weights length {len(weights_arr)} does not match "
                    f"number of bins {len(bins_arr)}"
                )

            bin_midpoints = bins_arr.mean(axis=1)

            valid_si = np.isfinite(bin_midpoints) & np.isfinite(weights_arr)
            bin_midpoints = bin_midpoints[valid_si]
            weights_arr = weights_arr[valid_si]

            if len(bin_midpoints) > 0 and np.sum(weights_arr) > 0:
                rebinned_hist, _ = np.histogram(
                    bin_midpoints,
                    bins=common_si_bins,
                    weights=weights_arr,
                )
                rebinned_hist = rebinned_hist.astype(float)

                if rebinned_hist.sum() > 0:
                    rebinned_hist /= rebinned_hist.sum()

                ax_si.bar(
                    common_si_bins[:-1],
                    rebinned_hist,
                    width=np.diff(common_si_bins),
                    align="edge",
                    facecolor=LIGHT,
                    edgecolor=BLACK,
                    linewidth=HIST_LW,
                )
                ax_si.set_xlim(common_si_bins[0], common_si_bins[-1])
                ax_si.set_xticks(si_ticks)
                plotted_si = True

        if plotted_si:
            if si_ymax is not None:
                si_ymax = float(si_ymax)
                ax_si.set_ylim(0.0, 0.5)
                ax_si.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            else:
                ymax_si = ax_si.get_ylim()[1]
                ax_si.set_ylim(0, 0.5)

            if si_mean is not None and np.isfinite(si_mean):
                add_vline_with_label(
                    ax_si,
                    si_mean,
                    f"{si_mean:.2f}",
                    y_frac=0.84,
                    x_add_frac=0.025,
                    log_x=False,
                    ha="left",
                    va="bottom",
                    fontsize=ANNOTSIZE,
                )

            ax_si.set_xlabel("Suppression Index", fontsize=LABELSIZE)
            ax_si.set_ylabel("Proportion of cells", fontsize=LABELSIZE)
            _apply_article_axes(ax_si)

            os.makedirs(os.path.dirname(si_save_path), exist_ok=True)
            fig.savefig(si_save_path, dpi=300, bbox_inches="tight")

            if show:
                plt.show()

        if close:
            plt.close(fig)

    # -------------------------
    # 3) Surround / GSF ratio histogram
    # -------------------------
    if ratio_save_path is not None:
        diameter_ratio = y / x
        diameter_ratio = diameter_ratio[np.isfinite(diameter_ratio) & (diameter_ratio > 0)]

        if len(diameter_ratio) > 0:
            fig, ax_ratio = plt.subplots(figsize=RATIO_FIGSIZE, constrained_layout=True)

            if log_axes:
                ratio_lo = 10 ** np.floor(np.log10(np.min(diameter_ratio)))
                ratio_hi = 10 ** np.ceil(np.log10(np.max(diameter_ratio)))
                ratio_bins = np.logspace(np.log10(ratio_lo), np.log10(ratio_hi), 10)
                ax_ratio.set_xscale("log")
                ax_ratio.xaxis.set_major_formatter(ScalarFormatter())
                ax_ratio.tick_params(which="minor", length=0)
            else:
                ratio_lo = np.min(diameter_ratio)
                ratio_hi = np.max(diameter_ratio)
                ratio_bins = np.linspace(ratio_lo, ratio_hi, 10)

            ratio_weights = np.ones_like(diameter_ratio) / len(diameter_ratio)
            ax_ratio.hist(
                diameter_ratio,
                bins=ratio_bins,
                weights=ratio_weights,
                facecolor="white",
                edgecolor="black",
                linewidth=HIST_LW,
            )

            ymax_ratio = ax_ratio.get_ylim()[1]
            ax_ratio.set_ylim(0, ymax_ratio * 1.08)

            if ratio_gmean is not None and np.isfinite(ratio_gmean) and ratio_gmean > 0:
                add_vline_with_label(
                    ax_ratio,
                    ratio_gmean,
                    f"{ratio_gmean:.2f}",
                    y_frac=0.86,
                    x_mul=1.05,
                    x_add_frac=0.015,
                    log_x=log_axes,
                    ha="left",
                    va="bottom",
                    fontsize=ANNOTSIZE,
                )

            ax_ratio.set_xlabel("Surround diameter / GSF diameter", fontsize=LABELSIZE)
            ax_ratio.set_ylabel("Proportion of cells", fontsize=LABELSIZE)
            _apply_article_axes(ax_ratio)

            os.makedirs(os.path.dirname(ratio_save_path), exist_ok=True)
            fig.savefig(ratio_save_path, dpi=300, bbox_inches="tight")

            if show:
                plt.show()

            if close:
                plt.close(fig)


def jitter_log_duplicates_only(x, y, scale=0.04, rng=None):
    """
    Jitter only duplicated (x, y) pairs, preserving unique points.

    Jitter is multiplicative, so it is appropriate for log-scaled axes.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Positive coordinates.
    scale : float
        Half-width of uniform jitter in log space. Typical range: 0.03-0.06.
    rng : np.random.Generator or None
        Random generator for reproducibility. If None, a fresh generator is used.

    Returns
    -------
    x_plot, y_plot : np.ndarray
        Jittered coordinates for plotting only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of the same length.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must be finite.")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("jitter_log_duplicates_only requires strictly positive x and y.")

    if rng is None:
        rng = np.random.default_rng(42)

    xy = np.column_stack([x, y])
    _, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)

    x_plot = x.copy()
    y_plot = y.copy()

    for group_idx, count in enumerate(counts):
        if count <= 1:
            continue

        idx = np.flatnonzero(inverse == group_idx)

        # independent multiplicative jitter in log space
        x_plot[idx] *= np.exp(rng.uniform(-scale, scale, size=len(idx)))
        y_plot[idx] *= np.exp(rng.uniform(-scale, scale, size=len(idx)))

    return x_plot, y_plot

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

    valid_ids = []
    shifts = []

    for nid in neuron_ids:
        nid = int(nid)

        if not recompute:
            d = by_id.get(nid)
            if d is None:
                continue
            shift = float(d["shift"])
        else:
            mat = cst_curves["by_id"].get(nid)
            if mat is None:
                continue

            low_curve = np.asarray(mat[low_contrast_id], dtype=float)
            high_curve = np.asarray(mat[high_contrast_id], dtype=float)

            GSF_low, _, _, _, _ = get_GSF_surround_AMRF(
                radii=radii, circular_tuning_curve=low_curve, annular_tuning_curve=None
            )
            GSF_high, _, _, _, _ = get_GSF_surround_AMRF(
                radii=radii, circular_tuning_curve=high_curve, annular_tuning_curve=None
            )

            if (
                GSF_high is None or not np.isfinite(GSF_high) or GSF_high <= 0
                or GSF_low is None or not np.isfinite(GSF_low) or GSF_low <= 0
            ):
                shift = np.nan
            else:
                shift = float(GSF_low / GSF_high)

        valid_ids.append(nid)
        shifts.append(shift)

    valid_ids = np.asarray(valid_ids, dtype=int)
    shifts = np.asarray(shifts, dtype=float)

    order = np.argsort(np.where(np.isfinite(shifts), shifts, np.inf))
    return valid_ids[order], shifts[order]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

    
def neuron_key(neuron_id: int) -> str:
    return f"neuron_{int(neuron_id)}"
