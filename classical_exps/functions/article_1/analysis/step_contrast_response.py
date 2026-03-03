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
from .shared import sort_by_spread, ensure_dir
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
from matplotlib.lines import Line2D


def plot_contrast_response(
    neuron_id,
    center_contrasts,
    surround_contrasts,
    contrast_response_curves,        # raw: shape (n_surround, n_center)
    contrast_response_fits=None,     # fit: same shape, optional
    title=None,
    save_dir="/project/results/nature_and_interactions/reponse_contrasts/",
    style="paper",                  # "paper" | "diagnostic"
    xscale="log",                   # "log" | "linear" | "symlog"
    symlog_linthresh=1e-3,
    sort_surround=True,
    show_points=True,
    show_legend=True,
    legend_ncols=2,
    dpi=300,
):
    x = np.asarray(center_contrasts, dtype=float)
    s = np.asarray(surround_contrasts, dtype=float)
    y_raw = np.asarray(contrast_response_curves, dtype=float)

    if y_raw.ndim != 2 or y_raw.shape != (len(s), len(x)):
        raise ValueError(
            f"contrast_response_curves must have shape (n_surround, n_center)=({len(s)},{len(x)}), got {y_raw.shape}"
        )

    y_fit = None
    if contrast_response_fits is not None:
        y_fit = np.asarray(contrast_response_fits, dtype=float)
        if y_fit.shape != y_raw.shape:
            raise ValueError(f"contrast_response_fits shape mismatch: expected {y_raw.shape}, got {y_fit.shape}")

    # Reorder curves for readability (does not change values)
    if sort_surround:
        order = np.argsort(s)
        s = s[order]
        y_raw = y_raw[order]
        if y_fit is not None:
            y_fit = y_fit[order]

    if title is None:
        title = f"Contrast response – neuron {neuron_id}"

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # STYLE PRESETS (no data ops)
    # -------------------------
    if style == "paper":
        # grayscale by surround contrast value 
        smin, smax = float(np.min(s)), float(np.max(s))
        den = (smax - smin) if (smax > smin) else 1.0

        # choose mapping: high surround lighter (more suppression) is often visually nice
        # intensity in [0.15..0.80]
        intensity = 0.80 - (0.80 - 0.15) * ((s - smin) / den)   # low light -> high dark
        colors = [(float(t), float(t), float(t)) for t in intensity]

        line_lw = 2.2
        fit_alpha = 0.95
        pt_alpha = 0.9
        pt_size = 22
        grid_alpha = 0.15

        # points: classic open/filled circles are optional; simplest is open circles
        marker_face = "none"
        marker_edge = None  # set per-curve color
        legend_title = "Surround contrast"

        # in paper mode, often legend is off or minimal
        if show_legend is None:
            show_legend = False

    elif style == "diagnostic":
        # colorful and loud enough to spot mistakes
        cmap = plt.get_cmap("turbo") if len(s) > 10 else plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(s))]

        line_lw = 2.4
        fit_alpha = 0.95
        pt_alpha = 0.8
        pt_size = 26
        grid_alpha = 0.25

        marker_face = None  # filled with curve color
        marker_edge = "none"
        legend_title = "Surround contrast"

        if show_legend is None:
            show_legend = True
    else:
        raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'diagnostic'.")

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    legend_handles = []

    for i in range(len(s)):
        color = colors[i]
        label = f"{float(s[i]):.2f}"

        # raw points
        if show_points:
            if marker_face == "none":
                ax.scatter(
                    x, y_raw[i],
                    s=pt_size,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    alpha=pt_alpha,
                    zorder=3
                )
            else:
                ax.scatter(
                    x, y_raw[i],
                    s=pt_size,
                    color=color,
                    edgecolors=marker_edge,
                    alpha=pt_alpha,
                    zorder=3
                )

        # fit line
        if y_fit is not None:
            ax.plot(
                x, y_fit[i],
                color=color,
                linewidth=line_lw,
                alpha=fit_alpha,
                zorder=4
            )
            legend_handles.append(Line2D(
                [0], [0],
                color=color, lw=line_lw,
                marker="o", markersize=5,
                markerfacecolor=("none" if marker_face == "none" else color),
                markeredgecolor=color,
                linestyle="-",
                label=label
            ))
        else:
            # fallback: connect raw with a line if no fit provided
            ax.plot(
                x, y_raw[i],
                color=color,
                linewidth=1.6,
                alpha=0.8,
                zorder=2
            )
            legend_handles.append(Line2D(
                [0], [0],
                color=color, lw=0.0,
                marker="o", markersize=5,
                markerfacecolor=("none" if marker_face == "none" else color),
                markeredgecolor=color,
                linestyle="None",
                label=label
            ))

    ax.set_title(title)
    ax.set_xlabel("Center contrast")
    ax.set_ylabel("Response (Δ from gray)")

    # xscale without altering x (but log cannot display <=0)
    if xscale == "log":
        if np.any(x <= 0):
            # Don't silently lie. Log scale simply doesn't work with 0.
            # Use symlog or linear if you have 0 contrast in the data.
            raise ValueError("xscale='log' but center_contrasts contains 0 or negative values. Use xscale='symlog' or 'linear'.")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    elif xscale == "symlog":
        ax.set_xscale("symlog", linthresh=symlog_linthresh)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    elif xscale == "linear":
        pass
    else:
        raise ValueError(f"Unknown xscale={xscale}")

    ax.grid(True, alpha=grid_alpha)
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            frameon=False,
            ncols=legend_ncols,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0
        )

    fig.tight_layout()

    out = os.path.join(save_dir, f"response_contrast_{neuron_id}_{style}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_contrast_response_both(
    *args, **kwargs
):
    """Convenience: emits two files: paper + diagnostic."""
    out_paper = plot_contrast_response(*args, style="paper", **kwargs)
    out_diag  = plot_contrast_response(*args, style="diagnostic", **kwargs)
    return out_paper, out_diag

def contrast_response_results_1(
    neuron_ids,
    filtered_neuron_ids,
    loaded,
    sort_by_std=False,
    spread_to_plot=(15, 50, 85),
    plot_all=True,
):
    ''' This function aims to visualize some representative contrast response curves for the requested neuron set :
            
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean spread ...)
            - 3) It sorts the array of neurons according to the how spread their curves are
            - 4) It plots the contrast response curves for neurons representing different contrasts

        To estimate how spread the curves are for a neuron, there are two possibilities :

            - Either take the last points of the curves and calculate the ratio : max_response/min_response
            - Either compute the mean standard deviation for every points

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_response_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - sort_by_std    : Bool, decides which method to use to estimate how the curves are spread. If set to 'True', it will use the second method and sort with the mean_std value
            - spread_to_plot : An array containing the position of the neuron in the sorted spread array (in percentage). 100 means that it's the neuron with the highest spread, 50 means that 50% of the neurons have a smaller spread.
            - other          : see 'size_tuning_results_1'
    '''
     
    if filtered_neuron_ids is None:
        raise ValueError("contrast_response_results_1 requires filtered_neuron_ids.")
    if loaded is None:
        raise ValueError("contrast_response_results_1 requires loaded dict.")
    if "contrast_response_results" not in loaded:
        raise ValueError("loaded missing 'contrast_response_results'.")
    if "contrast_response" not in loaded:
        raise ValueError("loaded missing 'contrast_response'.")

    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)

    spread_results = loaded["contrast_response_results"]
    contrast_response = loaded["contrast_response"]

    sorted_neuron_ids, sorted_spread = sort_by_spread(
        neuron_ids=filtered_neuron_ids,
        spread_results=spread_results,
        sort_by_std=sort_by_std,
    )

    n = len(neuron_ids)
    n_new = len(sorted_neuron_ids)
    median_spread = round(float(np.median(sorted_spread)), 2)

    print("--------------------------------------")
    print("Visualisation of representative contrast response curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    if sort_by_std:
        print(f"    > The median of mean_std is {median_spread}")
    else:
        print(f"    > The median of max/min is {median_spread}")
    print(f"    > Plots :")

    center_contrasts = contrast_response["center_contrasts"]
    surround_contrasts = contrast_response["surround_contrasts"]
    curves_by_id = contrast_response["by_id"]

    fits_by_id = loaded.get("contrast_response_fits", {}).get("by_id", {})

    # decide which neurons to plot
    if plot_all or (isinstance(spread_to_plot, str) and spread_to_plot.lower() == "all"):
        ids_to_plot = list(map(int, sorted_neuron_ids))
        print(f"    > Plotting ALL curves: {len(ids_to_plot)} neurons")
        for idx, neuron_id in enumerate(ids_to_plot):
            mat = curves_by_id.get(neuron_id)
            mat_raw = curves_by_id.get(neuron_id)
            mat_fit = fits_by_id.get(neuron_id, {}).get("y_fit", None)
            spread_percent = 100 * idx / (len(ids_to_plot) - 1) if len(ids_to_plot) > 1 else 0
            spread_percent = round(spread_percent, 1)

            title = (
                f"Response contrast for a spread above {spread_percent}% of the neurons "
                f"(neuron {neuron_id})"
            )

            if mat is None:
                raise RuntimeError(f"Neuron {neuron_id} missing in contrast_response curves loader.")
            plot_contrast_response_both(
                neuron_id=neuron_id,
                center_contrasts=center_contrasts,
                surround_contrasts=surround_contrasts,
                contrast_response_curves=mat_raw,
                contrast_response_fits=mat_fit,   # overlay if exists
                title=title,
            )
    else:
        spread_to_plot = list(spread_to_plot)
        for spread_percent in spread_to_plot:
            neuron_pos = int((spread_percent / 100) * (n_new - 1))
            neuron_id = int(sorted_neuron_ids[neuron_pos])

            print(f"    > {spread_percent}% of spread, neuron {neuron_id} :")

            title = (
                f"Response contrast for a spread above {spread_percent}% of the neurons "
                f"(neuron {neuron_id})"
            )

            mat = curves_by_id.get(neuron_id)
            mat_raw = curves_by_id.get(neuron_id)
            mat_fit = fits_by_id.get(neuron_id, {}).get("y_fit", None)
            if mat is None:
                raise RuntimeError(f"Neuron {neuron_id} missing in contrast_response curves loader.")

            plot_contrast_response_curves(
                neuron_id=neuron_id,
                center_contrasts=center_contrasts,
                surround_contrasts=surround_contrasts,
                contrast_response_curves=mat_raw,
                contrast_response_fits=mat_fit,   # overlay if exists
                title=title,
            )

    print("--------------------------------------")
    print()


def plot_contrast_response_curves(
    neuron: str,
    center_contrasts,
    surround_contrasts,
    response_mat,
    save_dir="/project/results/nature_and_interactions/contrast_response/plots/",
    title=None,
):
    """
    Plot contrast-response curves: x=center contrast (log), each curve is fixed surround contrast.
    response_mat shape: (len(surround_contrasts), len(center_contrasts))
    """
    ensure_dir(save_dir)

    x = np.asarray(center_contrasts, dtype=float)
    surr = np.asarray(surround_contrasts, dtype=float)
    mat = np.asarray(response_mat, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4))

    for i in range(len(surr)):
        ax.plot(x, mat[i, :], lw=2, label=f"{100*surr[i]:.0f}%")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Center contrast")
    ax.set_ylabel("Response (Δ from gray)")
    ax.set_title(title or f"Contrast response – {neuron}")

    # Put legend outside so it never clips
    ax.legend(
        title="Surround contrast",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    ax.grid(True, alpha=0.25)

    out_path = os.path.join(save_dir, f"{neuron}_contrast_response.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
