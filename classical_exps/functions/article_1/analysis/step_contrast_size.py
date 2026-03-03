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
from .shared import sort_by_shift, sort_by_spread, ensure_dir

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


def contrast_size_tuning_results_1(
    neuron_ids,
    filtered_neuron_ids,
    loaded,
    shift_to_plot=(15, 50, 85),
    low_contrast_id=None,
    high_contrast_id=None,
    plot_all=False,
):
    ''' This function aims to visualize what happens when a size tuning experiment is performed at different contrasts :

            - 1) It performs some filtering.
            - 2) It gets the ratio of the GSF at the lowest contrast divided by the GSF at the highest contrast (GSFlow/GSFhigh)
            - 3) It prints the mean value of the GSFlow/GSFhigh, which basically represents how the receptive field radius change when lowering the contrast
            - 4) It plots the curves for neurons representing different shifts
        NB : 'shift' refers to the 'GSFlow/GSFhigh ratio'

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - shift_to_plot    : An array containing the position of the neurons to plot in the sorted shift array (in percentage). 100 means that it's the neuron with the highest shift, 50 means that 50% of the neurons have a smaller shift.
            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            - other            : see 'size_tuning_results_1'
    '''
    
    if filtered_neuron_ids is None:
        raise ValueError("contrast_size_tuning_results_1 requires filtered_neuron_ids.")
    if loaded is None:
        raise ValueError("contrast_size_tuning_results_1 requires loaded dict.")
    if "contrast_size_tuning" not in loaded:
        raise ValueError("loaded missing 'contrast_size_tuning'.")
    if "contrast_size_tuning_results" not in loaded:
        raise ValueError("loaded missing 'contrast_size_tuning_results'.")

    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)

    cst_curves = loaded["contrast_size_tuning"]                # axes + by_id curves
    cst_results = loaded["contrast_size_tuning_results"]       # by_id results (shift)

    radii = cst_curves["radii"]
    contrasts = cst_curves["contrasts"]
    curves_by_id = cst_curves["by_id"]

    # Decide whether shift is recomputed from curves
    recompute = (low_contrast_id is not None) or (high_contrast_id is not None)

    sorted_neuron_ids, sorted_shift = sort_by_shift(
        neuron_ids=filtered_neuron_ids,
        cst_results=cst_results,
        cst_curves=cst_curves if recompute else None,
        radii=radii if recompute else None,
        low_contrast_id=low_contrast_id,
        high_contrast_id=high_contrast_id,
    )

    n = len(neuron_ids)
    n_new = len(sorted_neuron_ids)

    mean_shift = float(np.nanmean(sorted_shift)) if n_new > 0 else np.nan

    print("--------------------------------------")
    print("Visualisation of representative contrast size tuning curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean shift value is {round(mean_shift, 2) if np.isfinite(mean_shift) else 'nan'}")
    print(f"    > Plots :")

    # Normalize default low/high ids for plotting (markers)
    plot_low = 0 if low_contrast_id is None else int(low_contrast_id)
    plot_high = -1 if high_contrast_id is None else int(high_contrast_id)

    # ---- PLOT ALL (sorted) ----
    if plot_all or (isinstance(shift_to_plot, str) and shift_to_plot.lower() == "all"):
        ids_to_plot = list(map(int, sorted_neuron_ids))
        print(f"    > Plotting ALL curves (sorted by shift): {len(ids_to_plot)} neurons")

        for idx, nid in enumerate(ids_to_plot):
            mat = curves_by_id.get(nid)
            if mat is None:
                raise RuntimeError(f"Neuron {nid} missing in contrast_size_tuning curves loader.")

            sh = sorted_shift[idx]
            title = (
                f"Contrast size tuning – neuron {nid} (shift={sh:.2f})"
                if np.isfinite(sh) else
                f"Contrast size tuning – neuron {nid} (shift=nan)"
            )

            cst_raw = loaded["contrast_size_tuning"]
            cst_fit = loaded.get("contrast_size_tuning_fits", None)

            Y_raw = cst_raw["by_id"][nid]
            Y_fit = None
            if cst_fit is not None and nid in cst_fit.get("by_id", {}):
                Y_fit = cst_fit["by_id"][nid]["Y_fit"]

            plot_contrast_size_tuning_both(
                neuron=f"neuron_{nid}",
                radii=radii,
                contrasts=contrasts,
                Y_raw=Y_raw,
                Y_fit=Y_fit,
                low_contrast_id=plot_low,
                high_contrast_id=plot_high,
                save_dir="/project/results/nature_and_interactions/contrast_size_tuning/plots/",
                use_diameter=True,
                title=title,
            )

    # ---- REPRESENTATIVE PERCENTILES ----
    else:
        shift_to_plot = list(shift_to_plot)

        for shift_percent in shift_to_plot:
            neuron_pos = int((shift_percent / 100) * (n_new - 1))
            nid = int(sorted_neuron_ids[neuron_pos])
            sh = sorted_shift[neuron_pos]

            print(
                f"    > {shift_percent}% of shift, neuron {nid}, GSFlow/high = "
                f"{round(sh, 2) if np.isfinite(sh) else 'nan'} :"
            )

            mat = curves_by_id.get(nid)
            if mat is None:
                raise RuntimeError(f"Neuron {nid} missing in contrast_size_tuning curves loader.")

            title = (
                f"Contrast size tuning – neuron {nid} (shift={sh:.2f})"
                if np.isfinite(sh) else
                f"Contrast size tuning – neuron {nid} (shift=nan)"
            )

            # plot_contrast_size_tuning_both(
            #     neuron_id=nid,
            #     radii=radii,
            #     contrasts=contrasts,
            #     curves=mat,
            #     low_contrast_id=plot_low,
            #     high_contrast_id=plot_high,
            #     title=title,
            # )

    print("--------------------------------------")
    print() 



def plot_contrast_size_tuning_curves(
    neuron: str,
    *,
    radii,
    contrasts,
    Y_raw,          # (C,R)
    Y_fit=None,     # (C,R) or None
    low_contrast_id=0,
    high_contrast_id=-1,
    save_dir="/project/results/nature_and_interactions/contrast_size_tuning/plots/",
    use_diameter=True,
    title=None,
    style="paper",          # "paper" | "diagnostic"
    show_points=True,
    show_legend=None,       # None -> style default
    legend_ncols=2,
    dpi=300,
):
    """
    Plot size tuning at multiple contrasts (contrast-size tuning).
    curves: rows are contrasts, cols are radii
    """
    ensure_dir(save_dir)

    style = (style or "paper").lower()
    if style not in ("paper", "diagnostic"):
        raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'diagnostic'.")

    radii = np.asarray(radii, float)
    contrasts = np.asarray(contrasts, float)
    Y_raw = np.asarray(Y_raw, float)
    if Y_fit is not None:
        Y_fit = np.asarray(Y_fit, float)

    x = 2 * radii if use_diameter else radii
    xlab = "Diameter (deg)" if use_diameter else "Radius (deg)"

    C = len(contrasts)
    low_i = int(low_contrast_id) % C
    high_i = int(high_contrast_id) % C

    # choose which curves to use for markers/shift (prefer fit)
    Y_for_markers = Y_fit if (Y_fit is not None) else Y_raw
    low_curve = Y_for_markers[low_i]
    high_curve = Y_for_markers[high_i]

    GSF_low, *_  = get_GSF_surround_AMRF(radii=radii, circular_tuning_curve=low_curve, annular_tuning_curve=None)
    GSF_high, *_ = get_GSF_surround_AMRF(radii=radii, circular_tuning_curve=high_curve, annular_tuning_curve=None)

    shift = None
    if (GSF_low is not None) and (GSF_high is not None) and np.isfinite(GSF_high) and (GSF_high > 0):
        shift = float(GSF_low / GSF_high)

    # -------------------------
    # STYLE PRESETS (no data ops)
    # -------------------------
    if style == "paper":
        # grayscale by contrast value (low contrast lighter, high contrast darker)
        cmin, cmax = float(np.min(contrasts)), float(np.max(contrasts))
        den = (cmax - cmin) if (cmax > cmin) else 1.0
        intensity = 0.80 - (0.80 - 0.15) * ((contrasts - cmin) / den)  # 0.80..0.15
        colors = [(float(t), float(t), float(t)) for t in intensity]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        line_lw_emph, line_lw_other = 2.6, 1.8
        fit_alpha_emph, fit_alpha_other = 0.95, 0.60
        pt_alpha_emph, pt_alpha_other = 0.90, 0.55
        pt_size = 22
        grid_alpha = 0.15
        marker_face = "none"   # open circles
        if show_legend is None:
            show_legend = True  # paper often still wants legend here, keep it on

    else:  # diagnostic
        cmap = plt.get_cmap("turbo") if C > 10 else plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(C)]

        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        line_lw_emph, line_lw_other = 3.0, 2.0
        fit_alpha_emph, fit_alpha_other = 1.00, 0.75
        pt_alpha_emph, pt_alpha_other = 0.85, 0.60
        pt_size = 26
        grid_alpha = 0.25
        marker_face = None     # filled circles
        if show_legend is None:
            show_legend = True

    # -------------------------
    # Plot
    # -------------------------
    legend_handles = []

    for i, c in enumerate(contrasts):
        color = colors[i]
        is_emph = i in (low_i, high_i)

        # raw points
        if show_points:
            if marker_face == "none":
                ax.scatter(
                    x, Y_raw[i],
                    s=pt_size,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    alpha=pt_alpha_emph if is_emph else pt_alpha_other,
                    zorder=3,
                )
            else:
                ax.scatter(
                    x, Y_raw[i],
                    s=pt_size,
                    color=color,
                    edgecolors="none",
                    alpha=pt_alpha_emph if is_emph else pt_alpha_other,
                    zorder=3,
                )

        # fit line (preferred)
        if Y_fit is not None:
            ax.plot(
                x, Y_fit[i],
                color=color,
                lw=line_lw_emph if is_emph else line_lw_other,
                alpha=fit_alpha_emph if is_emph else fit_alpha_other,
                zorder=4,
            )
        else:
            # fallback: connect raw with line if no fit
            ax.plot(
                x, Y_raw[i],
                color=color,
                lw=1.6,
                alpha=0.8,
                zorder=2,
            )

        # legend handle for all curves (paper wants readability; diagnostic wants completeness)
        label = f"{100*c:.0f}%" if c <= 1.0 else f"{c:.2f}"
        legend_handles.append(Line2D(
            [0], [0],
            color=color,
            lw=(line_lw_emph if is_emph else line_lw_other) if (Y_fit is not None) else 0.0,
            marker="o",
            markersize=5,
            markerfacecolor=("none" if marker_face == "none" else color),
            markeredgecolor=color,
            linestyle="-" if (Y_fit is not None) else "None",
            label=label
        ))

    # markers
    def mark(val, label):
        if val is None:
            return
        if (not np.isfinite(val)) or (val <= 0):
            return
        xv = 2 * val if use_diameter else val
        ax.axvline(xv, lw=2, ls=":", alpha=0.9)
        ax.text(xv, ax.get_ylim()[1] * 0.98, label, rotation=90, va="top", ha="right")

    mark(GSF_low, "GSF low")
    mark(GSF_high, "GSF high")

    # title
    if title is None:
        title = f"{neuron} – contrast size tuning" + (f" (shift={shift:.2f})" if shift is not None else "")
    ax.set_title(title)

    ax.set_xlabel(xlab)
    ax.set_ylabel("Response (Δ from gray)")
    ax.grid(True, alpha=grid_alpha)
    ax.set_axisbelow(True)

    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            title="Contrast",
            ncols=legend_ncols,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    # IMPORTANT: style suffix so the two calls don't overwrite each other
    out = os.path.join(save_dir, f"{neuron}_contrast_size_tuning_overlay_{style}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # print (kept)
    msg = f"[{neuron}] "
    if GSF_low is not None:  msg += f"GSF_low={2*GSF_low:.3f}° "
    if GSF_high is not None: msg += f"GSF_high={2*GSF_high:.3f}° "
    if shift is not None:    msg += f"shift={shift:.3f}"
    print(msg.strip())

    return shift, out

def plot_contrast_size_tuning_both(*args, **kwargs):
    """Convenience wrapper: saves paper + diagnostic versions."""
    shift_paper, out_paper = plot_contrast_size_tuning_curves(*args, style="paper", **kwargs)
    shift_diag,  out_diag  = plot_contrast_size_tuning_curves(*args, style="diagnostic", **kwargs)
    # shift should match; return one + both paths
    return shift_paper, out_paper, out_diag