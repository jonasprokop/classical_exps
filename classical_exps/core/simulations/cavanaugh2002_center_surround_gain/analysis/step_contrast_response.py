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
from .shared import sort_by_spread, ensure_dir
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
from matplotlib.lines import Line2D


def contrast_response_results_bundle(
    bundles,
    *,
    default_sort_by_std=False,
    default_spread_to_plot=(15, 50, 85),
    default_xscale="linear",
    default_print_fit_summary=True,
):
    """
    Run contrast_response_results_1 over multiple datasets with one outer call.

    Each bundle entry is:
        {
            "label": str,
            "neuron_ids": [...],
            "filtered_neuron_ids": [...],
            "loaded": loaded_like,
            "sort_by_std": bool,              # optional
            "spread_to_plot": [...],          # optional
            "plot_all": bool,                 # optional
            "xscale": str,                    # optional
            "print_fit_summary": bool,        # optional
        }
    """
    for bundle in bundles:
        print(f"    > {bundle['label']}")
        contrast_response_results_1(
            neuron_ids=bundle["neuron_ids"],
            filtered_neuron_ids=bundle["filtered_neuron_ids"],
            loaded=bundle["loaded"],
            sort_by_std=bundle.get("sort_by_std", default_sort_by_std),
            spread_to_plot=bundle.get("spread_to_plot", list(default_spread_to_plot)),
            plot_all=bundle.get("plot_all", True),
            xscale=bundle.get("xscale", default_xscale),
            print_fit_summary=bundle.get("print_fit_summary", default_print_fit_summary),
        )

def plot_contrast_response(
    neuron_id,
    center_contrasts,
    surround_contrasts,
    contrast_response_curves,        # raw: shape (n_surround, n_center)
    contrast_response_fits=None,     # fit: same shape, optional
    title=None,
    save_dir="/project/results/nature_and_interactions/contrast_response/plots/",
    style="paper",                   # "paper" | "diagnostic"
    xscale="symlog",                 # "log" | "linear" | "symlog"
    symlog_linthresh=1e-3,
    sort_surround=True,
    show_points=True,
    show_legend=True,
    legend_ncols=2,
    dpi=300,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

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
            raise ValueError(
                f"contrast_response_fits shape mismatch: expected {y_raw.shape}, got {y_fit.shape}"
            )

    if sort_surround:
        order = np.argsort(s)
        s = s[order]
        y_raw = y_raw[order]
        if y_fit is not None:
            y_fit = y_fit[order]

    if title is None:
        title = f"Contrast response – neuron {neuron_id}"

    os.makedirs(save_dir, exist_ok=True)

    if style == "paper":
        smin, smax = float(np.min(s)), float(np.max(s))
        den = (smax - smin) if (smax > smin) else 1.0
        intensity = 0.80 - (0.80 - 0.15) * ((s - smin) / den)
        colors = [(float(t), float(t), float(t)) for t in intensity]

        line_lw = 2.2
        fit_alpha = 0.95
        pt_alpha = 0.9
        pt_size = 22
        grid_alpha = 0.15
        marker_face = "none"
        marker_edge = None
        legend_title = "Surround contrast"

    elif style == "diagnostic":
        cmap = plt.get_cmap("turbo") if len(s) > 10 else plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(s))]

        line_lw = 2.4
        fit_alpha = 0.95
        pt_alpha = 0.85
        pt_size = 26
        grid_alpha = 0.25
        marker_face = None
        marker_edge = "none"
        legend_title = "Surround contrast"
    else:
        raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'diagnostic'.")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    legend_handles = []

    for i in range(len(s)):
        color = colors[i]
        label = f"{float(s[i]):.2f}"

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

        if y_fit is not None and np.any(np.isfinite(y_fit[i])):
            ax.plot(
                x, y_fit[i],
                color=color,
                linewidth=line_lw,
                alpha=fit_alpha,
                zorder=4
            )

            legend_handles.append(Line2D(
                [0], [0],
                color=color,
                lw=line_lw,
                marker="o",
                markersize=5,
                markerfacecolor=("none" if marker_face == "none" else color),
                markeredgecolor=color,
                linestyle="-",
                label=label
            ))
        else:
            ax.plot(
                x, y_raw[i],
                color=color,
                linewidth=1.6,
                alpha=0.8,
                zorder=2
            )
            legend_handles.append(Line2D(
                [0], [0],
                color=color,
                lw=0.0,
                marker="o",
                markersize=5,
                markerfacecolor=("none" if marker_face == "none" else color),
                markeredgecolor=color,
                linestyle="None",
                label=label
            ))

    ax.set_title(title)
    ax.set_xlabel("Center contrast")
    ax.set_ylabel("Response (Δ from gray)")

    if xscale == "log":
        if np.any(x <= 0):
            raise ValueError(
                "xscale='log' but center_contrasts contains 0 or negative values. "
                "Use xscale='symlog' or 'linear'."
            )
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

    out = os.path.join(save_dir, f"contrast_response_{neuron_id}_{style}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_contrast_response_both(*args, **kwargs):
    out_paper = plot_contrast_response(*args, style="paper", **kwargs)
    out_diag = plot_contrast_response(*args, style="diagnostic", **kwargs)
    return out_paper, out_diag

def contrast_response_results_1(
    neuron_ids,
    filtered_neuron_ids,
    loaded,
    sort_by_std=False,
    spread_to_plot=(15, 50, 85),
    plot_all=True,
    xscale="symlog",
    print_fit_summary=True,
):
    """
    Visualize representative contrast response curves.

    Expected loaded structure:
        loaded["contrast_response"] = {
            "center_contrasts": 1D array-like,
            "surround_contrasts": 1D array-like,
            "by_id": {
                neuron_id: 2D array of shape (n_surround, n_center)
            }
        }

    Optional:
        loaded["contrast_response_fits"] = {
            "by_id": {
                neuron_id: {
                    "y_fit": 2D array same shape as raw
                }
            }
        }

        loaded["contrast_response_results"] = {
            neuron_id: spread_metric
        }

        loaded["contrast_response_fit_summary"] = {
            neuron_id: [...]
        }
    """

    if filtered_neuron_ids is None:
        raise ValueError("contrast_response_results_1 requires filtered_neuron_ids.")
    if loaded is None:
        raise ValueError("contrast_response_results_1 requires loaded dict.")
    if "contrast_response" not in loaded:
        raise ValueError("loaded missing 'contrast_response'.")

    contrast_response = loaded["contrast_response"]
    center_contrasts = np.asarray(contrast_response["center_contrasts"], dtype=float)
    surround_contrasts = np.asarray(contrast_response["surround_contrasts"], dtype=float)
    curves_by_id = contrast_response["by_id"]
    fits_by_id = loaded.get("contrast_response_fits", {}).get("by_id", {})
    fit_summary_by_id = loaded.get("contrast_response_fit_summary", {})

    filtered_neuron_ids = list(filtered_neuron_ids)

    # Keep it simple: preserve provided order unless explicit sortable metrics exist
    sorted_neuron_ids = [nid for nid in filtered_neuron_ids if nid in curves_by_id]
    sorted_spread = np.full(len(sorted_neuron_ids), np.nan, dtype=float)
    median_spread = np.nan

    # Optional simple sort for loaders that expose neuron_id -> scalar metric directly
    if sort_by_std and "contrast_response_results" in loaded:
        spread_results = loaded["contrast_response_results"]

        if isinstance(spread_results, dict) and all(
            not isinstance(v, dict) for v in spread_results.values()
        ):
            sortable = [(nid, spread_results.get(nid, np.nan)) for nid in sorted_neuron_ids]
            sortable = [(nid, val) for nid, val in sortable if np.isfinite(val)]
            unsortable = [nid for nid in sorted_neuron_ids if not np.isfinite(spread_results.get(nid, np.nan))]

            sortable.sort(key=lambda x: x[1])
            sorted_neuron_ids = [nid for nid, _ in sortable] + unsortable
            sorted_spread = np.asarray([val for _, val in sortable], dtype=float)
            median_spread = round(float(np.nanmedian(sorted_spread)), 2) if len(sorted_spread) else np.nan


    n = len(neuron_ids)
    n_new = len(sorted_neuron_ids)

    print("--------------------------------------")
    print("Visualisation of representative contrast response curves :")
    print(f"    > Analysis made on {n} neurons")
    if n > 0:
        print(f"    > {n_new} neurons ({round((n_new / n * 100), 2)}%) left after filtration")
    else:
        print(f"    > {n_new} neurons left after filtration")

    if "contrast_response_results" in loaded:
        if sort_by_std:
            print(f"    > The median of mean_std is {median_spread}")
        else:
            print(f"    > The median of max/min is {median_spread}")
    else:
        print("    > No spread metrics found; plotting in provided filtered order.")

    if "contrast_response_fit_summary" in loaded and print_fit_summary:
        print("    > Fit summaries available and will be printed per plotted neuron.")

    print("    > Plots :")

    if n_new == 0:
        print("    > No valid neurons to plot.")
        print("--------------------------------------")
        print()
        return

    if plot_all or (isinstance(spread_to_plot, str) and spread_to_plot.lower() == "all"):
        ids_to_plot = list(sorted_neuron_ids)
        print(f"    > Plotting ALL curves: {len(ids_to_plot)} neurons")
        label_mode = "ranked"
    else:
        ids_to_plot = []
        for spread_percent in list(spread_to_plot):
            neuron_pos = int((spread_percent / 100) * (n_new - 1))
            neuron_id = sorted_neuron_ids[neuron_pos]
            ids_to_plot.append(neuron_id)
        label_mode = "requested_percentiles"

    for idx, neuron_id in enumerate(ids_to_plot):
        mat_raw = curves_by_id.get(neuron_id)
        fit_entry = fits_by_id.get(neuron_id)
        mat_fit = None if fit_entry is None else fit_entry.get("y_fit")

        scraped_raw_all = loaded.get("contrast_response_scraped_raw", None)
        scraped_fit_all = loaded.get("contrast_response_scraped_fit", None)

        scraped_raw_curves = None
        scraped_fit_curves = None

        if scraped_raw_all is not None and neuron_id in scraped_raw_all:
            scraped_raw_curves = scraped_raw_all[neuron_id]

        if scraped_fit_all is not None and neuron_id in scraped_fit_all:
            scraped_fit_curves = scraped_fit_all[neuron_id]

        if mat_raw is None and scraped_raw_curves is None:
            print(f"    > Skipping neuron {neuron_id}: missing raw contrast-response data.")
            continue

        if label_mode == "ranked":
            spread_percent = 100 * idx / (len(ids_to_plot) - 1) if len(ids_to_plot) > 1 else 0.0
            spread_percent = round(spread_percent, 1)
            title = (
                f"Response contrast for a spread above {spread_percent}% of the neurons "
                f"(neuron {neuron_id})"
            )
        else:
            title = f"Contrast response – neuron {neuron_id}"

        print(f"    > neuron {neuron_id}")

        if neuron_id in fit_summary_by_id and print_fit_summary:
            summary = fit_summary_by_id[neuron_id]

            if isinstance(summary, list):
                print("        Fit rows:")
                for row in summary:
                    msg = (
                        f"            row={row.get('row_idx', 'NA')}, "
                        f"n={row.get('n_points', 'NA')}, "
                        f"status={row.get('status', 'NA')}, "
                        f"rmse={row.get('rmse', 'NA')}"
                    )
                    extras = []
                    for k, v in row.items():
                        if k not in {'row_idx', 'n_points', 'status', 'rmse'}:
                            extras.append(f"{k}={v}")
                    if extras:
                        msg += ", " + ", ".join(extras)
                    print(msg)

            elif isinstance(summary, dict):
                print("        Joint fit summary:")
                print(
                    f"            success={summary.get('success', 'NA')}, "
                    f"cost={summary.get('cost', 'NA')}, "
                    f"nfev={summary.get('nfev', 'NA')}"
                )
                params = summary.get("params", {})
                if isinstance(params, dict) and len(params) > 0:
                    print(f"            params={params}")

            else:
                print(f"        Unrecognized fit summary format: {type(summary)}")

        if scraped_raw_curves is not None:
            plot_scraped_contrast_response_both(
                neuron_id=neuron_id,
                scraped_raw_curves=scraped_raw_curves,
                scraped_fit_curves=scraped_fit_curves,
                title=title,
                xscale=xscale,
            )
        else:
            plot_contrast_response_both(
                neuron_id=neuron_id,
                center_contrasts=center_contrasts,
                surround_contrasts=surround_contrasts,
                contrast_response_curves=mat_raw,
                contrast_response_fits=mat_fit,
                title=title,
                xscale=xscale,
            )

    print("--------------------------------------")
    print()

def plot_scraped_contrast_response(
    neuron_id,
    scraped_raw_curves,
    *,
    scraped_fit_curves=None,
    title=None,
    save_dir="/project/results/nature_and_interactions/contrast_response/plots/",
    style="paper",
    xscale="linear",
    symlog_linthresh=1e-3,
    sort_surround=True,
    show_points=True,
    show_legend=True,
    legend_ncols=2,
    dpi=300,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    if not isinstance(scraped_raw_curves, dict) or len(scraped_raw_curves) == 0:
        raise ValueError(f"scraped_raw_curves must be a non-empty dict for neuron {neuron_id}")

    surround_keys = list(scraped_raw_curves.keys())
    surround_vals = np.asarray([float(k) for k in surround_keys], dtype=float)

    if sort_surround:
        order = np.argsort(surround_vals)
        surround_vals = surround_vals[order]
        surround_keys = [surround_keys[i] for i in order]

    if title is None:
        title = f"Contrast response – neuron {neuron_id}"

    os.makedirs(save_dir, exist_ok=True)

    if style == "paper":
        smin, smax = float(np.min(surround_vals)), float(np.max(surround_vals))
        den = (smax - smin) if (smax > smin) else 1.0
        intensity = 0.80 - (0.80 - 0.15) * ((surround_vals - smin) / den)
        colors = [(float(t), float(t), float(t)) for t in intensity]

        line_lw = 2.2
        fit_alpha = 1.0
        pt_alpha = 0.9
        pt_size = 22
        grid_alpha = 0.15
        marker_face = "none"
        marker_edge = None
        legend_title = "Surround contrast"

    elif style == "diagnostic":
        cmap = plt.get_cmap("turbo") if len(surround_vals) > 10 else plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(surround_vals))]

        line_lw = 2.4
        fit_alpha = 1.0
        pt_alpha = 0.85
        pt_size = 26
        grid_alpha = 0.25
        marker_face = None
        marker_edge = "none"
        legend_title = "Surround contrast"
    else:
        raise ValueError(f"Unknown style={style!r}. Use 'paper' or 'diagnostic'.")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    legend_handles = []

    for i, s_key in enumerate(surround_keys):
        color = colors[i]
        s_val = float(s_key)
        label = f"{s_val:.2f}"

        raw_entry = scraped_raw_curves[s_key]
        x_raw = np.asarray(raw_entry["center_contrast"], dtype=float)
        y_raw = np.asarray(raw_entry["response"], dtype=float)

        if x_raw.ndim != 1 or y_raw.ndim != 1 or len(x_raw) != len(y_raw):
            raise ValueError(
                f"Bad scraped raw curve for neuron {neuron_id}, surround {s_key}: "
                f"x shape {x_raw.shape}, y shape {y_raw.shape}"
            )

        order_raw = np.argsort(x_raw)
        x_raw = x_raw[order_raw]
        y_raw = y_raw[order_raw]

        if show_points:
            if marker_face == "none":
                ax.scatter(
                    x_raw, y_raw,
                    s=pt_size,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    alpha=pt_alpha,
                    zorder=3
                )
            else:
                ax.scatter(
                    x_raw, y_raw,
                    s=pt_size,
                    color=color,
                    edgecolors=marker_edge,
                    alpha=pt_alpha,
                    zorder=3
                )

        fit_drawn = False
        if scraped_fit_curves is not None and s_key in scraped_fit_curves:
            fit_entry = scraped_fit_curves[s_key]
            x_fit = np.asarray(fit_entry["center_contrast"], dtype=float)
            y_fit = np.asarray(fit_entry["response"], dtype=float)

            if x_fit.ndim != 1 or y_fit.ndim != 1 or len(x_fit) != len(y_fit):
                raise ValueError(
                    f"Bad scraped fit curve for neuron {neuron_id}, surround {s_key}: "
                    f"x shape {x_fit.shape}, y shape {y_fit.shape}"
                )

            order_fit = np.argsort(x_fit)
            x_fit = x_fit[order_fit]
            y_fit = y_fit[order_fit]

            valid_fit = np.isfinite(x_fit) & np.isfinite(y_fit)
            if np.any(valid_fit):
                ax.plot(
                    x_fit[valid_fit],
                    y_fit[valid_fit],
                    color=color,
                    linewidth=line_lw,
                    alpha=fit_alpha,
                    zorder=4
                )
                fit_drawn = True

        legend_handles.append(Line2D(
            [0], [0],
            color=color,
            lw=(line_lw if fit_drawn else 0.0),
            marker="o",
            markersize=5,
            markerfacecolor=("none" if marker_face == "none" else color),
            markeredgecolor=color,
            linestyle=("-" if fit_drawn else "None"),
            label=label
        ))

    ax.set_title(title)
    ax.set_xlabel("Center contrast")
    ax.set_ylabel("Response (Δ from gray)")

    if xscale == "log":
        positive_x = []
        for s_key in surround_keys:
            positive_x.extend([
                xx for xx in scraped_raw_curves[s_key]["center_contrast"]
                if float(xx) > 0
            ])
        if len(positive_x) == 0:
            raise ValueError("xscale='log' but scraped curves contain no positive x values.")
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

    out = os.path.join(save_dir, f"contrast_response_{neuron_id}_{style}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scraped_contrast_response_both(*args, **kwargs):
    out_paper = plot_scraped_contrast_response(*args, style="paper", **kwargs)
    out_diag = plot_scraped_contrast_response(*args, style="diagnostic", **kwargs)
    return out_paper, out_diag
