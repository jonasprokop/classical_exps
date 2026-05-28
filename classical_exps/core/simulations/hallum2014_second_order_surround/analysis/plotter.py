import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from classical_exps.core.simulations.hallum2014_second_order_surround.analysis.tools import (
    wrap_deg, 
    vec_from_deg_top0, 
    draw_degree_spokes, 
    _unit_circle_ax
)

def plot_second_order_orientation_preferences(
    second_order_preferences,
    results_sosi=None,
    bins=6,
    save_path="/project/results/modulation/second_order_preferences.png",
):
    """
    Plot distribution of second-order orientation preferences counted based on sosi significance of neurons.
    
    Parameters:
    - second_order_preferences: Array of orientation preferences
    - results_sosi: Array containing significance information
    - bins: Number of bins to use for histogram (default 5)
    """

    # radians -> degrees
    prefs_deg = np.degrees(second_order_preferences)

    # keep values inside [-90, 90]
    # useful if numerical precision gives e.g. 90.0000001
    prefs_deg = np.clip(prefs_deg, -90, 90)

    # Manual Cavanaugh-like bins:
    # narrow edge bins, wider inner bins
    edges = np.array([-90, -67.5, -22.5, 22.5, 67.5, 90])
    widths = np.diff(edges)

    counts, _ = np.histogram(prefs_deg, bins=edges)

    # Circular edge split:
    # values near -90 and +90 represent the same orientation,
    # so show the combined edge mass split across both sides.
    edge_sum = counts[0] + counts[-1]
    counts[0] = edge_sum / 2
    counts[-1] = edge_sum / 2

    props = counts / len(prefs_deg)

    # Save raw data
    np.save("/project/results/modulation/second_order_preferences.npy", prefs_deg)

    if results_sosi is not None:
        np.save("/project/results/modulation/results_sosi.npy", results_sosi)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    ax.bar(
        edges[:-1],
        props,
        width=widths,
        align="edge",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
    )

    # Axes limits
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, props.max() * 1.18)

    # Cavanaugh-like x ticks
    ax.set_xticks([-90, -45, 0, 45, 90])

    # Or, if you want fixed interval:
    # ax.xaxis.set_major_locator(MultipleLocator(45))

    # Labels
    ax.set_xlabel(
        "Second-order orientation preference\nrelative to carrier orientation (deg)",
        fontsize=11,
        fontstyle="italic",
    )

    ax.set_ylabel("Proportion of cells", fontsize=11)

    # Remove title for poster / article style
    ax.set_title("")

    # Minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # In the original, y-axis is visually de-emphasized.
    # Keep y ticks if you want quantitative readability:
    ax.tick_params(axis="y", labelsize=9, width=0.8, length=3)

    # X-axis styling
    ax.tick_params(axis="x", labelsize=10, width=0.8, length=5)

    # Optional: italic x tick labels like the source figure
    for label in ax.get_xticklabels():
        label.set_fontstyle("italic")

    # Make baseline clean
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


def plot_sosi_histogram(
    results_sosi,
    bins=8,
    save_path="/project/results/modulation/osi_distribution.png",
):
    """
    Plot SOSI distribution.

    If results_sosi is a full statistical-test array, only the first column
    is used for plotting. Significance is ignored here.
    """

    results_sosi = np.asarray(results_sosi)

    # Accept either:
    # - shape (N,)     -> SOSI values directly
    # - shape (N, >=1) -> SOSI values in first column
    if results_sosi.ndim == 1:
        sosi_values = results_sosi.astype(float)
    else:
        sosi_values = results_sosi[:, 0].astype(float)

    sosi_values = sosi_values[np.isfinite(sosi_values)]

    if len(sosi_values) == 0:
        raise ValueError("No finite SOSI values found.")

    min_val, max_val = sosi_values.min(), sosi_values.max()

    # Avoid zero-width range
    if min_val == max_val:
        pad = 0.5 if min_val == 0 else abs(min_val) * 0.1
        min_val -= pad
        max_val += pad

    counts, bin_edges = np.histogram(
        sosi_values,
        bins=bins,
        range=(min_val, max_val),
        density=False,
    )

    proportions = counts / len(sosi_values)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    for left, right, height in zip(bin_edges[:-1], bin_edges[1:], proportions):
        ax.add_patch(
            Rectangle(
                (left, 0),
                right - left,
                height,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
            )
        )

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, proportions.max() * 1.18 if proportions.max() > 0 else 1)

    ax.set_xlabel("SOSI", fontsize=11, fontstyle="italic")
    ax.set_ylabel("Proportion of cells", fontsize=11)

    # Optional fixed tick spacing.
    # Change 0.5 to whatever looks best for the actual SOSI range.
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    # Article-ish style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    ax.tick_params(axis="x", direction="out", length=5, width=0.8, labelsize=10)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8, labelsize=9)

    for label in ax.get_xticklabels():
        label.set_fontstyle("italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_circular_tunning_curves(raw_data, scaling_factor=100, neuron="0", show_only_averaged_phase=True):

    # For each neuron plots its circlular tunning

    relative_orientations, responses = raw_data

    unique_orientations = np.unique(relative_orientations)
    averaged_responses = []
    averaged_relative_ori = []
    for unique_orientation in unique_orientations:
        indices_orientation = tuple(np.argwhere(relative_orientations == unique_orientation)[0])
        resposes_orientation = np.take(responses, indices_orientation)
        average_response = np.average(resposes_orientation)
        averaged_responses.append(average_response)
        averaged_relative_ori.append(unique_orientation)
    averaged_responses = np.array(averaged_responses)
    averaged_relative_ori = np.array(averaged_relative_ori)
    responses = averaged_responses
    relative_orientations = averaged_relative_ori

    # Scale the responses for better vsibility
    area = scaling_factor * responses
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='polar')

    # Scatter plot with linear scaled area
    c = ax.scatter(relative_orientations, responses, 
                c=responses, s=area, alpha=0.75)

    # Add a colorbar to show response intensity
    plt.colorbar(c, label='Neuron Response')

    # Title and labels
    plt.title('Neuron Responses in Polar Coordinates')
    plt.tight_layout()

    directory = "/project/results/modulation/circular_tunning"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"{neuron}_circular_tunning_curve.png")
    plt.close()



def save_phase_vector_circle(phases, responses, outpath, neuron, rel_ori, max_points=120, show_weighted=True):
    phases = np.asarray(phases)
    responses = np.asarray(responses)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rel_ori_deg = wrap_deg(np.rad2deg(rel_ori), 180)
    _unit_circle_ax(ax, title=f"{neuron} | phase vectors (ϕ) @ rel ori={rel_ori_deg:.0f}°")

    draw_degree_spokes(ax, [0, 90, 180, 270])  # now correct

    w_norm = np.zeros_like(responses) if np.all(responses == 0) else responses / (np.max(responses) + 1e-12)

    V = 0 + 0j
    for phi, r, rn in zip(phases, responses, w_norm):
        phi_deg = wrap_deg(np.rad2deg(phi), 360)
        ux, uy = vec_from_deg_top0(phi_deg)

        ax.scatter([ux], [uy], s=25, alpha=0.75)
        ax.arrow(0, 0, ux*rn, uy*rn,
                 length_includes_head=True,
                 head_width=0.03, head_length=0.04,
                 linewidth=0.9, alpha=0.35)

        V += r * np.exp(1j * phi)

    mag = np.abs(V)
    phi_pref = np.angle(V)
    phi_pref_deg = wrap_deg(np.rad2deg(phi_pref), 360)

    rx, ry = vec_from_deg_top0(phi_pref_deg)
    ax.arrow(0, 0, rx, ry,
             length_includes_head=True,
             head_width=0.06, head_length=0.08,
             linewidth=2.6, alpha=0.95, label="resultant")

    ax.text(-1.15, -1.17, f"|Σ r·e^{{iϕ}}| = {mag:.3g} | ϕ_pref = {phi_pref_deg:.0f}°", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def _semi_circle_ax(ax, title=None):
    """Semicircle for orientation (θ in [0,180)), with 0° at top."""
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, linewidth=0.8, alpha=0.6)  # horizontal reference
    ax.axvline(0, linewidth=0.8, alpha=0.6)  # vertical reference

    # Semicircle: from top (0°) -> right (90°) -> bottom (180°)
    # Parameterize by θ in degrees: x=sin(θ), y=cos(θ)
    th = np.linspace(0, 180, 512)
    x = np.sin(np.deg2rad(th))
    y = np.cos(np.deg2rad(th))
    ax.plot(x, y, linewidth=1.2)

    # Only show right half-plane so it's visually "half a circle"
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)


def save_orientation_vector_circle(rel_oris, phase_invariant_amplitudes, outpath, neuron):
    rel_oris = np.asarray(rel_oris).astype(float)
    A = np.asarray(phase_invariant_amplitudes).astype(float)

    m = np.isfinite(rel_oris) & np.isfinite(A) & (A >= 0)
    rel_oris, A = rel_oris[m], A[m]
    if rel_oris.size == 0:
        raise ValueError("No valid orientation samples.")

    # θ ∈ [0, π)
    rel_oris = np.mod(rel_oris, np.pi)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _unit_circle_ax(ax, title=f"{neuron} | orientation tuning (θ), math uses 2θ")

    # Label θ only
    draw_degree_spokes(ax, [0, 45, 90, 135, 180])

    # Normalize for visuals
    A_norm = np.zeros_like(A) if np.all(A == 0) else A / (np.max(A) + 1e-12)

    # ---- computation in 2θ (unchanged, correct)
    V = np.sum(A * np.exp(2j * rel_oris))
    denom = np.sum(A) + 1e-12
    sosi = np.abs(V) / denom

    # ---- display in θ (unidirectional)
    xs = np.sin(rel_oris)
    ys = np.cos(rel_oris)

    ax.scatter(xs, ys, s=45, alpha=0.85, label=r"samples at $\theta$")

    for x, y, wn in zip(xs, ys, A_norm):
        ax.arrow(
            0, 0, x*wn, y*wn,
            length_includes_head=True,
            head_width=0.03, head_length=0.04,
            linewidth=1.2, alpha=0.45
        )

    # Resultant: show DIRECTION on the unit circle + show MAGNITUDE as a radial bar
    if np.abs(V) > 0:
        theta_pref = 0.5 * np.angle(V)
        theta_pref = np.mod(theta_pref, np.pi)
        theta_pref_deg = wrap_deg(np.rad2deg(theta_pref), 180)

        rx, ry = vec_from_deg_top0(theta_pref_deg)

        # 1) magnitude bar (0 -> sOSI)
        ax.arrow(
            0, 0, rx*sosi, ry*sosi,
            length_includes_head=True,
            head_width=0.05, head_length=0.07,
            linewidth=3.2, alpha=0.95, label="resultant magnitude"
        )

        # 2) direction arrow to the unit circle (always ends on the circle)
        # ax.arrow(
        #     0, 0, rx*1.0, ry*1.0,
        #     length_includes_head=True,
        #     head_width=0.03, head_length=0.05,
        #     linewidth=1.2, alpha=0.55, label="resultant direction"
        # )

        # # optional: mark the unit-circle tip so it's obvious
        # ax.scatter([rx], [ry], s=55, alpha=0.85)
    else:
        theta_pref_deg = np.nan

    ax.text(
        -1.15, -1.17,
        f"SOSI = {sosi:.3f} | θ_pref = {theta_pref_deg:.1f}°",
        fontsize=10
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def _get_scraped_distribution(scraped_config, key):
    """
    Accept either:
    - direct Hallum scraped config dict
    - wrapper dict containing hallum2014_second_order_surround_scraped_data
    """
    if scraped_config is None:
        return None

    if key in scraped_config:
        return scraped_config[key]

    wrapper_key = "hallum2014_second_order_surround_scraped_data"
    if wrapper_key in scraped_config and key in scraped_config[wrapper_key]:
        return scraped_config[wrapper_key][key]

    return None


def _infer_edges_from_centers(centers, *, xlim=None):
    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Need at least two bin centers to infer edges.")

    mids = (centers[:-1] + centers[1:]) / 2

    if xlim is None:
        first_width = centers[1] - centers[0]
        last_width = centers[-1] - centers[-2]
        left = centers[0] - first_width / 2
        right = centers[-1] + last_width / 2
    else:
        left, right = map(float, xlim)

    return np.r_[left, mids, right]

def _style_hallum_small_axis(ax, *, fontsize=8):
    """
    Shared compact black-and-white axis styling for Hallum comparison plots.
    Keeps tick labels, axis labels, legend, spines, and tick marks visually unified.
    """

    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        width=0.7,
        length=3.0,
        pad=2.0,
    )

    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)
def plot_second_order_orientation_preferences_scraped_comparison(
    second_order_preferences,
    results_sosi=None,
    *,
    scraped_config=None,
    save_path="/project/results/modulation/second_order_preferences_scraped_comparison.png",
):
    """
    Model-vs-Hallum comparison for second-order orientation preference.

    Important:
    This is a wrapped orientation-space histogram. The edge bars at -90 and +90
    are visual halves of the same circular bin. For clarity, we use categorical
    display positions with equal visual bin widths, while keeping degree labels.
    """

    fontsize = 8

    scraped = _get_scraped_distribution(
        scraped_config,
        "preferred_second_order_orientation_distribution",
    )
    if scraped is None:
        raise ValueError("Missing preferred_second_order_orientation_distribution.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    prefs_deg = np.degrees(second_order_preferences)
    prefs_deg = np.clip(prefs_deg, -90, 90)
    prefs_deg = prefs_deg[np.isfinite(prefs_deg)]

    if prefs_deg.size == 0:
        raise ValueError("No finite second-order preference values found.")

    # Real circular binning in degree space.
    # Edge bins are the split wraparound bin.
    degree_edges = np.array([-90, -67.5, -22.5, 22.5, 67.5, 90], dtype=float)

    counts, _ = np.histogram(prefs_deg, bins=degree_edges)

    edge_total = counts[0] + counts[-1]
    model_props = np.array(
        [
            edge_total / 2,
            counts[1],
            counts[2],
            counts[3],
            edge_total / 2,
        ],
        dtype=float,
    ) / prefs_deg.size

    exp_eval = np.asarray(
        scraped.get(
            "wrapped_proportion_of_cells_evaluated",
            scraped["wrapped_proportion_of_cells"],
        ),
        dtype=float,
    )

    if exp_eval.size != 4:
        raise ValueError("Expected four experimental orientation bins.")

    exp_props = np.array(
        [
            exp_eval[0] / 2,
            exp_eval[1],
            exp_eval[2],
            exp_eval[3],
            exp_eval[0] / 2,
        ],
        dtype=float,
    )

    # Display positions: equal visual slots.
    # This prevents the edge split-bin from looking like a broken half-width bin.
    x = np.arange(5, dtype=float)
    xlabels = ["−90", "−45", "0", "45", "90"]

    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    total_width = 0.78
    inner_gap = 0.08
    bar_width = (total_width - inner_gap) / 2

    exp_left = x - total_width / 2
    model_left = exp_left + bar_width + inner_gap

    ax.bar(
        exp_left,
        exp_props,
        width=bar_width,
        align="edge",
        facecolor="0.7",
        edgecolor="black",
        linewidth=0.8,
        label="Experimental",
    )

    ax.bar(
        model_left,
        model_props,
        width=bar_width,
        align="edge",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
        label="Model",
    )

    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ymax = max(np.max(exp_props), np.max(model_props))
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    ax.set_xlabel(
        scraped.get(
            "x_label",
            "second-order orientation preference (deg)",
        ),
        fontsize=fontsize,
    )
    ax.set_ylabel(
        scraped.get("y_label", "proportion of cells"),
        fontsize=fontsize,
    )

    _style_hallum_small_axis(ax, fontsize=fontsize)

    ax.legend(frameon=False, fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sosi_histogram_scraped_comparison(
    results_sosi,
    *,
    scraped_config=None,
    save_path="/project/results/modulation/sosi_distribution_scraped_comparison.png",
):
    """
    Model-vs-Hallum SOSI histogram.

    Same bin centers for experiment and model.
    Bars are side by side within each bin, with a small gap.
    """

    fontsize = 8

    scraped = _get_scraped_distribution(scraped_config, "sosi_distribution")
    if scraped is None:
        raise ValueError("Missing sosi_distribution.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results_sosi = np.asarray(results_sosi)

    if results_sosi.ndim == 1:
        sosi_values = results_sosi.astype(float)
    else:
        sosi_values = results_sosi[:, 0].astype(float)

    sosi_values = sosi_values[np.isfinite(sosi_values)]

    if sosi_values.size == 0:
        raise ValueError("No finite SOSI values found.")

    exp_props = np.asarray(scraped["proportion_of_cells"], dtype=float)

    xlim = scraped.get("xlim", [0.0, 0.70])
    edges = np.linspace(float(xlim[0]), float(xlim[1]), exp_props.size + 1)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2

    counts, _ = np.histogram(sosi_values, bins=edges)
    model_props = counts / sosi_values.size

    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    total_frac = 0.82
    gap_frac = 0.08

    total_used = widths * total_frac
    inner_gap = widths * gap_frac
    bar_width = (total_used - inner_gap) / 2

    exp_left = centers - total_used / 2
    model_left = exp_left + bar_width + inner_gap

    ax.bar(
        exp_left,
        exp_props,
        width=bar_width,
        align="edge",
        facecolor="0.7",
        edgecolor="black",
        linewidth=0.8,
        label="Experimental",
    )

    ax.bar(
        model_left,
        model_props,
        width=bar_width,
        align="edge",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
        label="Model",
    )

    ax.set_xlim(float(xlim[0]), float(xlim[1]))

    ymax = max(np.max(exp_props), np.max(model_props))
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    ax.set_xlabel(
        scraped.get("x_label", "SOSI"),
        fontsize=fontsize,
    )
    ax.set_ylabel(
        scraped.get("y_label", "proportion of cells"),
        fontsize=fontsize,
    )

    _style_hallum_small_axis(ax, fontsize=fontsize)

    ax.legend(frameon=False, fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)