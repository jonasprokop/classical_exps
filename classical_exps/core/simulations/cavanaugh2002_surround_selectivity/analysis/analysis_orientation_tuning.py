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
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.cohorts import get_selectivity_filtered_neuron_ids
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.loaders import load_orientation_tuning_bulk    

## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
import os

from matplotlib.patches import Rectangle




##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################

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


BIN_WIDTH_3A = 30.0

MODEL_BINS_3A = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0], dtype=float)
MODEL_XLIM_3A = (0.0, 180.0)
MODEL_XTICKS_3A = [0, 30, 60, 90, 120, 150, 180]

SCRAPED_BINS_3A = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0], dtype=float)
SCRAPED_XLIM_3A = (0.0, 180.0)
SCRAPED_XTICKS_3A = [0, 30, 60, 90, 120, 150, 180]

COMMON_YLIM_3A = (0.0, 0.7)



def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _article_hist(ax, data, bins, weights=None):
    ax.hist(
        data,
        bins=bins,
        weights=weights,
        edgecolor=BLACK,
        facecolor=LIGHT,
        linewidth=1.0,
    )



def _wrap_orientation_deg(x):
    x = np.asarray(x, dtype=float)
    return ((x + 90.0) % 180.0) - 90.0


def build_scraped_orientation_examples(scraped_orientation_tuning_data):
    """
    Convert scraped article example-neuron data into a simple plot-ready format.

    Expected input:
        {
            "directionality_tuning_neuron": {
                "neuron_A": {
                    "center_alone": {"direction": [...], "response": [...]},
                    "surround_influence": {"direction": [...], "response": [...]},
                },
                ...
            }
        }

    Output:
        {
            "labels": [...],
            "by_label": {
                label: {
                    "center_deg": np.ndarray,
                    "center_resp": np.ndarray,
                    "surround_deg": np.ndarray,
                    "surround_resp": np.ndarray,
                }
            }
        }
    """
    if scraped_orientation_tuning_data is None:
        return None

    raw = scraped_orientation_tuning_data.get("directionality_tuning_neuron", {})
    if not raw:
        return None

    by_label = {}
    labels = []

    for label, payload in raw.items():
        center = payload.get("center_alone", {})
        surround = payload.get("surround_influence", {})

        center_deg = _wrap_orientation_deg(center.get("direction", []))
        center_resp = np.asarray(center.get("response", []), dtype=float)

        surround_deg = _wrap_orientation_deg(surround.get("direction", []))
        surround_resp = np.asarray(surround.get("response", []), dtype=float)

        if center_deg.shape != center_resp.shape:
            raise ValueError(f"{label}: center_alone direction/response mismatch")
        if surround_deg.shape != surround_resp.shape:
            raise ValueError(f"{label}: surround_influence direction/response mismatch")

        center_order = np.argsort(center_deg)
        surround_order = np.argsort(surround_deg)

        by_label[label] = {
            "center_deg": center_deg[center_order],
            "center_resp": center_resp[center_order],
            "surround_deg": surround_deg[surround_order],
            "surround_resp": surround_resp[surround_order],
        }
        labels.append(label)

    return {
        "labels": labels,
        "by_label": by_label,
    }

def fold_direction_curve_to_orientation(curve):
    """
    Fold a 1D direction curve of length N (even) into orientation space of length N/2
    by averaging opposite directions: theta and theta + pi.
    """
    curve = np.asarray(curve, dtype=float)
    n = len(curve)
    if n % 2 != 0:
        raise ValueError(f"Expected even number of bins, got {n}")
    half = n // 2
    return 0.5 * (curve[:half] + curve[half:])


def fold_direction_matrix_to_orientation(matrix):
    """
    Fold a 2D direction matrix [center, surround] of shape (N, N) into
    orientation matrix of shape (N/2, N/2) by averaging opposite direction pairs
    along both axes.
    """
    matrix = np.asarray(matrix, dtype=float)
    n0, n1 = matrix.shape
    if n0 != n1:
        raise ValueError(f"Expected square matrix, got {matrix.shape}")
    if n0 % 2 != 0:
        raise ValueError(f"Expected even number of bins, got {n0}")

    half = n0 // 2

    # average along center axis
    m = 0.5 * (matrix[:half, :] + matrix[half:, :])
    # average along surround axis
    m = 0.5 * (m[:, :half] + m[:, half:])

    return m

def fold_loaded_orientation_dataset(loaded):
    """
    Post-loader fold from direction space to orientation space.

    This is deliberately detachable:
    - keep loader raw
    - fold only here
    - delete later if you switch to half-grating plotting upstream
    """
    present_ids = np.asarray(loaded["present_ids"], dtype=int)
    missing_ids = np.asarray(loaded["missing_ids"], dtype=int)
    by_id_raw = loaded["by_id"]

    ori_shifts_rad_raw = np.asarray(loaded["ori_shifts_rad"], dtype=float)
    n_raw = len(ori_shifts_rad_raw)
    if n_raw % 2 != 0:
        raise ValueError(f"Expected even number of direction bins, got {n_raw}")

    half = n_raw // 2

    ori_shifts_rad_folded = _wrap_orientation_rad(ori_shifts_rad_raw[:half])
    order = np.argsort(ori_shifts_rad_folded)
    ori_shifts_rad_folded = ori_shifts_rad_folded[order]
    ori_shifts_deg_folded = np.degrees(ori_shifts_rad_folded)

    zero_idx = int(np.argmin(np.abs(ori_shifts_rad_folded)))

    by_id = {}

    for nid in present_ids:
        d = by_id_raw[int(nid)]

        center = fold_direction_curve_to_orientation(d["center"])
        surround_only = fold_direction_curve_to_orientation(d["surround_only"])
        matrix = fold_direction_matrix_to_orientation(d["matrix"])

        center = center[order]
        surround_only = surround_only[order]
        matrix = matrix[order][:, order]

        surround_fixed_center = matrix[zero_idx, :].copy()
        center_fixed_surround = matrix[:, zero_idx].copy()

        by_id[int(nid)] = {
            "center": center,
            "surround_only": surround_only,
            "matrix": matrix,
            "center_surround_matrix": matrix,
            "surround_fixed_center": surround_fixed_center,
            "center_fixed_surround": center_fixed_surround,
        }

    return {
        "present_ids": present_ids,
        "missing_ids": missing_ids,
        "by_id": by_id,
        "ori_shifts_rad": ori_shifts_rad_folded,
        "ori_shifts_deg": ori_shifts_deg_folded,
        "zero_idx": zero_idx,
        "base_group": loaded.get("base_group", "/orientation_tuning"),
        "is_orientation_folded": True,
    }


def _orientation_xticks():
    return [-90, -45, 0, 45, 90]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

    
def neuron_key(neuron_id: int) -> str:
    return f"neuron_{int(neuron_id)}"


def _sizes_from_response(values, min_frac=0.08, max_frac=0.72, ref_max=None, area_scaled=True):
    """
    Convert responses to square side fractions inside each grid cell.

    If ref_max is given, normalize by that shared reference maximum.
    If area_scaled is True, square area is proportional to response.
    """
    v = np.asarray(values, dtype=float)

    if ref_max is None:
        vmax = np.nanmax(v)
    else:
        vmax = float(ref_max)

    if not np.isfinite(vmax) or vmax <= 0:
        return np.full_like(v, min_frac, dtype=float)

    vn = np.clip(v / vmax, 0.0, 1.0)

    if area_scaled:
        vn = np.sqrt(vn)

    return min_frac + (max_frac - min_frac) * vn


def plot_orientation_tuning_figure_1_matrix(
    neuron_id,
    neuron_data,
    ori_shifts_deg,
    out_path="/project/results/selectivity_and_spatial_distribution/single_neurons/orientation_tuning_figure_1_matrix",
):
    """
    Paper-style figure with:
        - left strip: surround alone
        - bottom strip: center alone
        - central square matrix: center x surround responses

    Rows: surround orientation
    Columns: center orientation
    """
    neuron = neuron_key(neuron_id)
    center_curve = np.asarray(neuron_data["center"], dtype=float)
    surround_only_curve = np.asarray(neuron_data["surround_only"], dtype=float)
    response_matrix = np.asarray(neuron_data["matrix"], dtype=float)

    shared_max = np.nanmax([
        np.nanmax(center_curve),
        np.nanmax(surround_only_curve),
        np.nanmax(response_matrix),
    ])

    # Display convention:
    # x-axis = center orientation
    # y-axis = surround orientation
    # stored matrix is [center, surround], so transpose it
    matrix_display = response_matrix.T

    n = len(ori_shifts_deg)

    fig = plt.figure(figsize=(8.2, 7.6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.3, 5.8],
        height_ratios=[5.8, 1.3],
        wspace=0.16,
        hspace=0.16,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_bottom = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # Main matrix
    # ------------------------------------------------------------------
    ax_main.set_xlim(-0.5, n - 0.5)
    ax_main.set_ylim(-0.5, n - 0.5)
    ax_main.invert_yaxis()
    ax_main.set_aspect("equal", adjustable="box")

    size_fracs = _sizes_from_response(
        matrix_display,
        min_frac=0.08,
        max_frac=0.72,
        ref_max=shared_max,
    )
    for iy in range(n):
        for ix in range(n):
            side = float(size_fracs[iy, ix])
            x0 = ix - side / 2
            y0 = iy - side / 2

            rect = Rectangle(
                (x0, y0),
                side,
                side,
                facecolor="white",
                edgecolor="0.40",
                linewidth=1.0,
            )
            ax_main.add_patch(rect)

    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels([f"{int(round(x))}" for x in ori_shifts_deg])
    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels([f"{int(round(y))}" for y in ori_shifts_deg])

    ax_main.set_xlabel("Center orientation (deg)")
    ax_main.set_ylabel("Surround orientation (deg)")
    # ax_main.set_title("Center × surround", pad=6)

    # ------------------------------------------------------------------
    # Left strip: surround alone
    # ------------------------------------------------------------------
    ax_left.set_xlim(0.0, 1.0)
    ax_left.set_ylim(-0.5, n - 0.5)
    ax_left.invert_yaxis()
    ax_left.set_aspect("equal", adjustable="box")

    surround_sizes = _sizes_from_response(
        surround_only_curve,
        min_frac=0.08,
        max_frac=0.72,
        ref_max=shared_max,
    )
    for iy in range(n):
        side = float(surround_sizes[iy])
        x0 = 0.5 - side / 2
        y0 = iy - side / 2

        rect = Rectangle(
            (x0, y0),
            side,
            side,
            facecolor="white",
            edgecolor="0.40",
            linewidth=1.0,
        )
        ax_left.add_patch(rect)

    ax_left.set_xticks([])
    ax_left.set_yticks(range(n))
    ax_left.set_yticklabels([f"{int(round(y))}" for y in ori_shifts_deg])
    # ax_left.set_title("Surround alone", pad=6)

    # ------------------------------------------------------------------
    # Bottom strip: center alone
    # ------------------------------------------------------------------
    ax_bottom.set_xlim(-0.5, n - 0.5)
    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.set_aspect("equal", adjustable="box")

    center_sizes = _sizes_from_response(
        center_curve,
        min_frac=0.08,
        max_frac=0.72,
        ref_max=shared_max,
    )
    for ix in range(n):
        side = float(center_sizes[ix])
        x0 = ix - side / 2
        y0 = 0.5 - side / 2

        rect = Rectangle(
            (x0, y0),
            side,
            side,
            facecolor="white",
            edgecolor="0.40",
            linewidth=1.0,
        )
        ax_bottom.add_patch(rect)

    ax_bottom.set_yticks([])
    ax_bottom.set_xticks(range(n))
    ax_bottom.set_xticklabels([f"{int(round(x))}" for x in ori_shifts_deg])
    ax_bottom.set_xlabel("Center orientation (deg)")
    ax_bottom.set_title("Center alone", y=0.75)
    # ------------------------------------------------------------------
    # Legend / size key
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Legend / size key
    # ------------------------------------------------------------------
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_aspect("equal", adjustable="box")
    ax_legend.axis("off")

    ax_legend.text(
        0.02, 0.96,
        "Max response",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    # exact displayed size for the maximum response in this neuron
    side_max = float(_sizes_from_response(
        np.array([shared_max], dtype=float),
        min_frac=0.08,
        max_frac=0.72,
        ref_max=shared_max,
    )[0])

    # center the square nicely in the legend panel
    x0 = 0.10
    y0 = 0.18

    rect_max = Rectangle(
        (x0, y0),
        side_max,
        side_max,
        facecolor="white",
        edgecolor="0.40",
        linewidth=1.0,
    )
    ax_legend.add_patch(rect_max)

    ax_legend.text(
        x0 + side_max + 0.08,
        y0 + side_max / 2,
        f"{shared_max:.3g}",
        va="center",
        ha="left",
        fontsize=9,
    )

    # clean spines
    for ax in [ax_main, ax_left, ax_bottom]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Orientation tuning Figure 1-style plot for neuron {neuron_id}", y=0.98)

    directory = os.path.join(out_path, neuron)
    ensure_dir(directory)

    _style_axis(ax)

    save_path = os.path.join(directory, f"orientation_tuning_figure_1_matrix_{neuron_id}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path

def plot_orientation_overview_figure(
    neuron_id,
    neuron_data,
    ori_shifts_deg,
    out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_overview",
):
    """
    Per-neuron overview figure for orientation tuning.

    Panels:
        A: center-only tuning
        B: surround-only tuning
        C: surround tuning with center fixed at 0 deg
        D: center tuning with fixed surround at 0 deg
    """
    neuron = neuron_key(neuron_id)

    center_curve = neuron_data["center"]
    surround_only_curve = neuron_data["surround_only"]
    surround_fixed_center = neuron_data["surround_fixed_center"]
    center_fixed_surround = neuron_data["center_fixed_surround"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ## Panel A: center only
    ax1.plot(ori_shifts_deg, center_curve, color=BLACK, linewidth=1.0)

    ax1.set_title("A. Center only")
    ax1.set_xlabel("Center orientation shift (degrees)")
    ax1.set_ylabel("Response")
    ax1.set_xticks(_orientation_xticks())
    ax1.set_ylim(np.min(center_curve) - 0.3, np.max(center_curve) + 0.3)

    ## Panel B: surround only
    ax2.plot(ori_shifts_deg, surround_only_curve, color=BLACK, linewidth=1.0)

    ax2.set_title("B. Surround only")
    ax2.set_xlabel("Surround orientation shift (degrees)")
    ax2.set_ylabel("Response")
    ax2.set_xticks(_orientation_xticks())
    ax2.set_ylim(np.min(surround_only_curve) - 0.3, np.max(surround_only_curve) + 0.3)

    ## Panel C: surround varies, center fixed at preferred
    max_val_center = np.max(center_curve)
    ax3.plot(
        ori_shifts_deg,
        surround_fixed_center,
        color=BLACK,
        linewidth=2.5,
    )
    ax3.plot(
        [np.min(ori_shifts_deg), np.max(ori_shifts_deg)],
        [max_val_center, max_val_center],
        linestyle="--",
        color=DARK,
        linewidth=1.0,
    )
    ax3.set_title("C. Surround tuning, center fixed at 0°")
    ax3.set_xlabel("Surround orientation shift (degrees)")
    ax3.set_ylabel("Response")
    ax3.set_xticks(_orientation_xticks())
    ax3.set_ylim(
        min(np.min(surround_fixed_center), np.min(center_curve)) - 0.3,
        max(np.max(surround_fixed_center), np.max(center_curve)) + 0.3,
    )
    # ax3.legend()

    ## Panel D: center varies, surround fixed at preferred
    max_val_surround = np.max(surround_only_curve)
    ax4.plot(
        ori_shifts_deg,
        center_fixed_surround,
        color=BLACK,
        linewidth=2.5,
    )
    ax4.plot(
        [np.min(ori_shifts_deg), np.max(ori_shifts_deg)],
        [max_val_surround, max_val_surround],
        linestyle="--",
        color=DARK,
        linewidth=1.0,
    )
    ax4.set_title("D. Center tuning, surround fixed at 0°")
    ax4.set_xlabel("Center orientation shift (degrees)")
    ax4.set_ylabel("Response")
    ax4.set_xticks(_orientation_xticks())
    ax4.set_ylim(
        min(np.min(center_fixed_surround), np.min(surround_only_curve)) - 0.3,
        max(np.max(center_fixed_surround), np.max(surround_only_curve)) + 0.3,
    )
    # ax4.legend()

    fig.suptitle(f"Orientation tuning overview for neuron {neuron_id}", y=0.98)

    directory = os.path.join(out_path, neuron)
    ensure_dir(directory)

    save_path = os.path.join(directory, f"orientation_tuning_overview_{neuron_id}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_orientation_tuning_curve(
    neuron_id,
    neuron_data,
    ori_shifts_deg,
    out_path="/project/results/selectivity_and_spatial_distribution/population/orientation_tuning_curve",
):
    """
    Plot center-only tuning and surround-with-fixed-center tuning for one neuron.
    """
    neuron = neuron_key(neuron_id)

    center_tuning_curve = neuron_data["center"]
    surround_tuning_curve = neuron_data["surround_fixed_center"]

    max_val_center = np.max(center_tuning_curve)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        ori_shifts_deg,
        center_tuning_curve,
        label="Center alone",
        color=BLACK,
        linewidth=1.0,
        marker="o",
        markersize=3,
        markerfacecolor=BLACK,
        markeredgecolor=BLACK,
    )
    ax.plot(
        ori_shifts_deg,
        surround_tuning_curve,
        label="Preffered center + surround",
        color=BLACK,
        linewidth=2.5,
        marker="o",
        markersize=4,
        markerfacecolor=WHITE,
        markeredgecolor=BLACK,
    )
    ax.plot(
        [np.min(ori_shifts_deg), np.max(ori_shifts_deg)],
        [max_val_center, max_val_center],
        linestyle="--",
        color=DARK,
        linewidth=1.0,
    )
    ax.set_xticks(_orientation_xticks())
    ax.set_ylim(
        min(np.min(center_tuning_curve), np.min(surround_tuning_curve)) - 0.3,
        max(np.max(center_tuning_curve), np.max(surround_tuning_curve)) + 0.3,
    )

    # ax.set_title(f"Orientation Tuning Curve for the neuron {neuron_id}")
    ax.set_xlabel("Orientation shift compared to the preferred orientation (degrees)")
    ax.set_ylabel("Response")
    ax.legend()
    _style_axis(ax)

    directory = os.path.join(out_path, neuron)
    ensure_dir(directory)

    save_path = os.path.join(directory, f"orientation_tuning_curve_{neuron_id}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_mean_orientation_tuning(
    ori_shifts_deg,
    mean_curve_center,
    mean_curve_surround,
    out_path,
):
    """
    Plot mean normalized center-only and surround-with-fixed-center
    orientation tuning curves across neurons.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        ori_shifts_deg,
        mean_curve_center,
        label="Center alone",
        color=BLACK,
        linewidth=1.0,
        marker="o",
        markersize=3,
        markerfacecolor=BLACK,
        markeredgecolor=BLACK,
    )

    ax.plot(
        ori_shifts_deg,
        mean_curve_surround,
        label="Preffered center + surround",
        color=BLACK,
        linewidth=2.5,
        marker="o",
        markersize=4,
        markerfacecolor=WHITE,
        markeredgecolor=BLACK,
    )

    ax.plot(
        [np.min(ori_shifts_deg), np.max(ori_shifts_deg)],
        [1.0, 1.0],
        linestyle="--",
        color=DARK,
        linewidth=1.0,
    )

    ax.set_xticks(_orientation_xticks())

    ax.set_ylim(
        min(np.min(mean_curve_center), np.min(mean_curve_surround)) - 0.05,
        max(np.max(mean_curve_center), np.max(mean_curve_surround)) + 0.05,
    )

    # ax.set_title("Mean normalized orientation tuning curves across neurons")
    ax.set_xlabel("Orientation shift compared to the preferred orientation (degrees)")
    ax.set_ylabel("Mean normalized response")
    ax.legend()
    _style_axis(ax)

    ensure_dir(out_path)
    save_path = os.path.join(out_path, "orientation_tuning_results_1.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def compute_most_suppressive_surround(
    loaded_orientation,
    center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
    min_center_response_frac=0.10,
    use_nearest_sampled=True,
):
    """
    Compute distributions of most suppressive surround orientations.

    Uses orientation-bias estimates of suppression in radians.
    """
    ori_shifts_rad = np.asarray(loaded_orientation["ori_shifts_rad"], dtype=float)
    by_id = loaded_orientation["by_id"]
    present_ids = np.asarray(loaded_orientation["present_ids"], dtype=int)
    center_angles_rad = np.asarray(center_angles_rad, dtype=float)

    center_idx = np.array([
        int(np.argmin(np.abs(ori_shifts_rad - a)))
        for a in center_angles_rad
    ], dtype=int)

    results = {float(a): [] for a in center_angles_rad}

    for nid in present_ids:
        d = by_id[int(nid)]
        center_curve = np.asarray(d["center"], dtype=float)
        matrix = np.asarray(d["matrix"], dtype=float)

        max_center = float(np.max(center_curve)) if len(center_curve) else np.nan
        if not np.isfinite(max_center) or max_center <= 0:
            continue

        for angle_rad, idx in zip(center_angles_rad, center_idx):
            center_resp = float(center_curve[idx])

            if center_resp < min_center_response_frac * max_center:
                continue

            surround_row = np.asarray(matrix[idx, :], dtype=float)
            bias = _orientation_bias_from_suppression(
                surround_curve=surround_row,
                surround_angles_rad=ori_shifts_rad,
                center_response=center_resp,
            )

            chosen_angle = (
                bias["nearest_sampled_bias_angle_rad"]
                if use_nearest_sampled
                else bias["bias_angle_rad"]
            )

            if np.isfinite(chosen_angle):
                results[float(angle_rad)].append(float(chosen_angle))

    for k in results:
        results[k] = np.asarray(results[k], dtype=float)

    return {
        "center_angles_rad": center_angles_rad,
        "ori_shifts_rad": ori_shifts_rad,
        "results": results,
    }


def plot_most_suppressive_surround_histograms(
    suppress_data,
    ori_shifts_rad,
    out_path,
    filename="most_suppressive_surround_histograms.png",
):
    """
    Plot histograms of most suppressive surround orientations in folded
    orientation space, using sampled-support-centered bins.

    Parameters
    ----------
    suppress_data : dict
        Output of compute_most_suppressive_surround(...), with:
            - "center_angles_rad"
            - "results"
    ori_shifts_rad : array-like
        Folded sampled orientation support in radians.
    out_path : str
        Output directory.
    filename : str
        Output filename.
    """
    ensure_dir(out_path)

    center_angles_rad = np.asarray(suppress_data["center_angles_rad"], dtype=float)
    results = suppress_data["results"]

    ori_shifts_rad = np.asarray(ori_shifts_rad, dtype=float)
    ori_shifts_deg = np.degrees(ori_shifts_rad)
    bins_rad = _sampled_orientation_bin_edges_rad(ori_shifts_rad)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, c_angle_rad in zip(axes, center_angles_rad):
        data = np.asarray(results[float(c_angle_rad)], dtype=float)

        if data.size > 0:
            weights = np.ones_like(data, dtype=float) / float(len(data))
            _article_hist(ax, data, bins=bins_rad, weights=weights)

        ax.axvline(c_angle_rad, linestyle="--", color=DARK, linewidth=1.0)  

        ymax = ax.get_ylim()[1]
        ax.text(
            c_angle_rad,
            ymax * 0.95 if ymax > 0 else 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)

        # ax.set_title(f"Center = {int(round(np.degrees(c_angle_rad)))}°")
        ax.set_xlabel("Most suppressive surround orientation (deg)")
        ax.set_ylabel("Proportion of neurons")

        _style_axis(ax)

    fig.tight_layout()

    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path

def _safe_mean(x, axis=0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    return np.nanmean(x, axis=axis)


def _safe_sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    valid = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = sd / np.sqrt(valid)
    sem = np.where(valid > 1, sem, np.nan)
    return sem


def _orientation_diff_rad(a_rad, b_rad):
    da = _wrap_orientation_rad(np.asarray(a_rad) - np.asarray(b_rad))
    return np.abs(da)


def _orientation_vector_from_curve(values, theta_rad):
    values = np.asarray(values, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)

    if values.ndim != 1 or theta_rad.ndim != 1 or len(values) != len(theta_rad):
        raise ValueError("values and theta_rad must be 1D arrays of the same length")

    return np.sum(values * np.exp(1j * 2.0 * theta_rad))


def _orientation_preference_and_si(values, theta_rad):
    """
    Preferred orientation and orientation selectivity index from vector sum.

    pref = 0.5 * angle(sum R_n exp(i 2 theta_n))
    SI   = |sum R_n exp(i 2 theta_n)| / sum R_n

    Assumes values are nonnegative response magnitudes.
    """
    values = np.asarray(values, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)

    if not np.all(np.isfinite(values)):
        return np.nan, np.nan

    denom = np.sum(values)
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan, np.nan

    vec = _orientation_vector_from_curve(values, theta_rad)
    pref_rad = 0.5 * np.angle(vec)
    pref_rad = float(_wrap_orientation_rad(pref_rad))
    si = float(np.abs(vec) / denom)

    return pref_rad, si


def _suppression_curve(reference_response, compound_curve):
    reference_response = float(reference_response)
    compound_curve = np.asarray(compound_curve, dtype=float)
    return np.clip(reference_response - compound_curve, 0.0, None)


def _make_panel_a_bins():
    """
    Equal-width bins for folded orientation-space Figure 3A.
    Collapse the paper's 0..180 direction-difference histogram
    into 0..90 orientation-difference space.
    """
    return np.array([0.0, 30.0, 60.0, 90.0], dtype=float)



def compute_orientation_figure_3_data(loaded):
    present_ids = np.asarray(loaded["present_ids"], dtype=int)
    by_id = loaded["by_id"]
    ori_shifts_deg = np.asarray(loaded["ori_shifts_deg"], dtype=float)
    ori_shifts_rad = np.asarray(loaded["ori_shifts_rad"], dtype=float)
    zero_idx = int(loaded["zero_idx"])

    n_ori = len(ori_shifts_rad)
    if n_ori < 3:
        raise ValueError(f"Need at least 3 orientation bins, got {n_ori}")

    delta_deg = []
    center_si = []
    surround_si = []
    aligned_center_norm = []
    aligned_compound_norm = []

    for nid in present_ids:
        d = by_id[int(nid)]

        center = np.asarray(d["center"], dtype=float)
        compound = np.asarray(d["surround_fixed_center"], dtype=float)

        if center.shape != (n_ori,) or compound.shape != (n_ori,):
            continue
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(compound))):
            continue

        pref_center_idx = int(np.argmax(center))
        pref_center_rad = float(ori_shifts_rad[pref_center_idx])

        ref_center_response = float(center[pref_center_idx])
        supp_curve = _suppression_curve(ref_center_response, compound)

        supp_idx = int(np.argmax(supp_curve))
        supp_rad = float(ori_shifts_rad[supp_idx])

        _, c_si = _orientation_preference_and_si(center, ori_shifts_rad)
        _, s_si = _orientation_preference_and_si(supp_curve, ori_shifts_rad)
        if not (np.isfinite(c_si) and np.isfinite(s_si)):
            continue

        max_center = float(np.max(center))
        if not np.isfinite(max_center) or np.isclose(max_center, 0.0):
            continue

        delta = _orientation_diff_rad(supp_rad, pref_center_rad)
        delta_deg.append(float(np.degrees(delta)))
        center_si.append(float(c_si))
        surround_si.append(float(s_si))

        shift = zero_idx - pref_center_idx
        aligned_center_norm.append(np.roll(center / max_center, shift))
        aligned_compound_norm.append(np.roll(compound / max_center, shift))

    aligned_center_norm = np.asarray(aligned_center_norm, dtype=float)
    aligned_compound_norm = np.asarray(aligned_compound_norm, dtype=float)

    rel_axis_deg = ori_shifts_deg - ori_shifts_deg[zero_idx]
    rel_axis_deg = ((rel_axis_deg + 90.0) % 180.0) - 90.0
    order = np.argsort(rel_axis_deg)

    if aligned_center_norm.size > 0:
        aligned_center_norm = aligned_center_norm[:, order]
        aligned_compound_norm = aligned_compound_norm[:, order]

    rel_axis_deg = rel_axis_deg[order]

    return {
        "delta_deg": np.asarray(delta_deg, dtype=float),
        "center_si": np.asarray(center_si, dtype=float),
        "surround_si": np.asarray(surround_si, dtype=float),
        "rel_axis_deg": np.asarray(rel_axis_deg, dtype=float),
        "mean_center_norm": _safe_mean(aligned_center_norm, axis=0),
        "sem_center_norm": _safe_sem(aligned_center_norm, axis=0),
        "mean_compound_norm": _safe_mean(aligned_compound_norm, axis=0),
        "sem_compound_norm": _safe_sem(aligned_compound_norm, axis=0),
        "panel_a_bins_deg": _make_panel_a_bins(),
    }


def plot_orientation_tuning_figure_3_all(
    fig3_data,
    out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_figure_3",
    prefix="orientation_tuning_figure_3",
):
    """
    Save all three figure-3 panels as separate files.
    """
    ensure_dir(out_path)

    save_a = plot_orientation_tuning_figure_3A(
        fig3_data,
        out_path=out_path,
        filename=f"{prefix}A.png",
    )
    save_b = plot_orientation_tuning_figure_3B(
        fig3_data,
        out_path=out_path,
        filename=f"{prefix}B.png",
    )
    save_d = plot_orientation_tuning_figure_3D(
        fig3_data,
        out_path=out_path,
        filename=f"{prefix}D.png",
    )

    return {
        "A": save_a,
        "B": save_b,
        "D": save_d,
    }

def plot_orientation_tuning_figure_3A(
    fig3_data,
    out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_figure_3",
    filename="orientation_tuning_figure_3A.png",
):
    ensure_dir(out_path)

    delta_deg = np.asarray(fig3_data["delta_deg"], dtype=float)

    # poster sizing
    FIGSIZE = (6.4, 5.2)
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if delta_deg.size > 0:
        weights = np.ones_like(delta_deg, dtype=float) / float(len(delta_deg))
        _article_hist(ax, delta_deg, bins=MODEL_BINS_3A, weights=weights)

    ax.set_xlabel("Difference in optimal orientation (deg)", fontsize=LABELSIZE)
    ax.set_ylabel("Proportion of neurons", fontsize=LABELSIZE)
    ax.set_xlim(*MODEL_XLIM_3A)
    ax.set_xticks(MODEL_XTICKS_3A)
    ax.set_ylim(*COMMON_YLIM_3A)
    ax.margins(x=0)
    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_orientation_tuning_figure_3B(
    fig3_data,
    out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_figure_3",
    filename="orientation_tuning_figure_3B.png",
):
    """
    Panel B:
    Center orientation SI vs surround-suppression orientation SI.
    """
    ensure_dir(out_path)

    center_si = np.asarray(fig3_data["center_si"], dtype=float)
    surround_si = np.asarray(fig3_data["surround_si"], dtype=float)

    valid = np.isfinite(center_si) & np.isfinite(surround_si)
    x = center_si[valid]
    y = surround_si[valid]

    lo, hi = 0.0, 1.0

    # poster sizing
    FIGSIZE = (6.4, 5.2)
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    SCATTER_SIZE = 34
    DIAG_LW = 1.8

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        x,
        y,
        s=SCATTER_SIZE,
        color=BLACK,
        linewidths=0,
    )
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color=BLACK,
        linewidth=DIAG_LW,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Center selectivity index", fontsize=LABELSIZE)
    ax.set_ylabel("Surround selectivity index", fontsize=LABELSIZE)
    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_orientation_tuning_figure_3D(
    fig3_data,
    out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_figure_3",
    filename="orientation_tuning_figure_3D.png",
):
    """
    Panel D:
    Mean normalized center-alone response and suppressed compound response.
    """
    ensure_dir(out_path)

    rel_axis_deg = np.asarray(fig3_data["rel_axis_deg"], dtype=float)
    mean_center_norm = np.asarray(fig3_data["mean_center_norm"], dtype=float)
    sem_center_norm = np.asarray(fig3_data["sem_center_norm"], dtype=float)
    mean_compound_norm = np.asarray(fig3_data["mean_compound_norm"], dtype=float)
    sem_compound_norm = np.asarray(fig3_data["sem_compound_norm"], dtype=float)

    # poster sizing
    FIGSIZE = (6.8, 5.4)
    LABELSIZE = 20
    TICKSIZE = 15
    LEGENDSIZE = 18
    TICKLEN = 6
    TICKWIDTH = 1.4
    CENTER_LW = 1.8
    COMPOUND_LW = 3.0
    CENTER_MS = 5.0
    COMPOUND_MS = 6.2
    ERR_LW = 1.2

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if mean_center_norm.size > 0 and mean_compound_norm.size > 0:
        ax.plot(
            rel_axis_deg,
            mean_center_norm,
            color=BLACK,
            linewidth=CENTER_LW,
            marker="o",
            markersize=CENTER_MS,
            markerfacecolor=BLACK,
            markeredgecolor=BLACK,
            label="Center alone",
        )
        ax.plot(
            rel_axis_deg,
            mean_compound_norm,
            color=BLACK,
            linewidth=COMPOUND_LW,
            marker="o",
            markersize=COMPOUND_MS,
            markerfacecolor=WHITE,
            markeredgecolor=BLACK,
            label="Compound",
        )

        if sem_center_norm.size == mean_center_norm.size:
            ax.errorbar(
                rel_axis_deg,
                mean_center_norm,
                yerr=sem_center_norm,
                fmt="none",
                ecolor=BLACK,
                elinewidth=ERR_LW,
                capsize=0,
            )

        if sem_compound_norm.size == mean_compound_norm.size:
            ax.errorbar(
                rel_axis_deg,
                mean_compound_norm,
                yerr=sem_compound_norm,
                fmt="none",
                ecolor=BLACK,
                elinewidth=ERR_LW,
                capsize=0,
            )

    ax.set_xlabel("Orientation relative to preferred (deg)", fontsize=LABELSIZE)
    ax.set_ylabel("Normalized response", fontsize=LABELSIZE)
    ax.set_xticks([-90, -45, 0, 45, 90])

    y_max = max(
        np.nanmax(mean_center_norm) if mean_center_norm.size else 0,
        np.nanmax(mean_compound_norm) if mean_compound_norm.size else 0,
    )
    y_max = max(1.05 * y_max, 1.05)

    ax.set_ylim(0.0, y_max)
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=LEGENDSIZE,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.2,
    )
    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
# =============================================================================
# Figure 4 remake: folded-orientation adaptation of Cavanaugh et al. 2002
# A-C = single-neuron panels
# D-F = population histograms
# no G
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Basic helpers: orientation geometry
# =============================================================================

def _wrap_orientation_rad(angle_rad):
    """
    Wrap to folded orientation space [-pi/2, pi/2).
    """
    a = np.asarray(angle_rad, dtype=float)
    return ((a + np.pi / 2.0) % np.pi) - np.pi / 2.0


def _orientation_distance_rad(a_rad, b_rad):
    """
    Absolute folded-orientation distance in radians.
    """
    return np.abs(
        _wrap_orientation_rad(np.asarray(a_rad, dtype=float) - np.asarray(b_rad, dtype=float))
    )


def _nearest_sampled_orientation_rad(angle_rad, sampled_angles_rad):
    """
    Snap a continuous folded-orientation angle to the nearest sampled angle.
    """
    sampled_angles_rad = np.asarray(sampled_angles_rad, dtype=float)
    angle_rad = float(_wrap_orientation_rad(angle_rad))
    idx = int(np.argmin(_orientation_distance_rad(sampled_angles_rad, angle_rad)))
    return float(sampled_angles_rad[idx]), idx


def _sampled_orientation_bin_edges_rad(sample_angles_rad):
    """
    Histogram bin edges centered on sampled folded-orientation support.
    """
    sample_angles_rad = np.sort(np.asarray(sample_angles_rad, dtype=float))
    if sample_angles_rad.ndim != 1 or len(sample_angles_rad) == 0:
        raise ValueError("sample_angles_rad must be a non-empty 1D array")

    if len(sample_angles_rad) == 1:
        step = np.pi / 8.0
        a = float(sample_angles_rad[0])
        return np.array([a - step / 2.0, a + step / 2.0], dtype=float)

    mids = 0.5 * (sample_angles_rad[:-1] + sample_angles_rad[1:])
    first_edge = sample_angles_rad[0] - 0.5 * (sample_angles_rad[1] - sample_angles_rad[0])
    last_edge = sample_angles_rad[-1] + 0.5 * (sample_angles_rad[-1] - sample_angles_rad[-2])

    edges = np.concatenate([[first_edge], mids, [last_edge]])
    edges[0] = max(edges[0], -np.pi / 2.0)
    edges[-1] = min(edges[-1], np.pi / 2.0)
    return edges


def _orientation_xticks_rad():
    return np.array(
        [-np.pi / 2.0, -np.pi / 4.0, 0.0, np.pi / 4.0, np.pi / 2.0],
        dtype=float,
    )


def _orientation_xtick_labels_deg():
    return ["-90", "-45", "0", "45", "90"]


# =============================================================================
# Small utility
# =============================================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =============================================================================
# Suppression-bias computation in radians
# =============================================================================

def _orientation_bias_from_suppression(
    surround_curve,
    surround_angles_rad,
    center_response,
    *,
    min_total_weight=1e-12,
):
    """
    Orientation-space analog of the paper's suppression-bias estimate.

        w_i = max(center_response - surround_curve[i], 0)
        V   = sum_i w_i * exp(1j * 2 * theta_i)

    Returns:
        bias_angle_rad
        nearest_sampled_bias_angle_rad
        nearest_sampled_bias_idx
        bias_strength
        weights
        total_suppression
        peak_suppression
        strongest_sample_angle_rad
        strongest_sample_idx
    """
    surround_curve = np.asarray(surround_curve, dtype=float)
    surround_angles_rad = np.asarray(surround_angles_rad, dtype=float)
    center_response = float(center_response)

    if surround_curve.ndim != 1 or surround_angles_rad.ndim != 1:
        raise ValueError("surround_curve and surround_angles_rad must be 1D")
    if len(surround_curve) != len(surround_angles_rad):
        raise ValueError("surround_curve and surround_angles_rad must have same length")

    suppression = np.maximum(center_response - surround_curve, 0.0)
    total = float(np.sum(suppression))

    if (not np.isfinite(total)) or total <= min_total_weight:
        return {
            "bias_angle_rad": np.nan,
            "nearest_sampled_bias_angle_rad": np.nan,
            "nearest_sampled_bias_idx": -1,
            "bias_strength": 0.0,
            "weights": suppression,
            "total_suppression": 0.0,
            "peak_suppression": 0.0,
            "strongest_sample_angle_rad": np.nan,
            "strongest_sample_idx": -1,
        }

    V = np.sum(suppression * np.exp(1j * 2.0 * surround_angles_rad))

    if not np.isfinite(V.real) or not np.isfinite(V.imag):
        bias_angle_rad = np.nan
        bias_strength = 0.0
        nearest_angle_rad = np.nan
        nearest_idx = -1
    else:
        bias_angle_rad = float(_wrap_orientation_rad(0.5 * np.angle(V)))
        bias_strength = float(np.abs(V) / total)
        nearest_angle_rad, nearest_idx = _nearest_sampled_orientation_rad(
            bias_angle_rad, surround_angles_rad
        )

    strongest_idx = int(np.argmax(suppression))
    strongest_angle_rad = float(surround_angles_rad[strongest_idx])
    peak_suppression = float(suppression[strongest_idx])

    return {
        "bias_angle_rad": bias_angle_rad,
        "nearest_sampled_bias_angle_rad": nearest_angle_rad,
        "nearest_sampled_bias_idx": nearest_idx,
        "bias_strength": bias_strength,
        "weights": suppression,
        "total_suppression": total,
        "peak_suppression": peak_suppression,
        "strongest_sample_angle_rad": strongest_angle_rad,
        "strongest_sample_idx": strongest_idx,
    }


# =============================================================================
# Optional helper for quick pre-filtering before prep
# =============================================================================

def filter_fig4_center_response_neurons(
    h5_file,
    neuron_ids,
    center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
    min_center_response_frac=0.10,
    require_all_centers=True,
    print_results=False,
):
    """
    Keep only neurons whose folded orientation center-only responses
    at the requested Figure 4 center offsets reach at least
    min_center_response_frac * max(center_curve).
    """
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    loaded = load_orientation_tuning_bulk(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        strict=False,
    )
    loaded = fold_loaded_orientation_dataset(loaded)

    ori_shifts_rad = np.asarray(loaded["ori_shifts_rad"], dtype=float)
    present_ids = np.asarray(loaded["present_ids"], dtype=int)
    by_id = loaded["by_id"]

    center_angles_rad = np.asarray(center_angles_rad, dtype=float)
    center_idx = np.array(
        [int(np.argmin(np.abs(ori_shifts_rad - a))) for a in center_angles_rad],
        dtype=int,
    )

    kept = []

    for neuron_id in present_ids:
        center_curve = np.asarray(by_id[int(neuron_id)]["center"], dtype=float)

        if center_curve.ndim != 1 or len(center_curve) == 0:
            continue

        max_center = float(np.nanmax(center_curve))
        if not np.isfinite(max_center) or max_center <= 0:
            continue

        flags = []
        for idx in center_idx:
            resp = float(center_curve[idx])
            flags.append(resp >= min_center_response_frac * max_center)

        keep = all(flags) if require_all_centers else any(flags)

        if keep:
            kept.append(int(neuron_id))

    kept = np.asarray(kept, dtype=int)

    if print_results:
        n = len(neuron_ids)
        new_n = len(kept)
        pct = round(100 * ((n - new_n) / n), 1) if n > 0 else 0.0
        print("--------------------------------------")
        print("Filtering neurons for Figure 4 center-response eligibility:")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n - new_n} neurons were removed ({pct}%)")
        print(f"    > There are {new_n} neurons left")
        print("--------------------------------------")
        print()

    return kept


# =============================================================================
# Unified Figure 4 data prep
# =============================================================================

def prepare_orientation_figure_4_data(
    loaded,
    center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
    min_center_response_frac=0.10,
):
    """
    Unified data prep for folded-orientation Figure 4 analysis.

    Produces:
    - per-neuron plotting payloads
    - population distributions of nearest-sampled suppression-bias angle
    - summary metrics for ranking / manual selection

    Notes
    -----
    This follows the Figure 4 caption logic most closely:
    D-F are based on bias estimates, then snapped to sampled support.
    """
    ori_shifts_rad = np.asarray(loaded["ori_shifts_rad"], dtype=float)
    present_ids = np.asarray(loaded["present_ids"], dtype=int)
    by_id = loaded["by_id"]

    center_angles_rad = np.asarray(center_angles_rad, dtype=float)
    center_idx = np.array(
        [int(np.argmin(np.abs(ori_shifts_rad - a))) for a in center_angles_rad],
        dtype=int,
    )

    suppressive_angles_by_center = {float(a): [] for a in center_angles_rad}
    valid_ids_by_center = {float(a): [] for a in center_angles_rad}

    neurons = []
    summary = []

    for nid in present_ids:
        d = by_id[int(nid)]
        center_curve = np.asarray(d["center"], dtype=float)
        matrix = np.asarray(d["matrix"], dtype=float)

        if center_curve.ndim != 1 or matrix.ndim != 2:
            continue
        if matrix.shape[0] != len(center_curve) or matrix.shape[1] != len(ori_shifts_rad):
            continue

        max_center = float(np.nanmax(center_curve)) if len(center_curve) else np.nan
        if not np.isfinite(max_center) or max_center <= 0:
            continue

        rows = []
        n_valid = 0

        for angle_rad, idx in zip(center_angles_rad, center_idx):
            center_resp = float(center_curve[idx])
            valid_center = bool(center_resp >= min_center_response_frac * max_center)

            surround_curve = np.asarray(matrix[idx, :], dtype=float)
            bias = _orientation_bias_from_suppression(
                surround_curve=surround_curve,
                surround_angles_rad=ori_shifts_rad,
                center_response=center_resp,
            )

            nearest_bias_angle_rad = bias["nearest_sampled_bias_angle_rad"]
            tracking_error_rad = (
                float(_orientation_distance_rad(nearest_bias_angle_rad, angle_rad))
                if np.isfinite(nearest_bias_angle_rad) else np.nan
            )

            row = {
                "center_angle_rad": float(angle_rad),
                "center_idx": int(idx),
                "center_response": center_resp,
                "surround_curve": surround_curve,
                "valid_center": valid_center,
                "bias_angle_rad": (
                    float(bias["bias_angle_rad"]) if np.isfinite(bias["bias_angle_rad"]) else np.nan
                ),
                "nearest_sampled_bias_angle_rad": (
                    float(nearest_bias_angle_rad) if np.isfinite(nearest_bias_angle_rad) else np.nan
                ),
                "nearest_sampled_bias_idx": int(bias["nearest_sampled_bias_idx"]),
                "bias_strength": float(bias["bias_strength"]),
                "weights": np.asarray(bias["weights"], dtype=float),
                "total_suppression": float(bias["total_suppression"]),
                "peak_suppression": float(bias["peak_suppression"]),
                "strongest_sample_angle_rad": (
                    float(bias["strongest_sample_angle_rad"])
                    if np.isfinite(bias["strongest_sample_angle_rad"]) else np.nan
                ),
                "strongest_sample_idx": int(bias["strongest_sample_idx"]),
                "tracking_error_rad": tracking_error_rad,
            }
            rows.append(row)

            if valid_center and np.isfinite(nearest_bias_angle_rad):
                n_valid += 1
                suppressive_angles_by_center[float(angle_rad)].append(float(nearest_bias_angle_rad))
                valid_ids_by_center[float(angle_rad)].append(int(nid))

        mean_err_rad = float(np.nanmean([r["tracking_error_rad"] for r in rows])) if rows else np.nan
        mean_bias_strength = float(np.nanmean([r["bias_strength"] for r in rows])) if rows else np.nan
        mean_peak_supp = float(np.nanmean([r["peak_suppression"] for r in rows])) if rows else np.nan
        mean_total_supp = float(np.nanmean([r["total_suppression"] for r in rows])) if rows else np.nan

        neuron_payload = {
            "neuron_id": int(nid),
            "center_curve": center_curve,
            "rows": rows,
            "n_valid_centers": int(n_valid),
            "mean_tracking_error_rad": mean_err_rad,
            "mean_bias_strength": mean_bias_strength,
            "mean_peak_suppression": mean_peak_supp,
            "mean_total_suppression": mean_total_supp,
        }
        neurons.append(neuron_payload)

        summary.append({
            "neuron_id": int(nid),
            "n_valid_centers": int(n_valid),
            "mean_tracking_error_rad": mean_err_rad,
            "mean_tracking_error_deg": float(np.degrees(mean_err_rad)) if np.isfinite(mean_err_rad) else np.nan,
            "mean_bias_strength": mean_bias_strength,
            "mean_peak_suppression": mean_peak_supp,
            "mean_total_suppression": mean_total_supp,
        })

    for k in suppressive_angles_by_center:
        suppressive_angles_by_center[k] = np.asarray(suppressive_angles_by_center[k], dtype=float)
        valid_ids_by_center[k] = np.asarray(valid_ids_by_center[k], dtype=int)

    summary.sort(
        key=lambda x: (
            -x["n_valid_centers"],
            x["mean_tracking_error_rad"] if np.isfinite(x["mean_tracking_error_rad"]) else np.inf,
            -(x["mean_bias_strength"] if np.isfinite(x["mean_bias_strength"]) else -np.inf),
            -(x["mean_peak_suppression"] if np.isfinite(x["mean_peak_suppression"]) else -np.inf),
        )
    )

    return {
        "ori_shifts_rad": ori_shifts_rad,
        "center_angles_rad": center_angles_rad,
        "center_idx": center_idx,
        "sampled_edges_rad": _sampled_orientation_bin_edges_rad(ori_shifts_rad),
        "suppressive_angles_by_center": suppressive_angles_by_center,
        "valid_ids_by_center": valid_ids_by_center,
        "neurons": neurons,
        "summary": summary,
        "n_present": len(present_ids),
    }


# =============================================================================
# Figure 4 A-C single-neuron panels
# =============================================================================

def _plot_fig4_ac_single_neuron_panels(
    neuron_payload,
    ori_shifts_rad,
    out_path,
    filename_prefix="figure_4",
):
    """
    Save A, B, C as separate single-neuron figures.
    """
    _ensure_dir(out_path)

    neuron_id = int(neuron_payload["neuron_id"])
    center_curve = np.asarray(neuron_payload["center_curve"], dtype=float)
    rows = neuron_payload["rows"]
    ori_shifts_rad = np.asarray(ori_shifts_rad, dtype=float)

    y_min = min(np.nanmin(center_curve), min(np.nanmin(r["surround_curve"]) for r in rows))
    y_max = max(np.nanmax(center_curve), max(np.nanmax(r["surround_curve"]) for r in rows))
    pad = 0.07 * (y_max - y_min) if y_max > y_min else 0.1

    saved = {}

    for row, panel_label in zip(rows, ["A", "B", "C"]):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))

        surround_curve = np.asarray(row["surround_curve"], dtype=float)
        center_angle_rad = float(row["center_angle_rad"])
        center_resp = float(row["center_response"])
        nearest_bias_angle_rad = row["nearest_sampled_bias_angle_rad"]
        nearest_bias_idx = int(row["nearest_sampled_bias_idx"])

        ax.plot(
            ori_shifts_rad,
            center_curve,
            color=BLACK,
            linewidth=1.0,
        )

        ax.plot(
            ori_shifts_rad,
            surround_curve,
            color=BLACK,
            linewidth=2.5,
            marker="o",
            markersize=4,
            markerfacecolor=WHITE,
            markeredgecolor=BLACK,
        )

        ax.axhline(
            center_resp,
            linestyle="--",
            color=DARK,
            linewidth=1.0,
        )
        # filled point on center curve
        ax.plot([center_angle_rad], [center_resp], marker="o", markersize=7, color="black")
        # asterisk on x-axis
        ax.text(
            center_angle_rad,
            y_min - 0.035 * (y_max - y_min + 1e-12),
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        # marker for nearest sampled bias used in D-F
        # if np.isfinite(nearest_bias_angle_rad) and nearest_bias_idx >= 0:
        #     ax.plot(
        #         [nearest_bias_angle_rad],
        #         [surround_curve[nearest_bias_idx]],
        #         marker="s",
        #         markersize=5,
        #         color="black",
        #     )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°")
        ax.set_xlabel("Surround orientation (deg)")
        ax.set_ylabel("Response")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        save_path = os.path.join(out_path, f"{filename_prefix}{panel_label}_neuron_{neuron_id}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved[panel_label] = save_path

    return saved


def _plot_fig4_ac_combined(
    neuron_payload,
    ori_shifts_rad,
    out_path,
    filename="figure_4_AC_combined.png",
):
    """
    Save A-C as one combined single-neuron figure.
    """
    _ensure_dir(out_path)

    neuron_id = int(neuron_payload["neuron_id"])
    center_curve = np.asarray(neuron_payload["center_curve"], dtype=float)
    rows = neuron_payload["rows"]
    ori_shifts_rad = np.asarray(ori_shifts_rad, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    y_min = min(np.nanmin(center_curve), min(np.nanmin(r["surround_curve"]) for r in rows))
    y_max = max(np.nanmax(center_curve), max(np.nanmax(r["surround_curve"]) for r in rows))
    pad = 0.07 * (y_max - y_min) if y_max > y_min else 0.1

    for ax, row, panel_label in zip(axes, rows, ["A", "B", "C"]):
        surround_curve = np.asarray(row["surround_curve"], dtype=float)
        center_angle_rad = float(row["center_angle_rad"])
        center_resp = float(row["center_response"])
        nearest_bias_angle_rad = row["nearest_sampled_bias_angle_rad"]
        nearest_bias_idx = int(row["nearest_sampled_bias_idx"])

        ax.plot(
            ori_shifts_rad,
            center_curve,
            color=BLACK,
            linewidth=1.0,
        )

        ax.plot(
            ori_shifts_rad,
            surround_curve,
            color=BLACK,
            linewidth=2.5,
            marker="o",
            markersize=4,
            markerfacecolor=WHITE,
            markeredgecolor=BLACK,
        )

        ax.axhline(
            center_resp,
            linestyle="--",
            color=DARK,
            linewidth=1.0,
        )
        ax.plot([center_angle_rad], [center_resp], marker="o", markersize=7, color="black")
        ax.text(
            center_angle_rad,
            y_min - 0.035 * (y_max - y_min + 1e-12),
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        # if np.isfinite(nearest_bias_angle_rad) and nearest_bias_idx >= 0:
        #     ax.plot(
        #         [nearest_bias_angle_rad],
        #         [surround_curve[nearest_bias_idx]],
        #         marker="s",
        #         markersize=5,
        #         color="black",
        #     )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°")
        ax.set_xlabel("Surround orientation (deg)")
        ax.set_ylabel("Response")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # fig.suptitle(f"Figure 4 A-C | neuron {neuron_id}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# Figure 4 D-F population histograms
# =============================================================================

def _plot_fig4_df_population_panels_separate(
    fig4_prep,
    out_path,
    filename_prefix="figure_4",
):
    """
    Save D, E, F as separate population figures.
    """
    _ensure_dir(out_path)

    center_angles_rad = np.asarray(fig4_prep["center_angles_rad"], dtype=float)
    sampled_edges_rad = np.asarray(fig4_prep["sampled_edges_rad"], dtype=float)
    suppressive_angles_by_center = fig4_prep["suppressive_angles_by_center"]
    valid_ids_by_center = fig4_prep["valid_ids_by_center"]

    saved = {}

    for center_angle_rad, panel_label in zip(center_angles_rad, ["D", "E", "F"]):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))

        data = np.asarray(suppressive_angles_by_center[float(center_angle_rad)], dtype=float)
        n_valid = len(valid_ids_by_center[float(center_angle_rad)])

        bins_rad = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])

        if len(data) > 0:
            weights = np.ones_like(data, dtype=float) / len(data)
            ax.hist(
                data,
                bins=bins_rad,
                weights=weights,
                edgecolor=BLACK,
                facecolor=LIGHT,
                linewidth=1.0,
                align="mid",
            )

        ax.axvline(center_angle_rad, linestyle="--", color=DARK, linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text(
            center_angle_rad,
            ymax * 0.95 if ymax > 0 else 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°\n n = {n_valid}")
        ax.set_xlabel("Most suppressive surround orientation (deg)")
        ax.set_ylabel("Proportion of neurons")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        _style_axis(ax)

        fig.tight_layout()
        save_path = os.path.join(out_path, f"{filename_prefix}{panel_label}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved[panel_label] = save_path

    return saved


def _plot_fig4_df_combined(
    fig4_prep,
    out_path,
    filename="figure_4_DF_combined.png",
):
    """
    Save D-F as one combined population figure.
    """
    _ensure_dir(out_path)

    center_angles_rad = np.asarray(fig4_prep["center_angles_rad"], dtype=float)
    sampled_edges_rad = np.asarray(fig4_prep["sampled_edges_rad"], dtype=float)
    suppressive_angles_by_center = fig4_prep["suppressive_angles_by_center"]
    valid_ids_by_center = fig4_prep["valid_ids_by_center"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    for ax, center_angle_rad, panel_label in zip(axes, center_angles_rad, ["D", "E", "F"]):
        data = np.asarray(suppressive_angles_by_center[float(center_angle_rad)], dtype=float)
        n_valid = len(valid_ids_by_center[float(center_angle_rad)])

        bins_rad = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])

        if len(data) > 0:
            weights = np.ones_like(data, dtype=float) / len(data)
            ax.hist(
                data,
                bins=bins_rad,
                weights=weights,
                edgecolor=BLACK,
                facecolor=LIGHT,
                linewidth=1.0,
                align="mid",
            )

        ax.axvline(center_angle_rad, linestyle="--", color=DARK, linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text(
            center_angle_rad,
            ymax * 0.95 if ymax > 0 else 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°\n n = {n_valid}")
        ax.set_xlabel("Most suppressive surround orientation (deg)")
        ax.set_ylabel("Proportion of neurons")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        _style_axis(ax)

    # fig.suptitle("Figure 4 D-F population histograms", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# Full combined A-F figure
# =============================================================================

def plot_orientation_tuning_figure_4_combined(
    fig4_prep,
    neuron_payload,
    out_path,
    filename="figure_4_combined.png",
):
    """
    Save full Figure 4 remake as one combined 2x3 figure:
    top row  A-C = single neuron
    bottom   D-F = population
    """
    _ensure_dir(out_path)

    neuron_id = int(neuron_payload["neuron_id"])
    center_curve = np.asarray(neuron_payload["center_curve"], dtype=float)
    rows = neuron_payload["rows"]
    ori_shifts_rad = np.asarray(fig4_prep["ori_shifts_rad"], dtype=float)

    center_angles_rad = np.asarray(fig4_prep["center_angles_rad"], dtype=float)
    sampled_edges_rad = np.asarray(fig4_prep["sampled_edges_rad"], dtype=float)
    suppressive_angles_by_center = fig4_prep["suppressive_angles_by_center"]
    valid_ids_by_center = fig4_prep["valid_ids_by_center"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    top_axes = axes[0]
    bottom_axes = axes[1]

    y_min = min(np.nanmin(center_curve), min(np.nanmin(r["surround_curve"]) for r in rows))
    y_max = max(np.nanmax(center_curve), max(np.nanmax(r["surround_curve"]) for r in rows))
    pad = 0.07 * (y_max - y_min) if y_max > y_min else 0.1

    # A-C
    for ax, row, panel_label in zip(top_axes, rows, ["A", "B", "C"]):
        surround_curve = np.asarray(row["surround_curve"], dtype=float)
        center_angle_rad = float(row["center_angle_rad"])
        center_resp = float(row["center_response"])
        nearest_bias_angle_rad = row["nearest_sampled_bias_angle_rad"]
        nearest_bias_idx = int(row["nearest_sampled_bias_idx"])
        ax.plot(
            ori_shifts_rad,
            center_curve,
            color=BLACK,
            linewidth=1.0,
        )

        ax.plot(
            ori_shifts_rad,
            surround_curve,
            color=BLACK,
            linewidth=2.5,
            marker="o",
            markersize=4,
            markerfacecolor=WHITE,
            markeredgecolor=BLACK,
        )

        ax.axhline(
            center_resp,
            linestyle="--",
            color=DARK,
            linewidth=1.0,
        )
        ax.plot([center_angle_rad], [center_resp], marker="o", markersize=7, color="black")
        ax.text(
            center_angle_rad,
            y_min - 0.035 * (y_max - y_min + 1e-12),
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        # if np.isfinite(nearest_bias_angle_rad) and nearest_bias_idx >= 0:
        #     ax.plot(
        #         [nearest_bias_angle_rad],
        #         [surround_curve[nearest_bias_idx]],
        #         marker="s",
        #         markersize=5,
        #         color="black",
        #     )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°")
        ax.set_xlabel("Surround orientation (deg)")
        ax.set_ylabel("Response")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # D-F
    for ax, center_angle_rad, panel_label in zip(bottom_axes, center_angles_rad, ["D", "E", "F"]):
        data = np.asarray(suppressive_angles_by_center[float(center_angle_rad)], dtype=float)
        n_valid = len(valid_ids_by_center[float(center_angle_rad)])

        bins_rad = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])
        if len(data) > 0:
            weights = np.ones_like(data, dtype=float) / len(data)
            ax.hist(
                data,
                bins_rad = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0]),
                weights=weights,
                edgecolor=BLACK,
                facecolor=LIGHT,
                linewidth=1.0,
                align="mid",
            )

        ax.axvline(center_angle_rad, linestyle="--", color=DARK, linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text(
            center_angle_rad,
            ymax * 0.95 if ymax > 0 else 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_title(f"{panel_label}. center = {np.degrees(center_angle_rad):.0f}°\n n = {n_valid}")
        ax.set_xlabel("Most suppressive surround orientation (deg)")
        ax.set_ylabel("Proportion of neurons")
        ax.set_xticks(_orientation_xticks_rad())
        ax.set_xticklabels(_orientation_xtick_labels_deg())
        ax.set_xlim(-np.pi / 2.0, np.pi / 2.0)
        _style_axis(ax)

    fig.suptitle(f"Figure 4 remake | example neuron {neuron_id}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# Dispatcher
# =============================================================================

def plot_orientation_tuning_figure_4(
    fig4_prep,
    neuron_payload,
    out_path,
    mode="combined",
):
    """
    mode:
        'combined' -> one 2x3 A-F figure
        'separate' -> A/B/C in neurons/, D/E/F in population/
    """
    _ensure_dir(out_path)

    if mode not in {"combined", "separate"}:
        raise ValueError("mode must be 'combined' or 'separate'")

    if mode == "combined":
        return {
            "combined": plot_orientation_tuning_figure_4_combined(
                fig4_prep=fig4_prep,
                neuron_payload=neuron_payload,
                out_path=out_path,
                filename=f"figure_4_combined_neuron_{int(neuron_payload['neuron_id'])}.png",
            )
        }

    neurons_dir = os.path.join(out_path, "neurons")
    population_dir = os.path.join(out_path, "population")
    _ensure_dir(neurons_dir)
    _ensure_dir(population_dir)

    ac_paths = _plot_fig4_ac_single_neuron_panels(
        neuron_payload=neuron_payload,
        ori_shifts_rad=fig4_prep["ori_shifts_rad"],
        out_path=neurons_dir,
        filename_prefix="figure_4",
    )

    df_paths = _plot_fig4_df_population_panels_separate(
        fig4_prep=fig4_prep,
        out_path=population_dir,
        filename_prefix="figure_4",
    )

    return {
        "neurons": ac_paths,
        "population": df_paths,
    }


# =============================================================================
# High-level convenience wrapper
# =============================================================================

def run_orientation_tuning_figure_4(
    h5_file,
    neuron_ids,
    out_path,
    mode="combined",
    example_neuron_id=None,
    center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
    min_center_response_frac=0.10,
    print_summary=True,
):
    """
    Full Figure 4 pipeline:
      1) load bulk orientation tuning
      2) fold to orientation space
      3) prepare Figure 4 data with internal 10% rule
      4) choose example neuron
      5) save combined or separate Figure 4 outputs

    Returns
    -------
    dict with:
        fig4_prep
        chosen_neuron_id
        plot_paths
    """
    _ensure_dir(out_path)

    neuron_ids = np.asarray(neuron_ids, dtype=int)

    loaded = load_orientation_tuning_bulk(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        strict=False,
    )
    loaded = fold_loaded_orientation_dataset(loaded)

    fig4_prep = prepare_orientation_figure_4_data(
        loaded=loaded,
        center_angles_rad=center_angles_rad,
        min_center_response_frac=min_center_response_frac,
    )

    summary = fig4_prep["summary"]
    if len(summary) == 0:
        raise RuntimeError("No valid neurons available for Figure 4 after internal 10% filter")

    if print_summary:
        print("\nTop Figure 4 candidates:\n")
        for rank, item in enumerate(summary[:15], start=1):
            print(
                f"{rank:2d}. neuron {item['neuron_id']:>4d} | "
                f"valid={item['n_valid_centers']}/3 | "
                f"mean_err={item['mean_tracking_error_deg']:.1f}° | "
                f"mean_|V|={item['mean_bias_strength']:.2f} | "
                f"mean_peak={item['mean_peak_suppression']:.3f}"
            )

    if example_neuron_id is None:
        chosen_neuron_id = int(summary[0]["neuron_id"])
    else:
        chosen_neuron_id = int(example_neuron_id)

    neuron_payload = None
    for payload in fig4_prep["neurons"]:
        if int(payload["neuron_id"]) == chosen_neuron_id:
            neuron_payload = payload
            break

    if neuron_payload is None:
        raise ValueError(f"Requested example_neuron_id={chosen_neuron_id} not found in Figure 4 cohort")

    plot_paths = plot_orientation_tuning_figure_4(
        fig4_prep=fig4_prep,
        neuron_payload=neuron_payload,
        out_path=out_path,
        mode=mode,
    )

    return {
        "fig4_prep": fig4_prep,
        "chosen_neuron_id": chosen_neuron_id,
        "plot_paths": plot_paths,
    }

def prepare_scraped_orientation_figure_4_df_population(fig4_df_population_data):
    """
    Prepare scraped article Figure 4 D-F population histogram data.

    Input config shape:
        {
            "source": "scraped_article",
            "panel": "4D-F",
            "n_cells": 27,
            "most_suppressive_surround_direction_deg": [...],
            "center_minus_45_response_proportion": [...],
            "center_0_response_proportion": [...],
            "center_plus_45_response_proportion": [...],
        }

    Returns
    -------
    dict with normalized proportions and plotting support.
    """
    data = fig4_df_population_data

    bin_centers_deg = np.asarray(
        data["most_suppressive_surround_direction_deg"],
        dtype=float,
    )

    if bin_centers_deg.ndim != 1 or len(bin_centers_deg) < 2:
        raise ValueError(
            "most_suppressive_surround_direction_deg must be a 1D array with at least 2 entries"
        )

    diffs = np.diff(bin_centers_deg)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("Expected equally spaced bin centers for scraped Figure 4 D-F")

    def _norm(props):
        props = np.asarray(props, dtype=float)
        if props.shape != bin_centers_deg.shape:
            raise ValueError(
                f"Proportion array shape {props.shape} does not match bin centers shape {bin_centers_deg.shape}"
            )
        props = np.where(np.isfinite(props), props, 0.0)
        props = np.clip(props, 0.0, None)

        s = props.sum()
        if s <= 0:
            raise ValueError("Scraped Proportion sum to zero")

        return props / s

    center_minus_45 = _norm(data["center_minus_45_response_proportion"])
    center_0 = _norm(data["center_0_response_proportion"])
    center_plus_45 = _norm(data["center_plus_45_response_proportion"])

    step_deg = float(diffs[0])
    half_step = step_deg / 2.0

    bin_edges_deg = np.concatenate([
        [bin_centers_deg[0] - half_step],
        0.5 * (bin_centers_deg[:-1] + bin_centers_deg[1:]),
        [bin_centers_deg[-1] + half_step],
    ])

    center_angles_deg = np.asarray([-45.0, 0.0, 45.0], dtype=float)

    proportions_by_center = {
        -45.0: center_minus_45,
        0.0: center_0,
        45.0: center_plus_45,
    }

    return {
        "source": data.get("source", "scraped_article"),
        "panel": data.get("panel", "4D-F"),
        "n_cells": int(data["n_cells"]),
        "center_angles_deg": center_angles_deg,
        "bin_centers_deg": bin_centers_deg,
        "bin_edges_deg": bin_edges_deg,
        "proportions_by_center": proportions_by_center,
    }
def _plot_fig4_df_population_panels_separate_scraped(
    fig4_scraped_prep,
    out_path,
    filename_prefix="figure_4_scraped",
):
    """
    Save scraped article Figure 4 D-F as separate population figures.
    """
    _ensure_dir(out_path)

    center_angles_deg = np.asarray(fig4_scraped_prep["center_angles_deg"], dtype=float)
    bin_centers_deg = np.asarray(fig4_scraped_prep["bin_centers_deg"], dtype=float)
    proportions_by_center = fig4_scraped_prep["proportions_by_center"]
    n_cells = int(fig4_scraped_prep["n_cells"])

    if len(bin_centers_deg) < 2:
        raise ValueError("Need at least two bin centers")

    FIGSIZE = (6.4, 5.2)
    LABELSIZE = 20
    TITLESIZE = 18
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    BAR_LW = 1.4
    VLINE_LW = 1.4
    STAR_SIZE = 18

    bar_width_deg = float(np.diff(bin_centers_deg)[0])

    saved = {}

    for center_deg, panel_label in zip(center_angles_deg, ["D", "E", "F"]):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        heights = np.asarray(proportions_by_center[float(center_deg)], dtype=float)

        ax.bar(
            bin_centers_deg,
            heights,
            width=bar_width_deg,
            edgecolor=BLACK,
            align="center",
            color=LIGHT,
            linewidth=BAR_LW,
        )

        ax.axvline(center_deg, linestyle="--", color=DARK, linewidth=VLINE_LW)

        ymax = ax.get_ylim()[1]
        ax.text(
            center_deg,
            ymax * 0.95 if ymax > 0 else 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=STAR_SIZE,
            fontweight="bold",
        )

        ax.set_title(f"{panel_label}. center = {center_deg:.0f}°\n n = {n_cells}", fontsize=TITLESIZE)
        ax.set_xlabel("Most suppressive surround orientation (deg)", fontsize=LABELSIZE)
        ax.set_ylabel("Proportion of neurons", fontsize=LABELSIZE)
        ax.set_xticks(bin_centers_deg)
        ax.set_xlim(bin_centers_deg[0] - bar_width_deg / 2.0, bin_centers_deg[-1] + bar_width_deg / 2.0)

        _style_axis(ax)
        ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

        fig.tight_layout()
        save_path = os.path.join(out_path, f"{filename_prefix}{panel_label}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved[panel_label] = save_path

    return saved
def _plot_fig4_df_combined_scraped(
    fig4_scraped_prep,
    out_path,
    filename="figure_4_DF_scraped_combined.png",
):
    """
    Save scraped article Figure 4 D-F as one combined population figure.
    """
    _ensure_dir(out_path)

    center_angles_deg = np.asarray(fig4_scraped_prep["center_angles_deg"], dtype=float)
    bin_centers_deg = np.asarray(fig4_scraped_prep["bin_centers_deg"], dtype=float)
    proportions_by_center = fig4_scraped_prep["proportions_by_center"]
    n_cells = int(fig4_scraped_prep["n_cells"])

    LABELSIZE = 20
    TITLESIZE = 18
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    BAR_LW = 1.4
    VLINE_LW = 1.4
    STAR_SIZE = 18

    bar_width_deg = float(np.diff(bin_centers_deg)[0])

    all_heights = np.concatenate([
        np.asarray(proportions_by_center[float(c)], dtype=float)
        for c in center_angles_deg
    ])
    y_max = 1.08 * float(np.max(all_heights)) if len(all_heights) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharey=True)

    for i, (ax, center_deg, panel_label) in enumerate(zip(axes, center_angles_deg, ["D", "E", "F"])):
        heights = np.asarray(proportions_by_center[float(center_deg)], dtype=float)

        ax.bar(
            bin_centers_deg,
            heights,
            width=bar_width_deg,
            edgecolor=BLACK,
            align="center",
            color=LIGHT,
            linewidth=BAR_LW,
        )

        ax.axvline(center_deg, linestyle="--", color=DARK, linewidth=VLINE_LW)

        ax.set_ylim(0.0, y_max)
        ax.text(
            center_deg,
            y_max * 0.95,
            "*",
            ha="center",
            va="top",
            fontsize=STAR_SIZE,
            fontweight="bold",
        )

        ax.set_title(f"{panel_label}. center = {center_deg:.0f}°\n n = {n_cells}", fontsize=TITLESIZE)
        ax.set_xlabel("Most suppressive surround orientation (deg)", fontsize=LABELSIZE)
        if i == 0:
            ax.set_ylabel("Proportion of neurons", fontsize=LABELSIZE)

        ax.set_xticks(bin_centers_deg)
        ax.set_xlim(
            bin_centers_deg[0] - bar_width_deg / 2.0,
            bin_centers_deg[-1] + bar_width_deg / 2.0,
        )

        _style_axis(ax)
        ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.suptitle("Figure 4 D-F population histograms", y=0.98, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path

def run_scraped_orientation_tuning_figure_4_df_population(
    fig4_df_population_data,
    out_path,
    mode="separate",
):
    """
    Plot scraped article Figure 4 D-F population histograms only.

    mode:
        'separate' -> D/E/F separate files
        'combined' -> one combined D-F figure
    """
    _ensure_dir(out_path)

    fig4_scraped_prep = prepare_scraped_orientation_figure_4_df_population(
        fig4_df_population_data
    )

    if mode not in {"separate", "combined"}:
        raise ValueError("mode must be 'separate' or 'combined'")

    if mode == "separate":
        plot_paths = _plot_fig4_df_population_panels_separate_scraped(
            fig4_scraped_prep=fig4_scraped_prep,
            out_path=out_path,
            filename_prefix="figure_4_scraped_",
        )
    else:
        plot_paths = {
            "combined": _plot_fig4_df_combined_scraped(
                fig4_scraped_prep=fig4_scraped_prep,
                out_path=out_path,
                filename="figure_4_DF_scraped_combined.png",
            )
        }

    return {
        "fig4_scraped_prep": fig4_scraped_prep,
        "plot_paths": plot_paths,
    }

def orientation_tuning_results(
    h5_file,
    neuron_ids,
    fit_err_thresh=0.2,
    supp_thresh=0.1,
    scraped_orietation_tuning_data=None,
    out_path="/project/results/selectivity_and_spatial_distribution/population/mean_orientation_tuning_curves",
    curve_out_path="/project/results/selectivity_and_spatial_distribution/single_neuron/orientation_tuning_curve",
    figure_1_out_path="/project/results/selectivity_and_spatial_distribution/population/orientation_tuning_overviews",
    figure_3_out_path="/project/results/selectivity_and_spatial_distribution/population/orientation_tuning_figure_3",
    figure_4_out_path="/project/results/selectivity_and_spatial_distribution/semi/orientation_tuning_figure_4",
    plot_tunning_curves=True,
    plot_histograms=True,
    plot_fig_1=True,
    plot_fig_4=True,
):
    """
    Visualise responses of neurons to different center and surround orientations.

    Workflow:
        1) Filter neurons
        2) Load orientation-tuning data
        3) Fold into orientation space
        4) Plot mean tuning curves
        5) Optionally save per-neuron tuning curves
        6) Optionally save Figure 1-style plots
        7) Optionally save Figure 3 plots
        8) Optionally save Figure 4 per-neuron inspection plots + population histograms

    Prerequisite:
        - size_tuning_experiment_all_phases executed for required neurons
        - orientation_tuning_experiment_all_phases executed for required neurons

    Filtering:
        - Exclude neurons with poor Gaussian RF fits
        - Exclude neurons with no surround suppression / saturation
        - Exclude neurons without valid GSF / AMRF
        - Optionally exclude low suppression neurons
    """

    group_path = "/orientation_tuning"

    # -------------------------------------------------------------------------
    # Check experiment presence
    # -------------------------------------------------------------------------
    check_group_exists_error(h5_file=h5_file, group_path=group_path)

    print(scraped_orietation_tuning_data)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    filtered_neuron_ids = get_selectivity_filtered_neuron_ids(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        apply_fit_error_filter=True,
        apply_no_supp_filter=True,
        apply_low_supp_filter=True,
        apply_valid_gsf_amrf_filter=True,
        verbose=False,
    )

    check_neurons_presence_error(
        h5_file=h5_file,
        list_group_path=[
            group_path + "/curves_center",
            group_path + "/curves_surround_only",
            group_path + "/curves_center_surround",
        ],
        neuron_ids=filtered_neuron_ids,
    )

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    # -------------------------------------------------------------------------
    # Load once, then fold once
    # -------------------------------------------------------------------------
    loaded = load_orientation_tuning_bulk(
        h5_file=h5_file,
        neuron_ids=filtered_neuron_ids,
        strict=False,
    )

    present_ids = np.asarray(loaded["present_ids"], dtype=int)
    missing_ids = np.asarray(loaded["missing_ids"], dtype=int)
    by_id = loaded["by_id"]
    ori_shifts_rad = np.asarray(loaded["ori_shifts_rad"], dtype=float)
    ori_shifts_deg = np.asarray(loaded["ori_shifts_deg"], dtype=float)

    n_loaded = len(present_ids)

    print("--------------------------------------")
    print("Visualisation of orientation tuning curves:")
    print(f"    > Analysis made on {n} neurons")
    if n > 0:
        print(f"    > {n_new} neurons ({round((n_new / n * 100), 2)}%) left after filtration")
    else:
        print("    > 0 neurons left after filtration")
    print(f"    > {n_loaded} neurons successfully loaded")
    if len(missing_ids) > 0:
        print(f"    > {len(missing_ids)} neurons were missing or malformed in orientation tuning datasets")
    print()
    print("    > The mean curves across every neuron:")

    if n_loaded == 0:
        print()
        print("    > No neurons available for plotting.")
        print("--------------------------------------")
        print()
        return {
            "filtered_neuron_ids": np.asarray(filtered_neuron_ids, dtype=int),
            "loaded_neuron_ids": np.asarray([], dtype=int),
            "missing_neuron_ids": np.asarray(missing_ids, dtype=int),
            "mean_plot_path": None,
            "curve_plot_paths": [],
            "figure_1_plot_paths": {"overview": [], "matrix": []},
            "hist_plot_path": None,
            "figure_3_plot_paths": {},
            "figure_4_plot_paths": {
                "per_neuron": [],
                "population_histograms": None,
                "summary": [],
            },
        }

    # -------------------------------------------------------------------------
    # Build stacked arrays for mean plots
    # -------------------------------------------------------------------------
    curves_center = np.asarray(
        [by_id[int(neuron_id)]["center"] for neuron_id in present_ids],
        dtype=float,
    )
    curves_surround = np.asarray(
        [by_id[int(neuron_id)]["surround_fixed_center"] for neuron_id in present_ids],
        dtype=float,
    )
    
    def _normalize_curves_by_row_max(curves, *, ref_curves=None):
        """
        Normalize each row by its own maximum, or by the maximum of ref_curves
        if provided. Returns NaN for rows with invalid / non-positive scale.
        """
        curves = np.asarray(curves, dtype=float)

        if ref_curves is None:
            ref_curves = curves
        else:
            ref_curves = np.asarray(ref_curves, dtype=float)

        scale = np.max(ref_curves, axis=1, keepdims=True)
        valid = np.isfinite(scale) & (scale > 0)
        scale = np.where(valid, scale, np.nan)

        return curves / scale
    
    curves_center_norm = _normalize_curves_by_row_max(curves_center)
    curves_surround_norm = _normalize_curves_by_row_max(curves_surround, ref_curves=curves_center)

    mean_curve_center = np.nanmean(curves_center_norm, axis=0)
    mean_curve_surround = np.nanmean(curves_surround_norm, axis=0)


    mean_plot_path = plot_mean_orientation_tuning(
        ori_shifts_deg=ori_shifts_deg,
        mean_curve_center=mean_curve_center,
        mean_curve_surround=mean_curve_surround,
        out_path=out_path,
    )

    # -------------------------------------------------------------------------
    # Per-neuron simple tuning curves
    # -------------------------------------------------------------------------
    curve_plot_paths = []
    scraped_curve_plot_paths = []

    if plot_tunning_curves:
        print()
        print("    > Saving simple orientation tuning curves for all loaded neurons:")

        for neuron_id in present_ids:
            save_path = plot_orientation_tuning_curve(
                neuron_id=int(neuron_id),
                neuron_data=by_id[int(neuron_id)],
                ori_shifts_deg=ori_shifts_deg,
                out_path=curve_out_path,
            )
            curve_plot_paths.append(save_path)

        print(f"    > Saved {len(curve_plot_paths)} simple per-neuron plots")

    # -------------------------------------------------------------------------
    # Figure 1
    # -------------------------------------------------------------------------
    overview_plot_paths = []
    figure_1_matrix_plot_paths = []

    if plot_fig_1:
        print()
        print("    > Saving Figure 1 overview plots for all loaded neurons:")

        figure_1_overview_out_path = os.path.join(figure_1_out_path, "overview_style")
        for neuron_id in present_ids:
            save_path = plot_orientation_overview_figure(
                neuron_id=int(neuron_id),
                neuron_data=by_id[int(neuron_id)],
                ori_shifts_deg=ori_shifts_deg,
                out_path=figure_1_overview_out_path,
            )
            overview_plot_paths.append(save_path)

        print(f"    > Saved {len(overview_plot_paths)} Figure 1 overview plots")

        print()
        print("    > Saving paper-style Figure 1 matrix plots:")

        figure_1_matrix_out_path = os.path.join(figure_1_out_path, "matrix_style")
        for neuron_id in present_ids:
            save_path = plot_orientation_tuning_figure_1_matrix(
                neuron_id=int(neuron_id),
                neuron_data=by_id[int(neuron_id)],
                ori_shifts_deg=ori_shifts_deg,
                out_path=figure_1_matrix_out_path,
            )
            figure_1_matrix_plot_paths.append(save_path)

        print(f"    > Saved {len(figure_1_matrix_plot_paths)} paper-style Figure 1 matrix plots")


     # -------------------------------------------------------------------------
    # Scraped article examples: article-style only
    # -------------------------------------------------------------------------
    if scraped_orietation_tuning_data is not None:
        print()
        print("    > Saving scraped article-style orientation tuning examples:")

        scraped_examples = build_scraped_orientation_examples(scraped_orietation_tuning_data)

        if scraped_examples is None or len(scraped_examples["labels"]) == 0:
            print("    > No scraped orientation examples found")
        else:
            scraped_out_path = os.path.join(curve_out_path, "scraped_article_style")

            for label in scraped_examples["labels"]:
                save_path = plot_scraped_article_orientation_example(
                    label=label,
                    scraped_neuron_data=scraped_examples["by_label"][label],
                    out_path=scraped_out_path,
                )
                scraped_curve_plot_paths.append(save_path)

            print(f"    > Saved {len(scraped_curve_plot_paths)} scraped article-style example plots")


    # -------------------------------------------------------------------------
    # Histograms + Figure 3
    # -------------------------------------------------------------------------
    hist_path = None
    figure_3_plot_paths = {}
    scraped_figure_3A_path = None
    scraped_figure_3B_path = None
    scraped_figure_3D_path = None

    if plot_histograms:
        suppress_data = compute_most_suppressive_surround(
            loaded_orientation=loaded,
            center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
            min_center_response_frac=0.10,
            use_nearest_sampled=True,
        )

        hist_path = plot_most_suppressive_surround_histograms(
            suppress_data,
            loaded["ori_shifts_rad"],
            out_path="/project/results/selectivity_and_spatial_distribution/orientation_tuning_histograms",
        )

        print()
        print("    > Saving adapted Figure 3 panels:")

        fig3_data = compute_orientation_figure_3_data(loaded)

        figure_3_plot_paths = plot_orientation_tuning_figure_3_all(
            fig3_data,
            out_path=figure_3_out_path,
            prefix="orientation_tuning_figure_3",
        )

        print(f"    > Saved Figure 3A plot to {figure_3_plot_paths['A']}")
        print(f"    > Saved Figure 3B plot to {figure_3_plot_paths['B']}")
        print(f"    > Saved Figure 3D plot to {figure_3_plot_paths['D']}")

        if scraped_orietation_tuning_data is not None:
            scraped_out_path = os.path.join(figure_3_out_path, "scraped")

            scraped_figure_3A_path = plot_scraped_orientation_figure_3A(
                scraped_orietation_tuning_data,
                out_path=scraped_out_path,
                filename="orientation_tuning_figure_3A_scraped.png",
            )
            if scraped_figure_3A_path is not None:
                print(f"    > Saved scraped Figure 3A plot to {scraped_figure_3A_path}")

            scraped_figure_3B_path = plot_scraped_orientation_figure_3B(
                scraped_orietation_tuning_data,
                out_path=scraped_out_path,
                filename="orientation_tuning_figure_3B_scraped.png",
            )
            if scraped_figure_3B_path is not None:
                print(f"    > Saved scraped Figure 3B plot to {scraped_figure_3B_path}")

            scraped_figure_3D_path = plot_scraped_orientation_figure_3D(
                scraped_orietation_tuning_data,
                out_path=scraped_out_path,
                filename="orientation_tuning_figure_3D_scraped.png",
            )
            if scraped_figure_3D_path is not None:
                print(f"    > Saved scraped Figure 3D plot to {scraped_figure_3D_path}")


    # -------------------------------------------------------------------------
    # Figure 4
    # -------------------------------------------------------------------------
    figure_4_plot_paths = {}

    if plot_fig_4:
        print()
        print("    > Preparing adapted Figure 4 data:")

        fig4_result = run_orientation_tuning_figure_4(
            h5_file=h5_file,
            neuron_ids=filtered_neuron_ids,
            out_path=figure_4_out_path,
            mode="separate",   # or "combined"
            example_neuron_id=None,
            center_angles_rad=(-np.pi / 4.0, 0.0, np.pi / 4.0),
            min_center_response_frac=0.10,
            print_summary=True,
        )

        figure_4_plot_paths = fig4_result["plot_paths"]

        print(f"    > Saved Figure 4 outputs to {figure_4_out_path}")

        fig4_scraped_result = run_scraped_orientation_tuning_figure_4_df_population(
        fig4_df_population_data=scraped_orietation_tuning_data["figure_4_df_population_data"],
        out_path="/project/results/selectivity_and_spatial_distribution/scraped_figure_4_population",
        mode="separate",   # or "combined"
)

    return {
        "filtered_neuron_ids": np.asarray(filtered_neuron_ids, dtype=int),
        "loaded_neuron_ids": np.asarray(present_ids, dtype=int),
        "missing_neuron_ids": np.asarray(missing_ids, dtype=int),
        "mean_plot_path": mean_plot_path,
        "curve_plot_paths": curve_plot_paths,
        "figure_1_plot_paths": {
            "overview": overview_plot_paths,
            "matrix": figure_1_matrix_plot_paths,
        },
        "hist_plot_path": hist_path,
        "figure_3_plot_paths": figure_3_plot_paths,
        "figure_4_plot_paths": figure_4_plot_paths,
    }

def plot_scraped_orientation_figure_3B(
    scraped_orientation_tuning_data,
    out_path,
    filename="orientation_tuning_figure_3B_scraped.png",
):
    """
    Plot scraped Figure 3B:
    center selectivity index vs surround selectivity index.
    """
    if scraped_orientation_tuning_data is None:
        return None

    raw = scraped_orientation_tuning_data.get("figure_3_panel_B_data", None)
    if raw is None:
        return None

    points = raw.get("points", {})
    x = np.asarray(points.get("center_selectivity_index", []), dtype=float)
    y = np.asarray(points.get("surround_selectivity_index", []), dtype=float)

    if x.size == 0 or y.size == 0:
        return None
    if x.shape != y.shape:
        raise ValueError("Scraped Figure 3B: x and y must have the same shape.")

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return None

    ensure_dir(out_path)

    FIGSIZE = (6.4, 5.2)
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    SCATTER_SIZE = 34
    DIAG_LW = 1.8

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.scatter(
        x,
        y,
        s=SCATTER_SIZE,
        color=BLACK,
        linewidths=0,
    )

    lo = 0.0
    hi = max(1.0, float(np.nanmax([np.nanmax(x), np.nanmax(y)])))
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color=DARK,
        linewidth=DIAG_LW,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Center selectivity index", fontsize=LABELSIZE)
    ax.set_ylabel("Surround selectivity index", fontsize=LABELSIZE)

    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _break_wrap(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # detect wrap (assuming sorted)
    if np.isclose(x[0], -90) and np.isclose(x[-1], 90):
        # insert NaN between endpoints to break the line
        x_new = np.concatenate([x, [np.nan]])
        y_new = np.concatenate([y, [np.nan]])
        return x_new, y_new

    return x, y


def plot_scraped_orientation_figure_3D(
    scraped_orientation_tuning_data,
    out_path,
    filename="orientation_tuning_figure_3D_scraped.png",
):
    """
    Plot scraped Figure 3D:
    mean normalized center and surround responses.
    """
    if scraped_orientation_tuning_data is None:
        return None

    raw = scraped_orientation_tuning_data.get("figure_3_panel_D_data", None)
    if raw is None:
        return None

    x = np.asarray(raw.get("direction_relative_to_preferred_deg", []), dtype=float)
    center = np.asarray(raw.get("mean_center_response", []), dtype=float)
    surround = np.asarray(raw.get("mean_surround_response", []), dtype=float)

    if x.size == 0 or center.size == 0 or surround.size == 0:
        return None
    if not (x.shape == center.shape == surround.shape):
        raise ValueError("Scraped Figure 3D: x, center, surround must have the same shape.")

    sem_center = raw.get("sem_center_response", None)
    sem_surround = raw.get("sem_surround_response", None)

    if sem_center is not None:
        sem_center = np.asarray(sem_center, dtype=float)
        if sem_center.shape != x.shape:
            raise ValueError("Scraped Figure 3D: sem_center_response shape mismatch.")
    if sem_surround is not None:
        sem_surround = np.asarray(sem_surround, dtype=float)
        if sem_surround.shape != x.shape:
            raise ValueError("Scraped Figure 3D: sem_surround_response shape mismatch.")

    order = np.argsort(x)
    x = x[order]
    center = center[order]
    surround = surround[order]
    if sem_center is not None:
        sem_center = sem_center[order]
    if sem_surround is not None:
        sem_surround = sem_surround[order]

    ensure_dir(out_path)

    FIGSIZE = (6.8, 5.4)
    LABELSIZE = 20
    TICKSIZE = 15
    LEGENDSIZE = 18
    TICKLEN = 6
    TICKWIDTH = 1.4
    CENTER_LW = 1.8
    COMPOUND_LW = 3.0
    CENTER_MS = 5.0
    COMPOUND_MS = 6.2
    ERR_LW = 1.2

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_c, center_c = _break_wrap(x, center)
    x_s, surround_s = _break_wrap(x, surround)

    ax.plot(
        x_c,
        center_c,
        color=BLACK,
        linewidth=CENTER_LW,
        marker="o",
        markersize=CENTER_MS,
        markerfacecolor=BLACK,
        markeredgecolor=BLACK,
        label="Center alone",
    )

    ax.plot(
        x_s,
        surround_s,
        color=BLACK,
        linewidth=COMPOUND_LW,
        marker="o",
        markersize=COMPOUND_MS,
        markerfacecolor=WHITE,
        markeredgecolor=BLACK,
        label="Compound",
    )

    if sem_center is not None:
        ax.errorbar(
            x,
            center,
            yerr=sem_center,
            fmt="none",
            ecolor=BLACK,
            elinewidth=ERR_LW,
            capsize=0,
        )

    if sem_surround is not None:
        ax.errorbar(
            x,
            surround,
            yerr=sem_surround,
            fmt="none",
            ecolor=BLACK,
            elinewidth=ERR_LW,
            capsize=0,
        )

    ax.set_xlabel("Direction relative to preferred (deg)", fontsize=LABELSIZE)
    ax.set_ylabel("Mean normalized response", fontsize=LABELSIZE)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_xticks(x)

    y_max = max(
        np.nanmax(center) if center.size else 0,
        np.nanmax(surround) if surround.size else 0,
    )
    y_max = max(1.05 * y_max, 1.05)

    ax.set_ylim(0.0, y_max)

    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=LEGENDSIZE,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.2,
    )

    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_scraped_article_orientation_example(
    label,
    scraped_neuron_data,
    out_path="/project/results/selectivity_and_spatial_distribution/scraped_orientation_tuning/article_style_examples",
):
    """
    Plot scraped article example in Figure-2-like style:
    thin line  = center alone
    thick line = surround influence with preferred center fixed
    """
    ensure_dir(out_path)

    center_deg = np.asarray(scraped_neuron_data["center_deg"], dtype=float)
    center_resp = np.asarray(scraped_neuron_data["center_resp"], dtype=float)
    surround_deg = np.asarray(scraped_neuron_data["surround_deg"], dtype=float)
    surround_resp = np.asarray(scraped_neuron_data["surround_resp"], dtype=float)

    FIGSIZE = (6.8, 5.0)
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    CENTER_LW = 1.8
    COMPOUND_LW = 3.0
    CENTER_MS = 5.0
    COMPOUND_MS = 6.2
    REF_LW = 1.4

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        center_deg,
        center_resp,
        color=BLACK,
        linewidth=CENTER_LW,
        marker="o",
        markersize=CENTER_MS,
        markerfacecolor=BLACK,
        markeredgecolor=BLACK,
    )

    ax.plot(
        surround_deg,
        surround_resp,
        color=BLACK,
        linewidth=COMPOUND_LW,
        marker="o",
        markersize=COMPOUND_MS,
        markerfacecolor=WHITE,
        markeredgecolor=BLACK,
    )

    if center_resp.size > 0 and np.any(np.isfinite(center_resp)):
        pref_idx = int(np.nanargmax(center_resp))
        pref_center_resp = float(center_resp[pref_idx])
        ax.axhline(pref_center_resp, linestyle="--", color=DARK, linewidth=REF_LW)

    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("Orientation relative to preferred (deg)", fontsize=LABELSIZE)
    ax.set_ylabel("Response", fontsize=LABELSIZE)

    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    save_path = os.path.join(out_path, f"{label}_article_style.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved scraped article-style example plot for {label} to {save_path}")
    plt.close(fig)

    return save_path


def plot_scraped_orientation_figure_3A(
    scraped_orientation_tuning_data,
    out_path="/project/results/selectivity_and_spatial_distribution/scraped_orientation_tuning/figure_3",
    filename="orientation_tuning_figure_3A_scraped.png",
):
    ensure_dir(out_path)

    raw = scraped_orientation_tuning_data.get("difference_in_optimal_direction", {})
    bins_deg = raw.get("bins_deg", None)
    proportions = raw.get("proportions", None)

    if bins_deg is None or proportions is None:
        return None

    bins_deg = np.asarray(bins_deg, dtype=float)
    proportions = np.asarray(proportions, dtype=float)

    if bins_deg.ndim != 2 or bins_deg.shape[1] != 2:
        raise ValueError("bins_deg must be shape (n_bins, 2)")
    if len(proportions) != len(bins_deg):
        raise ValueError("proportions length must match bins_deg")

    orig_left = bins_deg[:, 0]
    orig_right = bins_deg[:, 1]
    orig_width = orig_right - orig_left
    if np.any(orig_width <= 0):
        raise ValueError("All scraped bin widths must be positive")

    orig_density = proportions / orig_width

    rebinned = np.zeros(len(SCRAPED_BINS_3A) - 1, dtype=float)

    for i, (l_new, r_new) in enumerate(zip(SCRAPED_BINS_3A[:-1], SCRAPED_BINS_3A[1:])):
        for l_old, r_old, d_old in zip(orig_left, orig_right, orig_density):
            overlap = max(0.0, min(r_new, r_old) - max(l_new, l_old))
            if overlap > 0:
                rebinned[i] += d_old * overlap

    total = np.sum(rebinned)
    if total > 0:
        rebinned = rebinned / total

    FIGSIZE = (6.4, 5.2)
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4
    BAR_LW = 1.4

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        SCRAPED_BINS_3A[:-1],
        rebinned,
        width=np.diff(SCRAPED_BINS_3A),
        align="edge",
        edgecolor=BLACK,
        color=LIGHT,
        linewidth=BAR_LW,
    )

    ax.set_xlabel("Difference in optimal direction (deg)", fontsize=LABELSIZE)
    ax.set_ylabel("Proportion of cells", fontsize=LABELSIZE)
    ax.set_xlim(*SCRAPED_XLIM_3A)
    ax.set_xticks(SCRAPED_XTICKS_3A)
    ax.set_ylim(*COMMON_YLIM_3A)
    ax.margins(x=0)
    _style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

    fig.tight_layout()
    save_path = os.path.join(out_path, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path