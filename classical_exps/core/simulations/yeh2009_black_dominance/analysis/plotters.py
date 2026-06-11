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
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy

import os
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

from matplotlib import colors

from matplotlib.patches import Patch


BLACK_FILL = "0.35"
WHITE_FILL = "0.80"

BLACK = "black"
DARK = "0.20"
MID = "0.45"
LIGHT = "0.85"
WHITE = "white"

ARTICLE_COLOR = BLACK
MODEL_COLOR = "#2F5D8C"

def plot_logSNRwb(
    logSNR_OFF,
    logSNR_ON,
    neuron_depths_OFF = None,
    neuron_depths_ON = None
    ):
    ''' This functions plots the Signal Noise Ratios 

        Arguments : 

            - logSNR_OFF : the values of log(SNRwhite / SNRblack) that are < 0
            - logSNR_ON  :  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  "" > 0

        NB : make sure the depths are discrete values and not continuous to plot the correct mean
    '''

    logSNR_OFF = np.asarray(logSNR_OFF, dtype=float)
    logSNR_ON  = np.asarray(logSNR_ON, dtype=float)

    if neuron_depths_OFF is not None :
        y_OFF = np.asarray(neuron_depths_OFF, dtype=float)
        y_ON  = np.asarray(neuron_depths_ON, dtype=float)

    else :
        ## Create random y values to faciliate the reading of the plot
        rng = np.random.default_rng(42)
        y_OFF = rng.random(len(logSNR_OFF))
        y_ON  = rng.random(len(logSNR_ON))

    ## MAKE THE PLOT

    ## Start with a square Figure.
    fig = plt.figure(figsize=(8, 6))

    ## Name the fig
    plt.suptitle('Visualisation of the neurons color preference')

    ## Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    ## the size of the marginal axes and the main axes in both directions.
    ## Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 1,  height_ratios=(1, 4),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)
    
    ## Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[0, 0], sharex=ax)
    
    ## Scatter plot
    if len(logSNR_OFF) > 0:
        ax.scatter(logSNR_OFF, y_OFF, color='0', label = 'black-dominant')
    if len(logSNR_ON) > 0:
        ax.scatter(logSNR_ON, y_ON, color='1', edgecolor ='k', label = 'white-dominant')
    

    # ## Set nice y axis
    if neuron_depths_OFF is not None:
        max_y = max(np.max(y_OFF) if len(y_OFF) > 0 else 0,
                    np.max(y_ON)  if len(y_ON)  > 0 else 0)
        ax.set_ylim(max_y + 0.1 * max_y if max_y > 0 else 1.0, -0.10)
    else:
        ax.set_ylim(1.1, -0.10)

    ## Add vertical dashed line at zero
    ax.axvline(0, linestyle='--', color='k')
    
    # Calculate the bin edges (make an edge be 0)
    all_logSNR = np.concatenate([logSNR_OFF, logSNR_ON]) if (len(logSNR_OFF) + len(logSNR_ON)) > 0 else np.array([0.0])
    max_abs = max(abs(np.min(all_logSNR)), abs(np.max(all_logSNR)), 1e-6)
    bin_edges = np.linspace(-max_abs, max_abs, 21)

    
    weights = np.ones(len(all_logSNR)) / len(all_logSNR)
    ax_hist.hist(all_logSNR, bins=bin_edges, density = False, edgecolor='k', weights=weights)
    ax_hist.tick_params(axis="x", labelbottom=False)

    ax.set_xlabel('log(E_white / E_black)', fontsize=12)

    if neuron_depths_OFF is None :
        ax.set_yticks([])
    else :

        depth_unique = np.union1d(np.unique(neuron_depths_OFF),np.unique(neuron_depths_ON))
        
        ## Key of the dict is the depth, value is the count or the sum of SNR
        dict_count = {}
        dict_sum   = {}

        ## Initialise the dict values
        for depth in depth_unique :
            dict_count[depth] = 0
            dict_sum[depth]   = 0

        ## For every off neuron :
        for i, depth in enumerate(neuron_depths_OFF) :

            logSNR = logSNR_OFF[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        ## For every on neuron :
        for i, depth in enumerate(neuron_depths_ON) :

            logSNR = logSNR_ON[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        mean_val = np.zeros(len(depth_unique))

        for i, depth in enumerate(depth_unique) :
            
            mean_val[i] = dict_sum[depth] / dict_count[depth]

        ## Plot the mean curve
        ax.plot(mean_val, depth_unique, label = 'Mean curve')

        ## Name the y axis
        ax.set_ylabel('Depth')
    
    ax_hist.set_ylabel('Distribution')
    
    ax.legend()
    directory = f"/project/results/black_reponses_dominate" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"plot_logSNRwb.png")
    plt.close(fig)

def plot_black_white_difference_panels(
    true_maps,
    *,
    neuron_ids,
    energy_b=None,
    energy_w=None,
    log_energy_wb=None,
    save_dir="/project/results/black_and_white_experiment/example_maps/",
    prefix="bw_panels",
    true_cmap="cividis",
    diff_cmap="RdBu_r",
    eps=1e-8,
):
    """
    Per-neuron black/white response-map panels.

    Shows:
    - black response map
    - white response map
    - raw black - white difference
    - normalized black-white difference

    No noise maps. No fake SNR. Just the actual response maps.
    """
    os.makedirs(save_dir, exist_ok=True)

    true_maps = np.asarray(true_maps, dtype=float)
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    if true_maps.ndim != 4 or true_maps.shape[1] != 2:
        raise ValueError(f"Expected true_maps shape (N, 2, H, W), got {true_maps.shape}")

    n = len(neuron_ids)
    if n == 0:
        print("No neurons available for black/white/difference panels.")
        return

    if true_maps.shape[0] != n:
        raise ValueError(
            f"Number of maps and neuron_ids mismatch: {true_maps.shape[0]} vs {n}"
        )

    for idx, nid in enumerate(neuron_ids):
        true_b = true_maps[idx, 0]
        true_w = true_maps[idx, 1]

        raw_diff = true_b - true_w
        norm_diff = (true_b - true_w) / (true_b + true_w + eps)

        vmin_true = min(np.min(true_b), np.min(true_w))
        vmax_true = max(np.max(true_b), np.max(true_w))
        if np.isclose(vmin_true, vmax_true):
            vmax_true = vmin_true + 1e-6

        vmax_raw_diff = float(np.max(np.abs(raw_diff)))
        if np.isclose(vmax_raw_diff, 0):
            vmax_raw_diff = 1e-6

        vmax_norm_diff = float(np.max(np.abs(norm_diff)))
        if np.isclose(vmax_norm_diff, 0):
            vmax_norm_diff = 1e-6

        fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6), constrained_layout=True)

        im0 = axes[0].imshow(
            true_b,
            cmap=true_cmap,
            vmin=vmin_true,
            vmax=vmax_true,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        axes[0].set_title("Black response", fontsize=11)

        axes[1].imshow(
            true_w,
            cmap=true_cmap,
            vmin=vmin_true,
            vmax=vmax_true,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        axes[1].set_title("White response", fontsize=11)

        im2 = axes[2].imshow(
            raw_diff,
            cmap=diff_cmap,
            vmin=-vmax_raw_diff,
            vmax=vmax_raw_diff,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        axes[2].set_title("Black - white", fontsize=11)

        im3 = axes[3].imshow(
            norm_diff,
            cmap=diff_cmap,
            vmin=-vmax_norm_diff,
            vmax=vmax_norm_diff,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        axes[3].set_title("(Black - white) / sum", fontsize=11)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        title = f"Neuron {nid}"
        if energy_b is not None and energy_w is not None and log_energy_wb is not None:
            title += (
                f" | E_black={energy_b[idx]:.2f},"
                f" E_white={energy_w[idx]:.2f},"
                f" log10(Ew/Eb)={log_energy_wb[idx]:.2f}"
            )

        fig.suptitle(title, fontsize=12)

        cbar0 = fig.colorbar(im0, ax=axes[:2], shrink=0.82, pad=0.02)
        cbar0.set_label("Response magnitude", fontsize=10)
        cbar0.ax.tick_params(labelsize=8)

        cbar1 = fig.colorbar(im2, ax=axes[2], shrink=0.82, pad=0.02)
        cbar1.set_label("Raw difference", fontsize=10)
        cbar1.ax.tick_params(labelsize=8)

        cbar2 = fig.colorbar(im3, ax=axes[3], shrink=0.82, pad=0.02)
        cbar2.set_label("Normalized difference", fontsize=10)
        cbar2.ax.tick_params(labelsize=8)

        out = os.path.join(save_dir, f"{prefix}_neuron_{nid}.png")
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_black_white_maps(
    maps,
    *,
    neuron_ids=None,
    save_dir="/project/results/black_and_white_experiment/summary_maps/",
    prefix="bw",
    annotate_small=True,
    cmap="cividis",
    make_per_neuron=True,
    make_population_mean=True,
    make_convolution_demo=False,
    cell_px=7,
    conv_kernel_px=5,
    normalize_conv_kernel=False,
):
    """
    Plot black/white response maps.

    Additionally, if make_convolution_demo=True:
    - expands each 12x12 response map into a dense field where each grid value
      occupies a cell_px x cell_px block
    - applies a conv_kernel_px x conv_kernel_px convolution
    - saves a second figure showing how the old overlapping/convolution-style
      representation inflates/spreads values

    Expected maps shape:
        (N, 2, H, W)
        maps[:, 0] = black maps
        maps[:, 1] = white maps
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import ndimage as ndi

    maps = np.asarray(maps, dtype=float)

    if maps.ndim != 4 or maps.shape[1] != 2:
        raise ValueError(f"Expected maps shape (N, 2, H, W), got {maps.shape}")

    N, _, H, W = maps.shape

    if neuron_ids is None:
        neuron_ids = list(range(N))

    if len(neuron_ids) != N:
        raise ValueError("len(neuron_ids) must match maps.shape[0]")

    os.makedirs(save_dir, exist_ok=True)

    def _expand_grid_to_dense_field(grid_map, cell_px=7):
        """
        Convert a coarse response grid into a dense field.

        Example:
            12x12 map -> 84x84 field when cell_px=7

        Each original grid value becomes a cell_px x cell_px constant block.
        """
        return np.kron(grid_map, np.ones((cell_px, cell_px), dtype=float))

    def _convolve_dense_field(dense_field, kernel_px=5, normalize=False):
        """
        Apply old-style local convolution.

        normalize=False:
            sum convolution, shows value blow-up directly.

        normalize=True:
            average convolution, shows smoothing/spreading without scale blow-up.
        """
        kernel = np.ones((kernel_px, kernel_px), dtype=float)

        if normalize:
            kernel = kernel / kernel.sum()

        return ndi.convolve(
            dense_field,
            kernel,
            mode="constant",
            cval=0.0,
        )

    def _add_grid_annotations(ax, data, norm):
        if not (annotate_small and data.shape[0] <= 16 and data.shape[1] <= 16):
            return

        h, w = data.shape

        for i in range(h):
            for j in range(w):
                val = data[i, j]
                shade = norm(val)
                text_color = "black" if shade > 0.58 else "white"

                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=text_color,
                )

    def _plot_pair(arr, title, filepath):
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (2, H, W):
            raise ValueError(f"Expected arr shape (2, {H}, {W}), got {arr.shape}")

        # Shared scale within black/white pair
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        # Lift low-mid values so broad RF structure is visible
        norm = colors.PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        labels = ["Black response map", "White response map"]
        im = None

        for k, (ax, data, lab) in enumerate(zip(axes, [arr[0], arr[1]], labels)):
            im = ax.imshow(
                data,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="upper",
                aspect="equal",
            )

            ax.set_title(lab, fontsize=15, pad=8)
            ax.set_xlabel("x position", fontsize=11)

            if k == 0:
                ax.set_ylabel("y position", fontsize=11)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            ax.tick_params(axis="both", labelsize=9)
            _add_grid_annotations(ax, data, norm)

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="4.5%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Response magnitude", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        fig.suptitle(title, fontsize=18)

        plt.savefig(filepath, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _plot_convolution_demo(arr, title, filepath):
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (2, H, W):
            raise ValueError(f"Expected arr shape (2, {H}, {W}), got {arr.shape}")

        black_grid = arr[0]
        white_grid = arr[1]

        black_dense = _expand_grid_to_dense_field(black_grid, cell_px=cell_px)
        white_dense = _expand_grid_to_dense_field(white_grid, cell_px=cell_px)

        black_conv = _convolve_dense_field(
            black_dense,
            kernel_px=conv_kernel_px,
            normalize=normalize_conv_kernel,
        )

        white_conv = _convolve_dense_field(
            white_dense,
            kernel_px=conv_kernel_px,
            normalize=normalize_conv_kernel,
        )

        grid_pair = np.stack([black_grid, white_grid])
        conv_pair = np.stack([black_conv, white_conv])

        # Separate scales:
        # top row = original grid values
        # bottom row = convolved dense values
        grid_vmin = float(np.min(grid_pair))
        grid_vmax = float(np.max(grid_pair))
        if np.isclose(grid_vmin, grid_vmax):
            grid_vmax = grid_vmin + 1e-6

        conv_vmin = float(np.min(conv_pair))
        conv_vmax = float(np.max(conv_pair))
        if np.isclose(conv_vmin, conv_vmax):
            conv_vmax = conv_vmin + 1e-6

        grid_norm = colors.PowerNorm(gamma=0.65, vmin=grid_vmin, vmax=grid_vmax)
        conv_norm = colors.PowerNorm(gamma=0.65, vmin=conv_vmin, vmax=conv_vmax)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.5, 9.2),
            constrained_layout=True,
        )

        # Row 1: original 12x12 maps
        im_grid = None
        for k, (ax, data, lab) in enumerate(
            zip(
                axes[0],
                [black_grid, white_grid],
                ["Black grid map", "White grid map"],
            )
        ):
            im_grid = ax.imshow(
                data,
                cmap=cmap,
                norm=grid_norm,
                interpolation="nearest",
                origin="upper",
                aspect="equal",
            )
            ax.set_title(lab, fontsize=14, pad=8)
            ax.set_xlabel("grid x", fontsize=10)

            if k == 0:
                ax.set_ylabel("grid y", fontsize=10)
            else:
                ax.tick_params(axis="y", labelleft=False)

            ax.tick_params(axis="both", labelsize=8)
            _add_grid_annotations(ax, data, grid_norm)

        # Row 2: dense expanded + 5x5 convolution
        conv_label = (
            f"{conv_kernel_px}x{conv_kernel_px} "
            f"{'mean' if normalize_conv_kernel else 'sum'} convolution"
        )

        im_conv = None
        for k, (ax, data, lab) in enumerate(
            zip(
                axes[1],
                [black_conv, white_conv],
                [f"Black after {conv_label}", f"White after {conv_label}"],
            )
        ):
            im_conv = ax.imshow(
                data,
                cmap=cmap,
                norm=conv_norm,
                interpolation="nearest",
                origin="upper",
                aspect="equal",
            )
            ax.set_title(lab, fontsize=14, pad=8)
            ax.set_xlabel("pixel x", fontsize=10)

            if k == 0:
                ax.set_ylabel("pixel y", fontsize=10)
            else:
                ax.tick_params(axis="y", labelleft=False)

            ax.tick_params(axis="both", labelsize=8)

        # Colorbar for original grid row
        divider_grid = make_axes_locatable(axes[0, 1])
        cax_grid = divider_grid.append_axes("right", size="4.5%", pad=0.15)
        cbar_grid = fig.colorbar(im_grid, cax=cax_grid)
        cbar_grid.set_label("Original grid response", fontsize=10)
        cbar_grid.ax.tick_params(labelsize=8)

        # Colorbar for convolved row
        divider_conv = make_axes_locatable(axes[1, 1])
        cax_conv = divider_conv.append_axes("right", size="4.5%", pad=0.15)
        cbar_conv = fig.colorbar(im_conv, cax=cax_conv)
        cbar_conv.set_label("Convolved response", fontsize=10)
        cbar_conv.ax.tick_params(labelsize=8)

        fig.suptitle(
            (
                f"{title}\n"
                f"Grid values expanded to {cell_px}x{cell_px} pixel blocks, "
                f"then convolved with {conv_kernel_px}x{conv_kernel_px} kernel"
            ),
            fontsize=16,
        )

        plt.savefig(filepath, dpi=180, bbox_inches="tight")
        plt.close(fig)

    if make_per_neuron:
        per_neuron_dir = os.path.join(save_dir, "per_neuron")
        os.makedirs(per_neuron_dir, exist_ok=True)

        if make_convolution_demo:
            per_neuron_conv_dir = os.path.join(save_dir, "per_neuron_convolution_demo")
            os.makedirs(per_neuron_conv_dir, exist_ok=True)

        for idx, neuron_id in enumerate(neuron_ids):
            filepath = os.path.join(
                per_neuron_dir,
                f"{prefix}_neuron_{neuron_id}.png",
            )
            _plot_pair(maps[idx], f"Neuron {neuron_id}", filepath)

            if make_convolution_demo:
                conv_filepath = os.path.join(
                    per_neuron_conv_dir,
                    f"{prefix}_neuron_{neuron_id}_convolved_{conv_kernel_px}x{conv_kernel_px}.png",
                )
                _plot_convolution_demo(
                    maps[idx],
                    f"Neuron {neuron_id}",
                    conv_filepath,
                )

    if make_population_mean:
        mean_maps = np.mean(maps, axis=0)

        filepath = os.path.join(save_dir, f"{prefix}_population_mean.png")
        _plot_pair(
            mean_maps,
            f"Population mean over {len(neuron_ids)} neurons",
            filepath,
        )

        if make_convolution_demo:
            conv_filepath = os.path.join(
                save_dir,
                f"{prefix}_population_mean_convolved_{conv_kernel_px}x{conv_kernel_px}.png",
            )
            _plot_convolution_demo(
                mean_maps,
                f"Population mean over {len(neuron_ids)} neurons",
                conv_filepath,
            )



def plot_black_white_energy_cumulative(
    energy_black,
    energy_white,
    *,
    save_path="/project/results/black_and_white_experiment/summary_plots/E_black_white_cumulative.png",
    title="Cumulative black/white spatial energy distributions",
):
    """
    Cumulative percentage distributions for black and white spatial energy.
    Article-style analogue of cumulative SNR_black / SNR_white distributions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    energy_black = np.asarray(energy_black, dtype=float)
    energy_white = np.asarray(energy_white, dtype=float)

    energy_black = energy_black[np.isfinite(energy_black)]
    energy_white = energy_white[np.isfinite(energy_white)]

    if len(energy_black) == 0 or len(energy_white) == 0:
        print("No valid black/white energy values for cumulative plot.")
        return

    black_sorted = np.sort(energy_black)
    white_sorted = np.sort(energy_white)

    black_cum = 100.0 * np.arange(1, len(black_sorted) + 1) / len(black_sorted)
    white_cum = 100.0 * np.arange(1, len(white_sorted) + 1) / len(white_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        black_sorted,
        black_cum,
        color="black",
        linewidth=2,
        label="Black",
    )

    ax.plot(
        white_sorted,
        white_cum,
        color="0.55",
        linewidth=2,
        label="White",
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Spatial energy", fontsize=13)
    ax.set_ylabel("Cumulative percentage of neurons", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax.set_ylim(0, 100)

    xmax = max(np.max(black_sorted), np.max(white_sorted), 1e-6)
    ax.set_xlim(0, xmax * 1.05)

    txt = (
        f"n black = {len(black_sorted)}\n"
        f"n white = {len(white_sorted)}\n"
        f"median black = {np.median(black_sorted):.3f}\n"
        f"median white = {np.median(white_sorted):.3f}"
    )

    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.95,
            edgecolor="0.5",
        ),
    )

    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig) 


def _topcode_snr_for_display(values, *, top_code=30.0, overflow_value=35.0):
    """
    Display convention for Yeh 2009 SNR plots.

    Values <= top_code are plotted as-is.
    Values > top_code are plotted in a single overflow category at overflow_value.

    overflow_value is a categorical display position, not literal SNR.
    """
    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[out > top_code] = overflow_value
    out[out < 0] = 0.0
    return out


# =============================================================================
# Black/white histogram + scatter plotters
# Unified dataset API
# =============================================================================


def _safe_mkdir_for_file(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _black_fraction_from_counts(counts, bin_edges):
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = float(np.sum(counts))

    if total <= 0:
        return np.nan

    return float(np.sum(counts[centers < 0]) / total)


def _topcode_snr_for_display(values, *, top_code=30.0, overflow_value=35.0):
    """
    Display convention for Yeh 2009 SNR plots.

    Values <= top_code are plotted as-is.
    Values > top_code are plotted in a single overflow category at overflow_value.

    overflow_value is a categorical display position, not literal SNR.
    """

    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[out > top_code] = overflow_value
    out[out < 0] = 0.0
    return out


def _safe_layer_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "_")
    )


# =============================================================================
# Histogram plotters
# =============================================================================


def plot_black_white_histogram_dataset(
    hist_dataset,
    *,
    save_path,
    title=None,
    use_percent=False,
    show_legend=True,
):
    """
    Plot one black/white log-ratio histogram.

    Expected hist_dataset:
        {
            "label": str,
            "source": "model" | "experiment",
            "bin_edges": array-like,
            "counts": array-like,
            "percent": array-like,
            "n": int,
            "hatch": None | str,
        }

    Convention:
        bin center < 0  -> black-dominant
        bin center >= 0 -> white-dominant
    """

    _safe_mkdir_for_file(save_path)

    edges = np.asarray(hist_dataset["bin_edges"], dtype=float)
    counts = np.asarray(hist_dataset["counts"], dtype=float)

    if len(edges) != len(counts) + 1:
        raise ValueError(
            f"Expected len(bin_edges) == len(counts) + 1, got "
            f"{len(edges)} edges and {len(counts)} counts."
        )

    y_key = "percent" if use_percent else "counts"
    y = np.asarray(hist_dataset[y_key], dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    is_black = centers < 0
    is_white = centers >= 0

    black_frac = _black_fraction_from_counts(counts, edges)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.bar(
        centers[is_black],
        y[is_black],
        width=widths[is_black] * 0.82,
        color="0.25",
        edgecolor="black",
        linewidth=0.9,
        align="center",
        label="black-dominant",
    )

    ax.bar(
        centers[is_white],
        y[is_white],
        width=widths[is_white] * 0.82,
        color="white",
        edgecolor="black",
        linewidth=0.9,
        align="center",
        label="white-dominant",
    )

    ax.axvline(0, color="black", linewidth=1.1)

    ax.set_xlim(edges[0], edges[-1])

    ymax = float(np.max(y)) if len(y) else 1.0
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)

    if title is None:
        title = f"{hist_dataset['label']} black/white bias"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"log$_{10}$(white / black)", fontsize=13)
    ax.set_ylabel("% neurons" if use_percent else "Number of neurons", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    txt = (
        f"n = {hist_dataset['n']}\n"
        f"black = {black_frac * 100:.1f}%"
    )

    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.95,
            edgecolor="0.5",
        ),
    )

    if show_legend:
        ax.legend(frameon=False, fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved histogram: {save_path}")


def plot_black_white_histogram_dataset(
    hist_dataset,
    *,
    save_path,
    title=None,
    use_percent=False,
    show_legend=True,
):
    _safe_mkdir_for_file(save_path)

    edges = np.asarray(hist_dataset["bin_edges"], dtype=float)
    counts = np.asarray(hist_dataset["counts"], dtype=float)

    if len(edges) != len(counts) + 1:
        raise ValueError(
            f"Expected len(bin_edges) == len(counts) + 1, got "
            f"{len(edges)} edges and {len(counts)} counts."
        )

    y_key = "percent" if use_percent else "counts"
    y = np.asarray(hist_dataset[y_key], dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    is_black = centers < 0
    is_white = centers >= 0

    black_frac = _black_fraction_from_counts(counts, edges)
    hatch = hist_dataset.get("hatch", None)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.bar(
        centers[is_black],
        y[is_black],
        width=widths[is_black] * 0.82,
        color=BLACK_FILL,
        edgecolor="black",
        linewidth=0.9,
        hatch=hatch,
        align="center",
        label="black-dominant",
    )

    ax.bar(
        centers[is_white],
        y[is_white],
        width=widths[is_white] * 0.82,
        color=WHITE_FILL,
        edgecolor="black",
        linewidth=0.9,
        hatch=hatch,
        align="center",
        label="white-dominant",
    )

    ax.axvline(0, color="black", linewidth=1.1)

    ax.set_xlim(edges[0], edges[-1])

    ymax = float(np.max(y)) if len(y) else 1.0
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)

    ax.set_xlabel(r"log$_{10}$(white / black)", fontsize=13)
    ax.set_ylabel("% neurons" if use_percent else "Number of neurons", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    txt = (
        f"n = {hist_dataset['n']}\n"
        f"black = {black_frac * 100:.1f}%"
    )

    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.95,
            edgecolor="0.5",
        ),
    )

    if show_legend:
        handles = [
            Patch(facecolor=BLACK_FILL, edgecolor="black", hatch=hatch, label="black-dominant"),
            Patch(facecolor=WHITE_FILL, edgecolor="black", hatch=hatch, label="white-dominant"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved histogram: {save_path}")


def plot_black_white_histogram_comparison(
    hist_datasets,
    *,
    save_path,
    title=None,
    use_proportion=True,
    bar_width_fraction=0.36,
):
    """
    Side-by-side comparison of black/white dominance histograms.

    Visual convention:
        experiment = grayscale family
        model      = blue family

    Within each source:
        black-dominant bins = darker shade
        white-dominant bins = lighter shade

    No hatching.
    Experiment is always plotted first, then model.
    """
    if len(hist_datasets) == 0:
        raise ValueError("No histogram datasets provided.")

    _safe_mkdir_for_file(save_path)

    # ------------------------------------------------------------------
    # Enforce consistent source order:
    # experiment first, model second.
    # ------------------------------------------------------------------
    source_priority = {"experiment": 0, "model": 1}
    ordered_datasets = sorted(
        hist_datasets,
        key=lambda ds: source_priority.get(ds.get("source"), 99),
    )

    ref_edges = np.asarray(ordered_datasets[0]["bin_edges"], dtype=float)

    for ds in ordered_datasets:
        edges = np.asarray(ds["bin_edges"], dtype=float)
        if not np.allclose(edges, ref_edges):
            raise ValueError("All histogram datasets must use identical bin_edges.")

    centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
    widths = np.diff(ref_edges)

    is_black = centers < 0
    is_white = centers >= 0

    n_ds = len(ordered_datasets)
    offsets = np.array([0.0]) if n_ds == 1 else np.linspace(-0.5, 0.5, n_ds)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))

    max_y = 0.0

    for i, ds in enumerate(ordered_datasets):
        if use_proportion:
            if "proportion" in ds:
                y = np.asarray(ds["proportion"], dtype=float)
            elif "percent" in ds:
                y = np.asarray(ds["percent"], dtype=float) / 100.0
            else:
                counts = np.asarray(ds["counts"], dtype=float)
                total = np.sum(counts)
                y = counts / total if total > 0 else counts
        else:
            y = np.asarray(ds["counts"], dtype=float)        
        max_y = max(max_y, float(np.max(y)) if len(y) else 0.0)

        x = centers + offsets[i] * widths * bar_width_fraction
        bar_width = widths * bar_width_fraction

        source = ds.get("source")
        palette = _source_preference_palette(source)

        ax.bar(
            x[is_black],
            y[is_black],
            width=bar_width[is_black],
            color=palette["black"],
            edgecolor=BLACK,
            linewidth=0.9,
            align="center",
        )

        ax.bar(
            x[is_white],
            y[is_white],
            width=bar_width[is_white],
            color=palette["white"],
            edgecolor=BLACK,
            linewidth=0.9,
            align="center",
        )

    ax.axvline(0, color=BLACK, linewidth=1.1)

    ax.set_xlim(ref_edges[0], ref_edges[-1])

    if use_proportion:
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    else:
        ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1.0)

    ax.set_xlabel(r"log$_{10}$(E white / E black)", fontsize=13)
    ax.set_ylabel(
        "Proportion of neurons" if use_proportion else "Number of neurons",
        fontsize=13,
    )    
    ax.tick_params(axis="both", labelsize=11)


    # ------------------------------------------------------------------
    # Unified legend: experiment first, then model
    # ------------------------------------------------------------------
    model_palette = _source_preference_palette("model")
    experiment_palette = _source_preference_palette("experiment")

    legend_handles = [
        Patch(
            facecolor=experiment_palette["black"],
            edgecolor=BLACK,
            label="Exp. black",
        ),
        Patch(
            facecolor=model_palette["black"],
            edgecolor=BLACK,
            label="Model black",
        ),
        Patch(
            facecolor=experiment_palette["white"],
            edgecolor=BLACK,
            label="Exp. white",
        ),
        Patch(
            facecolor=model_palette["white"],
            edgecolor=BLACK,
            label="Model white",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=10,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.3,
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved histogram comparison: {save_path}")

# =============================================================================
# Scatter plotters
# =============================================================================


def _prepare_black_white_scatter_display(
    scatter_dataset,
    *,
    top_code=30.0,
    overflow_value=35.0,
    scale_model=False,
):
    """
    Prepare display coordinates from unified scatter dataset.

    Expected scatter_dataset:
        {
            "label": str,
            "source": "model" | "experiment",
            "white": array-like,
            "black": array-like,
            "n": int,
        }

    Experiment:
        top-code values >30 into overflow bin.

    Model:
        raw values by default.
        optionally min-max scaled to top_code for comparison panels.
    """

    white_raw = np.asarray(scatter_dataset["white"], dtype=float)
    black_raw = np.asarray(scatter_dataset["black"], dtype=float)

    mask = np.isfinite(white_raw) & np.isfinite(black_raw)
    white_raw = white_raw[mask]
    black_raw = black_raw[mask]

    if len(white_raw) == 0:
        raise ValueError(f"No finite scatter values for {scatter_dataset['label']}.")

    if scatter_dataset.get("source") == "experiment":
        white_display = _topcode_snr_for_display(
            white_raw,
            top_code=top_code,
            overflow_value=overflow_value,
        )
        black_display = _topcode_snr_for_display(
            black_raw,
            top_code=top_code,
            overflow_value=overflow_value,
        )

        ticks = [0, 5, 10, 15, 20, 25, top_code, overflow_value]
        tick_labels = [
            "0", "5", "10", "15", "20", "25",
            str(int(top_code)), f">{int(top_code)}",
        ]

        xlim = (0, overflow_value + 5)
        ylim = (0, overflow_value + 5)
        identity_max = top_code

    elif scale_model:
        vmax = max(float(np.max(white_raw)), float(np.max(black_raw)), 1e-12)

        white_display = white_raw / vmax * top_code
        black_display = black_raw / vmax * top_code

        ticks = [0, 5, 10, 15, 20, 25, top_code, overflow_value]
        tick_labels = ["0", "5", "10", "15", "20", "25", "30", ">30"]

        xlim = (0, overflow_value + 5)
        ylim = (0, overflow_value + 5)
        identity_max = top_code

    else:
        white_display = white_raw
        black_display = black_raw

        vmax = max(float(np.max(white_raw)), float(np.max(black_raw)), 1e-12)
        xlim = (0, vmax * 1.03)
        ylim = (0, vmax * 1.03)
        ticks = None
        tick_labels = None
        identity_max = vmax * 1.03

    return {
        "white_raw": white_raw,
        "black_raw": black_raw,
        "white_display": white_display,
        "black_display": black_display,
        "ticks": ticks,
        "tick_labels": tick_labels,
        "xlim": xlim,
        "ylim": ylim,
        "identity_max": identity_max,
    }


def plot_black_white_scatter_dataset(
    scatter_dataset,
    *,
    save_path,
    title=None,
    bins=24,
    top_code=30.0,
    overflow_value=35.0,
    scale_model=False,
    marker_size=24,
):
    """
    Plot one black/white scatter dataset with marginal histograms.

    x = white
    y = black
    y > x means black-dominant.
    """

    _safe_mkdir_for_file(save_path)

    prepared = _prepare_black_white_scatter_display(
        scatter_dataset,
        top_code=top_code,
        overflow_value=overflow_value,
        scale_model=scale_model,
    )

    white_raw = prepared["white_raw"]
    black_raw = prepared["black_raw"]
    white = prepared["white_display"]
    black = prepared["black_display"]

    frac_black = np.mean(black_raw > white_raw)

    if scatter_dataset.get("source") == "experiment":
        hist_bins = [0, 5, 10, 15, 20, 25, top_code, overflow_value]
        facecolor = "white"
        alpha = 0.95
        xlabel = r"SNR$_{white}$"
        ylabel = r"SNR$_{black}$"
    else:
        hist_bins = np.linspace(prepared["xlim"][0], prepared["xlim"][1], bins + 1)
        facecolor = "0.45"
        alpha = 0.75
        xlabel = scatter_dataset.get("white_label", "white response")
        ylabel = scatter_dataset.get("black_label", "black response")

    fig = plt.figure(figsize=(7.2, 7.2))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.12,
        right=0.92,
        bottom=0.11,
        top=0.90,
        wspace=0.04,
        hspace=0.04,
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(
        white,
        black,
        s=marker_size,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=0.85,
        alpha=alpha,
    )

    ax.plot(
        [0, prepared["identity_max"]],
        [0, prepared["identity_max"]],
        linestyle="--",
        color="0.35",
        linewidth=1.0,
    )

    ax.set_xlim(*prepared["xlim"])
    ax.set_ylim(*prepared["ylim"])
    ax.set_aspect("equal", adjustable="box")

    if prepared["ticks"] is not None:
        ax.set_xticks(prepared["ticks"])
        ax.set_yticks(prepared["ticks"])
        ax.set_xticklabels(prepared["tick_labels"])
        ax.set_yticklabels(prepared["tick_labels"])

    weights = np.ones(len(white)) * 100.0 / len(white)

    ax_histx.hist(
        white,
        bins=hist_bins,
        weights=weights,
        color="white",
        edgecolor="black",
        linewidth=0.9,
    )

    ax_histy.hist(
        black,
        bins=hist_bins,
        weights=weights,
        orientation="horizontal",
        color="black",
        edgecolor="black",
        linewidth=0.9,
    )

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", labelsize=9)
    ax_histx.set_ylabel("% cells", fontsize=10)

    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelsize=9)
    ax_histy.set_xlabel("% cells", fontsize=10)

    for a in (ax_histx, ax_histy):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    if title is None:
        title = f"{scatter_dataset['label']} black/white scatter"

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    txt = (
        f"n = {len(white)}\n"
        f"black > white = {frac_black * 100:.1f}%"
    )

    ax.text(
        0.04,
        0.96,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.95,
            edgecolor="0.5",
        ),
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved scatter: {save_path}")

def _scatter_hist_bins_for_display(
    *,
    top_code=30.0,
    overflow_value=35.0,
):
    """
    Display bins for Yeh-style scatter marginals.

    Last bin is the categorical overflow bin:
        30..35 means 30+
    """
    return np.asarray(
        [0, 5, 10, 15, 20, 25, top_code, overflow_value],
        dtype=float,
    )

def plot_black_white_scatter_comparison_panel(
    scatter_datasets,
    *,
    save_path,
    top_code=30.0,
    overflow_value=35.0,
    marker_size=22,
):
    """
    Side-by-side scatter comparison with per-panel marginal histograms.

    Scatter:
        dataset identity only
        model      = filled grey
        experiment = open white

    Marginals:
        response identity + dataset identity
        white response = light grey
        black response = dark grey
        model          = solid
        experiment     = hatched
    """

    _safe_mkdir_for_file(save_path)

    n_panels = len(scatter_datasets)

    hist_bins = _scatter_hist_bins_for_display(
        top_code=top_code,
        overflow_value=overflow_value,
    )

    tick_vals = hist_bins
    tick_labels = ["0", "5", "10", "15", "20", "25", "30", ">30"]

    fig = plt.figure(figsize=(5.7 * n_panels + 1.2, 5.8))

    outer = fig.add_gridspec(
        1,
        n_panels,
        left=0.07,
        right=0.84,   # leave room for legend
        bottom=0.10,
        top=0.95,
        wspace=0.28,
    )

    for j, ds in enumerate(scatter_datasets):
        sub = outer[j].subgridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            wspace=0.04,
            hspace=0.04,
        )

        ax_histx = fig.add_subplot(sub[0, 0])
        ax = fig.add_subplot(sub[1, 0], sharex=ax_histx)
        ax_histy = fig.add_subplot(sub[1, 1], sharey=ax)

        prepared = _prepare_black_white_scatter_display(
            ds,
            top_code=top_code,
            overflow_value=overflow_value,
            scale_model=True,
        )

        white_raw = prepared["white_raw"]
        black_raw = prepared["black_raw"]
        white = prepared["white_display"]
        black = prepared["black_display"]

        frac_black = np.mean(black_raw > white_raw)

        if ds.get("source") == "experiment":
            scatter_face = "white"
            scatter_edge = "black"
            scatter_alpha = 0.95
            hatch = "///"
        else:
            scatter_face = "0.35"
            scatter_edge = "black"
            scatter_alpha = 0.55
            hatch = None

        ax.scatter(
            white,
            black,
            s=marker_size,
            facecolor=scatter_face,
            edgecolor=scatter_edge,
            linewidth=0.85,
            alpha=scatter_alpha,
        )

        ax.plot(
            [0, top_code],
            [0, top_code],
            linestyle="--",
            color="0.35",
            linewidth=1.0,
        )

        ax.set_xlim(0, overflow_value + 5)
        ax.set_ylim(0, overflow_value + 5)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        # Same axis wording for both panels. Caption carries the scaling caveat.
        ax.set_xlabel("E white", fontsize=12)
        ax.set_ylabel("E black", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        weights = np.ones(len(white)) * 100.0 / len(white)

        ax_histx.hist(
            white,
            bins=hist_bins,
            weights=weights,
            color=WHITE_FILL,
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch,
        )

        ax_histy.hist(
            black,
            bins=hist_bins,
            weights=weights,
            orientation="horizontal",
            color=BLACK_FILL,
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch,
        )

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", labelsize=8)
        ax_histx.set_ylabel("% neurons", fontsize=9)

        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelsize=8)
        ax_histy.set_xlabel("% neurons", fontsize=9)

        for a in (ax_histx, ax_histy):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

    scatter_handles = [
        Patch(facecolor="0.35", edgecolor="black", label="Model points"),
        Patch(facecolor="white", edgecolor="black", label="Experimental points"),
    ]

    marginal_handles = [
        Patch(facecolor=WHITE_FILL, edgecolor="black", label="Model white marginal"),
        Patch(facecolor=BLACK_FILL, edgecolor="black", label="Model black marginal"),
        Patch(facecolor=WHITE_FILL, edgecolor="black", hatch="///", label="Experimental white marginal"),
        Patch(facecolor=BLACK_FILL, edgecolor="black", hatch="///", label="Experimental black marginal"),
    ]

    legend_handles = scatter_handles + marginal_handles

    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.52),
        frameon=False,
        fontsize=8,
        ncol=1,
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved scatter comparison panel: {save_path}")

def plot_black_white_scatter_comparison_overlay(
    scatter_datasets,
    *,
    save_path,
    top_code=30.0,
    overflow_value=35.0,
    marker_size=24,
    bar_width_fraction=0.36,
):
    """
    Overlay model and experiment scatter with shared marginal histograms.

    Scatter:
        source family + binary black/white preference tone

    Marginals:
        top   = projection of E white values only
        right = projection of E black values only

    Marginals are NOT split by preference class.
    Their colors only indicate which axis is being projected:
        top   = lighter / white-like source shade
        right = darker / black-like source shade

    For the side-by-side marginal bars, experiment is always plotted first,
    then model, regardless of input order.
    """

    _safe_mkdir_for_file(save_path)

    hist_bins = _scatter_hist_bins_for_display(
        top_code=top_code,
        overflow_value=overflow_value,
    )

    tick_vals = hist_bins
    tick_labels = ["0", "5", "10", "15", "20", "25", "30", ">30"]

    centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    widths = np.diff(hist_bins)

    # ------------------------------------------------------------------
    # Enforce dataset order for marginal bars:
    # experiment first, model second.
    # Keep any unknown sources at the end.
    # ------------------------------------------------------------------
    source_priority = {"experiment": 0, "model": 1}
    ordered_datasets = sorted(
        scatter_datasets,
        key=lambda ds: source_priority.get(ds.get("source"), 99),
    )

    n_ds = len(ordered_datasets)

    if n_ds == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.5, 0.5, n_ds)

    fig = plt.figure(figsize=(7.2, 7.0))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.12,
        right=0.93,
        bottom=0.11,
        top=0.94,
        wspace=0.04,
        hspace=0.04,
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    max_hist_y = 0.0
    max_hist_x = 0.0

    for i, ds in enumerate(ordered_datasets):
        prepared = _prepare_black_white_scatter_display(
            ds,
            top_code=top_code,
            overflow_value=overflow_value,
            scale_model=True,
        )

        white_raw = prepared["white_raw"]
        black_raw = prepared["black_raw"]

        white = prepared["white_display"]
        black = prepared["black_display"]

        source = ds.get("source")

        # Scatter palette:
        # source family + black/white preference tone.
        scatter_palette = _source_preference_palette(source)

        masks = _black_white_preference_masks(
            white_raw,
            black_raw,
        )

        point_colors = _colors_from_preference_masks(
            masks,
            scatter_palette,
        )
        point_colors = np.asarray(point_colors, dtype=object)

        # Marginal palette:
        # same source family, but axis-specific shade.
        # Top white projection gets white-ish shade.
        # Right black projection gets black-ish shade.
        projection_palette = _source_projection_palette(source)

        if source == "experiment":
            scatter_alpha = 0.92
            hist_alpha = 0.80
            zorder = 3
        else:
            scatter_alpha = 0.75
            hist_alpha = 0.88
            zorder = 2

        # --------------------------------------------------------------
        # Scatter
        # --------------------------------------------------------------
        if source == "experiment":
            point_edges = np.full(len(white), WHITE, dtype=object)
            point_linewidths = np.full(len(white), 0.70, dtype=float)

            # Experiment white-preferring gets black edge.
            point_edges[masks["white"]] = BLACK
            point_linewidths[masks["white"]] = 0.75

        else:
            # Model: black edge for all points.
            point_edges = np.full(len(white), BLACK, dtype=object)
            point_linewidths = np.full(len(white), 0.55, dtype=float)

        ax.scatter(
            white,
            black,
            s=marker_size,
            facecolors=point_colors,
            edgecolors=point_edges,
            linewidths=point_linewidths,
            alpha=scatter_alpha,
            label=f"{ds['label']} (n={ds['n']})",
            zorder=zorder,
        )

        weights = np.ones(len(white)) / len(white)

        # --------------------------------------------------------------
        # Plain axis projections
        # --------------------------------------------------------------
        # Top marginal: histogram of E white values only.
        white_counts, _ = np.histogram(
            white,
            bins=hist_bins,
            weights=weights,
        )

        # Right marginal: histogram of E black values only.
        black_counts, _ = np.histogram(
            black,
            bins=hist_bins,
            weights=weights,
        )

        max_hist_y = max(max_hist_y, float(np.max(white_counts)))
        max_hist_x = max(max_hist_x, float(np.max(black_counts)))

        x = centers + offsets[i] * widths * bar_width_fraction
        y = centers + offsets[i] * widths * bar_width_fraction
        bar_width = widths * bar_width_fraction

        ax_histx.bar(
            x,
            white_counts,
            width=bar_width,
            color=projection_palette["white_hist"],
            edgecolor=projection_palette["edge"],
            linewidth=0.6,
            align="center",
            alpha=hist_alpha,
        )

        ax_histy.barh(
            y,
            black_counts,
            height=bar_width,
            color=projection_palette["black_hist"],
            edgecolor=projection_palette["edge"],
            linewidth=0.6,
            align="center",
            alpha=hist_alpha,
        )

    # ------------------------------------------------------------------
    # Main scatter formatting
    # ------------------------------------------------------------------
    ax.plot(
        [0, top_code],
        [0, top_code],
        linestyle="--",
        color=MID,
        linewidth=1.0,
        zorder=1,
    )

    ax.set_xlim(0, overflow_value + 5)
    ax.set_ylim(0, overflow_value + 5)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("E white", fontsize=12)
    ax.set_ylabel("E black", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # ------------------------------------------------------------------
    # Marginal formatting
    # ------------------------------------------------------------------
    ax_histx.set_ylim(0, 1.0)
    ax_histy.set_xlim(0, 1.0)

    ax_histx.set_yticks([0.0, 0.5, 1.0])
    ax_histy.set_xticks([0.0, 0.5, 1.0])

    ax_histx.set_yticklabels(["0", "0.5", "1"])
    ax_histy.set_xticklabels(["0", "0.5", "1"])

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", labelsize=8)
    ax_histx.set_ylabel("Prop.", fontsize=9)
    ax_histx.set_title("E white", fontsize=9, pad=2)

    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelsize=8)
    ax_histy.set_xlabel("Prop.", fontsize=9)
    ax_histy.set_title("E black", fontsize=9, pad=2)

    for a in (ax_histx, ax_histy):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # One unified scatter legend, upper right.
    # Experiment first, then model.
    # ------------------------------------------------------------------
    model_palette = _source_preference_palette("model")
    experiment_palette = _source_preference_palette("experiment")

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=experiment_palette["black"],
            markeredgecolor=WHITE,
            markeredgewidth=0.7,
            markersize=5.5,
            label="Exp. black",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=experiment_palette["white"],
            markeredgecolor=BLACK,
            markeredgewidth=0.7,
            markersize=5.5,
            label="Exp. white",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=model_palette["black"],
            markeredgecolor=BLACK,
            markeredgewidth=0.55,
            markersize=5.5,
            label="Model black",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=model_palette["white"],
            markeredgecolor=BLACK,
            markeredgewidth=0.55,
            markersize=5.5,
            label="Model white",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7.4,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
        borderpad=0.2,
        labelspacing=0.25,
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved scatter comparison overlay: {save_path}")

# helper functions for scatter plotters color palettes and preference masks 
def _blend_color(color, target, amount):
    """
    Blend color toward target.

    amount = 0 -> color
    amount = 1 -> target

    Returns hex string, not RGB tuple/array.
    This avoids NumPy boolean-assignment nonsense.
    """
    base = np.asarray(colors.to_rgb(color), dtype=float)
    target = np.asarray(colors.to_rgb(target), dtype=float)
    out = base * (1.0 - amount) + target * amount
    out = np.clip(out, 0, 1)

    return "#{:02X}{:02X}{:02X}".format(
        int(round(out[0] * 255)),
        int(round(out[1] * 255)),
        int(round(out[2] * 255)),
    )


def _source_preference_palette(source):
    """
    Scatter / histogram preference palette.

    black = base source color
    white = lightened source color
    """
    if source == "model":
        return {
            "black": MODEL_COLOR,
            "white": _blend_color(MODEL_COLOR, WHITE, 0.38),
            "edge": BLACK,
        }

    return {
        "black": BLACK,
        "white": LIGHT,
        "edge": BLACK,
    }


def _black_white_preference_masks(white_raw, black_raw):
    """
    Binary article-style preference.

    black:
        E_black > E_white

    white:
        E_white >= E_black
    """
    white_raw = np.asarray(white_raw, dtype=float)
    black_raw = np.asarray(black_raw, dtype=float)

    black_mask = black_raw > white_raw
    white_mask = ~black_mask

    return {
        "black": black_mask,
        "white": white_mask,
    }


def _colors_from_preference_masks(masks, palette):
    point_colors = np.empty(len(masks["black"]), dtype=object)

    point_colors[masks["black"]] = palette["black"]
    point_colors[masks["white"]] = palette["white"]

    return point_colors


def _source_projection_palette(source):
    """
    Marginal projection colors.

    Top histogram:
        E white projection -> lightened source color

    Right histogram:
        E black projection -> base source color
    """
    if source == "model":
        return {
            "white_hist": _blend_color(MODEL_COLOR, WHITE, 0.38),
            "black_hist": MODEL_COLOR,
            "edge": BLACK,
        }

    return {
        "white_hist": LIGHT,
        "black_hist": BLACK,
        "edge": BLACK,
    }