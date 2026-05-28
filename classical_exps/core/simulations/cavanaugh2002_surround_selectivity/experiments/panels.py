############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.core.tools.utils import *
## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.core.tools.utils import SingleCellModel
## Data storage
import h5py
from scipy.optimize import curve_fit
import time
import cv2
import matplotlib.pyplot as plt



##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################

def six_panel_indices(n: int, panel_n: int = 6) -> np.ndarray:
    """
    Select 6 orientation indices for panel visualization,
    but ALWAYS include the Δθ ≈ 0 case.

    Why this exists:
    ----------------
    - Orientation tuning is π-periodic (bidirectional), so many
      sampled orientations are redundant for visualization.
    - We only need a sparse subset for sanity-check panels.
    - HOWEVER, the Δθ = 0 condition (same center & surround orientation)
      is the key diagnostic case:
          * it should produce seamless alignment at the boundary
            when AMRF ≈ GSF
          * if there's a discontinuity there, we likely have
            a phase/origin mismatch bug
    - Default linspace sampling can skip the exact zero index.
      This function forces it in.
    """


    n = int(n)
    panel_n = int(panel_n)

    if n <= 0:
        raise ValueError("n must be >= 1")
    if panel_n <= 0:
        raise ValueError("panel_n must be >= 1")

    # If we ask for more panels than available samples, just return all indices.
    if panel_n >= n:
        return np.arange(n, dtype=int)

    # Start with roughly evenly spaced indices
    idx = np.round(np.linspace(0, n - 1, panel_n)).astype(int)
    idx = np.unique(idx).tolist()

    # For symmetric grids like linspace(-π, π, 9),
    # the center index corresponds to Δθ ≈ 0
    zero_idx = n // 2

    # Ensure the Δθ=0 case is always included
    if zero_idx not in idx:
        # Replace one interior element (not extremes if possible)
        replace_pos = min(max(1, panel_n // 2), len(idx) - 2) if len(idx) >= 3 else 0
        idx[replace_pos] = zero_idx
        idx = sorted(set(idx))

    # If uniqueness reduced count below panel_n, refill deterministically
    while len(idx) < panel_n:
        for k in range(n):
            if k not in idx:
                idx.append(k)
                idx = sorted(idx)
                if len(idx) == panel_n:
                    break

    return np.array(idx[:panel_n], dtype=int)


def plot_orientation_controls_panel_raw(images, labels, out_path, title=""):
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if len(images) == 0:
        raise ValueError("`images` must contain at least one image.")
    if len(images) != len(labels):
        raise ValueError(
            f"`images` and `labels` must have the same length, got {len(images)} and {len(labels)}."
        )

    # raw global limits across all provided images (diagnostic, no manipulation)
    all_vals = np.concatenate([np.asarray(im).ravel() for im in images])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    n = len(images)

    # sensible compact layouts:
    # 1 -> 1x1
    # 2 -> 1x2
    # 3 -> 1x3
    # 4 -> 2x2
    # 5-6 -> 2x3
    # 7-9 -> 3x3
    if n <= 3:
        ncols = n
    else:
        ncols = 3
    nrows = math.ceil(n / ncols)

    fig_w = max(4.0, 3.2 * ncols)
    fig_h = max(3.2, 3.0 * nrows + 0.8)
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # Normalize axs to a flat list even for 1x1
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    im_artist = None
    for i, ax in enumerate(axs):
        if i < n:
            im_artist = ax.imshow(
                images[i],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="none",  # no smoothing lies
            )
            ax.set_title(labels[i], fontsize=10, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)

    # one shared colorbar, outside the grid
    cbar = fig.colorbar(
        im_artist,
        ax=axs,
        fraction=0.035,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("stimulus value (raw)", fontsize=10)

    # tiny diagnostic footer
    fig.text(0.01, 0.01, f"vmin={vmin:.3f}, vmax={vmax:.3f}", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_ccss_controls_panel_raw(rows, out_path, title=""):
    """
    Plot a diagnostic panel for the article-faithful CCSS experiment.

    Parameters
    ----------
    rows : list of dict
        One dict per surround contrast, with structure:
        {
            "surround_contrast": float,
            "stimuli": [
                {"center_contrast": float, "img": (H, W)},
                ...
            ]
        }

    The panel layout is:
        rows    -> surround contrast
        columns -> center contrast
    """

    if len(rows) == 0:
        raise ValueError("rows is empty")

    n_rows = len(rows)
    n_cols = len(rows[0]["stimuli"])

    all_imgs = []
    for row in rows:
        if len(row["stimuli"]) != n_cols:
            raise ValueError("Inconsistent number of center contrasts across rows.")
        for cell in row["stimuli"]:
            if cell["img"] is None:
                raise ValueError("Encountered None image in panel rows.")
            all_imgs.append(cell["img"].ravel())

    all_vals = np.concatenate(all_imgs)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[None, :]
    elif n_cols == 1:
        axs = axs[:, None]

    im_artist = None

    for r, row in enumerate(rows):
        surround_contrast = row["surround_contrast"]
        for c, cell in enumerate(row["stimuli"]):
            ax = axs[r, c]
            im_artist = ax.imshow(
                cell["img"],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="none"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0:
                ax.set_title(f"c={cell['center_contrast']:.2f}", fontsize=10, pad=6)

            if c == 0:
                ax.set_ylabel(
                    f"s={surround_contrast:.2f}",
                    fontsize=10,
                    rotation=0,
                    labelpad=28,
                    va="center"
                )

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    cbar = fig.colorbar(im_artist, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("stimulus value (raw)", fontsize=10)

    fig.text(0.01, 0.01, f"vmin={vmin:.3f}, vmax={vmax:.3f}", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)