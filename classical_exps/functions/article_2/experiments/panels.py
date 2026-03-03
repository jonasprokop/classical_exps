############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.functions.utils import *
## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.functions.utils import SingleCellModel
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

def plot_orientation_controls_panel_raw(images6, labels6, out_path, title=""):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # raw global limits across the six (diagnostic, no manipulation)
    all_vals = np.concatenate([im.ravel() for im in images6])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(2, 3, figsize=(10, 6.2))
    axs = axs.flatten()

    im_artist = None
    for i, ax in enumerate(axs):
        im_artist = ax.imshow(
            images6[i],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",  # no smoothing lies
        )
        ax.set_title(labels6[i], fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)

    # one shared colorbar, outside the grid, not overlaying anything
    cbar = fig.colorbar(
        im_artist,
        ax=axs.tolist(),
        fraction=0.035,
        pad=0.02
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("stimulus value (raw)", fontsize=10)

    # tiny diagnostic footer (actual range used)
    fig.text(0.01, 0.01, f"vmin={vmin:.3f}, vmax={vmax:.3f}", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_ccss_controls_panel_raw(rows, out_path, title=""):
    """
    rows: list of dicts, one per center_contrast, each dict contains:
        {
          "contrast": float,
          "iso": (H,W) raw,
          "ortho": (H,W) raw,
          "center": (H,W) raw
        }
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # global limits across all tiles (diagnostic)
    all_vals = np.concatenate([d[k].ravel() for d in rows for k in ("iso", "ortho", "center")])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    n = len(rows)
    fig, axs = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axs = np.array([axs])  # force 2D

    col_titles = ["iso (Δθ=0)", "ortho (Δθ=π/2)", "center-only (surround=0)"]

    im_artist = None
    for r, d in enumerate(rows):
        imgs = [d["iso"], d["ortho"], d["center"]]
        for c in range(3):
            ax = axs[r, c]
            im_artist = ax.imshow(imgs[c], cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, pad=6)
        axs[r, 0].set_ylabel(f"c={d['contrast']:.2f}", fontsize=10, rotation=0, labelpad=30, va="center")

    if title:
        fig.suptitle(title, fontsize=13, y=0.99)

    cbar = fig.colorbar(im_artist, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("stimulus value (raw)", fontsize=10)
    fig.text(0.01, 0.01, f"vmin={vmin:.3f}, vmax={vmax:.3f}", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
