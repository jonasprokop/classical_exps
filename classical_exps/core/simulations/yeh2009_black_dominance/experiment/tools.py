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
import io
import scipy.ndimage


# Saving an example stimuli panel:

def save_generated_stimuli_panel_png(
    black_stimuli,
    white_stimuli,
    sample_indices,
    sample_positions,
    save_path,
):
    """
    Save an example panel from the exact generated stimuli tensors.

    Parameters
    ----------
    black_stimuli, white_stimuli : torch.Tensor or np.ndarray
        Shape (num_dots, H, W)
    sample_indices : list[int]
        Indices into the generated stimuli tensors
    sample_positions : list[tuple[int, int]]
        Matching (y, x) top-left positions for labels
    save_path : str
        Output .png path
    """
    if isinstance(black_stimuli, torch.Tensor):
        black_np = black_stimuli.detach().cpu().numpy()
    else:
        black_np = np.asarray(black_stimuli)

    if isinstance(white_stimuli, torch.Tensor):
        white_np = white_stimuli.detach().cpu().numpy()
    else:
        white_np = np.asarray(white_stimuli)

    black_imgs = [black_np[i] for i in sample_indices]
    white_imgs = [white_np[i] for i in sample_indices]

    fig = plt.figure(figsize=(10, 5))

    for i, (img, (y, x)) in enumerate(zip(black_imgs, sample_positions), start=1):
        ax = fig.add_subplot(2, 4, i)
        ax.imshow(img, interpolation="nearest", cmap="gray", vmax=1, vmin=-1)
        ax.set_title(f"B (y{y}, x{x})", fontsize=9)
        ax.axis("off")

    for i, (img, (y, x)) in enumerate(zip(white_imgs, sample_positions), start=5):
        ax = fig.add_subplot(2, 4, i)
        ax.imshow(img, interpolation="nearest", cmap="gray", vmax=1, vmin=-1)
        ax.set_title(f"W (y{y}, x{x})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

import os
import numpy as np
import torch
import h5py
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
from classical_exps.core.tools.utils import *
from classical_exps.core.tools.utils import SingleCellModel
from classical_exps.core.tools.filtering_functions import *


# ============================================================
# Core utility
# ============================================================

def _map_variance(arr, apply_smoothing=False, sigma=1.0):
    """
    Spatial variance of a response map.

    Default is raw variance, which is closer to the article's definition.
    Optional Gaussian smoothing can be enabled as a robustness check.
    """
    arr = np.asarray(arr, dtype=float)
    if apply_smoothing:
        arr = scipy.ndimage.gaussian_filter(arr, sigma=sigma)
    return float(np.var(arr))

# ============================================================
# Core utility
# ============================================================

def _map_variance(arr, apply_smoothing=False, sigma=1.0):
    """
    Spatial variance of a response map.

    Default is raw variance, which is closer to the article's definition.
    Optional Gaussian smoothing can be enabled as a robustness check.
    """
    arr = np.asarray(arr, dtype=float)
    if apply_smoothing:
        arr = scipy.ndimage.gaussian_filter(arr, sigma=sigma)
    return float(np.var(arr))


# ============================================================
# Exact generated stimuli sanity panel
# ============================================================

def save_generated_stimuli_panel_png(
    black_stimuli,
    white_stimuli,
    sample_indices,
    sample_positions,
    save_path,
):
    if isinstance(black_stimuli, torch.Tensor):
        black_np = black_stimuli.detach().cpu().numpy()
    else:
        black_np = np.asarray(black_stimuli)

    if isinstance(white_stimuli, torch.Tensor):
        white_np = white_stimuli.detach().cpu().numpy()
    else:
        white_np = np.asarray(white_stimuli)

    black_imgs = [black_np[i] for i in sample_indices]
    white_imgs = [white_np[i] for i in sample_indices]

    fig = plt.figure(figsize=(10, 5))

    for i, (img, (y, x)) in enumerate(zip(black_imgs, sample_positions), start=1):
        ax = fig.add_subplot(2, 4, i)
        ax.imshow(img, interpolation="nearest", cmap="gray", vmax=1, vmin=-1)
        ax.set_title(f"B (y{y}, x{x})", fontsize=9)
        ax.axis("off")

    for i, (img, (y, x)) in enumerate(zip(white_imgs, sample_positions), start=5):
        ax = fig.add_subplot(2, 4, i)
        ax.imshow(img, interpolation="nearest", cmap="gray", vmax=1, vmin=-1)
        ax.set_title(f"W (y{y}, x{x})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


