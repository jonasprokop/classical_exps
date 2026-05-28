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

import re
import hashlib
from pathlib import Path


from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.audit import audit_texture_noise_images



def center_crop_img(img, target_res=[93, 93]):
    """
    Deterministic center crop.
    For benchmarking a static network.
    """
    y_res, x_res = img.shape[:2]
    target_y, target_x = target_res

    if y_res < target_y or x_res < target_x:
        raise ValueError(
            f"Image too small for crop: image={img.shape}, target_res={target_res}"
        )

    init_y = (y_res - target_y) // 2
    init_x = (x_res - target_x) // 2

    return img[init_y:init_y + target_y, init_x:init_x + target_x]


def random_crop_img(img, target_res=[93, 93], rng=None):
    """
    Seeded random crop. Use only if you explicitly want crop variability.
    """
    y_res, x_res = img.shape[:2]
    target_y, target_x = target_res

    if y_res < target_y or x_res < target_x:
        raise ValueError(
            f"Image too small for crop: image={img.shape}, target_res={target_res}"
        )

    if rng is None:
        rng = np.random.default_rng(0)

    init_y = rng.integers(0, y_res - target_y + 1)
    init_x = rng.integers(0, x_res - target_x + 1)

    return img[init_y:init_y + target_y, init_x:init_x + target_x]



def load_imgs(
    directory_imgs,
    target_res=[93, 93],
    contrast=1,
    pixel_min=-1.7876,
    pixel_max=2.1919,
    num_samples=15,
    device=None,
    crop_mode="center",
    seed=0,
    expected_num_families=15,
):
    """
    Loads Freeman texture/noise images deterministically.

    Expected image name format:
        tex-320x320-im13-smp2.png
        noise-320x320-im13-smp2.png

    Output:
        tex_imgs   shape = [n_family, n_sample, y, x]
        noise_imgs shape = [n_family, n_sample, y, x]
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    directory_imgs = Path(directory_imgs)

    # Structural audit first. Fail loudly.
    records, families = audit_texture_noise_images(
        directory_imgs=directory_imgs,
        num_samples=num_samples,
        expected_num_families=expected_num_families,
        require_complete=True,
    )

    dict_fam = {family: i for i, family in enumerate(families)}

    tex_imgs = torch.zeros(
        (len(families), num_samples, *target_res),
        dtype=torch.float32,
        device=device,
    )

    noise_imgs = torch.zeros_like(tex_imgs)

    rng = np.random.default_rng(seed)

    print("   > loading images ...")

    # Deterministic order: family -> sample -> category
    records = sorted(
        records,
        key=lambda r: (r["family"], r["sample"], r["category"])
    )

    for r in tqdm(records):
        img_name = r["file"]
        category = r["category"]
        family = r["family"]
        sample = r["sample"]

        fam_id = dict_fam[family]
        smp_id = sample - 1

        img_path = directory_imgs / img_name
        img = mpimg.imread(str(img_path))

        # Convert RGB/RGBA to grayscale if needed.
        # Your model appears to expect [H, W], not [C, H, W].
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)

        img = torch.tensor(img, dtype=torch.float32)

        if crop_mode == "center":
            cropped_img = center_crop_img(img=img, target_res=target_res)
        elif crop_mode == "random":
            cropped_img = random_crop_img(img=img, target_res=target_res, rng=rng)
        elif crop_mode is None or crop_mode == "none":
            if list(img.shape) != list(target_res):
                raise ValueError(
                    f"crop_mode='none' but image has shape {img.shape}, "
                    f"expected {target_res}. File: {img_name}"
                )
            cropped_img = img
        else:
            raise ValueError(
                f"Unknown crop_mode={crop_mode}. Use 'center', 'random', or 'none'."
            )

        cropped_img = cropped_img.to(device)

        if category == "tex":
            tex_imgs[fam_id, smp_id] = cropped_img
        elif category == "noise":
            noise_imgs[fam_id, smp_id] = cropped_img
        else:
            # Should be impossible because audit already checks this.
            raise ValueError(f"Unknown image category: {category}")

    # Fixed global rescaling across the loaded stimulus set.
    # This keeps your original behavior, but now deterministic.
    min_val = torch.min(torch.min(tex_imgs), torch.min(noise_imgs))
    max_val = torch.max(torch.max(tex_imgs), torch.max(noise_imgs))

    if torch.isclose(max_val, min_val):
        raise ValueError(
            f"Invalid image intensity range: min_val={min_val}, max_val={max_val}"
        )

    for i in range(len(tex_imgs)):
        for j in range(num_samples):
            tex_img = tex_imgs[i, j]
            noise_img = noise_imgs[i, j]

            tex_img = rescale(tex_img, min_val, max_val, 0, 1) * contrast
            tex_img = rescale(tex_img, 0, 1, pixel_min, pixel_max)

            noise_img = rescale(noise_img, min_val, max_val, 0, 1) * contrast
            noise_img = rescale(noise_img, 0, 1, pixel_min, pixel_max)

            tex_imgs[i, j] = tex_img
            noise_imgs[i, j] = noise_img

    return tex_imgs, noise_imgs, dict_fam