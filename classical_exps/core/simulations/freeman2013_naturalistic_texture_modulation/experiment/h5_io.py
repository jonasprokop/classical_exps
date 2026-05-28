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

from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.audit import compute_selected_lowlevel_family_stats, compute_texture_noise_image_statistics



def save_selected_lowlevel_family_stats_to_h5(
    h5_file,
    tex_imgs,
    noise_imgs,
    dict_fam,
    group_path="/texture_noise_response/lowlevel_family_stats",
    overwrite=True,
):
    """
    Save compact low-level family statistics into the same HDF5 file.
    """
    family_ids, feature_names, family_features = compute_selected_lowlevel_family_stats(
        tex_imgs=tex_imgs,
        noise_imgs=noise_imgs,
        dict_fam=dict_fam,
    )

    with h5py.File(h5_file, "a") as f:
        if group_path in f and overwrite:
            del f[group_path]

        grp = f.require_group(group_path)

        grp.create_dataset("family_ids", data=family_ids)
        grp.create_dataset("family_features", data=family_features)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "feature_names",
            data=np.asarray(feature_names, dtype=object),
            dtype=string_dtype,
        )

    print(f"   > low-level family image stats saved to HDF5: {group_path}")

    

def save_texture_noise_image_statistics_to_h5(
    h5_file,
    tex_imgs,
    noise_imgs,
    dict_fam,
    group_path="/texture_noise_response/image_statistics",
    overwrite=True,
):
    """
    Save family-level and sample-level low-level image statistics into HDF5.
    """
    family_ids, feature_names, family_features, sample_features = (
        compute_texture_noise_image_statistics(
            tex_imgs=tex_imgs,
            noise_imgs=noise_imgs,
            dict_fam=dict_fam,
        )
    )

    with h5py.File(h5_file, "a") as f:
        if group_path in f and overwrite:
            del f[group_path]

        grp = f.require_group(group_path)

        grp.create_dataset("family_ids", data=family_ids)
        grp.create_dataset("family_features", data=family_features)
        grp.create_dataset("sample_features", data=sample_features)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "feature_names",
            data=np.asarray(feature_names, dtype=object),
            dtype=string_dtype,
        )

    print(f"   > image statistics saved to HDF5: {group_path}")

 
