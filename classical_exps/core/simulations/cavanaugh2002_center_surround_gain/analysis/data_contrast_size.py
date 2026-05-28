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
from .shared import neuron_key

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

def load_contrast_size_tuning_fits_bulk(
    h5_file: str,
    neuron_ids,
    curves_path: str,
    strict: bool = False,
    expected_shape=None,  # (n_contrasts, n_radii)
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    y_fit = {}

    with h5py.File(h5_file, "r") as f:
        if curves_path not in f:
            # fits are optional
            return {
                "present_ids": np.array([], dtype=int),
                "missing_ids": neuron_ids.copy(),
                "by_id": {},
                "y_fit": {},
                "curves_path": curves_path,
            }

        grp = f[curves_path]
        for nid in neuron_ids:
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid))
                continue

            mat = np.asarray(grp[k][:], dtype=float)

            if strict and expected_shape is not None and mat.shape != tuple(expected_shape):
                raise ValueError(
                    f"{curves_path}/{k} shape mismatch: got {mat.shape}, expected {expected_shape}"
                )

            present.append(int(nid))
            y_fit[int(nid)] = mat

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": y_fit,
        "y_fit": y_fit,
        "curves_path": curves_path,
    }


def load_contrast_size_tuning_curves_bulk(
    h5_file: str,
    neuron_ids,
    curves_path: str = "/contrast_size_tuning/curves",
    curves_f1_path: str = "/contrast_size_tuning/curves_f1",
    strict: bool = True,
    load_f1: bool = False,
    require_f1: bool = False,
    expected_shape=None,  # (n_contrasts, n_radii)
):
    """
    Load contrast-size tuning curves with inferred global shape.

    Rules:
      - Each neuron dataset in curves_path must be 2D: (C, R).
      - All neurons must share the SAME (C, R) shape, inferred from the first loaded neuron.
      - If F1 is loaded and present, it must have exactly the same shape as F0 for that neuron.
      - If require_f1=True, missing F1 for any present neuron is an error (when strict).

    Returns:
      dict with y_f0 (and optionally y_f1), keyed by neuron_id, plus inferred_shape.
    """
    import numpy as np
    import h5py

    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    missing_f1 = []
    y_f0, y_f1 = {}, {}

    inferred_shape = None  

    with h5py.File(h5_file, "r") as f:
        if curves_path not in f:
            raise KeyError(f"Missing group: {curves_path}")
        grp0 = f[curves_path]

        grp1 = None
        if load_f1:
            if curves_f1_path in f:
                grp1 = f[curves_f1_path]
            elif require_f1:
                raise KeyError(f"Missing group (require_f1=True): {curves_f1_path}")

        for nid in neuron_ids:
            k = neuron_key(nid)

            if k not in grp0:
                missing.append(int(nid))
                continue

            mat0 = np.asarray(grp0[k][:], dtype=float)

            # Must be 2D (C,R)
            if mat0.ndim != 2:
                if strict:
                    raise ValueError(f"{curves_path}/{k} expected 2D (C,R), got {mat0.shape}")
                missing.append(int(nid))
                continue

            # Infer global shape from first valid neuron
            if inferred_shape is None:
                inferred_shape = tuple(mat0.shape)
            else:
                if tuple(mat0.shape) != inferred_shape:
                    if strict:
                        raise ValueError(
                            f"{curves_path}/{k} shape mismatch: got {mat0.shape}, expected {inferred_shape}"
                        )
                    missing.append(int(nid))
                    continue

            # Optional F1
            mat1 = None
            if load_f1 and grp1 is not None:
                if k in grp1:
                    mat1 = np.asarray(grp1[k][:], dtype=float)
                    if mat1.ndim != 2:
                        if strict:
                            raise ValueError(f"{curves_f1_path}/{k} expected 2D (C,R), got {mat1.shape}")
                        mat1 = None
                    elif tuple(mat1.shape) != inferred_shape:
                        if strict:
                            raise ValueError(
                                f"{curves_f1_path}/{k} shape mismatch vs F0: got {mat1.shape}, expected {inferred_shape}"
                            )
                        mat1 = None
                else:
                    if require_f1:
                        if strict:
                            raise KeyError(f"Missing dataset: {curves_f1_path}/{k}")
                        missing_f1.append(int(nid))

            present.append(int(nid))
            y_f0[int(nid)] = mat0
            if mat1 is not None:
                y_f1[int(nid)] = mat1

    out = {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "y_f0": y_f0,
        "by_id": y_f0,          # backward compatibility for code expecting by_id/y_raw
        "curves_path": curves_path,
        "inferred_shape": inferred_shape,
    }
    if load_f1:
        out["y_f1"] = y_f1
        out["missing_f1_ids"] = np.asarray(missing_f1, dtype=int)
        out["curves_f1_path"] = curves_f1_path

    return out

def load_contrast_size_tuning_results_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/contrast_size_tuning/results",
    strict: bool = True,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    by_id = {}

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")
        grp = f[group_path]

        for nid in neuron_ids:
            k = neuron_key(nid)
            if k not in grp:
                missing.append(int(nid))
                continue

            arr = np.asarray(grp[k][:], dtype=float).ravel()

            try:
                if arr.size < 3:
                    raise ValueError(f"Expected [GSF_low, GSF_high, shift], got size={arr.size}")

                gsf_low = float(arr[0])
                gsf_high = float(arr[1])
                shift = float(arr[2])
            except Exception as e:
                if strict:
                    raise ValueError(
                        f"Bad array for {group_path}/{k}, shape={arr.shape}, data={arr}"
                    ) from e
                missing.append(int(nid))
                continue

            present.append(int(nid))
            by_id[int(nid)] = {
                "GSF_low": gsf_low,
                "GSF_high": gsf_high,
                "shift": shift,
            }

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": by_id,
    }