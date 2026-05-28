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

def load_contrast_response_curves_bulk(
    h5_file: str,
    neuron_ids,
    curves_path: str = "/contrast_response/curves",
    strict: bool = False,
    expected_shape=None,  # (n_surround, n_center) 
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing, invalid = [], [], []
    y_raw = {}

    with h5py.File(h5_file, "r") as f:
        if curves_path not in f:
            raise KeyError(f"Missing group: {curves_path}")
        grp = f[curves_path]

        for nid in neuron_ids:
            nid_i = int(nid)
            k = neuron_key(nid_i)

            # missing dataset
            if k not in grp:
                missing.append(nid_i)
                y_raw[nid_i] = None
                continue

            mat = np.asarray(grp[k][:], dtype=float)

            # must be 2D (S,C)
            if mat.ndim != 2:
                msg = f"{curves_path}/{k} expected 2D (S,C), got {mat.shape}"
                if strict:
                    raise ValueError(msg)
                invalid.append(nid_i)
                y_raw[nid_i] = None
                continue

            # shape check
            if expected_shape is not None and tuple(mat.shape) != tuple(expected_shape):
                msg = f"{curves_path}/{k} shape mismatch: got {mat.shape}, expected {expected_shape}"
                if strict:
                    raise ValueError(msg)
                invalid.append(nid_i)
                y_raw[nid_i] = None
                continue

            present.append(nid_i)
            y_raw[nid_i] = mat

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "invalid_ids": np.asarray(invalid, dtype=int),
        "y_raw": y_raw,
        # backward compat
        "by_id": y_raw,
    }




def load_contrast_response_results_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/contrast_response/results",
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

            arr = np.asarray(grp[k][:], dtype=float)
            try:
                maxmin = float(arr[0])
                mean_std = float(arr[1])
            except Exception as e:
                if strict:
                    raise ValueError(f"Bad array for {group_path}/{k}, shape={arr.shape}") from e
                missing.append(int(nid))
                continue

            present.append(int(nid))
            by_id[int(nid)] = {"maxmin": maxmin, "mean_std": mean_std}

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": by_id,
    }

