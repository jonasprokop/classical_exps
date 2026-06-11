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


def load_size_tuning_results_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/size_tuning/results",
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

            try:
                arr = np.asarray(grp[k][:], dtype=float).reshape(-1)

                if arr.size == 6:
                    GSF, surr_ext, AMRF, SI, Ropt, Rsupp = map(float, arr)
                elif arr.size == 4:
                    GSF, surr_ext, AMRF, SI = map(float, arr)
                    Ropt = float("nan")
                    Rsupp = float("nan")
                else:
                    raise ValueError(f"Unexpected result length {arr.size}")

            except Exception as e:
                if strict:
                    raise ValueError(
                        f"Bad array for {group_path}/{k}, shape={arr.shape}"
                    ) from e
                missing.append(int(nid))
                continue

            present.append(int(nid))
            by_id[int(nid)] = {
                "GSF": GSF,
                "surround_extent": surr_ext,
                "AMRF": AMRF,
                "SI": SI,
                "Ropt": Ropt,
                "Rsupp": Rsupp,
            }

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": by_id,
    }

def load_size_tuning_curves_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/size_tuning/curves",
    strict: bool = True,
    expected_len: int = None,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    present, missing = [], []
    circular_by_id, annular_by_id = {}, {}
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
            if arr.ndim != 2 or arr.shape[0] != 2:
                if strict:
                    raise ValueError(f"{group_path}/{k} expected shape (2,R), got {arr.shape}")
                missing.append(int(nid))
                continue

            if expected_len is not None and arr.shape[1] != expected_len:
                if strict:
                    raise ValueError(
                        f"{group_path}/{k} length mismatch: got R={arr.shape[1]}, expected {expected_len}"
                    )
                missing.append(int(nid))
                continue

            present.append(int(nid))
            c = arr[0].copy()
            a = arr[1].copy()
            circular_by_id[int(nid)] = c
            annular_by_id[int(nid)]  = a
            by_id[int(nid)] = {"circular": c, "annular": a}

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "circular_by_id": circular_by_id,
        "annular_by_id": annular_by_id,   
        "by_id": by_id,                   
        "group_path": group_path,
    }