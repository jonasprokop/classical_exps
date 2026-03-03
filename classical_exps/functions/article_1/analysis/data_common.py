############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF
from .data_size_tuning import load_size_tuning_curves_bulk, load_size_tuning_results_bulk
from .data_contrast_response import load_contrast_response_curves_bulk, load_contrast_response_results_bulk
from .data_contrast_size import load_contrast_size_tuning_curves_bulk, load_contrast_size_tuning_results_bulk
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
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

def load_data(
    h5_file: str,
    neuron_ids,
    run: dict,
    *,
    # axes passed in from the single entrypoint
    center_contrasts=None,
    surround_contrasts=None,
    contrasts=None,
    radii=None,
    strict: bool = True,
    print_inventory: bool = True,
):
    """
    Central loader: loads everything needed into `loaded`.
    No parsing of H5 attrs['arguments'].
    Axes are passed in (because you run everything from one entrypoint anyway).
    """
    loaded = {}
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    def need(*flags):
        return any(run.get(f, False) for f in flags)
    
    # ---- size tuning curves ----
    if need("load_size_tuning_curves", "plot_size_tuning_curves", "fit_size_tuning", "use_size_tuning_fits"):
        if radii is None:
            raise ValueError("load_data needs radii for size_tuning_curves loading.")
        loaded["size_tuning_curves"] = load_size_tuning_curves_bulk(
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            strict=strict,
            expected_len=len(radii),
        )
        _fail_if_missing(loaded["size_tuning_curves"], "size_tuning_curves", neuron_ids)
        loaded["size_tuning_curves"]["radii"] = np.asarray(radii, float)


    # # ---- size tuning summary ----
    # if need("size_results_1", "size_results_2", "load_size_results"):
    #     loaded["size_results"] = load_size_tuning_results_bulk(
    #         h5_file=h5_file,
    #         neuron_ids=neuron_ids,
    #         strict=strict,
    #     )
    #     _fail_if_missing(loaded["size_results"], "size_results", neuron_ids)

    # ---- contrast response curves + spread ----
    if need("contrast_response_results_1", "load_contrast_response"):
        if center_contrasts is None or surround_contrasts is None:
            raise ValueError("load_data needs center_contrasts and surround_contrasts for contrast_response loading.")
        expected = (len(surround_contrasts), len(center_contrasts))

        loaded["contrast_response"] = load_contrast_response_curves_bulk(
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            strict=strict,
            expected_shape=expected,
        )
        _fail_if_missing(loaded["contrast_response"], "contrast_response", neuron_ids)

        # attach axes (fit-friendly)
        loaded["contrast_response"]["x"] = np.asarray(center_contrasts, dtype=float)
        loaded["contrast_response"]["s"] = np.asarray(surround_contrasts, dtype=float)
        # backward compat
        loaded["contrast_response"]["center_contrasts"] = loaded["contrast_response"]["x"]
        loaded["contrast_response"]["surround_contrasts"] = loaded["contrast_response"]["s"]

    if need("contrast_response_results_1", "load_contrast_response_results"):
        loaded["contrast_response_results"] = load_contrast_response_results_bulk(
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            strict=strict,
        )
        _fail_if_missing(loaded["contrast_response_results"], "contrast_response_results", neuron_ids)

    # ---- contrast-size tuning curves + shift ----
    if need("contrast_size_tuning_results_1", "load_contrast_size_tuning"):
        if contrasts is None or radii is None:
            raise ValueError("load_data needs contrasts and radii for contrast_size_tuning loading.")
        expected = (len(contrasts), len(radii))

        loaded["contrast_size_tuning"] = load_contrast_size_tuning_curves_bulk(
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            strict=strict,
            expected_shape=expected,
            )
        
        _fail_if_missing(loaded["contrast_size_tuning"], "contrast_size_tuning", neuron_ids)

        loaded["contrast_size_tuning"]["contrasts"] = np.asarray(contrasts, dtype=float)
        loaded["contrast_size_tuning"]["radii"] = np.asarray(radii, dtype=float)

    if need("contrast_size_tuning_results_1", "load_contrast_size_tuning_results"):
        loaded["contrast_size_tuning_results"] = load_contrast_size_tuning_results_bulk(
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            strict=strict,
        )
        _fail_if_missing(loaded["contrast_size_tuning_results"], "contrast_size_tuning_results", neuron_ids)

    if print_inventory:
        print(">> loaded inventory:")
        for k in sorted(loaded.keys()):
            pi = loaded[k].get("present_ids", [])
            mi = loaded[k].get("missing_ids", [])
            print(f"   - {k}: {len(pi)}/{len(neuron_ids)} present, {len(mi)} missing")

    return loaded



def _fail_if_missing(dataset: dict, name: str, neuron_ids):
    missing = dataset.get("missing_ids", None)
    if missing is None:
        raise RuntimeError(f"Loader '{name}' did not return 'missing_ids'.")
    if len(missing) > 0:
        raise RuntimeError(
            f"Loader '{name}' missing {len(missing)}/{len(neuron_ids)} neurons. "
            f"Example missing: {missing[:10].tolist()}"
        )


