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

from matplotlib import colors


def load_black_white_maps_bulk(
    h5_file,
    neuron_ids,
    group_path="/black_white_preference/position_response_img",
    strict=True,
):
    present = []
    missing = []
    maps = []

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")
        grp = f[group_path]

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"
            if neuron not in grp:
                missing.append(int(neuron_id))
                continue

            arr = np.asarray(grp[neuron][:], dtype=float)
            if arr.ndim != 3 or arr.shape[0] != 2:
                if strict:
                    raise ValueError(f"Bad shape for {group_path}/{neuron}: {arr.shape}")
                missing.append(int(neuron_id))
                continue

            present.append(int(neuron_id))
            maps.append(arr)

    if len(maps) == 0:
        maps = np.empty((0, 2, 0, 0), dtype=float)
    else:
        maps = np.stack(maps, axis=0)

    return {
        "maps": maps,
        "neuron_ids": np.asarray(present, dtype=int),
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
    }


def load_black_white_results_bulk(
    h5_file,
    neuron_ids,
    group_path="/black_white_preference/results",
    strict=True,
):
    present = []
    missing = []

    energy_b = []
    energy_w = []
    log_energy_wb = []

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")

        grp = f[group_path]

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"

            if neuron not in grp:
                missing.append(int(neuron_id))
                continue

            arr = np.asarray(grp[neuron][:], dtype=float).reshape(-1)

            if arr.shape[0] < 3:
                if strict:
                    raise ValueError(f"Bad shape for {group_path}/{neuron}: {arr.shape}")
                missing.append(int(neuron_id))
                continue

            present.append(int(neuron_id))
            energy_b.append(float(arr[0]))
            energy_w.append(float(arr[1]))
            log_energy_wb.append(float(arr[2]))

    return {
        "neuron_ids": np.asarray(present, dtype=int),
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "energy_b": np.asarray(energy_b, dtype=float),
        "energy_w": np.asarray(energy_w, dtype=float),
        "log_energy_wb": np.asarray(log_energy_wb, dtype=float),
    }

def load_model_black_white_histogram_dataset(
    loaded_results,
    *,
    bin_edges,
    label="Model",
    hatch=None,
):
    """
    Convert loaded model H5 results into a standard histogram dataset.

    Input:
        loaded_results from load_black_white_results_bulk()

    Uses:
        loaded_results["log_energy_wb"] = log10(E_white / E_black)

    Output schema:
        {
            label, source,
            bin_edges, counts, percent, n,
            hatch, alpha,
            values,
        }
    """

    if "log_energy_wb" not in loaded_results:
        raise KeyError("loaded_results must contain 'log_energy_wb'.")

    values = np.asarray(loaded_results["log_energy_wb"], dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite model log_energy_wb values.")

    bin_edges = np.asarray(bin_edges, dtype=float)

    if len(bin_edges) < 2:
        raise ValueError("bin_edges must contain at least two values.")

    if not np.any(np.isclose(bin_edges, 0.0)):
        raise ValueError("bin_edges must contain 0 as an exact edge.")

    counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts.astype(float)

    return {
        "label": label,
        "source": "model",
        "bin_edges": bin_edges,
        "counts": counts,
        "percent": counts * 100.0 / len(values),
        "n": int(len(values)),
        "hatch": hatch,
        "alpha": 1.0,
        "values": values,
        "x_label": "log10(E_white / E_black)",
    }


def load_scraped_black_white_histogram_dataset(
    scraped_config,
    *,
    histogram_key="histogram",
    layers=None,
    label="Yeh experiment",
    hatch="///",
):
    """
    Convert scraped Yeh histogram config into a standard histogram dataset.

    By default sums all layers:
        L2/3 + L4A/B + L4C + L5/6

    Input config:
        scraped_config["histogram"]["bin_edges"]
        scraped_config["histogram"]["layers"][layer]["counts"]

    Output schema matches load_model_black_white_histogram_dataset().
    """

    if histogram_key not in scraped_config:
        raise KeyError(
            f"Missing histogram key {histogram_key!r}. "
            f"Available keys: {list(scraped_config.keys())}"
        )

    hist = scraped_config[histogram_key]
    all_layers = hist["layers"]

    if layers is None:
        layers = list(all_layers.keys())

    bin_edges = np.asarray(hist["bin_edges"], dtype=float)

    if len(bin_edges) < 2:
        raise ValueError("histogram bin_edges must contain at least two values.")

    if not np.any(np.isclose(bin_edges, 0.0)):
        raise ValueError("histogram bin_edges must contain 0 as an exact edge.")

    counts_sum = np.zeros(len(bin_edges) - 1, dtype=float)
    n_reported = 0

    for layer_name in layers:
        if layer_name not in all_layers:
            raise KeyError(
                f"Missing histogram layer {layer_name!r}. "
                f"Available layers: {list(all_layers.keys())}"
            )

        layer = all_layers[layer_name]
        counts = np.asarray(layer["counts"], dtype=float)

        if len(counts) != len(counts_sum):
            raise ValueError(
                f"Bad counts length for {layer_name}: "
                f"{len(counts)} vs expected {len(counts_sum)}"
            )

        counts_sum += counts
        n_reported += int(layer.get("n_cells_reported", int(np.sum(counts))))

    total = float(np.sum(counts_sum))

    if total <= 0:
        raise ValueError("Summed scraped histogram has zero total count.")

    return {
        "label": label,
        "source": "experiment",
        "layers": tuple(layers),
        "bin_edges": bin_edges,
        "counts": counts_sum,
        "percent": counts_sum * 100.0 / total,
        "n": int(round(total)),
        "n_reported": n_reported,
        "hatch": hatch,
        "alpha": 1.0,
        "values": None,
        "x_label": hist.get("x_label", "log(SNR_white / SNR_black)"),
    }


def load_model_black_white_scatter_dataset(
    loaded_results,
    *,
    label="Model",
):
    """
    Convert loaded model H5 results into a standard scatter dataset.

    Input:
        loaded_results from load_black_white_results_bulk()

    Uses:
        x = white = loaded_results["energy_w"]
        y = black = loaded_results["energy_b"]

    Output schema:
        {
            label, source,
            white, black,
            white_label, black_label,
            n,
        }
    """

    required = {"energy_w", "energy_b"}

    missing = required - set(loaded_results.keys())
    if missing:
        raise KeyError(f"loaded_results missing required keys: {sorted(missing)}")

    white = np.asarray(loaded_results["energy_w"], dtype=float)
    black = np.asarray(loaded_results["energy_b"], dtype=float)

    mask = np.isfinite(white) & np.isfinite(black)
    white = white[mask]
    black = black[mask]

    if len(white) == 0:
        raise ValueError("No finite model scatter values.")

    return {
        "label": label,
        "source": "model",
        "white": white,
        "black": black,
        "white_label": "E_white",
        "black_label": "E_black",
        "n": int(len(white)),
    }


def load_scraped_black_white_scatter_dataset(
    scraped_config,
    *,
    scatter_key="scatter",
    layers=None,
    label="Yeh experiment",
):
    """
    Convert scraped Yeh scatter config into a standard scatter dataset.

    By default concatenates all scatter layers available in the config:
        usually L2/3 + L4C

    Input config:
        scraped_config["scatter"]["layers"][layer]["SNR_white"]
        scraped_config["scatter"]["layers"][layer]["SNR_black"]

    Output schema matches load_model_black_white_scatter_dataset().
    """

    if scatter_key not in scraped_config:
        raise KeyError(
            f"Missing scatter key {scatter_key!r}. "
            f"Available keys: {list(scraped_config.keys())}"
        )

    scatter = scraped_config[scatter_key]
    all_layers = scatter["layers"]

    if layers is None:
        layers = list(all_layers.keys())

    white_all = []
    black_all = []
    layer_labels = []

    for layer_name in layers:
        if layer_name not in all_layers:
            raise KeyError(
                f"Missing scatter layer {layer_name!r}. "
                f"Available layers: {list(all_layers.keys())}"
            )

        layer = all_layers[layer_name]

        white = np.asarray(layer["SNR_white"], dtype=float)
        black = np.asarray(layer["SNR_black"], dtype=float)

        mask = np.isfinite(white) & np.isfinite(black)
        white = white[mask]
        black = black[mask]

        white_all.append(white)
        black_all.append(black)
        layer_labels.extend([layer_name] * len(white))

    if len(white_all) == 0:
        raise ValueError("No scraped scatter layers selected.")

    white_all = np.concatenate(white_all)
    black_all = np.concatenate(black_all)

    if len(white_all) == 0:
        raise ValueError("No finite scraped scatter values.")

    return {
        "label": label,
        "source": "experiment",
        "layers": tuple(layers),
        "white": white_all,
        "black": black_all,
        "white_label": scatter.get("x_label", "SNR_white"),
        "black_label": scatter.get("y_label", "SNR_black"),
        "axis_limits": scatter.get("axis_limits", {}),
        "layer_labels": np.asarray(layer_labels, dtype=object),
        "n": int(len(white_all)),
    }