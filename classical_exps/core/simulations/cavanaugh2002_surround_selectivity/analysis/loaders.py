import numpy as np
import h5py

from classical_exps.core.tools.utils import (
    check_group_exists_error,
    check_neurons_presence_error,
    get_arguments_from_str,
    get_list_from_str,
)


def neuron_key(neuron_id: int) -> str:
    return f"neuron_{int(neuron_id)}"

def load_orientation_tuning_neuron(
    h5_file: str,
    neuron_id: int,
    base_group: str = "/orientation_tuning",
):
    """
    Load all orientation-tuning outputs for one neuron in a canonical format.

    Expected HDF5 structure
    -----------------------
    {base_group}
        attrs["arguments"]                        -> serialized arguments containing ori_shifts
    {base_group}/curves_center/neuron_X          -> shape (N,)
    {base_group}/curves_surround_only/neuron_X   -> shape (N,)
    {base_group}/curves_center_surround/neuron_X -> shape (N, N)

    Returns
    -------
    data : dict
        {
            "neuron_id": int,
            "neuron_key": str,
            "ori_shifts_rad": np.ndarray of shape (N,),
            "ori_shifts_deg": np.ndarray of shape (N,),
            "zero_idx": int,
            "center_curve": np.ndarray of shape (N,),
            "surround_only_curve": np.ndarray of shape (N,),
            "center_surround_matrix": np.ndarray of shape (N, N),
            "surround_fixed_center": np.ndarray of shape (N,),
            "center_fixed_surround": np.ndarray of shape (N,),
        }

    Notes
    -----
    - `surround_fixed_center` is the row of the full center x surround matrix
      where center orientation shift is closest to 0.
    - `center_fixed_surround` is the column of the full center x surround matrix
      where surround orientation shift is closest to 0.
    """

    grp_center = f"{base_group}/curves_center"
    grp_surround_only = f"{base_group}/curves_surround_only"
    grp_matrix = f"{base_group}/curves_center_surround"

    for grp in (base_group, grp_center, grp_surround_only, grp_matrix):
        check_group_exists_error(h5_file, grp)

    key = neuron_key(neuron_id)
    check_neurons_presence_error(
        h5_file,
        [grp_center, grp_surround_only, grp_matrix],
        [neuron_id],
    )

    with h5py.File(h5_file, "r") as f:
        args_str = f[base_group].attrs["arguments"]
        args = get_arguments_from_str(args_str)
        ori_shifts = np.asarray(get_list_from_str(args["ori_shifts"]), dtype=float)

        center_curve = np.asarray(f[grp_center][key][:], dtype=float)
        surround_only_curve = np.asarray(f[grp_surround_only][key][:], dtype=float)
        center_surround_matrix = np.asarray(f[grp_matrix][key][:], dtype=float)

    n_ori = len(ori_shifts)
    zero_idx = int(np.argmin(np.abs(ori_shifts)))

    if center_curve.shape != (n_ori,):
        raise ValueError(
            f"{grp_center}/{key} has shape {center_curve.shape}, expected {(n_ori,)}"
        )

    if surround_only_curve.shape != (n_ori,):
        raise ValueError(
            f"{grp_surround_only}/{key} has shape {surround_only_curve.shape}, "
            f"expected {(n_ori,)}"
        )

    if center_surround_matrix.shape != (n_ori, n_ori):
        raise ValueError(
            f"{grp_matrix}/{key} has shape {center_surround_matrix.shape}, "
            f"expected {(n_ori, n_ori)}"
        )

    surround_fixed_center = center_surround_matrix[zero_idx, :].copy()
    center_fixed_surround = center_surround_matrix[:, zero_idx].copy()

    return {
        "neuron_id": int(neuron_id),
        "neuron_key": key,
        "ori_shifts_rad": ori_shifts,
        "ori_shifts_deg": np.degrees(ori_shifts),
        "zero_idx": zero_idx,
        "center_curve": center_curve,
        "surround_only_curve": surround_only_curve,
        "center_surround_matrix": center_surround_matrix,
        "surround_fixed_center": surround_fixed_center,
        "center_fixed_surround": center_fixed_surround,
    }


def load_orientation_tuning_bulk(
    h5_file: str,
    neuron_ids,
    base_group: str = "/orientation_tuning",
    strict: bool = True,
):
    """
    Load orientation tuning results for multiple neurons.

    Expected HDF5 structure
    -----------------------
    /orientation_tuning
        attrs["arguments"] -> contains ori_shifts
    /orientation_tuning/curves_center/neuron_X
    /orientation_tuning/curves_surround_only/neuron_X
    /orientation_tuning/curves_center_surround/neuron_X

    Returns
    -------
    dict with keys

    present_ids
    missing_ids
    by_id
    ori_shifts_rad
    ori_shifts_deg
    zero_idx
    """

    neuron_ids = np.asarray(neuron_ids, dtype=int)

    grp_center = f"{base_group}/curves_center"
    grp_surround = f"{base_group}/curves_surround_only"
    grp_matrix = f"{base_group}/curves_center_surround"

    present, missing = [], []
    by_id = {}

    with h5py.File(h5_file, "r") as f:

        if base_group not in f:
            raise KeyError(f"Missing group: {base_group}")

        args_str = f[base_group].attrs["arguments"]
        args = get_arguments_from_str(args_str)
        ori_shifts = np.asarray(get_list_from_str(args["ori_shifts"]), dtype=float)

        zero_idx = int(np.argmin(np.abs(ori_shifts)))
        n_ori = len(ori_shifts)

        for grp in (grp_center, grp_surround, grp_matrix):
            if grp not in f:
                raise KeyError(f"Missing group: {grp}")

        grpC = f[grp_center]
        grpS = f[grp_surround]
        grpM = f[grp_matrix]

        for nid in neuron_ids:

            k = neuron_key(nid)

            if k not in grpC or k not in grpS or k not in grpM:
                missing.append(int(nid))
                continue

            center_curve = np.asarray(grpC[k][:], dtype=float)
            surround_curve = np.asarray(grpS[k][:], dtype=float)
            center_surround_matrix = np.asarray(grpM[k][:], dtype=float)

            if center_curve.shape != (n_ori,):
                if strict:
                    raise ValueError(
                        f"{grp_center}/{k} shape {center_curve.shape}, expected {(n_ori,)}"
                    )
                missing.append(int(nid))
                continue

            if surround_curve.shape != (n_ori,):
                if strict:
                    raise ValueError(
                        f"{grp_surround}/{k} shape {surround_curve.shape}, expected {(n_ori,)}"
                    )
                missing.append(int(nid))
                continue

            if center_surround_matrix.shape != (n_ori, n_ori):
                if strict:
                    raise ValueError(
                        f"{grp_matrix}/{k} shape {center_surround_matrix.shape}, expected {(n_ori, n_ori)}"
                    )
                missing.append(int(nid))
                continue

            surround_fixed_center = center_surround_matrix[zero_idx, :].copy()
            center_fixed_surround = center_surround_matrix[:, zero_idx].copy()

            present.append(int(nid))

            by_id[int(nid)] = {
                "center": center_curve,
                "surround_only": surround_curve,
                "matrix": center_surround_matrix,
                "center_surround_matrix": center_surround_matrix,
                "surround_fixed_center": surround_fixed_center,
                "center_fixed_surround": center_fixed_surround,
            }

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "by_id": by_id,
        "ori_shifts_rad": ori_shifts,
        "ori_shifts_deg": np.degrees(ori_shifts),
        "zero_idx": zero_idx,
        "base_group": base_group,
    }

def load_ccss_curves_bulk(
    h5_file: str,
    neuron_ids,
    group_path: str = "/center_contrast_surround_suppression",
    strict: bool = True,
):
    """
    Load article-faithful CCSS contrast-family data and its saved config.

    Expected HDF5 structure
    -----------------------
    /center_contrast_surround_suppression
        attrs:
            pixel_min
            pixel_max
            experiment_version
            stimulus_extent_units
            center_orientation_mode
            surround_orientation_mode_same
            surround_orientation_mode_orthogonal
            center_sf_mode
            surround_sf_mode
            center_radius_source
            surround_radius_source

        /config
            center_contrasts_ccss   -> shape (C,)
            surround_contrasts_ccss -> shape (S,)
            phases                  -> shape (P,)
            size                    -> scalar or shape (2,)
            img_res                 -> shape (2,)

        /results_same
            neuron_X                -> array of shape (S, C)

        /results_orthogonal
            neuron_X                -> array of shape (S, C)

    Convention
    ----------
    responses[s, c] =
        response for surround contrast index s
        and center contrast index c

    Returns
    -------
    {
        "present_ids": np.ndarray,
        "missing_ids": np.ndarray,
        "responses_same_by_id": dict[int, np.ndarray],
        "responses_orth_by_id": dict[int, np.ndarray],
        "by_id": dict[int, dict],
        "center_contrasts": np.ndarray,
        "surround_contrasts": np.ndarray,
        "config": dict,
        "group_path": str,
        "config_path": str,
        "results_same_path": str,
        "results_orth_path": str,
    }
    """
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    # Accept either base group or subgroup paths, because apparently we enjoy chaos
    for suffix in ("/results", "/results_same", "/results_orthogonal", "/config"):
        if group_path.endswith(suffix):
            group_path = group_path[:-len(suffix)]
            break

    config_path = f"{group_path}/config"
    results_same_path = f"{group_path}/results_same"
    results_orth_path = f"{group_path}/results_orthogonal"

    present = []
    missing = []
    responses_same_by_id = {}
    responses_orth_by_id = {}
    by_id = {}

    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"Missing group: {group_path}")
        if config_path not in f:
            raise KeyError(f"Missing config group: {config_path}")
        if results_same_path not in f:
            raise KeyError(f"Missing results group: {results_same_path}")
        if results_orth_path not in f:
            raise KeyError(f"Missing results group: {results_orth_path}")

        grp = f[group_path]
        cfg = f[config_path]
        results_same_grp = f[results_same_path]
        results_orth_grp = f[results_orth_path]

        required_cfg_keys = [
            "center_contrasts_ccss",
            "surround_contrasts_ccss",
            "phases",
            "size",
            "img_res",
        ]
        for key in required_cfg_keys:
            if key not in cfg:
                raise KeyError(f"Missing config dataset: {config_path}/{key}")

        center_contrasts = np.asarray(cfg["center_contrasts_ccss"][:], dtype=float)
        surround_contrasts = np.asarray(cfg["surround_contrasts_ccss"][:], dtype=float)

        config = {
            "center_contrasts_ccss": center_contrasts,
            "surround_contrasts_ccss": surround_contrasts,
            "phases": np.asarray(cfg["phases"][:], dtype=float),
            "size": np.asarray(cfg["size"][()], dtype=float),
            "img_res": np.asarray(cfg["img_res"][:], dtype=int),
        }

        for key in [
            "pixel_min",
            "pixel_max",
            "experiment_version",
            "stimulus_extent_units",
            "center_orientation_mode",
            "surround_orientation_mode_same",
            "surround_orientation_mode_orthogonal",
            "center_sf_mode",
            "surround_sf_mode",
            "center_radius_source",
            "surround_radius_source",
        ]:
            if key in grp.attrs:
                config[key] = grp.attrs[key]

        n_surround = len(surround_contrasts)
        n_center = len(center_contrasts)
        expected_shape = (n_surround, n_center)

        for nid in neuron_ids:
            k = neuron_key(nid)

            if k not in results_same_grp or k not in results_orth_grp:
                missing.append(int(nid))
                continue

            arr_same = np.asarray(results_same_grp[k][:], dtype=float)
            arr_orth = np.asarray(results_orth_grp[k][:], dtype=float)

            if arr_same.shape != expected_shape or arr_orth.shape != expected_shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch for {k}: "
                        f"same={arr_same.shape}, orth={arr_orth.shape}, "
                        f"expected={expected_shape}"
                    )
                missing.append(int(nid))
                continue

            present.append(int(nid))

            responses_same_by_id[int(nid)] = arr_same.copy()
            responses_orth_by_id[int(nid)] = arr_orth.copy()

            by_id[int(nid)] = {
                "responses_same": arr_same.copy(),
                "responses_orth": arr_orth.copy(),
                "same": arr_same.copy(),
                "orthogonal": arr_orth.copy(),
            }

    return {
        "present_ids": np.asarray(present, dtype=int),
        "missing_ids": np.asarray(missing, dtype=int),
        "responses_same_by_id": responses_same_by_id,
        "responses_orth_by_id": responses_orth_by_id,
        "by_id": by_id,
        "center_contrasts": center_contrasts,
        "surround_contrasts": surround_contrasts,
        "config": config,
        "group_path": group_path,
        "config_path": config_path,
        "results_same_path": results_same_path,
        "results_orth_path": results_orth_path,
    }


    