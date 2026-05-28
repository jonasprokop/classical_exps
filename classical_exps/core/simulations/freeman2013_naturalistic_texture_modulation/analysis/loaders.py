import h5py
import numpy as np
from classical_exps.core.tools.utils import check_group_exists_error, check_neurons_presence_error


def load_texture_noise_responses(h5_file, neuron_ids):
    """
    Load texture/noise responses.

    Returns:
        all_tex_resp   [n_neuron, n_family, n_sample]
        all_noise_resp [n_neuron, n_family, n_sample]
        family_ids     [n_family]
    """
    group_path = "/texture_noise_response"
    subgroup_tex_path = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    check_neurons_presence_error(
        h5_file=h5_file,
        list_group_path=[subgroup_noise_path, subgroup_tex_path],
        neuron_ids=neuron_ids,
    )

    neuron_ids = np.asarray(neuron_ids, dtype=int)

    all_tex_resp = []
    all_noise_resp = []

    with h5py.File(h5_file, "r") as file:
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        description = subgroup_tex.attrs["description"]
        family_ids = np.asarray(description.split("-"))

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"

            tex = np.asarray(subgroup_tex[neuron][:], dtype=float)
            noise = np.asarray(subgroup_noise[neuron][:], dtype=float)

            if tex.shape != noise.shape:
                raise ValueError(
                    f"Texture/noise shape mismatch for {neuron}: "
                    f"tex={tex.shape}, noise={noise.shape}"
                )

            if tex.ndim != 2:
                raise ValueError(
                    f"Expected [family, sample] for {neuron}, got {tex.shape}"
                )

            all_tex_resp.append(tex)
            all_noise_resp.append(noise)

    all_tex_resp = np.stack(all_tex_resp, axis=0)
    all_noise_resp = np.stack(all_noise_resp, axis=0)

    if all_tex_resp.shape[1] != len(family_ids):
        raise ValueError(
            f"Family ID count does not match response shape: "
            f"{len(family_ids)} vs {all_tex_resp.shape[1]}"
        )

    return all_tex_resp, all_noise_resp, family_ids




def load_selected_lowlevel_family_stats(
    h5_file,
    group_path="/texture_noise_response/lowlevel_family_stats",
):
    """
    Load five low-level family image statistics saved by the experiment branch.
    """
    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(
                f"Missing group: {group_path}. "
                "Run texture_noise_response_experiment again after adding low-level stat saving."
            )

        grp = f[group_path]

        family_ids = np.asarray(grp["family_ids"][:], dtype=int)
        family_features = np.asarray(grp["family_features"][:], dtype=float)

        raw_names = grp["feature_names"][:]
        feature_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in raw_names
        ]

    return family_ids, feature_names, family_features


def load_texture_noise_image_statistics(
    h5_file,
    group_path="/texture_noise_response/image_statistics",
):
    """
    Load low-level image statistics saved during experiment.
    """
    with h5py.File(h5_file, "r") as f:
        if group_path not in f:
            raise KeyError(
                f"Missing group: {group_path}. "
                "Run texture_noise_response_experiment again after adding image-stat saving."
            )

        grp = f[group_path]

        family_ids = np.asarray(grp["family_ids"][:], dtype=int)
        family_features = np.asarray(grp["family_features"][:], dtype=float)

        raw_names = grp["feature_names"][:]
        feature_names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in raw_names
        ]

    return family_ids, feature_names, family_features