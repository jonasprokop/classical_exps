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
from .panels import plot_ccss_controls_panel_raw
## Data storage
import h5py
from scipy.optimize import curve_fit
import time
import cv2
import matplotlib.pyplot as plt
from classical_exps.core.tools.metadata import save_experiment_config


##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################


def center_contrast_surround_suppression_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    center_contrasts_ccss=np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5]),
    surround_contrasts_ccss=np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5]),
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
    img_res=[93, 93],
    save_panel=True,
):
    """
    Article-faithful center/surround contrast-family experiment
    inspired by Cavanaugh et al. (2002).

    For each neuron:
        - center orientation = preferred orientation
        - surround orientation (same condition) = preferred orientation
        - surround orientation (orthogonal condition) = preferred orientation + pi/2
        - center SF = preferred SF
        - surround SF = preferred SF
        - center radius = GSF
        - surround inner radius = max(AMRF, GSF)

    The contrasts of center and surround are varied independently.

    Saved dataset per neuron:
        shape = (n_surround_contrasts, n_center_contrasts)

    Axis convention:
        axis 0 -> surround contrast
        axis 1 -> center contrast

    Saved groups:
        /center_contrast_surround_suppression/results_same
        /center_contrast_surround_suppression/results_orthogonal
    """
    print(" > Center contrast surround suppression experiment")

    # Required groups
    group_path_ff_params = "/full_field_params"
    group_path_pos = "/preferred_pos"
    group_path_st_results = "/size_tuning/results"

    # Groups to fill
    group_path = "/center_contrast_surround_suppression"
    subgroup_path_same = group_path + "/results_same"
    subgroup_path_orth = group_path + "/results_orthogonal"

    if overwrite:
        clear_group(h5_file, group_path)

    args_str = (
        f"center_contrasts_ccss={center_contrasts_ccss}/"
        f"surround_contrasts_ccss={surround_contrasts_ccss}/"
        f"phases={phases}/"
        f"pixel_min={pixel_min}/"
        f"pixel_max={pixel_max}/"
        f"size={size}/"
        f"img_res={img_res}"
    )

    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_same, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_orth, group_args_str=args_str)

    save_experiment_config(
        h5_file=h5_file,
        group_path=group_path,
        args_str=args_str,
        overwrite=overwrite,
        array_fields={
            "center_contrasts_ccss": center_contrasts_ccss,
            "surround_contrasts_ccss": surround_contrasts_ccss,
            "phases": phases,
            "size": size,
            "img_res": img_res,
        },
        attr_fields={
            "pixel_min": float(pixel_min),
            "pixel_max": float(pixel_max),
            "experiment_version": "ccss_v3_same_and_orthogonal",
            "stimulus_extent_units": "radii",
            "center_orientation_mode": "preferred",
            "surround_orientation_mode_same": "preferred",
            "surround_orientation_mode_orthogonal": "preferred+pi/2",
            "center_sf_mode": "preferred",
            "surround_sf_mode": "preferred",
            "center_radius_source": "GSF",
            "surround_radius_source": "max(AMRF,GSF)",
        },
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(size, (int, float)):
        size = [size] * 2

    with h5py.File(h5_file, "a") as f:
        subgroup_same = f[subgroup_path_same]
        subgroup_orth = f[subgroup_path_orth]

        for neuron_id in tqdm(neuron_ids):
            neuron = f"neuron_{neuron_id}"

            # Skip only if BOTH are already present
            if neuron in subgroup_same and neuron in subgroup_orth:
                continue

            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            with torch.no_grad():
                # Calcuate the response to a gray stimulus for this neuron (used for normalization)
                gray_stim = torch.ones((1, 93, 93), device=device) * ((pixel_min + pixel_max) / 2)
                gray_resp = float(single_model(gray_stim).item())

                try:
                    preferred_ori = f[group_path_ff_params][neuron][:][0]
                    preferred_sf = f[group_path_ff_params][neuron][:][1]
                    x_pix = f[group_path_pos][neuron][:][0]
                    y_pix = f[group_path_pos][neuron][:][1]
                    GSF = f[group_path_st_results][neuron][0]
                    AMRF = f[group_path_st_results][neuron][2]
                except Exception as e:
                    print(f"Error retrieving parameters for {neuron}: {e}")
                    continue

                if (
                    GSF is None or AMRF is None
                    or not np.isfinite(GSF) or not np.isfinite(AMRF)
                    or GSF <= 0 or AMRF <= 0
                ):
                    print(f"Invalid GSF or AMRF for {neuron} (GSF={GSF}, AMRF={AMRF}). Skipping.")
                    continue

                surround_radius = max(AMRF, GSF)
                center_radius = GSF

                response_matrix_same = np.zeros(
                    (len(surround_contrasts_ccss), len(center_contrasts_ccss)),
                    dtype=float,
                )
                response_matrix_orth = np.zeros(
                    (len(surround_contrasts_ccss), len(center_contrasts_ccss)),
                    dtype=float,
                )

                panel_rows_same = []
                panel_rows_ortho = []
                panel_phase_index = len(phases) // 2


                for s_idx, surround_contrast in enumerate(surround_contrasts_ccss):
                    row_imgs_same = {
                        "surround_contrast": float(surround_contrast),
                        "stimuli": [],
                    }
                    row_imgs_ortho = {
                        "surround_contrast": float(surround_contrast),
                        "stimuli": [],
                    }

                    for c_idx, center_contrast in enumerate(center_contrasts_ccss):
                        resp_sum_same = 0.0
                        resp_sum_orth = 0.0

                        panel_img_same = None
                        panel_img_ortho = None

                        for p_i, phase in enumerate(phases):
                            # --------------------------------------------------
                            # SAME-ORIENTATION SURROUND (numeric + panel)
                            # --------------------------------------------------
                            stimulus_same, img_raw_same = get_center_surround_stimulus(
                                center_radius=center_radius,
                                center_ori=preferred_ori,
                                center_sf=preferred_sf,
                                center_phase=phase,
                                center_contrast=center_contrast,
                                surround_radius=surround_radius,
                                surround_ori=preferred_ori,
                                surround_sf=preferred_sf,
                                surround_phase=phase,
                                surround_contrast=surround_contrast,
                                x_pix=x_pix,
                                y_pix=y_pix,
                                pixel_min=pixel_min,
                                pixel_max=pixel_max,
                                size=size,
                                img_res=img_res,
                                device=device,
                            )

                            resp_sum_same += single_model(stimulus_same).item()

                            # --------------------------------------------------
                            # ORTHOGONAL SURROUND (numeric + panel)
                            # --------------------------------------------------
                            stimulus_orth, img_raw_orth = get_center_surround_stimulus(
                                center_radius=center_radius,
                                center_ori=preferred_ori,
                                center_sf=preferred_sf,
                                center_phase=phase,
                                center_contrast=center_contrast,
                                surround_radius=surround_radius,
                                surround_ori=preferred_ori + np.pi / 2,
                                surround_sf=preferred_sf,
                                surround_phase=phase,
                                surround_contrast=surround_contrast,
                                x_pix=x_pix,
                                y_pix=y_pix,
                                pixel_min=pixel_min,
                                pixel_max=pixel_max,
                                size=size,
                                img_res=img_res,
                                device=device,
                            )

                            resp_sum_orth += single_model(stimulus_orth).item()

                            if save_panel and p_i == panel_phase_index:
                                panel_img_same = img_raw_same.detach().cpu().numpy()
                                panel_img_ortho = img_raw_orth.detach().cpu().numpy()

                        response_matrix_same[s_idx, c_idx] = resp_sum_same / float(len(phases))
                        response_matrix_orth[s_idx, c_idx] = resp_sum_orth / float(len(phases))

                    

                        if save_panel:
                            row_imgs_same["stimuli"].append({
                                "center_contrast": float(center_contrast),
                                "img": panel_img_same,
                            })
                            row_imgs_ortho["stimuli"].append({
                                "center_contrast": float(center_contrast),
                                "img": panel_img_ortho,
                            })

                    if save_panel:
                        panel_rows_same.append(row_imgs_same)
                        panel_rows_ortho.append(row_imgs_ortho)

                response_matrix_same = response_matrix_same - gray_resp
                response_matrix_orth = response_matrix_orth - gray_resp

            # remove old neuron dataset if only one subgroup existed before a partial rerun
            if neuron in subgroup_same:
                del subgroup_same[neuron]
            if neuron in subgroup_orth:
                del subgroup_orth[neuron]

            subgroup_same.create_dataset(name=neuron, data=response_matrix_same)
            subgroup_orth.create_dataset(name=neuron, data=response_matrix_orth)

            if save_panel:
                out_dir = "/project/results/selectivity_and_spatial_distribution/stimuli/ccss_panels/"
                os.makedirs(out_dir, exist_ok=True)

                out_path_same = os.path.join(out_dir, f"{neuron}_ccss_panel_same.png")
                plot_ccss_controls_panel_raw(
                    rows=panel_rows_same,
                    out_path=out_path_same,
                    title=f"{neuron} CCSS same | ori={preferred_ori:.2f} sf={preferred_sf:.2f}"
                )

                out_path_ortho = os.path.join(out_dir, f"{neuron}_ccss_panel_ortho.png")
                plot_ccss_controls_panel_raw(
                    rows=panel_rows_ortho,
                    out_path=out_path_ortho,
                    title=f"{neuron} CCSS orthogonal | ori={preferred_ori:.2f} sf={preferred_sf:.2f}"
                )