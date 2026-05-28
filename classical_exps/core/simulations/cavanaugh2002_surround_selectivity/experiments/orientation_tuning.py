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
from .panels import plot_orientation_controls_panel_raw, six_panel_indices
import time
import cv2
import matplotlib.pyplot as plt

##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################

def get_orientation_tuning_curves_all_phases(
    single_model,
    preferred_ori,
    preferred_sf,
    x_pix,
    y_pix,
    GSF,
    AMRF,
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts=np.linspace(-np.pi, np.pi, 9, endpoint=False),
    contrast=1,
    pixel_min=-1.7876,
    pixel_max=2.1919,
    size=2.67,
    img_res=[93, 93],
    device=None,
    stimulus_type="experiment",   # "experiment" or "control"
    do_surround_fixed_center=True,
    center_shift=0,
    surround_shift=0,
    save_panel=False,
    panel_dir="/project/results/selectivity_and_spatial_distribution/stimuli/orientation_panels/",
    panel_tag="",
    panel_phase_index=None,  # default -> phase 0
    panel_n=6,
    panel_ori_indices=None,  # if None, defaults to six_panel_indices(len(ori_shifts), panel_n)
    neuron="",
    gray_resp=None,
    subtract_gray_baseline=True,
):
    """
    Compute orientation tuning curves across phases.

    Has built in logic to handle both control and experimental conditions.

    stimulus_type = "control" or "experiment"
        control    = only center or only surround
        experiment = center and surround together

    do_surround_fixed_center = True or False
        if True, the center is fixed and the surround changes
        if False, the surround is fixed and the center changes

    Responses can optionally be baseline-subtracted using a gray stimulus
    response provided via gray_resp.
    """

    if stimulus_type not in {"experiment", "control"}:
        raise ValueError(
            f"Invalid stimulus_type={stimulus_type!r}. Expected 'experiment' or 'control'."
        )

    if subtract_gray_baseline and gray_resp is None:
        raise ValueError(
            "subtract_gray_baseline=True requires gray_resp to be provided."
        )

    if gray_resp is not None:
        gray_resp = float(gray_resp)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)
    single_model.eval()

    if isinstance(size, (int, float)):
        size = [size] * 2

    if GSF > AMRF:
        AMRF = GSF

    capture = bool(save_panel)
    if capture:
        if panel_ori_indices is None:
            panel_idx = six_panel_indices(len(ori_shifts), panel_n)
        else:
            panel_idx = np.asarray(panel_ori_indices, dtype=int)

        idx_to_slot = {int(k): s for s, k in enumerate(panel_idx)}

        if panel_phase_index is None:
            panel_phase_index = 0
        panel_phase_index = int(np.clip(panel_phase_index, 0, len(phases) - 1))

        images6 = [None] * len(panel_idx)
        labels6 = [f"Δθ={ori_shifts[i]:+.2f} rad" for i in panel_idx]

    all_orientation_curves = torch.zeros((len(phases), len(ori_shifts)), dtype=torch.float32)

    for num_phase, phase in enumerate(phases):
        for num_ori, ori_shift in enumerate(ori_shifts):

            # ---------------------------------------------------------
            # CASE A: CONTROL / CENTER ONLY
            # ---------------------------------------------------------
            if stimulus_type == "control" and do_surround_fixed_center is False:

                center_ori = preferred_ori + ori_shift

                stimulus = torch.Tensor(imagen.SineGrating(
                    mask_shape=imagen.Disk(smoothing=0.0, size=GSF * 2.0),
                    orientation=center_ori,
                    frequency=preferred_sf,
                    phase=phase,
                    bounds=BoundingBox(points=((-size[1] / 2, -size[0] / 2), (size[1] / 2, size[0] / 2))),
                    offset=-1,
                    scale=2,
                    xdensity=img_res[1] / size[1],
                    ydensity=img_res[0] / size[0],
                    x=get_offset_in_degr(x_pix, img_res[1], size[1]),
                    y=-get_offset_in_degr(y_pix, img_res[0], size[0]),
                )())

                center_mask = torch.Tensor(imagen.Disk(
                    smoothing=0.0,
                    size=GSF * 2.0,
                    bounds=BoundingBox(points=((-size[1] / 2, -size[0] / 2), (size[1] / 2, size[0] / 2))),
                    xdensity=img_res[1] / size[1],
                    ydensity=img_res[0] / size[0],
                    x=get_offset_in_degr(x_pix, img_res[1], size[1]),
                    y=-get_offset_in_degr(y_pix, img_res[0], size[0]),
                )())

                stimulus = stimulus * center_mask
                stimulus *= contrast
                image_for_panel = stimulus.detach().cpu().numpy()
                stimulus = rescale(stimulus, -1, 1, pixel_min, pixel_max).reshape(1, *img_res).to(device)

            # ---------------------------------------------------------
            # CASE B: CONTROL / SURROUND ONLY
            # ---------------------------------------------------------
            elif stimulus_type == "control" and do_surround_fixed_center is True:

                surround_ori = preferred_ori + ori_shift

                grating_background = torch.Tensor(imagen.SineGrating(
                    orientation=surround_ori,
                    frequency=preferred_sf,
                    phase=phase,
                    bounds=BoundingBox(points=((-size[1] / 2, -size[0] / 2), (size[1] / 2, size[0] / 2))),
                    offset=-1,
                    scale=2,
                    xdensity=img_res[1] / size[1],
                    ydensity=img_res[0] / size[0],
                    x=get_offset_in_degr(x_pix, img_res[1], size[1]),
                    y=-get_offset_in_degr(y_pix, img_res[0], size[0]),
                )())

                inner_edge = (torch.Tensor(imagen.Disk(
                    smoothing=0.0,
                    size=AMRF * 2.0,
                    bounds=BoundingBox(points=((-size[1] / 2, -size[0] / 2), (size[1] / 2, size[0] / 2))),
                    xdensity=img_res[1] / size[1],
                    ydensity=img_res[0] / size[0],
                    x=get_offset_in_degr(x_pix, img_res[1], size[1]),
                    y=-get_offset_in_degr(y_pix, img_res[0], size[0]),
                )()) * -1) + 1

                stimulus = grating_background * inner_edge
                stimulus *= contrast
                image_for_panel = stimulus.detach().cpu().numpy()
                stimulus = rescale(stimulus, -1, 1, pixel_min, pixel_max).reshape(1, *img_res).to(device)

            # ---------------------------------------------------------
            # CASE C: EXPERIMENT / CENTER FIXED, SURROUND MOVES
            # ---------------------------------------------------------
            elif stimulus_type == "experiment" and do_surround_fixed_center is True:

                surround_ori = preferred_ori + ori_shift
                center_ori = preferred_ori if center_shift is None else preferred_ori + center_shift

                stimulus, image_denormalised = get_center_surround_stimulus(
                    center_radius=GSF,
                    center_ori=center_ori,
                    center_sf=preferred_sf,
                    center_phase=phase,
                    center_contrast=contrast,
                    surround_radius=AMRF,
                    surround_ori=surround_ori,
                    surround_sf=preferred_sf,
                    surround_phase=phase,
                    surround_contrast=contrast,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max,
                    size=size,
                    img_res=img_res,
                    device=device,
                )
                image_for_panel = image_denormalised.detach().cpu().numpy()

            # ---------------------------------------------------------
            # CASE D: EXPERIMENT / SURROUND FIXED, CENTER MOVES
            # ---------------------------------------------------------
            elif stimulus_type == "experiment" and do_surround_fixed_center is False:

                center_ori = preferred_ori + ori_shift
                surround_ori = preferred_ori if surround_shift is None else preferred_ori + surround_shift

                stimulus, image_denormalised = get_center_surround_stimulus(
                    center_radius=GSF,
                    center_ori=center_ori,
                    center_sf=preferred_sf,
                    center_phase=phase,
                    center_contrast=contrast,
                    surround_radius=AMRF,
                    surround_ori=surround_ori,
                    surround_sf=preferred_sf,
                    surround_phase=phase,
                    surround_contrast=contrast,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max,
                    size=size,
                    img_res=img_res,
                    device=device,
                )
                image_for_panel = image_denormalised.detach().cpu().numpy()

            else:
                raise RuntimeError(
                    f"Unhandled branch for stimulus_type={stimulus_type!r}, "
                    f"do_surround_fixed_center={do_surround_fixed_center!r}"
                )

            if capture and (num_phase == panel_phase_index) and (num_ori in idx_to_slot):
                images6[idx_to_slot[num_ori]] = image_for_panel

            with torch.no_grad():
                response = float(single_model(stimulus).item())

            if subtract_gray_baseline:
                response = response - gray_resp

            all_orientation_curves[num_phase, num_ori] = response

    orientation_tuning_curve = torch.mean(all_orientation_curves, dim=0)

    if capture:
        if any(im is None for im in images6):
            raise RuntimeError(
                f"Orientation panel capture failed for phase index {panel_phase_index}: "
                f"some panel slots were not filled."
            )

        os.makedirs(panel_dir, exist_ok=True)
        tag = f"_{panel_tag}" if panel_tag else ""
        neuron_name = neuron if neuron else "neuron"

        if stimulus_type == "control":
            mode = "surround_only" if do_surround_fixed_center else "center_only"
        else:
            mode = "surround_moves" if do_surround_fixed_center else "center_moves"

        out_path = os.path.join(
            panel_dir,
            f"{neuron_name}_{mode}_ori_panel{tag}.png"
        )
        title = (
            f"{neuron_name} {mode} | "
            f"phase={float(phases[panel_phase_index]):.2f} | "
            f"center_shift={center_shift} | surround_shift={surround_shift}"
        )
        plot_orientation_controls_panel_raw(images6, labels6, out_path, title=title)

    return orientation_tuning_curve

def orientation_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts=np.linspace(-np.pi, np.pi, 9, endpoint=False),
    contrast=1,
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
    img_res=[93, 93]
):
    '''
    Perform orientation tuning experiments across multiple phases and save:

        /orientation_tuning/curves_center
            -> center-only control curve, shape (n_ori,)

        /orientation_tuning/curves_surround_only
            -> surround-only control curve, shape (n_ori,)

        /orientation_tuning/curves_center_surround
            -> full center x surround matrix, shape (n_ori, n_ori)

    Notes
    -----
    Responses saved here are baseline-subtracted by the response to a
    uniform gray image at the midpoint of pixel_min and pixel_max.
    '''

    def get_debug_panel_ori_indices(ori_shifts):
        ori_shifts = np.asarray(ori_shifts, dtype=float)
        idx_neg_pi = int(np.argmin(np.abs(ori_shifts - (-np.pi))))
        idx_zero = int(np.argmin(np.abs(ori_shifts - 0.0)))
        idx_pi_over_2 = int(np.argmin(np.abs(ori_shifts - (np.pi / 2))))
        return [idx_neg_pi, idx_zero, idx_pi_over_2]

    print(' > Orientation tuning experiment')

    group_path_ff_params = "/full_field_params"
    group_path_pos = "/preferred_pos"
    group_path_st_results = "/size_tuning/results"

    group_path_orientation_tuning = "/orientation_tuning"
    subgroup_path_results = group_path_orientation_tuning + "/results"
    subgroup_path_curves_center = group_path_orientation_tuning + "/curves_center"
    subgroup_path_curves_surround_only = group_path_orientation_tuning + "/curves_surround_only"
    subgroup_path_curves_center_surround = group_path_orientation_tuning + "/curves_center_surround"

    if overwrite:
        clear_group(h5_file, group_path_orientation_tuning)

    args_str = (
        f"phases={phases}/ori_shifts={ori_shifts}/contrast={contrast}/"
        f"pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    )
    group_init(h5_file=h5_file, group_path=group_path_orientation_tuning, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_center, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_surround_only, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_center_surround, group_args_str=args_str)

    with h5py.File(h5_file, 'a') as file:
        subgroup_results = file[subgroup_path_results]
        subgroup_curves_center = file[subgroup_path_curves_center]
        subgroup_curves_surround_only = file[subgroup_path_curves_surround_only]
        subgroup_curves_center_surround = file[subgroup_path_curves_center_surround]

        for neuron_id in tqdm(neuron_ids):
            neuron = f"neuron_{neuron_id}"

            if neuron in subgroup_curves_center_surround:
                continue

            single_model = SingleCellModel(all_neurons_model, neuron_id)

            if device is None:
                local_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                local_device = device

            single_model.to(local_device)
            single_model.eval()

            with torch.no_grad():
                gray_stim = torch.ones((1, 93, 93), device=local_device) * ((pixel_min + pixel_max) / 2)
                gray_resp = float(single_model(gray_stim).item())

            try:
                preferred_ori = file[group_path_ff_params][neuron][:][0]
                preferred_sf = file[group_path_ff_params][neuron][:][1]
                x_pix = file[group_path_pos][neuron][:][0]
                y_pix = file[group_path_pos][neuron][:][1]
                GSF = file[group_path_st_results][neuron][0]
                AMRF = file[group_path_st_results][neuron][2]

            except KeyError as e:
                print(f"Missing data for {neuron} in the required groups. Skipping this neuron. Error details: {e}")
                continue

            if GSF > AMRF:
                AMRF = GSF

            if (
                GSF is None or AMRF is None or
                not np.isfinite(GSF) or not np.isfinite(AMRF) or
                GSF <= 0 or AMRF <= 0
            ):
                print(f"Invalid GSF or AMRF for {neuron} (GSF={GSF}, AMRF={AMRF}). Skipping this neuron.")
                continue

            debug_panel_idx = get_debug_panel_ori_indices(ori_shifts)

            # A) CENTER-ONLY CONTROL
            center_tuning_curve = get_orientation_tuning_curves_all_phases(
                single_model=single_model,
                preferred_ori=preferred_ori,
                preferred_sf=preferred_sf,
                contrast=contrast,
                x_pix=x_pix,
                y_pix=y_pix,
                GSF=GSF,
                AMRF=AMRF,
                phases=phases,
                ori_shifts=ori_shifts,
                pixel_min=pixel_min,
                pixel_max=pixel_max,
                size=size,
                img_res=img_res,
                device=local_device,
                stimulus_type="control",
                do_surround_fixed_center=False,
                center_shift=0,
                save_panel=True,
                panel_phase_index=0,
                panel_ori_indices=debug_panel_idx,
                neuron=neuron,
                panel_tag="debug_center_only",
                gray_resp=gray_resp,
                subtract_gray_baseline=True,
            )

            # B) SURROUND-ONLY CONTROL
            surround_only_curve = get_orientation_tuning_curves_all_phases(
                single_model=single_model,
                preferred_ori=preferred_ori,
                preferred_sf=preferred_sf,
                contrast=contrast,
                x_pix=x_pix,
                y_pix=y_pix,
                GSF=GSF,
                AMRF=AMRF,
                phases=phases,
                ori_shifts=ori_shifts,
                pixel_min=pixel_min,
                pixel_max=pixel_max,
                size=size,
                img_res=img_res,
                device=local_device,
                stimulus_type="control",
                do_surround_fixed_center=True,
                center_shift=0,
                save_panel=True,
                panel_phase_index=0,
                panel_ori_indices=debug_panel_idx,
                neuron=neuron,
                panel_tag="debug_surround_only",
                gray_resp=gray_resp,
                subtract_gray_baseline=True,
            )

            # C) DEBUG PANELS: SURROUND MOVES, CENTER FIXED
            for k, fixed_center_shift in enumerate([-np.pi, 0.0, np.pi / 2]):
                _ = get_orientation_tuning_curves_all_phases(
                    single_model=single_model,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    GSF=GSF,
                    AMRF=AMRF,
                    phases=phases,
                    ori_shifts=ori_shifts,
                    contrast=contrast,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max,
                    size=size,
                    img_res=img_res,
                    device=local_device,
                    stimulus_type="experiment",
                    do_surround_fixed_center=True,
                    center_shift=fixed_center_shift,
                    save_panel=True,
                    panel_phase_index=0,
                    panel_ori_indices=debug_panel_idx,
                    neuron=neuron,
                    panel_tag=f"debug_surround_moves_center_{k}",
                    gray_resp=gray_resp,
                    subtract_gray_baseline=True,
                )

            # D) DEBUG PANELS: CENTER MOVES, SURROUND FIXED
            for k, fixed_surround_shift in enumerate([-np.pi, 0.0, np.pi / 2]):
                _ = get_orientation_tuning_curves_all_phases(
                    single_model=single_model,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    GSF=GSF,
                    AMRF=AMRF,
                    phases=phases,
                    ori_shifts=ori_shifts,
                    contrast=contrast,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max,
                    size=size,
                    img_res=img_res,
                    device=local_device,
                    stimulus_type="experiment",
                    do_surround_fixed_center=False,
                    surround_shift=fixed_surround_shift,
                    save_panel=True,
                    panel_phase_index=0,
                    panel_ori_indices=debug_panel_idx,
                    neuron=neuron,
                    panel_tag=f"debug_center_moves_surround_{k}",
                    gray_resp=gray_resp,
                    subtract_gray_baseline=True,
                )

            # E) FULL CENTER x SURROUND MATRIX
            center_shifts = np.asarray(ori_shifts)
            center_surround_matrix = torch.zeros(len(center_shifts), len(ori_shifts), dtype=torch.float32)

            for i, center_shift in enumerate(center_shifts):
                surround_tuning_curve = get_orientation_tuning_curves_all_phases(
                    single_model=single_model,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    GSF=GSF,
                    AMRF=AMRF,
                    phases=phases,
                    ori_shifts=ori_shifts,
                    contrast=contrast,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max,
                    size=size,
                    img_res=img_res,
                    device=local_device,
                    stimulus_type="experiment",
                    do_surround_fixed_center=True,
                    center_shift=center_shift,
                    save_panel=False,
                    neuron=neuron,
                    panel_tag="debug_center_surround",
                    gray_resp=gray_resp,
                    subtract_gray_baseline=True,
                )

                center_surround_matrix[i] = surround_tuning_curve

            subgroup_curves_center.create_dataset(name=neuron, data=center_tuning_curve.detach().cpu().numpy())
            subgroup_curves_surround_only.create_dataset(name=neuron, data=surround_only_curve.detach().cpu().numpy())
            subgroup_curves_center_surround.create_dataset(name=neuron, data=center_surround_matrix.detach().cpu().numpy())