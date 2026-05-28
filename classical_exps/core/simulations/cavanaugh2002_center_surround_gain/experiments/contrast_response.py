"""Experiment stimulus generation utilities.

This package contains stimulus generation + experiment runner functions
"""

from __future__ import annotations

# Useful
import os
import time

import numpy as np
import torch
from tqdm import tqdm

# Utils (project)
from classical_exps.core.tools.utils import *  # noqa: F401,F403
from classical_exps.core.tools.utils import SingleCellModel  # noqa: F401

# Image generation
import imagen
from imagen.image import BoundingBox

# Data / optimization / plotting
import h5py
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .utils import ensure_dir, fmt_pct

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def get_contrast_response(
    single_model,
    x_pix,
    y_pix,
    center_radius,
    surround_radius,
    preferred_ori,
    preferred_sf,
    center_contrasts = np.logspace(-2,np.log10(1),18),  
    surround_contrasts = np.logspace(-2,np.log10(1),18),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,
    neuron="0",
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    phase_stat="F0",
    ):
    
    ''' This function takes the preferred parameters for the orientation, spatial frequency and phase to create images with a circular center 
        and an annular surround. The radius of the center and annular is fixed, we advice to take the GSF for the center radius and the AMRF 
        for the surround radius (or the GSF if the AMRF is smaller than the GSF). 
        The goal of this function is to try different combination of center and surround contrast for the given model (unique neuron),
        get their response, then normalise the values by substracting the response to gray stimulus.
        Optionally (if neg_val == False) it replaces the negative values by zero (neg values are coming from the normalisation).

        NB  : This function uses the SingleNeuronModel class and the v1_convnext_ensemble
        NB2 : Surround_radius is the radius from the center to the inner edge of the ring
        NB3 : For this function, you can't use surround diameters smaller than center diameter (if overlapping, the center takes advantage)

        Arguments :

            - neg_val : Bool, if set to false, replace every negative value to 0
        
        Outputs : 

            - contrast_resps_mat : A matrix that contains the response of the model (neuron) for every surround contrast (rows) and center contrast (cols)
    '''
    
    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

    contrast_responses_images_mat = []
    contrast_responses_surround_contrast_mat = []
    contrast_responses_center_contrast_mat = []

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    ## Avoid surround and center to overlap
    if surround_radius < center_radius :
        surround_radius = center_radius


    contrast_resps_mat = torch.zeros((len(surround_contrasts), len(center_contrasts))).to(device)

    with torch.no_grad():
        ## Get the response of the model to gray stimulus
        gray_stim = torch.ones((1,93,93)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = single_model(gray_stim)

        ## For every combination of contrasts
        for i, surround_contrast in enumerate(surround_contrasts):
            for j, center_contrast in enumerate(center_contrasts) : 

                resp_list = []

                for k, phi in enumerate(phases):
                    image, image_denormalised = get_center_surround_stimulus(
                        center_radius=center_radius,
                        center_ori=preferred_ori,
                        center_sf=preferred_sf,
                        center_phase=phi,
                        center_contrast=center_contrast,
                        surround_radius=surround_radius,
                        surround_ori=preferred_ori,
                        surround_sf=preferred_sf,
                        surround_phase=phi,
                        surround_contrast=surround_contrast,
                        x_pix=x_pix,
                        y_pix=y_pix,
                        pixel_min=pixel_min,
                        pixel_max=pixel_max,
                        size=size,
                        img_res=img_res,
                        device=device
                    )

                    if k==0: 
                        contrast_responses_images_mat.append(image_denormalised.cpu().numpy().squeeze())
                        contrast_responses_surround_contrast_mat.append(surround_contrast)
                        contrast_responses_center_contrast_mat.append(center_contrast)

                    resp = single_model(image)
                    resp_list.append(resp - gray_resp)

                resp_stack = torch.stack([r.view(-1)[0] for r in resp_list], dim=0)

                if phase_stat in ("F0", "MEAN"):
                    resp_norm = resp_stack.mean()
                elif phase_stat == "MAX":
                    resp_norm = resp_stack.max()
                else:
                    raise ValueError("Unsupported phase_stat")

                contrast_resps_mat[i, j] = resp_norm


        plot_contrast_response_stimuli_panel(
            neuron=neuron,  # rotate outside later
            images_flat=contrast_responses_images_mat,
            surround_contrasts_flat=contrast_responses_surround_contrast_mat,
            center_contrasts_flat=contrast_responses_center_contrast_mat,
            save_dir="/project/results/nature_and_interactions/stimuli/contrast_response_panels/",
            n_center_samples=3,
            n_surround_samples=3,
        )


    ## Optional : replace negative values by zero
    if neg_val == False :     
        contrast_resps_mat = torch.maximum(contrast_resps_mat, torch.zeros(contrast_resps_mat.shape).to(device))

    return contrast_resps_mat

def contrast_response_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    center_contrasts = np.logspace(-2,np.log10(1),18),
    surround_contrasts = np.logspace(-2,np.log10(1),6),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,
    phases=np.linspace(0, 2*np.pi, 37)[:-1],
    phase_stat="F0",
    ):
    ''' For the selected neurons this function :

            1) Performs 'get_contrast_response' function to get the response of each neuron to different center and surround contrasts.

            2) Analysis for each result : it gets the last point of every curves and calculate the difference between the highest and
            the lowest value ('max/min'), also computes the mean standard deviation accross every points ('mean_std')
            Those values aim to characterise how the neuron is affected by the surround contrast (if the points are far from each other, that means that the response is affected by the surround contrast, the std and max/min ratio values will then be higher)

            3) For each neuron it saves the contrast response curves and the results of the analysis in different subgroups

        It saves the data with this architecture : 


                                      _________ SubGroup /contrast_response/curves   --> neuron datasets
                                      |
        Group /contrast_response  ____|
                                      |
                                      |________ SubGroup /contrast_response/results  --> neuron datasets
        
        Prerequisite : 

            - function 'get_all_grating_parameters'        executed for the required neurons with matching arguments
            - function 'get_preferred_position'            executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments

        Arguments :

            - center_contrasts   : An array containing the contrast values to test for the circular (center) stiumulus
            - surround_contrasts :      ""      ""      ""      ""      ""      ""      "" annular (surround) stiumulus
            - others             : Described in other functions

        Outputs :


            - datasets in ../curves     : A matrix that contains the response of the model (neuron) for every surround contrast (rows) and center contrast (cols) (see 'get_contrast_response')

            - datasets in ../results    : An array containing the results of this analysis for each neuron
                                          format = [max/min, mean_std] 
                                                
    '''
    print(' > Contrast response experiment')
    
    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path_cr            = "/contrast_response"
    subgroup_path_cr_results = group_path_cr + "/results"
    subgroup_path_cr_curves  = group_path_cr + "/curves"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_cr)

    ## Initialize the Group and the subgroups
    args_str = f"center_contrasts={center_contrasts}/surround_contrasts={surround_contrasts}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_cr, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cr_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cr_curves, group_args_str=args_str)

    # ## Check compatibility between every experiment
    # check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos,group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)


    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        group_cr            = file[group_path_cr]
        subgroup_cr_results = file[subgroup_path_cr_results]
        subgroup_cr_curves  = file[subgroup_path_cr_curves]
        group_ff_params     = file[group_path_ff_params]
        group_pos           = file[group_path_pos]
        group_st_results    = file[group_path_st_results]

        
        for neuron_id in tqdm(neuron_ids) :
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_cr_results :

                ## Get the single_model of the neuron
                single_model = SingleCellModel(all_neurons_model, neuron_id)

                
                ## Get the preferred parameters of the neuron
                try:
                    arr = group_st_results[neuron][:].reshape(-1)
                    GSF  = float(arr[0])
                    AMRF = float(arr[2])
                    preferred_ori   = group_ff_params[neuron][:][0]
                    preferred_sf    = group_ff_params[neuron][:][1]
                    x_pix           = group_pos[neuron][:][0]
                    y_pix           = group_pos[neuron][:][1]
                except Exception as e:
                    print(f"Error loading parameters for neuron {neuron_id}: {e}")
                    subgroup_cr_curves.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    subgroup_cr_results.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    continue

                def _bad(*xs):
                    xs = np.array(xs, dtype=float)
                    return (not np.all(np.isfinite(xs)))

                # ... inside loop, after loading GSF/AMRF/etc.

                # Defensive casts
                GSF = float(GSF)
                AMRF = float(AMRF)
                preferred_ori = float(preferred_ori)
                preferred_sf  = float(preferred_sf)
                x_pix = float(x_pix)
                y_pix = float(y_pix)

                # Core validity checks
                if _bad(GSF, AMRF, preferred_ori, preferred_sf, x_pix, y_pix):
                    print(f"Skip {neuron}: non-finite params (GSF/AMRF/or/sf/x/y).")
                    subgroup_cr_curves.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    subgroup_cr_results.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    continue

                # Radii must be > 0 for imagen.Disk(... size=radius*2.0)
                if GSF <= 0.0 or AMRF <= 0.0:
                    print(f"Skip {neuron}: invalid radii (GSF={GSF}, AMRF={AMRF}).")
                    subgroup_cr_curves.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    subgroup_cr_results.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    continue

                # SF must be > 0 (negative/zero SF is also nonsense for gratings)
                if preferred_sf <= 0.0:
                    print(f"Skip {neuron}: invalid preferred_sf={preferred_sf}.")
                    subgroup_cr_curves.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    subgroup_cr_results.create_dataset(name=neuron, data=[float("nan"), float("nan")])
                    continue

                # Avoid overlap (your existing logic, but AFTER validation)
                if GSF > AMRF:
                    AMRF = GSF
                
                ## Get the contrast response curves (matrix)
                contrast_response_mat = get_contrast_response(
                    single_model = single_model,
                    x_pix = x_pix,
                    y_pix = y_pix,
                    center_radius = GSF,
                    surround_radius = AMRF,
                    preferred_ori = preferred_ori,
                    preferred_sf = preferred_sf,
                    center_contrasts = center_contrasts,
                    surround_contrasts = surround_contrasts,
                    pixel_min = pixel_min,
                    pixel_max = pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = neg_val,
                    neuron=neuron,
                    phases=phases,
                    phase_stat=phase_stat,
                    )
                

                ## Capture how the curves are spread
                last_points = contrast_response_mat[:,-1]
                min = torch.min(last_points)
                max = torch.max(last_points)
                ratio = (max/min).cpu().item()
                
                mean_std = torch.mean(torch.std(contrast_response_mat, axis = 0)).cpu().item()
                
                ## Save the curves
                subgroup_cr_curves.create_dataset(name=neuron, data=contrast_response_mat.cpu())

                ## Save the results
                data = [ratio, mean_std]
                subgroup_cr_results.create_dataset(name=neuron, data=data)



def plot_contrast_response_stimuli_panel(
    neuron: str,
    images_flat,                  # list of (H,W) arrays, len = nS*nC
    surround_contrasts_flat,       # list, same len
    center_contrasts_flat,         # list, same len
    save_dir,
    filename=None,
    cmap="gray",
    n_center_samples=3,
    n_surround_samples=3,
):
    """
    Build a (n_surround_samples x n_center_samples) panel of stimuli with a shared colorbar.
    Inputs are flat lists collected during the scan.

    Assumes images are in display space (whatever you put into images_flat).
    """
    os.makedirs(save_dir, exist_ok=True)

    images_flat = list(images_flat)
    sc_flat = np.asarray(surround_contrasts_flat, dtype=float)
    cc_flat = np.asarray(center_contrasts_flat, dtype=float)

    if len(images_flat) == 0:
        return None
    if not (len(images_flat) == len(sc_flat) == len(cc_flat)):
        raise ValueError("images_flat, surround_contrasts_flat, center_contrasts_flat must have same length")

    # Unique contrasts (sorted for predictable panels)
    sc_unique = np.unique(sc_flat)
    cc_unique = np.unique(cc_flat)

    # pick low/mid/high (or fewer if not available)
    def pick_subset(arr, k):
        k = int(min(max(1, k), len(arr)))
        if len(arr) == k:
            return arr
        idx = np.linspace(0, len(arr)-1, k, dtype=int)
        return arr[idx]

    sc_sel = pick_subset(sc_unique, n_surround_samples)
    cc_sel = pick_subset(cc_unique, n_center_samples)

    # map (s,c) -> first image encountered (your scan order makes this deterministic)
    im_map = {}
    for im, s, c in zip(images_flat, sc_flat, cc_flat):
        key = (float(s), float(c))
        if key not in im_map:
            im_map[key] = np.asarray(im)

    # gather selected grid
    grid = []
    for s in sc_sel:
        row = []
        for c in cc_sel:
            key = (float(s), float(c))
            if key not in im_map:
                raise KeyError(f"Missing stimulus for surround={s}, center={c}")
            row.append(im_map[key])
        grid.append(row)

    # shared color scale
    all_imgs = [grid[i][j] for i in range(len(sc_sel)) for j in range(len(cc_sel))]
    vmin = float(min(im.min() for im in all_imgs))
    vmax = float(max(im.max() for im in all_imgs))

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    nrows, ncols = len(sc_sel), len(cc_sel)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1*ncols, 2.1*nrows), dpi=220, sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, s in enumerate(sc_sel):
        for j, c in enumerate(cc_sel):
            ax = axes[i, j]
            ax.imshow(grid[i][j], cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"C {100*c:.0f}%", fontsize=9)
            if j == 0:
                ax.text(-0.08, 0.5, f"S {100*s:.0f}%", transform=ax.transAxes,
                        va="center", ha="right", fontsize=9, rotation=90)

    fig.suptitle(f"{neuron} – contrast response stimuli", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.92])

    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.05, pad=0.02)
    cbar.set_label("stimulus", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if filename is None:
        filename = f"{neuron}_contrast_response_stimuli.png"
    out = os.path.join(save_dir, filename)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out