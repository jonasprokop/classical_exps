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
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF
from .size_tuning import get_size_tuning_curves

# Image generation
import imagen
from imagen.image import BoundingBox

# Data / optimization / plotting
import h5py
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .utils import ensure_dir

def get_contrast_size_tuning_curve_all_phases(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.array([
    0.06,
    0.10,
    0.18,
    0.30,
    0.45,
    0.65,
    1.0
    ]),
    radii = np.logspace(-2,np.log10(2),40) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,
    phase_stat="F0",          # "F0", "F1", "MAX", "BOTH"
    assume_uniform_phases=True
    ):
    ''' For a given neuron, this function aims to create multiple size tuning curves, for different contrasts.
        This function uses the 'get_size_tuning_curves' to get the tuning curves.
        The outputs of this function are two matrices, one for the circular stimulus tuning curve, the other for the annular stimulus one.
        The matrices are created as so : every row correspond to a contrast value, every column correspond to a radius.

        Arguments :

            - contrasts : Every contrast to test (will be the rows of the matrices)
            - others    : See 'get_size_tuning_curves' arguments

        Outputs :

            - all_circular_curves : For the circular stimuli : the matrix containing the response of the neuron for every combination of contrast (row) and radius (column)
            - all_annular_curves  : For the annular stimuli  : Same

        NB : since we perform accross every phase, the responses set in the matrices are the best response accross every phase
    '''

    all_circular_curves = torch.zeros((len(phases),len(contrasts), len(radii)))
    
    ## For every phase :
    for num_phase in range(len(phases)) :
        
        ## For every contrast (every row of the matrices)
        for i,contrast in enumerate(contrasts) : 

            ## Get the size tuning curve
            circular_curve = get_size_tuning_curves(
                single_model = single_model,
                x_pix = x_pix,
                y_pix = y_pix,
                preferred_ori = preferred_ori,
                preferred_sf = preferred_sf,
                preferred_phase = phases[num_phase],
                radii = radii,
                contrast = contrast,
                pixel_min = pixel_min,
                pixel_max =  pixel_max, 
                device = device,
                size = size,
                img_res = img_res,
                neg_val = neg_val,
                compute_annular = False, ## This function does not need the annular tuning curve
                save_stimuli_panel=False,
                stimuli_panel_dir="/project/results/nature_and_interactions/stimuli/size_tuning_panels/",
                stimuli_panel_tag=f"contrast_{contrast}_phase_{phases[num_phase]}",
                )

            ## Fill the corresponding rows of the matrices
            all_circular_curves[num_phase ,i ,:] = circular_curve


    stat = phase_stat.upper()

    # Default, most "article-like" proxy for drifting mean rate
    F0 = all_circular_curves.mean(dim=0)  # [C, R]

    if stat in ("F0", "MEAN"):
        return F0

    if stat == "MAX":
        return all_circular_curves.max(dim=0).values

    # F1 amplitude from phase sweep: 2 * |c1|
    # Valid if phases are uniform over [0, 2π); with your default linspace it is.
    if stat == "F1" or stat == "BOTH":
        if assume_uniform_phases:
            phi = torch.as_tensor(phases, dtype=all_circular_curves.dtype, device=all_circular_curves.device)
            ejphi = torch.exp(1j * phi)[:, None, None]  # [P,1,1]
            c1 = (all_circular_curves.to(torch.complex64) * ejphi).mean(dim=0)  # [C,R]
            F1 = (2.0 * torch.abs(c1)).to(all_circular_curves.dtype)            # [C,R]
        else:
            # If phases ever become non-uniform, do least-squares fit instead.
            raise ValueError("Non-uniform phases not supported for F1 in this implementation.")

        if stat == "F1":
            return F1
        else:
            # BOTH
            return F0, F1

    raise ValueError(f"Unknown phase_stat={phase_stat} (use F0/MEAN, F1, MAX, BOTH)")

def contrast_size_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.logspace(np.log10(0.06),np.log10(1),5),
    radii = np.logspace(-2,np.log10(2),40) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,
    contrast=None,
):
    ''' For the selected neurons this function :

            1) Performs 'get_contrast_size_tuning_curve_all_phases' function to get the size tuning curves of each neuron for different contrasts

            2) Analysis for each result : This function gets for each neuron the GSF values computed from size tuning experiments for the highest contrast (100%) and for the lowest contrast (6%)
                                          It then computes the shift of the GSF as a ratio GSFlow/GSFhigh
            
            3) For each neuron it saves the curves and the results of the analysis in different subgroups

        It saves the data with this architecture : 


                                         _________ SubGroup /contrast_size_tuning/curves   --> neuron datasets
                                         |
        Group /contrast_size_tuning  ____|
                                         |
                                         |________ SubGroup /contrast_size_tuning/results  --> neuron datasets
        
        Prerequisite : 

            - function 'get_all_grating_parameters'        executed for the required neurons with matching arguments
            - function 'get_preferred_position'            executed for the required neurons with matching arguments

        Arguments :

            - contrasts : Every contrast to test (will be the rows of the matrices)
            - others    : See other functions

        Outputs :


            - datasets in ../curves     : A matrix that contains the response of the model (neuron) for every contrast (row) and radius (col)

            - datasets in ../results    : An array containing the results of this analysis for each neuron
                                          format = [GSFs_ratio] with GSFs_ratio = GSFlow/GSFhigh
                                                
    '''
    print(' > Contrast size tuning experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    ## Groups to fill
    group_path_cst            = "/contrast_size_tuning"
    subgroup_path_cst_results = group_path_cst + "/results"
    subgroup_path_cst_curves  = group_path_cst + "/curves"
    subgroup_path_cst_curves_F1  = group_path_cst + "/curves_f1"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_cst)

    ## Initialize the Group and the subgroups
    args_str = f"phases={phases}/contrasts={contrasts}/radii={radii}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_cst, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_curves, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_curves_F1, group_args_str=args_str)


    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos], neuron_ids=neuron_ids, group_args_str=args_str)

    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        group_cst                 = file[group_path_cst]
        subgroup_cst_results      = file[subgroup_path_cst_results]
        subgroup_cst_curves  = file[subgroup_path_cst_curves]
        group_ff_params     = file[group_path_ff_params]
        group_pos           = file[group_path_pos]
        subgroup_cst_curves_F1  = file[subgroup_path_cst_curves_F1]

        for neuron_id in tqdm(neuron_ids) :
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_cst_results :

                ## Get the single_model of the neuron
                single_model = SingleCellModel(all_neurons_model, neuron_id)

                ## Get the preferred parameters of the neuron
                preferred_ori   = group_ff_params[neuron][:][0]
                preferred_sf    = group_ff_params[neuron][:][1]
                x_pix           = group_pos[neuron][:][0]
                y_pix           = group_pos[neuron][:][1]
                
                cst_curves, cst_curves_F1 = get_contrast_size_tuning_curve_all_phases(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    phases=phases,
                    contrasts=contrasts,
                    radii=radii,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max, 
                    device=device,
                    size=size,
                    img_res=img_res,
                    neg_val=neg_val,
                    phase_stat="BOTH",
                    )


                ## For low contrast  
                low_contrast_curve = cst_curves[0]
                GSF_low,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
                ## For contrast =1
                high_contrast_curve = cst_curves[-1]
                GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)

                if GSF_low is None or GSF_high is None or GSF_high == 0:
                    GSFs_ratio = np.nan
                else:
                    GSFs_ratio = GSF_low / GSF_high

                ## Save the curves
                subgroup_cst_curves.create_dataset(name=neuron, data=cst_curves.cpu())
                subgroup_cst_curves_F1.create_dataset(name=neuron, data=cst_curves_F1.cpu())
                ## Save the results
                data = [GSF_low if GSF_low is not None else np.nan,
                        GSF_high if GSF_high is not None else np.nan,
                        GSFs_ratio]
                subgroup_cst_results.create_dataset(name=neuron, data=data)
