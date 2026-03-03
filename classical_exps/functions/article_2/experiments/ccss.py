############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.functions.utils import *
## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.functions.utils import SingleCellModel
from .panels import plot_ccss_controls_panel_raw
## Data storage
import h5py
from scipy.optimize import curve_fit
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

def center_contrast_surround_suppression_experiment(
            h5_file,
            all_neurons_model,
            neuron_ids,
            overwrite = False,
            center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5]),
            surround_contrast = 0.5,
            phases = np.linspace(0, 2*np.pi, 37)[:-1],
            pixel_min = -1.7876,
            pixel_max =  2.1919,
            device = None,
            size = 2.67,
            img_res = [93,93] ,
            save_panel=True,
        ):
    ''' This function aims to determine the link between the surround suppression and the center stimulus contrast
        For every center_contrast in center_contrasts_ccss :

            1) The center patch of each stimulus is at the preferred orientation and with a contrast = center_contrast
            2) It gets the response of each neuron to this center and surround at the parallel and orthogonal orientation + the response of the center alone
            3) It saves the best responses for the parallel, orthogonal and center alone stimulus accross every phase
            4) It saves the results in the HDF5 file as a matrix where every row is a center contrast 
        

        It saves the data with this architecture : 
                                                            
            Group /center_contrast_surround_suppression _______ SubGroup ../results     --> neuron datasets

        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments    

        Outputs :

            - datasets in ../results  : A matrix (shape [len(center_contrasts_ccss), 3]) containing the response of each neuron for a surround at the preferred orientation, orthogonal position and the response of the center alone.
                                        Each column is a condition and each row correspond to a center contrast

                                                format :             resp_iso | resp_ortho | resp_center_alone
                                                        _______________________________________________________
                                                        contrast 0 |          |            |
                                                        ______________________|____________|___________________
                                                        contrast...|          |            |
                                                        _______________________________________________________
    '''                                                 
    print(' > Center contrast surround suppression experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path      = "/center_contrast_surround_suppression"
    subgroup_path   = group_path + "/results"

    ## Clear the group if requested    
    if overwrite : 
        clear_group(h5_file,group_path)

    ## Initialize the Group and subgroup
    args_str = f"center_contrasts_ccss={center_contrasts_ccss}/surround_contrast={surround_contrast}/phases={phases}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos, group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    with h5py.File(h5_file, 'a') as f :
        
        ## Access the subgroup
        subgroup = f[subgroup_path]
        for neuron_id in tqdm(neuron_ids) :

            neuron = f"neuron_{neuron_id}"

            rows_for_panel = []  # for panel visualization, optional


            ## Check if the neuron data is not already present
            if neuron not in subgroup :
                
                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id)
                single_model.to(device)
                single_model.eval()

                ## Get the neuron's preferred parameters 
                preferred_ori = f[group_path_ff_params][neuron][:][0]
                preferred_sf  = f[group_path_ff_params][neuron][:][1]
                x_pix         = f[group_path_pos][neuron][:][0]
                y_pix         = f[group_path_pos][neuron][:][1]
                GSF           = f[group_path_st_results][neuron][1]
                AMRF          = f[group_path_st_results][neuron][3]
                
                if AMRF<GSF :
                    AMRF=GSF

                mean_response = np.zeros((len(center_contrasts_ccss), 3))
                rows_for_panel = []

                panel_phase_index = len(phases) // 2  # deterministic phase for saved stimuli

                with torch.no_grad():
                    for num_contrast, center_contrast in enumerate(center_contrasts_ccss):

                        # store mean responses for this contrast: [iso, ortho, center]
                        mean_row = np.zeros(3, dtype=float)

                        # panel images for this contrast (raw generator-space)
                        panel_imgs = {"iso": None, "ortho": None, "center": None}

                        # ---- ISO + ORTHO (i = 0,1) ----
                        for i, shift_ori in enumerate([0, np.pi/2]):
                            surround_ori = preferred_ori + shift_ori

                            resp_sum = 0.0
                            for p_i, phase in enumerate(phases):
                                stimulus, img_raw = get_center_surround_stimulus(
                                    center_radius=GSF,
                                    center_ori=preferred_ori,
                                    center_sf=preferred_sf,
                                    center_phase=phase,
                                    center_contrast=center_contrast,
                                    surround_radius=AMRF,
                                    surround_ori=surround_ori,
                                    surround_sf=preferred_sf,
                                    surround_phase=phase,
                                    surround_contrast=surround_contrast,
                                    x_pix=x_pix, y_pix=y_pix,
                                    pixel_min=pixel_min, pixel_max=pixel_max,
                                    size=size, img_res=img_res, device=device
                                )

                                resp = single_model(stimulus).item()
                                resp_sum += resp

                                # save deterministic phase stimulus for diagnostics
                                if save_panel and (p_i == panel_phase_index):
                                    panel_imgs["iso" if i == 0 else "ortho"] = img_raw.detach().cpu().numpy()

                            mean_row[i] = resp_sum / float(len(phases))

                        # ---- CENTER ONLY (i = 2) ----
                        resp_sum = 0.0
                        surround_ori_dummy = preferred_ori  # explicit dummy
                        for p_i, phase in enumerate(phases):
                            stimulus, img_raw = get_center_surround_stimulus(
                                center_radius=GSF,
                                center_ori=preferred_ori,
                                center_sf=preferred_sf,
                                center_phase=phase,
                                center_contrast=center_contrast,
                                surround_radius=AMRF,
                                surround_ori=surround_ori_dummy,
                                surround_sf=preferred_sf,
                                surround_phase=phase,
                                surround_contrast=0,  # center-only
                                x_pix=x_pix, y_pix=y_pix,
                                pixel_min=pixel_min, pixel_max=pixel_max,
                                size=size, img_res=img_res, device=device
                            )

                            resp = single_model(stimulus).item()
                            resp_sum += resp

                            if save_panel and (p_i == panel_phase_index):
                                panel_imgs["center"] = img_raw.detach().cpu().numpy()

                        mean_row[2] = resp_sum / float(len(phases))

                        # save numeric row
                        mean_response[num_contrast, :] = mean_row

                        # save panel row (optional)
                        if save_panel:
                            rows_for_panel.append({
                                "contrast": float(center_contrast),
                                "iso": panel_imgs["iso"],
                                "ortho": panel_imgs["ortho"],
                                "center": panel_imgs["center"],
                            })

                # save matrix ONCE per neuron
                subgroup.create_dataset(name=neuron, data=mean_response)

                # after neuron loop, plot panel once (if save_panel)
                if save_panel:
                    out_dir = "/project/results/selectivity_and_spatial_distribution/stimuli/ccss_panels/"
                    out_path = os.path.join(out_dir, f"{neuron}_ccss_panel.png")
                    plot_ccss_controls_panel_raw(
                        rows_for_panel,
                        out_path,
                        title=f"{neuron} CCSS | ori={preferred_ori:.2f} sf={preferred_sf:.2f}"
                    )