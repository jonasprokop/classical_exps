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
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts = np.linspace(-np.pi,np.pi,9, endpoint=False),   
    contrast = 1,
    pixel_min = -1.7876, 
    pixel_max =  2.1919, 
    size = 2.67,
    img_res = [93,93],
    device = None,
    do_surround_fixed_center = True,
    center_shift = 0,
    save_panel=False,
    panel_dir="/project/results/selectivity_and_spatial_distribution/stimuli/orientation_panels/",
    panel_tag="",
    panel_phase_index=None,  # None -> middle phase
    panel_n=6,
    neuron="",
):
    ''' For a given neuron, this function does either :

            If do_surround_fixed_center == True :

                Computes an orientation tuning curve for the surround orientation with the center's orientation fixed

            Else :
                
                Computes the orientation tuning curve for the center with no surround

        - Returns a curve of response for a center stimulus alone at different orientations
        - Returns a curve of response for a center stimulus with a fixed orientation and a surround stimulus at different orientations

    Arguments :

        - single_model              : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
        - preferred_ori             : The preferred orientation (the one that elicited the greatest response). 
        - preferred_sf              : The preferred spatial frequency
        - x_pix                     : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
        - y_pix                     : The y coordinates   ""        ""        ""
        - GSF                       : Grating Summation Field, it will be the center's radius
        - AMRF                      : Annular Minimum Response Field, it will be the inner edge radius of the surround (since the surround is a ring)
        - phases                    : An array containing the phases to try
        - ori_shifts                : An array containing the shift of orientation in Radiant to try (realtive to the preferred orientation) (Make sure that the middle value is 0 for better visualisation, don't expend over +-180  deg (+-pi))
        - pixel_min                 : Value of the minimal pixel that will serve as the black reference
        - pixel_max                 : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
        - size                      : The size of the image in terms of visual field degrees
        - img_res                   : Resolution of the image in term of pixels shape : [pix_y, pix_x]
        - device                    : The device on which to execute the code, if set to "None" it will take the available one
        - do_surround_fixed_center  : Bool, whether or not the output should be the orientation tuning curve of the center alone or the fixed center and the surround
        - center_shift              : (Optional), The orientation shift to apply on the center for the surround orientation tuning with fixed center
    
    Outputs :

        If do_surround_fixed_center == True

            - orientation_tuning_curve : An array containing the response of the neuron dor the image which is a center grating patch with a fixed orientation and a surround with changing orientation

        If do_surround_fixed_center == False

            - orientation_tuning_curve : An array containing the response of the neuron for the image which is a center grating patch with changing orientation  
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## Avoid overlapping
    if GSF>AMRF :
        AMRF = GSF

    # panel capture setup (fixed 6 slots)
    capture = bool(save_panel)
    if capture:
        panel_idx = six_panel_indices(len(ori_shifts), panel_n)
        idx_to_slot = {int(k): s for s, k in enumerate(panel_idx)}
        if panel_phase_index is None:
            panel_phase_index = len(phases)//2
        panel_phase_index = int(np.clip(panel_phase_index, 0, len(phases)-1))
        images6 = [None] * len(panel_idx)
        labels6 = [f"Δθ={ori_shifts[i]:+.2f} rad" for i in panel_idx]

    ## This will store the model responses for every phase (row) and every orientation (column) 
    all_orientation_curves = torch.zeros((len(phases), len(ori_shifts)))

    ## For every phase :
    for num_phase, phase in enumerate(phases) :

        ## For every orientation shift 
        for num_ori, ori_shift in enumerate(ori_shifts) : 
            
            ## Change the orientation
            orientation = preferred_ori + ori_shift

            ## Stimulus = center only
            if do_surround_fixed_center == False :
                
                ## Create the center stimulus
                stimulus = torch.Tensor(imagen.SineGrating(
                                    mask_shape=imagen.Disk(smoothing=0.0, size=GSF*2.0),
                                    orientation=orientation,
                                    frequency=preferred_sf,
                                    phase=phase,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    offset = -1,
                                    scale=2,  
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
                
                ## Mask the circular stimulus 
                center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                        size= GSF*2.0, 
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
                ## Add the gray background to the circular stimulus
                stimulus = stimulus * center_mask

                ## Change the contrast 
                stimulus *= contrast
                
                # capture raw generator-space image (no clamp, no renorm)
                if capture and (num_phase == panel_phase_index) and (num_ori in idx_to_slot):
                    images6[idx_to_slot[num_ori]] = stimulus.detach().cpu().numpy()


                ## Convert to the right shape for the model
                stimulus = rescale(stimulus,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)
                
            else :

                center_ori = preferred_ori

                ## Change the center's orientation 
                if center_shift is not None :
                    center_ori = center_ori + center_shift
    
                stimulus, image_denormalised = get_center_surround_stimulus(center_radius=GSF, center_ori=center_ori, center_sf=preferred_sf, center_phase=phase, center_contrast=contrast, surround_radius=AMRF, surround_ori=orientation, surround_sf=preferred_sf, surround_phase=phase, surround_contrast=contrast, x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)    
            
                # capture raw generator-space image (no clamp, no renorm)
                if capture and (num_phase == panel_phase_index) and (num_ori in idx_to_slot):
                    images6[idx_to_slot[num_ori]] = image_denormalised.detach().cpu().numpy()

            with torch.no_grad():
                
                ## Get the model's response
                response = single_model(stimulus).item()

            ## Save the response at the correct place
            all_orientation_curves[num_phase, num_ori] = response

    ## Get the maximum response accross phases
    orientation_tuning_curve = torch.mean(all_orientation_curves, dim=0)

    # save panel at end (once)
    if capture:
        if any(im is None for im in images6):
            raise RuntimeError("Orientation panel capture failed: some of the 6 slots were not filled.")
        os.makedirs(panel_dir, exist_ok=True)
        tag = f"_{panel_tag}" if panel_tag else ""
        mode = "surround" if do_surround_fixed_center else "center"
        neuron_name = neuron if neuron else "neuron"
        out_path = os.path.join(panel_dir, f"{neuron_name}_{mode}_ori_panel{tag}.png")
        title = f"{neuron_name} {mode} | phase={float(phases[panel_phase_index]):.2f} | center_shift={center_shift}"
        plot_orientation_controls_panel_raw(images6, labels6, out_path, title=title)


    return orientation_tuning_curve

def orientation_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts = np.linspace(-np.pi,np.pi,9),
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93]
):
    ''' This function performs an orientation tuning experiment for the neurons accross multiple phases :

            - 1) It computes the orientation tuning curve for the center alone and save it in the '/orientation_tuning/curves_center/' folder
            - 2) It computes the orientation tuning curves for the surround orientation with a fixed center orientation
            - 3) It performs 2) for a center orientation of -45°, 0° and +45° relative to the preferred orientation and save the results in the '/orientation_tuning/curves_surround_fixed_center/' folder
            - 4) It saves the most suppressing surround orientation (don't look for orientations above 90 and below -90 degrees) for all 3 center orientation in the /orientation_tuning/results/' folder
        
        It saves the data with this architecture : 


                                        _________ SubGroup /orientation_tuning/curves_center                 --> neuron datasets
                                        |
        Group /orientation_tuning ______|________ SubGroup /orientation_tuning/curves_surround_fixed_center  --> neuron datasets
                                        |
                                        |________ SubGroup /orientation_tuning/results                       --> neuron datasets
        
        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments        

        Arguments :

            - ori_shifts : An array containing the shift of orientation (in radiant) to try (realtive to the preferred orientation)
            - other      : See the 'get_orientation_tuning_curves_all_phase' function

        Outputs :

            - datasets in ../curves_center                : An array containing the responses of the neuron to all orientation

            - datasets in ../curves_surround_fixed_center : A matrix containing the orientation tuning curves for the surround (col) for 3 differnet center orientation (-45°, +0°, +45°) (row) 

            - datasets in ../results                      : An array containing the most suppressive orientation (only for orientations in [-90°,+90°])
    '''

    print(' > Orientation tuning experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path_orientation_tuning = "/orientation_tuning"
    subgroup_path_results         = group_path_orientation_tuning + "/results"
    subgroup_path_curves_center   = group_path_orientation_tuning + "/curves_center"
    subgroup_path_curves_surround = group_path_orientation_tuning + "/curves_surround_fixed_center"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_orientation_tuning)

    ## Initialize the Group and the subgroups
    args_str = f"phases={phases}/ori_shifts={ori_shifts}/contrast={contrast}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path_orientation_tuning, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_center, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_surround, group_args_str=args_str)


    # ## Check compatibility between every experiment
    # check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos, group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)

    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        subgroup_results         = file[subgroup_path_results]
        subgroup_curves_center   = file[subgroup_path_curves_center]
        subgroup_curves_surround = file[subgroup_path_curves_surround]

        ## For every neuron
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_results :

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id )

                ## Get the neuron's preferred parameters 
                preferred_ori = file[group_path_ff_params][neuron][:][0]
                preferred_sf  = file[group_path_ff_params][neuron][:][1]
                x_pix         = file[group_path_pos][neuron][:][0]
                y_pix         = file[group_path_pos][neuron][:][1]
                GSF           = file[group_path_st_results][neuron][1]
                AMRF          = file[group_path_st_results][neuron][3]

                ## 1) We set 'do_surround_fixed_center' to False in order to have the center orientation tuning curve

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
                            device=device,
                            center_shift=None,
                            do_surround_fixed_center=False,
                             
                            save_panel=True,  # only once per neuron, optional
                        neuron=neuron,
                        panel_tag="debug_center",
                        )

                ## 2-3)
                ## Perform for three center orientation
                center_shifts = np.array([-(np.pi/4),0,np.pi/4])
                center_oris = preferred_ori + center_shifts

                ## This will be where the results will be saved, every row is a different center orientation (-45,0,+45 degrees), every column correspond to a surround orientation
                surround_tuning_curves = torch.zeros(len(center_oris), len(ori_shifts))

                ## This will be where the most suppressing surround orientation will be saved. 
                most_suppr_surr = torch.zeros(len(center_oris))


                for i, center_shift in enumerate(center_shifts):

                    ## Set 'do_surround_fixed_center' to True

                    surround_tuning_curve = get_orientation_tuning_curves_all_phases(
                        single_model=single_model,
                        preferred_ori=preferred_ori,
                        preferred_sf=preferred_sf,
                        x_pix=x_pix, y_pix=y_pix,
                        GSF=GSF, AMRF=AMRF,
                        phases=phases, ori_shifts=ori_shifts,
                        contrast=contrast,
                        pixel_min=pixel_min, pixel_max=pixel_max,
                        size=size, img_res=img_res,
                        device=device,
                        center_shift=center_shift,
                        do_surround_fixed_center=True,
                        save_panel=(center_shift == 0),  # only once per neuron, optional
                        neuron=neuron,
                        panel_tag="debug_surround",
                    )

                    surround_tuning_curves[i] = surround_tuning_curve

                    ## 4) Select the values for a orientation shift >= -90 and <= 90 degrees

                    sub_array_id = np.where(np.abs(ori_shifts) < np.pi/2 + 0.01 )[0]
                    sub_ori = np.copy(ori_shifts[sub_array_id])
                    sub_tuning_curve = np.copy(surround_tuning_curve[sub_array_id])

                    ## Get the index of the most suppressing contrast orientation
                    argmin = np.argmin(sub_tuning_curve)

                    ## Get the corresponding orientation  
                    most_suppr_surr[i] = (sub_ori[argmin])

                ## Save every result in the HDF5 file 
                subgroup_curves_center.create_dataset(name=neuron, data=center_tuning_curve)
                subgroup_curves_surround.create_dataset(name=neuron, data=surround_tuning_curves)
                subgroup_results.create_dataset(name=neuron, data=most_suppr_surr)
