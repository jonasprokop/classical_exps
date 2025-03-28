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
import time
import cv2
import matplotlib.pyplot as plt
from classical_exps.functions.experiments.b_1st_article import get_size_tuning_curves, get_GSF_surround_AMRF


###############################################################################
#####  PART VII :   #####
#####   ---------------------------------------------------------------   #####
#####          #####
#####                                     #####
#####   ---------------------------------------------------------------   #####
#####      DOI :           #####
###############################################################################

def get_low_and_high_contrast_contrast(
    single_model, 
    preferred_ori = 0, 
    preferred_sf = 0, 
    preferred_phase = 0,
    contrasts =  np.linspace(0, 1, 40, endpoint=True),         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,   
    x_pix = 93/2,
    y_pix = 93/2,  
    mRF_radius = 93/4, 
    max_response = 1,     
    ):

    gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
    single_model.to(device)
    single_model.eval()
    gray_resp = single_model(gray_stim)

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    reponses = {}
    for contrast in contrasts:
        ## Create full-field grating
        grating = torch.Tensor(imagen.SineGrating(
                            orientation=preferred_ori,
                            frequency=preferred_sf,
                            phase=preferred_phase,
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            offset=-1,
                            scale=2,  
                            xdensity=img_res[1]/size[1],
                            ydensity=img_res[0]/size[0], 
                            x=get_offset_in_degr(x_pix, img_res[1], size[1]), 
                            y=-get_offset_in_degr(y_pix, img_res[0], size[0])
                        )())

        ## Create mRF mask
        mRF_mask = torch.Tensor(imagen.Disk(
                            smoothing=0.0,
                            size=mRF_radius * 2.0,  
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=img_res[1]/size[1],
                            ydensity=img_res[0]/size[0], 
                            x=get_offset_in_degr(x_pix, img_res[1], size[1]), 
                            y=-get_offset_in_degr(y_pix, img_res[0], size[0])
                        )())
        
        # plt.imsave(f'/project/results/facilitation/mask{contrast}.png', 
        # mRF_mask.squeeze(), cmap='gray', format='png')

        # plt.imsave(f'/project/results/facilitation/grating{contrast}.png', 
        # grating.squeeze(), cmap='gray', format='png')


        masked_grating = grating * mRF_mask

        masked_grating = torch.tensor(masked_grating, dtype=torch.float32)
        masked_grating = masked_grating.reshape(1,*img_res).to(device)

        ## Rescale because the output of imagen has values from 0 to 1 and we want 
        ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
        masked_grating = rescale(masked_grating, 0, 1, -1, 1)*contrast
        masked_grating = rescale(masked_grating, -1, 1, pixel_min, pixel_max)

        # plt.imsave(f'/project/results/facilitation/masked_grating{contrast}.png', 
        # masked_grating.cpu().squeeze(), cmap='gray', format='png')

        resp = single_model(masked_grating)

        reponses[contrast] = resp

    low_contrast = 0.04
    low_contrast_response = 0
    high_contrast = 0.8
    high_contrast_response = 0

    for contrast, response in reponses.items():

        if response < 0.9 * max_response and response > high_contrast_response and 0.5 < contrast < 0.8:
            high_contrast = contrast
            high_contrast_response = response
        
        elif response > 0.5 * max_response and response > low_contrast_response and response > gray_resp and 0.04 < contrast < 0.3:
            low_contrast = contrast
            low_contrast_response = response

    return low_contrast, high_contrast


def get_size_tuning_surround_annulus_curve(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    size = 2.67,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    img_res = [93,93],
    neg_val = False,     #TODO never used
    GSF_low=0.5,
    GSF_high=1,
    contrast_center=0.8,
    contrast_far_surround=0.4,
    gray_resp=1,
    radii_list_annular_tunning_curve= np.linspace(0,2,100, endpoint=False),
    neuron=1,
    ):
    
    '''
    '''
    outer_annulus_radii = radii_list_annular_tunning_curve[radii_list_annular_tunning_curve > GSF_low][::-1]

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## To save the responses
    circular_tuning_curve   = torch.zeros(len(outer_annulus_radii))

    with torch.no_grad():

        for i, outer_annulus_radius in enumerate(outer_annulus_radii):

            ## Make the circular stimulus - the inner circle grating
            grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=GSF_high*2.0),
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            ## Mask the circular stimulus - the inner circle mask
            center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                                size=GSF_high*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            # surround grating, created for separete control of surround contrast
            surround = torch.Tensor(imagen.SineGrating(
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())

            surround_mask = (torch.Tensor(imagen.Disk(
                                smoothing=0.0, 
                                size=outer_annulus_radius * 2.0,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
            

            # print(f"GSF_low={GSF_low}, GSF_high={GSF_high}, Contrasts={contrast_center, contrast_far_surround}, - radius={radius} - Annular_thickness={annular_thickness} ")

            directory = "/project/results/facilitation/grating_centers" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_grating_center{outer_annulus_radius}.png", 
                grating_center.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/center_masks" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_center_mask{outer_annulus_radius}.png", 
                center_mask.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/surround_masks" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_surround_mask{outer_annulus_radius}.png", 
                surround_mask.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/surround" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_surround{outer_annulus_radius}.png", 
                surround.squeeze(), cmap='gray', format='png')

            # Combine grating_center with background 
            grating = grating_center * center_mask * contrast_center + surround * surround_mask * contrast_far_surround
            
            directory = "/project/results/facilitation/gratings"  + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_grating_array_{outer_annulus_radius}.png", 
                grating.squeeze(), cmap='gray', format='png')
            
            
            grating = torch.tensor(grating, dtype=torch.float32)
            grating = grating.reshape(1,*img_res).to(device)

            grating = rescale(grating, -1, 1, pixel_min, pixel_max)

            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(single_model(grating) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = single_model(grating) - gray_resp

    return outer_annulus_radii, circular_tuning_curve    

def get_center_alone_stimulus(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    size = 2.67,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    img_res = [93,93],
    GSF_high=1,
    contrast_center=0.8,
    neuron=1,
    gray_resp=1
    ):

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2


    ## Make the circular stimulus - the inner circle grating
    grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=GSF_high*2.0),
                        orientation=preferred_ori,
                        frequency=preferred_sf,
                        phase=preferred_phase,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        offset = -1,
                        scale=2,  
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
    ## Mask the circular stimulus - the inner circle mask
    center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                        size=GSF_high*2.0, 
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())

                
    grating = grating_center * center_mask * contrast_center


    directory = "/project/results/facilitation/center_only"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.imsave(directory + "center_only.png", 
        grating.squeeze(), cmap='gray', format='png')
    
    grating = torch.tensor(grating, dtype=torch.float32)
    grating = grating.reshape(1,*img_res).to(device)

    grating = rescale(grating, -1, 1, pixel_min, pixel_max)

    response = single_model(grating)

    return response - gray_resp

# def get_surround_alone_stimulus(
#     single_model,
#     x_pix,
#     y_pix,
#     preferred_ori,
#     preferred_sf,
#     preferred_phase,
#     size = 2.67,
#     pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
#     pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
#     device = None,
#     img_res = [93,93],
#     GSF_high=1,
#     contrast_center=0.8,
#     ):

#             ## Convert the size to the right shape ( [size_height, size_width] )
#     if type(size) is int or type(size) is float: 
#         size = [size]*2


#     ## Make the circular stimulus - the inner circle grating
#     grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=GSF_high*2.0),
#                         orientation=preferred_ori,
#                         frequency=preferred_sf,
#                         phase=preferred_phase,
#                         bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
#                         offset = -1,
#                         scale=2,  
#                         xdensity=img_res[1]/size[1],
#                         ydensity=img_res[0]/size[0], 
#                         x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
#                         y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
#     ## Mask the circular stimulus - the inner circle mask
#     center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
#                         size=GSF_high*2.0, 
#                         bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
#                         xdensity=img_res[1]/size[1],
#                         ydensity=img_res[0]/size[0], 
#                         x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
#                         y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
#     grating = grating_center * center_mask * contrast_center

    
#     grating = torch.tensor(grating, dtype=torch.float32)
#     grating = grating.reshape(1,*img_res).to(device)

#     grating = rescale(grating, -1, 1, pixel_min, pixel_max)

#     response = single_model(grating)

#     return response.item()


def get_surround_contrast_facilitation(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    img_res=[93, 93],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
    percentage_of_guassian_energy=0.95,
    contrasts = np.arange(0, 0.89, 0.02),
    radii_list_annular_tunning_curve= np.linspace(0,2,100, endpoint=False),
    radii_list_contrast_tunning_curve = np.logspace(-2,np.log10(2),40),
    save_minimal_receptive_fields=True,
):
    '''
    '''
    print(' > Get preferred orientation contrast stimulation')

    # Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    # Groups to fill
    group_facilitation = "/surround_contrast_group_facilitation" 
    subgroup_facilitation_st_results_high_path = group_facilitation + "/results/high"
    subgroup_facilitation_st_results_low_path  = group_facilitation + "/results/low"
    subgroup_facilitation_st_curves_high_path  = group_facilitation + "/curves/high"
    subgroup_facilitation_st_curves_low_path  = group_facilitation +"/curves/low"

    subgroup_facilitation_st_curves_HH_path  = group_facilitation +"/curves/HH"
    subgroup_facilitation_st_curves_LH_path  = group_facilitation +"/curves/LH"
    subgroup_facilitation_st_curves_LL_path  = group_facilitation +"/curves/LL"

    subgroup_facilitation_st_results_HH_path  = group_facilitation +"/results/HH"
    subgroup_facilitation_st_results_LH_path  = group_facilitation +"/results/LH"
    subgroup_facilitation_st_results_LL_path  = group_facilitation +"/results/LL"

    minimal_receptive_fields_path = group_facilitation + "/minimal_receptive_fields"
    
    ## Clear the group if requested
    #if overwrite : 
    clear_group(h5_file,group_facilitation)

    args_str = f"percentage_of_guassian_energy={percentage_of_guassian_energy}/contrasts={contrasts}/radii={radii_list_contrast_tunning_curve}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_facilitation, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_high_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_low_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_high_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_low_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_HH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_LH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_LL_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_HH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_LH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_LL_path, group_args_str=args_str)


    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    with h5py.File(h5_file, "a") as file:

        ## Access the groups 
        subgroup_facilitation_st_results_high = file[subgroup_facilitation_st_results_high_path]
        subgroup_facilitation_st_results_low   = file[subgroup_facilitation_st_results_low_path]
        subgroup_facilitation_st_curves_high = file[subgroup_facilitation_st_curves_high_path]
        subgroup_facilitation_st_curves_low = file[subgroup_facilitation_st_curves_low_path]

        subgroup_facilitation_st_curves_HH = file[subgroup_facilitation_st_curves_HH_path]
        subgroup_facilitation_st_curves_LH = file[subgroup_facilitation_st_curves_LH_path]
        subgroup_facilitation_st_curves_LL = file[subgroup_facilitation_st_curves_LL_path]

        subgroup_facilitation_st_results_HH = file[subgroup_facilitation_st_results_HH_path]
        subgroup_facilitation_st_results_LH = file[subgroup_facilitation_st_results_LH_path]
        subgroup_facilitation_st_results_LL = file[subgroup_facilitation_st_results_LL_path]
        minimal_receptive_fields = file[minimal_receptive_fields_path]
        

        for neuron_id in tqdm(neuron_ids, desc="Applying Orientation Contrasts"):
            neuron = f"neuron_{neuron_id}"

            # Get the single_model of the neuron
            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            ## Create Gray stimulus to substract to the non gray stimuli
            gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
            gray_resp = single_model(gray_stim)

            try:

                # Get preffered params
                preferred_ori = file[group_path_ff_params][neuron][:][0]
                preferred_sf  = file[group_path_ff_params][neuron][:][1]
                preferred_phase = file[group_path_ff_params][neuron][:][2]
                max_response = file[group_path_ff_params][neuron][:][3]

            except:
                print(f"Loading values for preffered params for {neuron} has failed")
                continue

            try:
                # get center coordinates
                x_pix         = file[group_path_pos][neuron][:][0]
                y_pix         = file[group_path_pos][neuron][:][1]

                # get the sigmas for coordinates
                sigma_x_opt   = file[group_path_pos][neuron][:][2]
                sigma_y_opt   = file[group_path_pos][neuron][:][3]

            except:
                print(f"Loading values for center params for {neuron} has failed")
                continue
                

            try:
                mRF_radius = np.sqrt(-2 * np.log(1 - percentage_of_guassian_energy)) * np.sqrt((sigma_x_opt**2 * sigma_y_opt**2)/2)
                if np.isnan(mRF_radius):
                    print(f"For {neuron} the mRF_radius was not calculated")
                    continue
            except:
                    print(f"For {neuron} the mRF_radius was not calculated")
                    continue
            
            minimal_receptive_fields.create_dataset(name=neuron, data=mRF_radius)

                        
            if save_minimal_receptive_fields:
                mRF_mask = torch.Tensor(imagen.Disk(
                                    smoothing=0.0,
                                    size=mRF_radius * 2.0,  
                                    bounds=BoundingBox(points = ((-size/2, -size/2), (size/2, size/2))),
                                    xdensity=img_res[1]/size,
                                    ydensity=img_res[0]/size, 
                                    x=get_offset_in_degr(x_pix, img_res[1], size), 
                                    y=-get_offset_in_degr(y_pix, img_res[0], size)
                                )())
                
                directory = "/project/results/facilitation/minimal_receptive_field"  + "/" + neuron + "/"
                os.makedirs(directory, exist_ok=True)
                plt.imsave(directory + f"{neuron}_minimal_receptive_field_{mRF_radius:.2f}.png", 
                    mRF_mask.squeeze(), cmap='gray', format='png', )

            # get low and high contrast values for the neuron
            low_contrast, high_contrast = get_low_and_high_contrast_contrast(
                                            single_model, 
                                            preferred_ori = preferred_ori, 
                                            preferred_sf = preferred_sf, 
                                            preferred_phase = preferred_phase,
                                            contrasts = contrasts,         
                                            img_res = img_res, 
                                            pixel_min = pixel_min, 
                                            pixel_max =  pixel_max,
                                            device = device,
                                            size = size,   
                                            x_pix = x_pix,
                                            y_pix = y_pix,  
                                            mRF_radius = mRF_radius, 
                                            max_response = max_response,     
                                        )

            # get patch-size tunning curve 
            low_contrast_circular_size_tunning_curve = get_size_tuning_curves(
                                                single_model,
                                                x_pix,
                                                y_pix,
                                                preferred_ori,
                                                preferred_sf, 
                                                preferred_phase,
                                                radii = radii_list_contrast_tunning_curve,
                                                contrast = low_contrast,
                                                pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
                                                pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
                                                device = None,
                                                size = 2.67,
                                                img_res = [93,93],
                                                neg_val = True,     #TODO never used
                                                compute_annular = False
                                            )
            high_contrast_circular_size_tunning_curve = get_size_tuning_curves(
                                                single_model,
                                                x_pix,
                                                y_pix,
                                                preferred_ori,
                                                preferred_sf, 
                                                preferred_phase,
                                                radii = radii_list_contrast_tunning_curve,
                                                contrast = high_contrast,
                                                pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
                                                pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
                                                device = None,
                                                size = 2.67,
                                                img_res = [93,93],
                                                neg_val = True,     #TODO never used
                                                compute_annular = False
                                            )
                    
            ## Save the tuning curve in the subgroup '/size_tuning/curves
            subgroup_facilitation_st_curves_low.create_dataset(name=neuron, data=[radii_list_contrast_tunning_curve, low_contrast_circular_size_tunning_curve.cpu()])
            subgroup_facilitation_st_curves_high.create_dataset(name=neuron, data=[radii_list_contrast_tunning_curve, high_contrast_circular_size_tunning_curve.cpu()])

            subgroup_facilitation_st_results_low.create_dataset(name=neuron, data=low_contrast)
            subgroup_facilitation_st_results_high.create_dataset(name=neuron, data=high_contrast)

            ## Perform some analysis
            GSF_low, surround_extent_low, SI_low, Ropt_low, _ = get_GSF_surround_AMRF(
                radii = radii_list_contrast_tunning_curve,
                circular_tuning_curve = low_contrast_circular_size_tunning_curve,
                annular_tuning_curve = None
                )
            
            GSF_high, surround_extent_high, SI_high, Ropt_high, _ = get_GSF_surround_AMRF(
                radii = radii_list_contrast_tunning_curve,
                circular_tuning_curve = high_contrast_circular_size_tunning_curve,
                annular_tuning_curve = None
                )
            
            if GSF_low <= 1.e-9 or GSF_high <= 1.e-9 or GSF_high >= GSF_low:
                continue

            outer_annulus_radii_HH, size_tuning_surround_annulus_curve_HH = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=high_contrast,
                    contrast_far_surround=high_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron,
                    )
            
            outer_annulus_radii_LH, size_tuning_surround_annulus_curve_LH = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    contrast_far_surround=high_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron
                    )
            
            outer_annulus_radii_LL, size_tuning_surround_annulus_curve_LL = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    contrast_far_surround=low_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron
                    )
            
            
            response_center_alone_high_contrast = get_center_alone_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=high_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )

            response_center_alone_low_contrast = get_center_alone_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )

            gray_resp = gray_resp.item()
            response_center_alone_high_contrast = response_center_alone_high_contrast.item()
            response_center_alone_low_contrast = response_center_alone_low_contrast.item()

            subgroup_facilitation_st_curves_HH.create_dataset(name=neuron, data=[outer_annulus_radii_HH, size_tuning_surround_annulus_curve_HH.cpu()])
            subgroup_facilitation_st_curves_LH.create_dataset(name=neuron, data=[outer_annulus_radii_LH, size_tuning_surround_annulus_curve_LH.cpu()])
            subgroup_facilitation_st_curves_LL.create_dataset(name=neuron, data=[outer_annulus_radii_LL, size_tuning_surround_annulus_curve_LL.cpu()])

            subgroup_facilitation_st_results_HH.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high, response_center_alone_high_contrast])
            subgroup_facilitation_st_results_LH.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high,  response_center_alone_low_contrast])
            subgroup_facilitation_st_results_LL.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high, response_center_alone_low_contrast])

                                            
            print(f"for neuron {neuron} the calculation was a success")                                                





            






        
        


