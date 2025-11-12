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

##################################################################################
#####  PART II : Experiments for the first article, Cavanaugh et al., 2002  #####
#####   -----------------------------------------------------------------   #####
#####        Nature and Interaction of Signals From the Receptive Field     #####
#####               Center and Surround in Macaque V1 Neurons               #####
#####   -----------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001              #####
##################################################################################


def get_size_tuning_curves(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    radii = np.logspace(-2,np.log10(2),40) ,
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,     #TODO never used
    compute_annular = True,
    neuron = "",
    contrast_keyword="",
    ):
    
    ''' This function is a size tuning function that makes the size of circular and annular grating images vary 
        with their prefered parameters fixed. The difference between the response of the model for each image and the response for a gray image 
        is calculated and saved into an array of the same size of the argument "radii". You can afterwards plot these arrays
        to visualise the effect of the radii on the model's response. If neg_val is set to 'False', the negative values
        are set to 0.
        

        NB : We substract the response to gray image to the response of stimuli because it is necessary for the rest of our analysis, it normalises the values.
        NB2: What is called here 'annular' is actually not a ring since there is no outer boundary, the 'ring' fills the outer space of the image. 

        Arguments : 

            - single_model    : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix           : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
            - y_pix           : The y coordinates   ""        ""        ""
            - preferred_ori   : The preferred orientation (the one that elicited the greatest response)
            - preferred_sf    : The preferred spatial frequency
            - preferred_phase : The preferred phase
            - radii           : An array containing every radius size to test, NB : here the radius is from the center to the inner edge of the ring
            - Thickness       : The thickness of the annular item (half of the full thickness)
            - contrast        : The value that will multiply the image's values 
            - pixel_min       : Value of the minimal pixel that will serve as the black reference
            - pixel_max       : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device          : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size            : The size of the image in terms of visual field degrees
            - img_res         : Resolution of the image in term of pixels shape : [pix_y, pix_x]
            - neg_val         : If set to 'False', the outputs won't include negative values but will replace them by 0.
            - compute_annular : If set to 'False', the function will only perform size tuning on a circular grating image

        Outputs : 

            - circular_tuning_curve  : An array containing the responses for every center stimulus minus the response to gray image
            
            - If compute_annular == True :
                
                - annular_tuning_curve   : An array containing the responses for every surround stimulus minus the response to gray image
    '''

    grating_images_for_panel = []  
    grating_radii_for_panel = []  

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
    circular_tuning_curve   = torch.zeros(len(radii))
    annular_tuning_curve = torch.zeros(len(radii))


    with torch.no_grad():

        ## Create Gray stimulus to substract to the non gray stimuli
        gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = single_model(gray_stim)

        for i, radius in enumerate(radii) :
            
            ## Make the circular stimulus
            grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=radius*2.0),
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
            
            ## Mask the circular stimulus 
            center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                                size=radius*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            
            ## Add the gray background to the circular stimulus
            grating_center = grating_center * center_mask

            ## Change the contrast for the circle
            grating_center *= contrast
        

            ## Convert to the right shape for the model
            # fixed version
            grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

            grating_images_for_panel.append(grating_center.cpu().numpy().squeeze())
            grating_radii_for_panel.append(radius)

            
            # ## Convert to the right shape for the model
            # grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)


            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(single_model(grating_center) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = single_model(grating_center) - gray_resp

            ## Same for the annular stimulus
            if compute_annular : 
                ## Make the annular stimulus
                ## Create a grating background
                grating_background = torch.Tensor(imagen.SineGrating(
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
                

                ## Create the inner edge of the ring
                inner_edge = (torch.Tensor(imagen.Disk(
                                    smoothing=0.0, 
                                    size=(radius)*2.0,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
                grating_ring = grating_background * inner_edge

                ## Change the contrast for the ring
                grating_ring   *= contrast
                
                ## Convert to the right shape for the model 
                grating_ring = rescale(grating_ring,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)
                
                ## Save responses for this radius, substract the gray response
                if neg_val == False :
                    ## Avoid negative values
                    annular_tuning_curve[i]  = torch.maximum(single_model(grating_ring) - gray_resp,torch.tensor(0))   

                else : 
                    ## Allow negative values
                    annular_tuning_curve[i]  = single_model(grating_ring) - gray_resp


    # radii and responses arrays
    radii = np.array(radii)
    responses = np.array(circular_tuning_curve.cpu())

    # Define your "informative" range
    min_radius = 0.1 
    max_radius = 2.0

    # Filter
    valid_indices = np.where((radii >= min_radius) & (radii <= max_radius))[0]

    # Sample n points evenly from the filtered indices
    n_samples = 12
    if len(valid_indices) > n_samples:
        sampled_indices = valid_indices[np.linspace(0, len(valid_indices)-1, n_samples, dtype=int)]
    else:
        sampled_indices = valid_indices

    # Pick radii and images
    radii_sampled = radii[sampled_indices]
    images_sampled = [grating_images_for_panel[i] for i in sampled_indices]

    plot_size_tuning_gratings(
        neuron=f"{neuron}", 
        grating_images=images_sampled,
        radii=radii_sampled,
        n_samples=n_samples,
        contrast_keyword=contrast_keyword,
        contrast_level=contrast,
    )

    if compute_annular :
        return circular_tuning_curve, annular_tuning_curve
    
    else :
        return circular_tuning_curve
    

def get_GSF_surround_AMRF(
    radii,
    circular_tuning_curve,
    annular_tuning_curve = None
    ): 

    ''' This function takes the circular and annular tuning curves outputs of the function "get_size_tuning_curves".
        For the circular stimuli : 
            -It finds the Grating Summation Field (GSF) defined as the "diameter of the smallest stimulus that elicited at least 95% of the
            neuron’s maximum response".
            -It finds the surround extent radius defined as the "inhibitory surround extent as the diameter of the smallest stimulus 
            for which the neuron’s response was reduced to within 5% of its asymptotic value for the largest gratings".
            -It computes the supression index defined as "(Ropt - Rsupp) / Ropt", where Ropt is the response at the GSF and Rsupp is the asymptotic response.
        For the annular stimuli :
            -It finds the Annular Minimum Response Field (AMRF) defined as "the point at which the response to the annular stimulus reached
            a value of at most 5% of the neuron’s maximum response to a circular patch of grating."
        
        Optionally, if no annular_tuning_curve is given, this function will not returns the AMRF

        Arguments : 

            - radii                  : The array that contains the radii that were used to obtain circular_tuning_curve and annular_tuning_curve
            - circular_tuning_curve  : An array, the center stimulus size tuning curve
            - annular_tuning_curve   : An array, the surround stimulus size tuning curve

        Outputs : 

            - GSF             : Optimal circular radius that elicits approximately the maximal response
            - surround_extent : Optimal circular radius that elicits approximately the asymptotic response (suppression)
            - AMRF            : Optimal annular radius that elicits approximately the lowest response
            - Ropt            : The response of the model at the GSF for the circular stimuli
            - Rsupp           : The asymptotic response of the model for the circular stimuli

            NB : outputs are radii, not diameters
    '''
    
    ## Make sure the elements are tensors
    circular_tuning_curve = torch.as_tensor(circular_tuning_curve)
    if annular_tuning_curve is not None :
        annular_tuning_curve = torch.as_tensor(annular_tuning_curve)

    ## Avoid zeros
    GSF             = 1.e-9
    surround_extent = 1.e-9
    AMRF            = 1.e-9
    Ropt            = 0
    Rsupp           = circular_tuning_curve[-1] #The last response = asymptotic value

    ## Select the thresholds 
    GSF_thresh   = (95 * torch.max(circular_tuning_curve)) / 100
    a = Rsupp - ((5 * Rsupp) /100)
    b = Rsupp + ((5 * Rsupp) /100)
    surround_thresh_min = min(a,b)
    surround_thresh_max = max(a,b)
    AMRF_thresh  = 5 * torch.max(circular_tuning_curve) / 100
    
    ## For the GSF
    for i, resp in enumerate(circular_tuning_curve):
        if resp > GSF_thresh :
            GSF = radii[i]
            Ropt = resp
            break

    ## For the surround extent
    for i, resp in enumerate(circular_tuning_curve):
        ## Avoid to take values with smaller radius
        if radii[i] > GSF :
            ## Assert that the response is decreasing
            if resp < Ropt :
                if resp >= surround_thresh_min and resp <= surround_thresh_max:
                    surround_extent = radii[i]
                    break
    
    if annular_tuning_curve != None :
        ## For the AMRF
        for i, resp in enumerate(annular_tuning_curve):
            if resp < AMRF_thresh:
                AMRF = radii[i]
                break
    
    
    SI = ((Ropt - Rsupp) / Ropt).item()


    if annular_tuning_curve is not None :
        return GSF, surround_extent, AMRF, SI, Ropt, Rsupp
    
    else :
        
        return GSF, surround_extent, SI, Ropt, Rsupp
    

def size_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False, 
    radii = np.logspace(-2,np.log10(2),40),
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
    ):

    ''' This function performs a size tuning experiment for the neurons accross multiple phases :
        (we have to compute accross every phase because changing the images radius shifts the phase, and we want to counter that)  

            1) For each phase this function :                                                              

                - Gets the tuning curves for two grating images, one circular and one annular (see 'get_size_tuning_curves')

                - Computes the Grating Summation Field (GSF) and other elements (see 'get_GSF_surround_AMRF')
            
            2) It then takes the results obtained with the phase that lead to the maximal response 

            3) Finally it saves the size tuning curves and the results for each neuron in two subgroups in the HDF5 file.

        It saves the data with this architecture : 


                                _________ SubGroup /size_tuning/curves   --> neuron datasets
                                |
        Group /size_tuning  ____|
                                |
                                |________ SubGroup /size_tuning/results  --> neuron datasets


        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments


        Arguments :

            - radii                     : An array containing every radius to test
            - others                    : Described in other functions
            - neg_val                   : If set to 'False', this will set the negative responses encountered to zero (it should not change the results, only the curves). Negative values occurs when a stimulus elicits less response than the baseline (gray screen)
            

        Outputs :

            - datasets in ../results       : An array containing the results of this analysis for a neuron 
                                             format = [preferred_phase, GSF, surround_extent, AMRF, SI]
            NB  : The Value of GSF, surround_extant and AMRF are radii size (in deg), not diameters

            - datasets in ../curves : A matrix containing the circular patch tuning curve in the first row
                                                                 the annular patch tuning curve in the second row
    '''
    
    print(' > Size tuning experiment')
    
  
    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    ## Groups to fill
    group_path_size_tuning      = "/size_tuning"
    subgroup_path_results       = group_path_size_tuning + "/results"
    subgroup_path_tuning_curves = group_path_size_tuning + "/curves"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_size_tuning)

    ## Initialize the Group and the subgroups
    args_str = f"radii={radii}/phases={phases}/contrast={contrast}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_size_tuning, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_tuning_curves, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos], neuron_ids=neuron_ids, group_args_str=args_str)


    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        subgroup_results  = file[subgroup_path_results]
        subgroup_tc       = file[subgroup_path_tuning_curves]
        group_size_tuning = file[group_path_size_tuning]
        group_ff_params   = file[group_path_ff_params]
        group_pos         = file[group_path_pos]

        ## For every neuron
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_results :

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id )

                ## Get the neuron's preferred parameters 
                max_ori = group_ff_params[neuron][:][0]
                max_sf  = group_ff_params[neuron][:][1]
                max_x   = group_pos[neuron][:][0]
                max_y   = group_pos[neuron][:][1]
                            
                ## Initialise the parameters to find
                GSF_preferred_phase = 0
                surround_extent_preferred_phase = 0
                AMRF_preferred_phase = 0
                SI_preferred_phase = 0

                Rmax = 0 

                for phase in phases :
                    ## Get the tuning curve for the current phase
                    circular_tuning_curve, annular_tuning_curve = get_size_tuning_curves(
                        single_model = single_model,
                        x_pix = max_x,
                        y_pix = max_y,
                        preferred_ori = max_ori,
                        preferred_sf = max_sf,
                        preferred_phase = phase,
                        radii = radii,
                        contrast = contrast,
                        pixel_min = pixel_min,
                        pixel_max = pixel_max,
                        device = device,
                        size = size,
                        img_res = img_res,
                        neg_val = neg_val,
                        compute_annular = True ## This function needs the annular tuning curve
                        )
                    
                    ## Perform some analysis
                    GSF, surround_extent, AMRF, SI, Ropt, _ = get_GSF_surround_AMRF(
                        radii = radii,
                        circular_tuning_curve = circular_tuning_curve,
                        annular_tuning_curve = annular_tuning_curve
                        )
                    
                    if Ropt>Rmax :
                        ## Results
                        preferred_phase = phase
                        GSF_preferred_phase = GSF
                        surround_extent_preferred_phase = surround_extent
                        AMRF_preferred_phase = AMRF
                        SI_preferred_phase = SI
                        Rmax = Ropt

                        ## Curves
                        circular_tc_pref = circular_tuning_curve
                        annular_tc_pref  = annular_tuning_curve



                ## Merge the circular and annular tuning curves together (row 1 = circular, row 2 = annular)
                both_curves = torch.stack((circular_tc_pref, annular_tc_pref))
 
                ## Save the tuning curve in the subgroup '/size_tuning/curves
                subgroup_tc.create_dataset(name=neuron, data=both_curves.cpu())

                ## Save the results ([preferred_phase, GSF, surround_extent, AMRF, SI])
                subgroup_results.create_dataset(name=neuron, data=[preferred_phase, GSF_preferred_phase, surround_extent_preferred_phase, AMRF_preferred_phase, SI_preferred_phase])


def get_contrast_response(
    single_model,
    x_pix,
    y_pix,
    center_radius,
    surround_radius,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    center_contrasts = np.logspace(-2,np.log10(1),18),  
    surround_contrasts = np.logspace(-2,np.log10(1),18),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
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
              
                ## Get the image in the right shape for the model
                image = get_center_surround_stimulus(center_radius=center_radius, center_ori=preferred_ori, center_sf=preferred_sf, center_phase=preferred_phase, center_contrast=center_contrast, surround_radius=surround_radius, surround_ori=preferred_ori, surround_sf=preferred_sf, surround_phase=preferred_phase, surround_contrast=surround_contrast, x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)

                ## Get the model's response 
                resp = single_model(image)

                ## Normalise the response
                resp_norm = resp - gray_resp

                ## Save the response
                contrast_resps_mat[i,j] = resp_norm

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
    neg_val = True
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

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos,group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)


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
                preferred_phase = group_st_results[neuron][:][0]
                GSF             = group_st_results[neuron][:][1]
                AMRF            = group_st_results[neuron][:][3]
                preferred_ori   = group_ff_params[neuron][:][0]
                preferred_sf    = group_ff_params[neuron][:][1]
                x_pix           = group_pos[neuron][:][0]
                y_pix           = group_pos[neuron][:][1]

                ## Avoid overlapping
                if GSF>AMRF :
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
                    preferred_phase = preferred_phase,
                    center_contrasts = center_contrasts,
                    surround_contrasts = surround_contrasts,
                    pixel_min = pixel_min,
                    pixel_max = pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = neg_val
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


def get_contrast_size_tuning_curve_all_phases(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.logspace(np.log10(0.06),np.log10(1),5),
    radii = np.logspace(-2,np.log10(2),40) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
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
                compute_annular = False ## This function does not need the annular tuning curve
                )

            ## Fill the corresponding rows of the matrices
            all_circular_curves[num_phase ,i ,:] = circular_curve

    ## Get the maximum response accross phases
    all_circular_curves = torch.max(all_circular_curves, dim=0).values
    
    return all_circular_curves


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
    neg_val = True
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

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_cst)

    ## Initialize the Group and the subgroups
    args_str = f"phases={phases}/contrasts={contrasts}/radii={radii}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_cst, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_curves, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos], neuron_ids=neuron_ids, group_args_str=args_str)

    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        group_cst                 = file[group_path_cst]
        subgroup_cst_results      = file[subgroup_path_cst_results]
        subgroup_cst_curves  = file[subgroup_path_cst_curves]
        group_ff_params     = file[group_path_ff_params]
        group_pos           = file[group_path_pos]

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
                
                cst_curves = get_contrast_size_tuning_curve_all_phases(
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
                    neg_val=neg_val
                    )


                ## For low contrast  
                low_contrast_curve = cst_curves[0]
                GSF_low,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
                ## For contrast =1
                high_contrast_curve = cst_curves[-1]
                GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)

                GSFs_ratio = GSF_low/GSF_high

                ## Save the curves
                subgroup_cst_curves.create_dataset(name=neuron, data=cst_curves.cpu())

                ## Save the results
                data = [GSFs_ratio]
                subgroup_cst_results.create_dataset(name=neuron, data=data)


def plot_size_tuning_gratings(
    neuron,
    grating_images,
    radii=None,           # optional, used for titles
    n_samples=12,         # number of images to show
    save_dir="/project/results/facilitation/classical_contrast_size_tunning_grating_panels/",
    contrast_keyword="",
    contrast_level=1.0
):
    """
    Show a grid of sampled gratings only, no curves.
    All gratings share the same color scale (vmin/vmax) for consistent background tone.
    Each grating gets its own colorbar.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(save_dir, exist_ok=True)

    # --- Sample evenly across available gratings ---
    n_total = len(grating_images)
    n_samples = min(n_samples, n_total)
    idxs = np.linspace(0, n_total - 1, n_samples, dtype=int)
    sampled_gratings = [grating_images[i] for i in idxs]
    sampled_radii = [radii[i] for i in idxs] if radii is not None else [None] * n_samples

    # --- Determine shared color scale ---
    all_min = min(img.min() for img in sampled_gratings)
    all_max = max(img.max() for img in sampled_gratings)

    # --- Grid layout ---
    n_rows = 3
    n_cols = int(np.ceil(n_samples / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    axes = axes.ravel()

    for i, (ax, img, r) in enumerate(zip(axes, sampled_gratings, sampled_radii)):
        im = ax.imshow(img, cmap='gray', origin='lower', vmin=all_min, vmax=all_max)
        title = f"R={r:.2f}" if r is not None else ""
        ax.set_title(title, fontsize=8)
        ax.axis('off')

        # --- individual colorbar ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, fraction=0.046, pad=0.04)

    # Hide any empty axes if n_samples < n_rows * n_cols
    for ax in axes[len(sampled_gratings):]:
        ax.axis("off")

    # --- Title and save ---
    fig.suptitle(
        f"Sampled size tuning – {neuron}, {contrast_keyword} contrast: {contrast_level*100:.0f}%",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f"{neuron}_{contrast_keyword}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved panel to {save_path}")