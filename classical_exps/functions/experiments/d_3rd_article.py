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
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------  #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1     #####
#####   -------------------------------------------------------------------  #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009      #####
##################################################################################


def black_white_preference_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    dot_size_in_pixels=5,
    contrast = 1, 
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    seed = 42
    ):  
    ''' This function aims to find the Signal Noise Ratio (SNR) for black and white dot stimuli for the selected neurons :

            - 1) It creates the stimulations for every possible dot position
            - 2) It gets the response of the neuron of the model to the stimulations
            - 3) It creates a "position-response" image (matrix) where every pixel represents the summed response of the neuron to a dot at this position
            - 4) It creates a noise image similarly to 3) but the response is being shuffled
            - 5) Performs 1-4 for black and white stimuli and computes the Signal Noise Ratio as the variance of the position-response matrix divided by the variance of the noise
            - 6) Saves the position-response images, the noise images and the SNR in three subgroups in the HDF5 file

    
    It saves the data with this architecture : 


                                                _________ SubGroup ../position_response_img   --> neuron datasets
                                                |
        Group /black_white_preference   ________|________ SubGroup ../noise_img               --> neuron datasets
                                                |
                                                |________ SubGroup ../results                 --> neuron datasets
    Prerequisite :

        - None

    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - seed               : Random seed for reproducibility
        - others             : explained in other functions

    Outputs :

        - datasets in ../position_response_img  : A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                  every pixel of the images correspond to the summed response of the neuron to this pixel

        - datasets in ../noise_img              :  A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                   every pixel of the images correspond to the summed shuffled response of the neuron to this pixel (a shuffled response means that the response assigned to every dot can now be the response to another dot)


        - datasets in ../results                : An array containing the Signal noise ratios values and the log10(SNRw / SNRb)
                                         format : [SNR_b, SNR_w, logSNRwb]

    '''

    np.random.seed(seed)

    print(' > Black or white preference experiment')

    ## Groups to fill
    group_path = "/black_white_preference"
    subgroup_path_resp_img  = group_path + "/position_response_img"
    subgroup_path_noise_img = group_path + "/noise_img"
    subgroup_path_results   = group_path + "/results"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## Initialize the Group and the subgroups
    args_str = f"dot_size_in_pixels={dot_size_in_pixels}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_resp_img, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_noise_img, group_args_str=args_str)


    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    all_neurons_model.eval()
    

    ## Generate the objects to fill
    sum_dot_pos   = torch.zeros((img_res[1], img_res[0]), dtype=int).to(device)
    black_stimuli = []
    white_stimuli = []
    resp_b = []
    resp_w = []

    with torch.no_grad():
 
        ## Perform the dot stimulation
        for y in range(img_res[0] - dot_size_in_pixels + 1):
            
            for x in range(img_res[1] - dot_size_in_pixels + 1) :
            
                ## Create an image with values = 0
                black_stim = np.zeros(img_res, dtype=int)
                white_stim = np.zeros(img_res, dtype=int)

                ## Save the dot position
                sum_dot_pos[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] += 1

                ## Do one black and one white stimulus
                black_value = -1
                white_value = 1

                ## Create the dots
                black_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = black_value*contrast
                white_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = white_value*contrast

                ## Save the dots
                black_stimuli.append(torch.Tensor(black_stim))
                white_stimuli.append(torch.Tensor(white_stim))

        ## Convert the list of tensors to a unique tensor
        black_stimuli = torch.stack(black_stimuli)
        white_stimuli = torch.stack(white_stimuli)

        num_dots = len(black_stimuli)

        ## For every input get the responses of the all_neurons_model
        ## Rescale the image to pixel_min and pixel_max
        model_input_b = rescale(black_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, 93,93).to(device)
        model_input_w = rescale(white_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, 93,93).to(device)

        ## Get the responses
        resp_b = all_neurons_model(model_input_b)
        resp_w = all_neurons_model(model_input_w)
        
        ## Tensor 1 is a tensor containing every dot position of the images (1 if the dot is here, else 0). shape = (num_dots * img_res) 
        ## Tensor 2 is a tensor containing the responses of every neuron to the dot images,  shape = (num_dots * nNeuron) #nNeuron = len(output_of_all_neurons_model)
        ## This function creates for every dot (first dimension = 'b') the following tensor :
        ## 'nxy' Basically, for each neuron, resp_to_the_image * dot_position_matrix_in_image
        ## It can be visualised as an array of matrices. Each matrix contains zeros where there isn't a dot and the value of the response where the dot is
        ## Then it sums this Tensor for every dot image that was presented
        ## The result image is what is called 'position response image' and baically contains the summed response of the dots at every position (pixel)
        all_position_resp_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (resp_b).to(device)
                            )
        
        all_position_resp_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),
                            (resp_w).to(device)
                            )
        
        ## Shuffle the responses for each neuron to create noise
        shuffle_resp_b = np.copy(resp_b.cpu())
        shuffle_resp_w = np.copy(resp_w.cpu())
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_b)
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_w)


        ## Put everything back in a tensor on the correct device
        shuffle_resp_b = torch.Tensor(shuffle_resp_b).to(device)
        shuffle_resp_w = torch.Tensor(shuffle_resp_w).to(device)
        
        all_noise_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (shuffle_resp_b).to(device)
                            )
        
        all_noise_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),  
                            (shuffle_resp_w).to(device)
                            )
        
    ## Fill the HDF5 file 
    with h5py.File(h5_file, 'a') as f :

        ## Save everything in a dictionnary where every key is a neuron index
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            if neuron not in f[subgroup_path_results] :
                
                ## Create datasets
                pos_resp_imgs = torch.stack([all_position_resp_b[neuron_id], all_position_resp_w[neuron_id]]).to('cpu')
                noises        = torch.stack([all_noise_b[neuron_id], all_noise_w[neuron_id]]).to('cpu')
        
                SNR_b = torch.var(pos_resp_imgs[0]).item() / torch.var(noises[0]).item()
                SNR_w = torch.var(pos_resp_imgs[1]).item() / torch.var(noises[1]).item()

                logSNRwb = np.log10(SNR_w/SNR_b)
                
                f[subgroup_path_resp_img].create_dataset(name=neuron, data=pos_resp_imgs)
                f[subgroup_path_noise_img].create_dataset(name=neuron, data=noises)
                f[subgroup_path_results].create_dataset(name=neuron, data=[SNR_b, SNR_w, logSNRwb])

