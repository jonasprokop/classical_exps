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
import time
import cv2
import matplotlib.pyplot as plt

import re
import hashlib
from pathlib import Path

from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.h5_io import save_selected_lowlevel_family_stats_to_h5, save_texture_noise_image_statistics_to_h5
from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.tools import load_imgs
from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.audit import save_one_stimulus_pair_per_family



###############################################################################
#####  PART V : Experiments for the fourth article, Freeman et al., 2013  #####
#####   ---------------------------------------------------------------   #####
#####             A functional and perceptual signature of the            #####
#####                    second visual area in primates                   #####
#####   ---------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1038/nn.3402                  #####
###############################################################################


def texture_noise_response_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,
    directory_imgs,
    overwrite = False,
    contrast = 1,
    pixel_min = -1.7876, 
    pixel_max =  2.1919, 
    num_samples = 15,
    img_res = [93,93],
    device = None,
    save_stimulus_samples=True,
    stimulus_sample_strategy="random",
    stimulus_scale_mode="global",

):
    ''' This function aims to get the responses of a model to texture and noise images :

            1) Take a folder directory and load every images into two tensors, one for the textures one fore the noises.
            2) Get the responses of every image
            3) Save the responses in a HDF5 file

        It saves the data with this architecture :         

                                                    _________ SubGroup ../texture   --> neuron datasets
                                                    |                        
            Group /texture_noise_response    _______| 
                                                    |
                                                    |________ SubGroup ../noise     --> neuron datasets
                                                  
        Prerequisite :

            - None

        Arguments : 

            - h5_file      : Path to the HDF5 file
            - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
            - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
            - overwrite         : If set to True, will erase the data in the group "full_field_params" and then fill it again. 
            - other             : See 'load_imgs' description

        Outputs :

            - datasets in ../texture       : A matrix containing the responses of a neuron to the texture images. Each row correspond to a texture family, each column correspond to a sample

            - datasets in ../noise         : Same for noise images

    '''
    
    print(' > Texture and noise response experiment')

    ## Groups to fill
    group_path          = "/texture_noise_response"
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Clear the group if requested    
    if overwrite : 
        clear_group(h5_file,group_path)


    ## Initialize the Group and subgroup
    args_str = f"contrast={contrast}/pixel_min={pixel_min}/pixel_max{pixel_min}/num_samples={num_samples}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_tex_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_noise_path, group_args_str=args_str)

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)

    ## Evaluation mode
    all_neurons_model.eval()

    ## Load the images
    tex_imgs, noise_imgs, dict_fam = load_imgs(
        directory_imgs=directory_imgs,
        target_res=img_res,
        contrast=contrast,
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        num_samples=num_samples,
        device=device,
        crop_mode="center",
        seed=0,
        expected_num_families=15,
    )
    tex_resp   = []
    noise_resp = []


    if save_stimulus_samples:
        save_one_stimulus_pair_per_family(
            tex_imgs=tex_imgs,
            noise_imgs=noise_imgs,
            dict_fam=dict_fam,
            save_dir="/project/results/texture_noise_response/stimulus_samples/",
            sample_strategy=stimulus_sample_strategy,
            seed=42,
            scale_mode=stimulus_scale_mode,
            save_npy=True,
            save_pair_png=True,
            save_montage=True,
        )

    save_texture_noise_image_statistics_to_h5(
        h5_file=h5_file,
        tex_imgs=tex_imgs,
        noise_imgs=noise_imgs,
        dict_fam=dict_fam,
        group_path="/texture_noise_response/image_statistics",
        overwrite=True,
    )

    save_selected_lowlevel_family_stats_to_h5(
        h5_file=h5_file,
        tex_imgs=tex_imgs,
        noise_imgs=noise_imgs,
        dict_fam=dict_fam,
        group_path="/texture_noise_response/lowlevel_family_stats",
        overwrite=True,
    )

    ## Get the model's responses
    with torch.no_grad():

        ## Get the responses
        for i in range(len(tex_imgs)) :
            
            tex_resp.append(all_neurons_model(tex_imgs[i]))
            noise_resp.append(all_neurons_model(noise_imgs[i]))

    ## Convert the lists of tensors into single tensors
    tex_resp   = torch.stack(tex_resp).cpu()
    noise_resp = torch.stack(noise_resp).cpu()

    ## Fill the HDF5 file
    with h5py.File(h5_file, 'a') as f :
            
        ## Access the groups
        subgroup_tex = f[subgroup_tex_path]
        subgroup_noise = f[subgroup_noise_path]

        ## Add a description for the families
        description = ''
        for family in dict_fam.keys() :
            description += f'{family}-'
        description = description[:-1]
            

        f[group_path].attrs["description"]  = description
        subgroup_tex.attrs["description"]   = description
        subgroup_noise.attrs["description"] = description

        for neuron_id in tqdm(neuron_ids) : 
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_noise :

                neuron_results_tex   = tex_resp[:,:,neuron_id]
                neuron_results_noise = noise_resp[:,:,neuron_id]

                subgroup_tex.create_dataset(name=neuron, data=neuron_results_tex)
                subgroup_noise.create_dataset(name=neuron, data=neuron_results_noise)
