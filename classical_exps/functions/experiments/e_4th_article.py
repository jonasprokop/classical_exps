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

###############################################################################
#####  PART V : Experiments for the fourth article, Freeman et al., 2013  #####
#####   ---------------------------------------------------------------   #####
#####             A functional and perceptual signature of the            #####
#####                    second visual area in primates                   #####
#####   ---------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1038/nn.3402                  #####
###############################################################################


def random_crop_img(
    img, 
    target_res=[93,93]
):
    ''' This function aims to crop a subpart of the image in order to have the texture at the desired resolution 

        Arguments : 

            - img         : The source img, it's resolution should be greater than the target resolution
            - target_res  : The desired resolution for the cropped image

        Outputs :

            - cropped_img : The cropped img
    '''

    ## Get the former resolution
    y_res, x_res = img.shape

    ## Get a random initial position (top left pixel of the image)
    init_y = np.random.randint(0,y_res - target_res[0] +1)
    init_x = np.random.randint(0,x_res - target_res[1] +1)

    ## Crop the image
    cropped_img = img[init_y:init_y+target_res[0], init_x:init_x+target_res[1]]

    return cropped_img    


def load_imgs(
    directory_imgs, 
    target_res = [93,93],
    contrast = 1,  
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    num_samples = 15,
    device = None
):  
    ''' This function takes a folder path, loads and sorts every images by category (texture or noise), family, and sample
        For the images it crops them to the correct resolution, changes the contrast and rescale the images to the wanted values
    
        The name of the images should be formated as so : 

            'tex-320x320-im13-smp2.png'
            1     2      3    4   5

            - 1 : 'tex' if texture, 'noise' if noise
            - 2 : 'resolution_y' + 'x' + 'resolution_x'
            - 3 : 'im'  + number, the number shows to which texture familiy it corresponds
            - 4 : 'smp' + number, the number shows to which sample inside this family it corresponds
            - 5 : '.png' image should be in png
            - between each, there should be '-'
        
        Important :

            - In your folder, there should be exactly num_samples samples for every family
            - The resolution selected in 'target_res' should be equal or lower than the loaded images resolution

        Arguments :

            - directory_imgs : The path of the folder containing every images to load
            - target_res     : The desired resolution of the images (if the loaded images are in a higher dimention, it crops the image)
            - contrast       : The contrast to apply on the images, 1 means it does not change
            - pixel_min      : Value of the minimal pixel that will serve as the black reference
            - pixel_max      : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - num_samples    : The number of samples for each texture family
            - device         : The device on which to execute the code, if set to "None" it will take the available one
            
            - TODO           : Add random seed to avoid randomness

        Outputs : 

            - tex_imgs       : The tensor containing the texture images,              shape = [number of family, number of samples, resolution y, resolution x]
            - noise_imgs     : The tensor containing the corresponding noise images,     ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''
    
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get a list of all files in the directory
    list_imgs_names = os.listdir(directory_imgs)
    
    # Filter out only the image files
    list_imgs_names = [f for f in list_imgs_names if f.endswith(('.png'))] #TODO maybe allow other formats 
    
    ## Get every unique family
    dict_fam = {}
    for i, img_name in enumerate(list_imgs_names):

        ## Get the family for each image
        _, _, family, _ = img_name.split('-')
        family = int(family[2:])
        
        dict_fam[family] = 0

    ## Set the ids for every family
    for i, family in enumerate(dict_fam.keys()) : 
        dict_fam[family] = i

    ## Create a Tensor containing every image and noise sorted correctly
    tex_imgs   = torch.zeros((len(dict_fam.keys()), num_samples, *target_res)).to(device)
    noise_imgs = torch.zeros(tex_imgs.shape).to(device)

    print("   > loading images ...")
    ## Get every unique family
    for i, img_name in tqdm(enumerate(list_imgs_names)):
                
        ## Get the image informations
        category, _, family, sample = img_name.split('-')
        # load_img_res = [int(load_img_res.split('x')[0]), int(load_img_res.split('x')[1])]
        family = int(family[2:])
        sample = int(sample[3:-4])
        fam_id = dict_fam[family]
        smp_id = sample -1

        img_path = directory_imgs + '/' + img_name

        img = torch.Tensor(mpimg.imread(img_path))

        ## Crop the image to the correct dimention
        cropped_img = random_crop_img(img=img,target_res=target_res)

        if category == 'tex' :
            tex_imgs[fam_id, smp_id] = cropped_img
        else : 
            noise_imgs[fam_id, smp_id] = cropped_img

    ## Change the contrast and rescale to the wanted values
    min_val = torch.min(torch.min(tex_imgs), torch.min(noise_imgs))
    max_val = torch.max(torch.max(tex_imgs), torch.max(noise_imgs))
    
    for i in range(len(tex_imgs)) :

        for j in range(num_samples) :
            
            ## Get the images
            tex_img   = tex_imgs[i,j]
            noise_img = noise_imgs[i,j]

            # ## Change the contrast and rescale
            tex_img   = rescale(tex_img, min_val, max_val, -1, 1)*contrast
            tex_img   = rescale(tex_img, -1, 1, pixel_min, pixel_max)
            noise_img = rescale(noise_img, min_val, max_val, -1, 1)*contrast
            noise_img = rescale(noise_img, -1, 1, pixel_min, pixel_max)

            ## Update the tensor
            tex_imgs[i,j]   = tex_img
            noise_imgs[i,j] = noise_img

    return tex_imgs, noise_imgs, dict_fam

 
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
    device = None

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
    tex_imgs, noise_imgs, dict_fam = load_imgs(directory_imgs=directory_imgs, target_res=img_res, contrast=contrast, pixel_min=pixel_min, pixel_max=pixel_max, num_samples=num_samples, device=device)

    tex_resp   = []
    noise_resp = []

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
