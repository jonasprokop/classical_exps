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

########################################################
##### PART I : Functions that perform pre-analyses #####
########################################################


def find_preferred_grating_parameters_full_field(
    single_model, 
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,           
    ):

    ''' 
        This function generates full field grating images with every parameters combination and 
        returns the parameters that lead to the greatest activation in the Single Neuron Model.

        Arguments : 

            - single_model        : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - orientations        : An array containing the orientations of the grating to test (in rad)
            - spatial_frequencies : An array containing the spatial frequencies to test (it correspond to the number of grating cycles per degree of excentricity)
            - phases              : An array containing the phases of the grating to test
            - contrast            : The value that will multiply the grating image's values (lower than 1 will reduce the contrast and greater than 1 will increase the contrast)
            - img_res             : Resolution of the image in term of pixels [nb_y_pix, nb_x_pix]
            - pixel_min           : Value of the minimal pixel that will serve as the black reference
            - pixel_max           : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device              : The device on which to execute the code, if set to "None" it will take the available one
            - size                : The size of the image in terms of degree of visual angle (DVA)

        Outputs : 

            - max_ori             : Preferred orientation
            - max_sf              : Preferred spatial frequency
            - max_phase           : Preferred phase
            - stim_max            : Preferred grating image (with the preferred parameters)
            - resp_max            : Response of the model to the preferred image

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
    
    ## Initialisation
    resp_max = 0

    ## Get every parameters combination
    with torch.no_grad():
        for orientation in orientations :
            for sf in spatial_frequencies:
                for phase in phases: 
                
                    ## Creation of the grating image
                    grating = torch.Tensor(imagen.SineGrating(
                        orientation = orientation, 
                        frequency = sf,
                        phase = phase,
                        bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), #The bounds of the image. Warning, the shape is : ((width_left_side, height_bottom_side) , width_right_side, height_top_side)). 
                        offset = 0, #?
                        scale = 1,  #?
                        xdensity = img_res[1]/size[1], #pixels/degree
                        ydensity = img_res[0]/size[0],
                    )())

                    grating = grating.reshape(1,*img_res).to(device)
                    ## Rescale because the output of imagen has values from 0 to 1 and we want 
                    ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                    grating = rescale(grating, 0, 1, -1, 1)*contrast
                    grating = rescale(grating, -1, 1, pixel_min, pixel_max)

                    resp = single_model(grating.reshape(1,1,*img_res))

                    if resp>resp_max:
                        resp_max = resp
                        max_ori = orientation
                        max_phase = phase
                        max_sf = sf
                        stim_max = grating
        
    return max_ori, max_sf, max_phase, stim_max, resp_max


def get_all_grating_parameters(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False,
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,           
    ):

    '''
        This function uses the 'find_preferred_grating_parameters_full_field' on every single cell model (neuron)
        and write the preferred parameters for the full field grating image of the selected neurons in the HDF5 file
        

        Arguments :

            - h5_file           : The HDF5 file containing the data
            - overwrite         : If set to True, will erase the data in the group "full_field_params" and then fill it again. 
                                If set to False, the function will conserve the existing data and only add the one not already present
            - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
            - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
            - others            : Explained in 'find_preferred_grating_parameters_full_field'

        Outputs :

            - datasets in /full_field_params : An array containing the preferred full field parameters
                                               Format = [pref_orientation, pref_spatial_frequency, pref_phase]
    '''
    print(' > Get grating parameters')

    ## Create the group path
    group_path = "/full_field_params"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## This will serve to create and verify the arguments of the group
    args_str = f"orientations={orientations}/spatial_frequencies={spatial_frequencies}/phases={phases}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, group_path, args_str)
    
    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)

    ## Evaluation mode
    all_neurons_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    ## Initialisation of the maximal response of every neuron
    resp_max = torch.zeros(len(neuron_ids)).to(device)

    ## Save the preferred parameters (an array where each row is a neuron and each column is a parameter)
    preferred_params = torch.zeros((len(neuron_ids),3)) ## Columns = [ori, sf, phase,max_response]

    ## If the neurons are not all in the data : 
    if check_neurons_presence(h5_file, [group_path], neuron_ids) : 
        return
        
    else :

        ## Get every parameters combination
        with torch.no_grad():
            for orientation in tqdm(orientations) :
                for sf in spatial_frequencies:
                    for phase in phases: 
                    
                        ## Creation of the grating image
                        grating = torch.Tensor(imagen.SineGrating(
                            orientation = orientation, 
                            frequency = sf,
                            phase = phase,
                            bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), #The bounds of the image. Warning, the shape is : ((width_left_side, height_bottom_side) , width_right_side, height_top_side)). 
                            offset = 0, #?
                            scale = 1,  #?
                            xdensity = img_res[1]/size[1], #pixels/degree
                            ydensity = img_res[0]/size[0],
                        )())

                        grating = grating.reshape(1,*img_res).to(device)
                        ## Rescale because the output of imagen has values from 0 to 1 and we want 
                        ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                        grating = rescale(grating, 0, 1, -1, 1)*contrast
                        grating = rescale(grating, -1, 1, pixel_min, pixel_max)

                        ## Get the response for the selected neurons
                        resp = all_neurons_model(grating.reshape(1,1,*img_res))[0][neuron_ids]
                        
                        ## Get the indices of the neurons that yield the maximum response
                        condition = torch.where(resp>resp_max)[0]

                        ## Get the current parameters 
                        current_params = torch.Tensor([orientation,sf,phase])

                        ## Save the preferred parameters for these neurons
                        preferred_params[condition] = current_params

                        ## Update the maximal response
                        resp_max[condition] = resp[condition]

        ## Now save the parameters in the file
        with h5py.File(h5_file, 'a') as file :

            ## Access the group
            group = file[group_path]

            ## Add description
            group.attrs["description"] = "datasets = [orientation, sf, phase_full_field, max_response]"

            for i, id in enumerate(neuron_ids):

                ## Put everything into a list
                data = torch.cat([preferred_params[i], resp_max[i].cpu().unsqueeze(0)])
            
                ## Create a dataset for the neuron if it doen't already exists
                try :
                    group.create_dataset(f"neuron_{id}", data=data)

                except ValueError :
                    pass


def dot_stimulation(
    all_neurons_model,
    neuron_ids,  
    dot_size_in_pixels=4,
    contrast = 1, 
    num_dots=200000,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    bs = 40,
    seed = 0,
    ):  
    ''' This function aims to find, for the neurons of the all_neurons_model,
    their sensivity to points presented at different positions of the image.

    
    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - num_dots           : IMPORTANT : must be a multiple of bs | Number of dot images to present to the neurons. 
        - bs                 : IMPORTANT : num_dots%bs == 0.        | Batch size
        - seed               : random seed 
        - others             : explained in other functions

    Output :

        - dot_stim_dict : dictionnary containing the results of the dot stimulation for every selected neuron. The key are the neuron ids and the values are images where the responses of neurons to the dots where summed 
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    
    ## Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate the tensors
    tensors = []
    resp = []
    

    with torch.no_grad():
        ## Generate the responses to gray screens
        no_stim_resp= all_neurons_model(torch.ones(1,1,*img_res, device=device)* (pixel_min + pixel_max)/2)

        ## Do the dot stimulation
        for _ in range(num_dots):

            ## Create an image with values = 0
            tensor = np.zeros(img_res, dtype=int)

            ## Get a random position for the dot 
            x = np.random.randint(0, img_res[0] - dot_size_in_pixels + 1)
            y = np.random.randint(0, img_res[1] - dot_size_in_pixels + 1)

            ## Make the dot either black or white
            square_values = np.random.choice([-1, 1])

            ## Create the dot
            tensor[x:x+dot_size_in_pixels, y:y+dot_size_in_pixels] = square_values*contrast

            ## Save the dot
            tensors.append(torch.Tensor(tensor))
        
        ## Convert the list of tensors to a unique tensor
        tensors_array = torch.stack(tensors)
    
        ## For every batch, get the responses of the all_neurons_model
        for i in tqdm(range(0, num_dots, bs)):
            ## Rescale the image to pixel_min and pixel_max
            model_input = rescale(tensors_array[i:i+bs], -1, 1, pixel_min, pixel_max).reshape(bs, 1, *img_res).to(device)
            resp.append(all_neurons_model(model_input))
        
        ## Merge the batches together
        resp = torch.cat(resp, dim=0)


        ## Tensor 1 is a tensor containing every dot position of the images (1 if the dot is here, else 0)                                         shape = (num_dots * img_res) 
        ## Tensor 2 is a tensor containing the responses of every neuron to the dot images, with a substraction of the response to gray stimulus   shape = (num_dots * nNeuron) #nNeuron = len(output_of_all_neurons_model)
        ## This function creates for every dot (first dimension = 'b') the following tensor :
        ## 'nxy' Basically, for each neuron, resp_to_the_image * dot_position_in_image
        ## It can be visualised as an array of matrices. Each matrix contains of zeros where there isn't a dot and the normalised value of the resp where the dot is
        ## Then it sums this Tensor for every dot image that was presented
        all_neurons_outputs = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(tensors_array).to(device),
                            (resp-no_stim_resp).to(device)
                            )
        ## For this we assume that a large amount of pixels presented would lead to a uniform presentation accross every position,
        ## So the results we obtain after this are accounting for the strength of the response of the neurons to every position, and thus their receptive field 

    dot_stim_dict = {}

    ## Save everything in a dictionnary where every key is a neuron index
    for id in neuron_ids:
        dot_stim_dict[id] = all_neurons_outputs[id]

    return dot_stim_dict


def gaussian2D_with_correlation(xy, A, x0, y0, sigma_x, sigma_y, rho):
    ''' The model of the gaussian function'''
    x, y = xy
    a = 1.0 / (2 * (1 - rho**2))
    b = ((x - x0)**2) / (sigma_x**2)
    c = 2 * rho * (x - x0) * (y - y0) / (sigma_x * sigma_y)
    d = ((y - y0)**2) / (sigma_y**2)
    return A * np.exp(-a * (b - c + d))


def gauss_fit(dot_stim_img, img_size=[93,93]): 
    ''' This function takes images coming from a dot stimulation experiment and tries to 
    make a 2D gaussian model fit to it.
    The images are the output of the dot_stimulation function, contained as values in the dictionnary
    
    Arguments :

        - dot_stim_img : Should be a numpy array. This is the values contained in the dictionary which is the output of the 'dot_stimulation' function

    Outputs :

        - A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt : Fitted parameters

    '''

    with torch.no_grad():
        data = np.clip(dot_stim_img, a_min=0, a_max=None)
        data = data/data.std()
        x_data = np.arange(0, img_size[1])
        y_data = np.arange(0, img_size[0])
        x, y = np.meshgrid(x_data, y_data)

        # Flatten for fitting
        x_data, y_data = x.ravel(), y.ravel()
        z_data = data.ravel()

        # Initial guess [A, x0, y0, sigma_x, sigma_y, rho]
        initial_guess = [5, 93/2, 93/2, 10, 10, 0]

        # Fit the model
        params, covariance = curve_fit(gaussian2D_with_correlation, (x_data, y_data), z_data, p0=initial_guess)

        # Extract the optimized parameters
        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt = params

        fitted_data = gaussian2D_with_correlation((x_data, y_data), *params)
        fitted_data = fitted_data.reshape(*img_size)

        return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data


def get_preferred_position(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False,
    dot_size_in_pixels=4,
    contrast = 1, 
    num_dots=200000,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    bs = 40,
    seed = 0
):
    ''' This function performs multiple things :
        1) Multiple verifications to see if the group "perferred_pos" exists in the file and if so, check if the parameters are compatible
        2) Uses the function "dot_stimulation" to obtain a dictionnary containing the excitatory pixels for every neuron in "neuron_ids"
        3) Uses the function "gauss_fit" to try to fit the data to a 2D Gaussian model
        4) If it fits : save x0_opt and y0_opt and the error. x0_opt and y0_opt are the preferred position for the stimulus
        5) If the fitting doesn't converges : Save an error = np.nan and the middle point of the image for x0_opt and y0_opt
        6) Save the results in HDF5 file

        Outputs :

            - datasets in /preferred_pos : An array containing the prefered position (x and y) and the fitting error 
                                           Format = [x_pix, y_pix, error]
        '''
    
    print(' > Get preferred position')

    ## Step 1

    ## Create the group path
    group_path = "/preferred_pos"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)
    
    ## This will serve to create and verify the arguments of the group
    args_str = f"dot_size_in_pixels={dot_size_in_pixels}/contrast={contrast}/num_dots={num_dots}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/bs={bs}/seed={seed}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, group_path, args_str)

    ## Step 2)
    dot_stim_dict = dot_stimulation(
        all_neurons_model = all_neurons_model,
        neuron_ids = neuron_ids,  
        dot_size_in_pixels = dot_size_in_pixels,
        contrast = contrast, 
        num_dots = num_dots,
        img_res = img_res, 
        pixel_min = pixel_min, 
        pixel_max = pixel_max, 
        device = device,
        bs = bs,
        seed = seed)

    with h5py.File(h5_file, 'a') as file :

        ## Select the group
        group = file[group_path]
        ## Add description
        group.attrs["description"] = "datasets = [x_pix, y_pix, error, sigma_x_opt, sigma_y_opt]"     

        for id in neuron_ids:

            ## Steps 3)
            ## Try to fit
            try: 
                ## Normalise the dot stimulation result and put them in the right format
                dot_stim_norm = dot_stim_dict[id].detach().cpu().numpy()/dot_stim_dict[id].detach().cpu().numpy().std()
                
                ## Fit to the 2D Gaussian model
                A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_dot_stim = gauss_fit(dot_stim_norm)
                
                ## Calculate the error between the model and the data
                error = np.mean((fitted_dot_stim - dot_stim_norm)**2)

                ## Step 5) Save x0_opt, y0_opt and error
        
                try :
                    ## Create a dataset for the neuron if it doen't already exists
                    data = [x0_opt,y0_opt,error,sigma_x_opt, sigma_y_opt]
                    group.create_dataset(f"neuron_{id}", data=data)

                except ValueError :
                    pass
                
      
            ## Step 4)
            ## If it do not converge, that mean that the receptive field could not be fitted to a gaussian model
            except: 
                ## Step 5) Save x0_opt, y0_opt and error = np.nan
                try :
                    ## Create a dataset for the neuron if it doen't already exists
                    data=[img_res[1]/2,img_res[0]/2,np.nan]
                    group.create_dataset(f"neuron_{id}", data=data )
                    

                except ValueError :
                    pass

