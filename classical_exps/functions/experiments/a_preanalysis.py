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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d import Axes3D  # for older matplotlib versions
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares




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


                        grating_np = grating.squeeze().detach().cpu().numpy()
                        carrier_min_val, carrier_max_val = grating_np.min(), grating_np.max()
                        directory = "/project/results/get_all_grating_parameters/"
                        os.makedirs(directory, exist_ok=True)
                        fig, ax = plt.subplots()
                        im = ax.imshow(grating_np.squeeze(),  cmap='gray', vmin=carrier_min_val, vmax=carrier_max_val)
                        fig.colorbar(im, ax=ax)
                        plt.savefig(directory +  f"grating_ori_{orientation}_sf_{sf}_phase_{phase}.png")
                        plt.close()

                        grating = grating.to(device)
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
    dot_size_in_pixels=1,
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
            model_input = rescale(tensors_array[i:i+bs], 0, 1, pixel_min, pixel_max).reshape(bs, 1, *img_res).to(device)
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


def gauss_fit(dot_stim_img, img_size=[93,93], neuron=0, visualize=True): 
    ''' This function takes images coming from a dot stimulation experiment and tries to 
    make a 2D gaussian model fit to it.
    The images are the output of the dot_stimulation function, contained as values in the dictionnary
    
    Arguments :

        - dot_stim_img : Should be a numpy array. This is the values contained in the dictionary which is the output of the 'dot_stimulation' function

    Outputs :

        - A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt : Fitted parameters

    '''

    img_h, img_w = img_size
    max_sigma_x = img_w / 4  
    max_sigma_y = img_h / 4
    min_sigma = 2.0            
    n_coarse = 10
    n_refine = 3

    with torch.no_grad():
        # Preprocess data
        data = np.clip(dot_stim_img, a_min=0, a_max=None)
        baseline = np.percentile(data, 20)  
        data = data - baseline
        data /= (data.std() + 1e-9)

        # Meshgrid
        x = np.arange(img_size[1])
        y = np.arange(img_size[0])
        X, Y = np.meshgrid(x, y)
        x_data, y_data, z_data = X.ravel(), Y.ravel(), data.ravel()

        # Initial guess
        y_max, x_max = np.unravel_index(np.argmax(data), data.shape)
        A_guess = float(data.max() - np.median(data))
        mean_sigma = np.mean([min_sigma, np.mean([max_sigma_x, max_sigma_y])])
        base_guess = np.array([A_guess, x_max, y_max, mean_sigma, mean_sigma, 0.0])
 
        # Define bounds
        A_lower = data.max() * 0.2
        A_upper = max(A_guess * 5.0, (data.max() - data.min()) * 5.0)

        margin_x = img_size[1] * 0.1
        margin_y = img_size[0] * 0.1
        bounds = (
            [A_lower, margin_x, margin_y, min_sigma, min_sigma, -0.6],
            [A_upper, img_w - margin_x, img_h - margin_y, max_sigma_x, max_sigma_y, 0.6],
        )

        weight = (z_data / (z_data.max() + 1e-8)) ** 2  
        def resid(p):
            return weight * (gaussian2D_with_correlation((x_data, y_data), *p) - z_data)

        # --- Phase 1: Coarse multi-start ---
        best_cost, best_params = np.inf, None
        for _ in range(n_coarse):
            jitter = np.random.uniform(-0.5, 0.5, 6) * np.array([A_guess*0.2, 2, 2, 5, 5, 0.1])
            start = np.clip(base_guess + jitter, bounds[0], bounds[1])
            res = least_squares(resid, start, bounds=bounds, method='trf', loss='soft_l1', max_nfev=5000)
            if res.cost < best_cost:
                best_cost, best_params = res.cost, res.x

        # --- Phase 2: Refinement around best fit ---
        for _ in range(n_refine):
            jitter = np.random.normal(0, 0.1, 6) * np.array([A_guess*0.05, 1, 1, 2, 2, 0.02])
            start = np.clip(best_params + jitter, bounds[0], bounds[1])
            res = least_squares(resid, start, bounds=bounds, method='trf', loss='soft_l1', max_nfev=20000)
            if res.cost < best_cost:
                best_cost, best_params = res.cost, res.x

        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt = best_params

        # Fitted data
        fitted_data = gaussian2D_with_correlation((x_data, y_data), *best_params)
        fitted_data = fitted_data.reshape(*img_size)

        # Compute metrics
        y = z_data
        yhat = fitted_data.ravel()

        mse = float(np.mean((y - yhat) ** 2))
        var_y = np.var(y)
        r2 = 1.0 - mse / (var_y if var_y > 0 else 1.0)

        # Normalized residuals in 0–1
        resid_norm = (y - yhat - (y - yhat).min()) / ((y - yhat).ptp() + 1e-9)

        return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data, mse, r2, resid_norm


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
    seed = 0,
    plot = True,
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

        print("Top-level groups:", list(file.keys()))
        if '/preferred_pos' in file:
            group = file['/preferred_pos']
            print("Datasets in /preferred_pos:", list(group.keys()))
            print("Attributes:", dict(group.attrs))

        ## Select the group
        group = file[group_path]
        ## Add description
        group.attrs["description"] = "datasets = [x_pix, y_pix, sigma_x_opt, sigma_y_opt, rho_opt, mse, r2]"     

        unrealistic_fits = []
        for id in neuron_ids:

            ## Steps 3)
            ## Try to fit
            try: 
                ## Normalise the dot stimulation result and put them in the right format
                dot_stim_norm = dot_stim_dict[id].detach().cpu().numpy()/dot_stim_dict[id].detach().cpu().numpy().std()

                print(f"Fitting neuron {id}...")
                
                ## Fit to the 2D Gaussian model
                A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_dot_stim, mse, r2, resid_norm = gauss_fit(dot_stim_norm, neuron=id)
                
                ## Step 5) Save x0_opt, y0_opt and error
        
                try :
                    ## Create a dataset for the neuron if it doen't already exists

                    # from scipy.ndimage import gaussian_filter
                    # from scipy.signal import find_peaks
                    # # Smooth lightly to remove noise
                    # smoothed = gaussian_filter(dot_stim_norm, sigma=1.5)
                    # flat = smoothed.flatten()

                    # # --- Peak structure check ---
                    # # Find peaks and their heights
                    # peaks, props = find_peaks(flat, height=np.max(flat)*0.3, distance=img_res[0]//8)
                    # peak_heights = props['peak_heights'] if len(peaks) else []

                    # # Define "too many peaks" = 2+ peaks of comparable height
                    # too_many_peaks = (
                    #     len(peak_heights) >= 2
                    #     and np.max(peak_heights) * 0.9 <= np.sort(peak_heights)[-2]  
                    # )

                    # # --- Hole-like (annular) check ---
                    # center_val = smoothed[int(y0_opt), int(x0_opt)]
                    # mean_surround = np.mean(smoothed)
                    # hole_like = center_val < 0.4 * mean_surround

                    # # --- Decision ---
                    # unrealistic = too_many_peaks or hole_like

                    # if unrealistic:
                    #     unrealistic_fits.append([id, r2, too_many_peaks, hole_like])

                    if f'neuron_{id}' in group:
                        del group[f'neuron_{id}']
                        print(f"Neuron {id} already present, erasing and rewriting it")

                    data = [x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, mse, r2]

                    group.create_dataset(f"neuron_{id}", data=data)

                    print(f"Neuron {id} fitted with error = {mse, r2}")
                    print(f"Fitted parameters : A={A_opt}, x0={x0_opt}, y0={y0_opt}, sigma_x={sigma_x_opt}, sigma_y={sigma_y_opt}, rho={rho_opt}")

                    if plot:

                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        from matplotlib import cm
                        from matplotlib.colors import Normalize
                        from matplotlib.colors import LightSource
                        print("Starting visualization...")
                        
                        img_res_x, img_res_y = img_res
                        xs, ys = np.meshgrid(np.arange(img_res_x), np.arange(img_res_y))
                        x_data = xs.flatten()
                        y_data = ys.flatten()
                        z_data = dot_stim_norm.flatten()

                        color_2d = "viridis"
                        color_3d = "viridis"

                        # Flip Y-axis convention for consistency (matplotlib origin='lower' vs data origin)
                        mu_x = x0_opt
                        mu_y = img_res_y - 1 - y0_opt  

                        # ---- Covariance ellipse ----
                        cov = np.array([[sigma_x_opt**2, rho_opt*sigma_x_opt*sigma_y_opt],
                                        [rho_opt*sigma_x_opt*sigma_y_opt, sigma_y_opt**2]])
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        width, height = 4 * np.sqrt(eigvals)   # 2σ ellipse
                        theta_deg = np.degrees(np.arctan2(eigvecs[1,1], eigvecs[0,1]))

                        # Flip the angle because image y-axis grows upward (but matrix indexing goes downward)
                        theta_deg = -theta_deg

                        # ---- Gaussian function ----
                        def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
                            X = x - mu_x
                            Y = y - mu_y
                            denom = 2 * (1 - rho**2)
                            Z = np.exp(-(X**2 / sigma_x**2 + Y**2 / sigma_y**2 - 2 * rho * X * Y / (sigma_x * sigma_y)) / denom)
                            return Z

                        Z_gauss = gaussian_2d(xs, ys, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt)

                        fig = plt.figure(figsize=(14, 10))

                        # 1–5: 2D plots
                        axs = [fig.add_subplot(2, 3, i+1) for i in range(5)]

                        # 1. Original
                        im0 = axs[0].imshow(dot_stim_norm, cmap=color_2d, origin='lower')
                        axs[0].set_title("Original dot stimulation")


                        # 2. Fitted Gaussian
                        im1 = axs[1].imshow(fitted_dot_stim, cmap=color_2d, origin='lower')
                        axs[1].set_title(f"Fitted Gaussian (r²={r2:.3f})")

                        cmap = color_2d
                        vmin = min(np.nanmin(dot_stim_norm), np.nanmin(fitted_dot_stim))
                        vmax = max(np.nanmax(dot_stim_norm), np.nanmax(fitted_dot_stim))
                        norm = Normalize(vmin=float(vmin), vmax=float(vmax))

                        # 3. Contours
                        im2 = axs[2].imshow(dot_stim_norm, cmap=color_2d, origin='lower')
                        axs[2].contour(fitted_dot_stim / fitted_dot_stim.max(),
                                        levels=sorted([np.exp(-0.5*k**2) for k in [1,2,3]]),
                                        colors='lime')
                        axs[2].set_title("Gaussian contours of the fit (1σ, 2σ, 3σ)")


                        # 1) compute plot coords center
                        mu_x_plot = float(x0_opt)
                        mu_y_plot = float(img_res_y - 1 - y0_opt)   # IMPORTANT: use this for all overlays & gaussian

                        # 2) compute Z_gauss using plot coords (NOT array coords)
                        def gaussian_2d_plotcoords(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
                            X = x - mu_x
                            Y = y - mu_y
                            denom = 2.0 * (1.0 - rho**2)
                            return np.exp(-(X**2 / sigma_x**2 + Y**2 / sigma_y**2 - 2.0 * rho * X * Y / (sigma_x * sigma_y)) / denom)

                        Z_gauss = gaussian_2d_plotcoords(xs, ys, mu_x_plot, mu_y_plot, sigma_x_opt, sigma_y_opt, rho_opt)

                        # 3) compute ellipse parameters (plot coords)
                        cx, cy, width, height, angle_deg = ellipse_from_fit(
                            x0 = x0_opt, y0 = y0_opt,
                            sigma_x = sigma_x_opt, sigma_y = sigma_y_opt, rho = rho_opt,
                            img_h = img_res_y, img_w = img_res_x,
                            n_sigma = 2, origin = 'lower', force_major_point_right = True
                        )

                        # 4) verify, with slightly relaxed tolerance
                        max_err, ok = verify_ellipse_vs_gaussian(
                            cx, cy, width, height, angle_deg,
                            xs = xs, ys = ys,
                            mu_x_plot = mu_x_plot, mu_y_plot = mu_y_plot,
                            sigma_x = sigma_x_opt, sigma_y = sigma_y_opt, rho = rho_opt,
                            n_sigma = 2,
                            atol = 0.08  # relaxed tolerance; tune down if everything matches
                        )

                        if not ok:
                            # debug overlay: show where largest errors are, and draw eigenvectors
                            print(f"Warning: ellipse vs gaussian mismatch (max_err={max_err:.4f}). Showing debug overlay and falling back to contour.")

                            # compute ellipse boundary points (for debug)
                            a = width / 2.0
                            b = height / 2.0
                            th = np.radians(angle_deg)
                            thetas = np.linspace(0, 2*np.pi, 180)
                            x_ell = a * np.cos(thetas)
                            y_ell = b * np.sin(thetas)
                            xr = cx + (x_ell * np.cos(th) - y_ell * np.sin(th))
                            yr = cy + (x_ell * np.sin(th) + y_ell * np.cos(th))

                            # evaluate gaussian on those points
                            Xr = xr - mu_x_plot
                            Yr = yr - mu_y_plot
                            denom = 2.0 * (1.0 - rho_opt**2)
                            Zr = np.exp(-(Xr**2 / sigma_x_opt**2 + Yr**2 / sigma_y_opt**2 - 2*rho_opt*Xr*Yr/(sigma_x_opt*sigma_y_opt)) / denom)
                            level = np.exp(-0.5 * (2.0**2))

                            # find largest mismatch index
                            idx_max = np.argmax(np.abs(Zr - level))

                            # Now draw debug figure (you can save it or show it)
                            fig_dbg, ax_dbg = plt.subplots(figsize=(6,6))
                            im_dbg = ax_dbg.imshow(dot_stim_norm, origin='lower', cmap=color_2d)
                            ax_dbg.plot(xr, yr, '-', color='yellow', lw=1)               # ellipse boundary
                            ax_dbg.scatter([xr[idx_max]], [yr[idx_max]], c='red', s=50)  # worst point
                            # draw eigenvectors for visual check
                            cov = np.array([[sigma_x_opt**2, rho_opt*sigma_x_opt*sigma_y_opt],
                                            [rho_opt*sigma_x_opt*sigma_y_opt, sigma_y_opt**2]])
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            order = eigvals.argsort()[::-1]
                            eigvecs = eigvecs[:, order]
                            # convert eigenvectors to plot coords (flip y)
                            v1 = eigvecs[:,0].copy()
                            v1_plot = np.array([v1[0], -v1[1]])  # flip y
                            v2 = eigvecs[:,1].copy()
                            v2_plot = np.array([v2[0], -v2[1]])
                            # scale for visualization
                            ax_dbg.arrow(cx, cy, v1_plot[0]* (width/4), v1_plot[1]* (width/4), color='white', width=0.8)
                            ax_dbg.arrow(cx, cy, v2_plot[0]* (height/4), v2_plot[1]* (height/4), color='white', width=0.8)
                            ax_dbg.set_title(f"Ellipse debug (max_err={max_err:.4f})")
                            fig_dbg.colorbar(im_dbg, ax=ax_dbg)
                            # save debug figure next to main outputs
                            dbg_dir = f"/project/results/black_and_white_experiment/debug/{id}/"  # reuse your directory var
                            os.makedirs(dbg_dir, exist_ok=True)
                            fig_dbg.savefig(os.path.join(dbg_dir, f"dot_stim_debug_ellipse_{id}.png"), dpi=200, bbox_inches='tight')
                            plt.close(fig_dbg)

                            # finally: fall back to drawing contours (since ellipse unreliable)
                            # (your existing contour plotting code here)
                            # e.g. axs[2].contour(...) or save a contour-only fig
                        else:
                            # OK: ellipse verified; draw it on your axes
                            ellipse = Ellipse((cx, cy), width=width, height=height, angle=angle_deg,
                                            edgecolor='lime', facecolor='none', lw=2)

                        # Draw ellipse
                        im3 = axs[3].imshow(dot_stim_norm, cmap=color_2d, origin='lower')
                        ellipse = Ellipse((x0_opt, y0_opt), width=width, height=height,
                                        angle=-theta_deg, edgecolor='lime', facecolor='none', lw=2)
                        axs[3].add_patch(ellipse)
                        axs[3].set_title("2σ Ellipse from fit parameters")

                        # 5. mRF radius
                        percentage_of_gaussian_energy = 0.865
                        mRF_radius = np.sqrt(-2 * np.log(1 - percentage_of_gaussian_energy)) * np.sqrt(sigma_x_opt * sigma_y_opt)
                        im4 = axs[4].imshow(dot_stim_norm, cmap=color_2d, origin='lower')
                        circle = plt.Circle((x0_opt, y0_opt), mRF_radius, color='lime', fill=False, lw=2)
                        axs[4].add_patch(circle)
                        axs[4].set_title("mRF radius from fit parameters")

                        for ax in (axs[0], axs[1], axs[2], axs[3], axs[4]):
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="4%", pad=0.04)
                            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Response magnitude")
                            
                            
                        # 6. 3D plot
                        ax3d = fig.add_subplot(2, 3, 6, projection='3d')
                        sc = ax3d.scatter(xs.flatten(), ys.flatten(), z_data, c=z_data, cmap=color_3d, s=10)
                        ax3d.plot_surface(xs, ys, Z_gauss, cmap=color_3d, alpha=0.5)
                        ax3d.set_title("3D plot of the dot stimulation results")
                        ax3d.view_init(elev=45, azim=60)
                        fig.colorbar(sc, ax=ax3d, shrink=0.6, label="Response magnitude")

                        plt.tight_layout()
                        directory = f"/project/results/black_and_white_experiment/dot_stimulation/{id}/"
                        os.makedirs(directory, exist_ok=True)
                        plt.savefig(directory + f"dot_stimulation_fit_visual_{id}_3x2.png", dpi=300)
                        plt.close(fig)
                        print(f" > Neuron {id}: visualisation saved in {directory}")

                except ValueError as e:
                    print(e)
                    pass

            ## Step 4)
            ## If it do not converge, that mean that the receptive field could not be fitted to a gaussian model
            except: 
                ## Step 5) Save x0_opt, y0_opt and error = np.nan
                try :
                    ## Create a dataset for the neuron if it doen't already exists
                    data=[img_res[1]/2,img_res[0]/2,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan]

                    group.create_dataset(f"neuron_{id}", data=data )
                
                except ValueError as e:
                    print(e)
                    pass

    print(f" > Unrealistic fits detected for neurons : {unrealistic_fits}")





def ellipse_from_fit(x0, y0, sigma_x, sigma_y, rho,
                     img_h, img_w,
                     n_sigma=2,
                     origin='lower',
                     force_major_point_right=True):
    """
    Return (center_x_plot, center_y_plot, width, height, angle_deg) suitable
    for matplotlib.patches.Ellipse so that it overlays correctly on imshow(..., origin=origin).

    - x0,y0 : fit outputs in array coordinates (x = col, y = row, origin top-left)
    - sigma_x, sigma_y, rho : Gaussian fit params (stddevs and correlation)
    - img_h, img_w : image shape (height, width)
    - n_sigma : contour level (e.g. 2 for 2σ)
    - origin : 'lower' (default) if you plot imshow(..., origin='lower') OR 'upper' if you use origin='upper'
    - force_major_point_right : ensures consistent eigenvector direction (avoids ±180° flips)
    """
    # 1) Covariance matrix in (x,y) coords where x is horizontal, y vertical (array coords)
    cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                    [rho*sigma_x*sigma_y, sigma_y**2]])

    # 2) Eigen-decomposition, sort descending so index 0 is major axis
    eigvals, eigvecs = np.linalg.eigh(cov)                # ascending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]                          # columns are eigenvectors

    # 3) Semi-axis lengths (sigma along eigenvectors): sqrt(eigvals)
    #    Matplotlib Ellipse expects full axis lengths (width,height) = 2 * semi-axis
    #    For n_sigma contour: semi-axis = n_sigma * sqrt(eigval) -> full length = 2 * n_sigma * sqrt(eigval)
    full_lengths = 2.0 * n_sigma * np.sqrt(eigvals)      # [major_full, minor_full]

    width = float(full_lengths[0])
    height = float(full_lengths[1])

    # 4) Eigenvector for major axis
    vx, vy = eigvecs[0, 0], eigvecs[1, 0]  # major eigenvector components in array coords

    # 5) Convert fit center (array coords) --> plotting coords depending on origin
    if origin == 'lower':
        # imshow(..., origin='lower') shows row 0 at bottom => convert y
        cx = float(x0)
        cy = float(img_h - 1 - y0)
        # when flipping y, the eigenvector's y component must be flipped to compute angle in plot coords
        vy_plot = -vy
        vx_plot = vx
    elif origin == 'upper':
        # imshow(..., origin='upper') displays array coords directly (row 0 top)
        cx = float(x0)
        cy = float(y0)
        vy_plot = vy
        vx_plot = vx
    else:
        raise ValueError("origin must be 'lower' or 'upper'")

    # 6) Angle in degrees measured CCW from +x axis for matplotlib
    angle_deg = np.degrees(np.arctan2(vy_plot, vx_plot))

    # 7) Stabilize orientation: make angle in [0,180)
    angle_deg = angle_deg % 180.0

    # 8) Force major axis point-right (optional): if the x component of plot eigenvector is negative, flip 180°.
    #    This removes the arbitrary eigenvector sign ambiguity across neurons.
    if force_major_point_right:
        # compute unit vector of (vx_plot, vy_plot)
        norm = np.hypot(vx_plot, vy_plot)
        if norm == 0:
            pass
        else:
            ux, uy = vx_plot / norm, vy_plot / norm
            # if major axis's x-component is < 0, rotate by 180 deg (flip direction)
            if ux < 0:
                angle_deg = (angle_deg + 180.0) % 360.0
                # keep angle in [0,180)
                angle_deg = angle_deg % 180.0

    # 9) Guarantee width is major axis (defensive)
    if width < height:
        width, height = height, width
        angle_deg = (angle_deg + 90.0) % 180.0

    # Return ellipse parameters in plotting coordinates
    return cx, cy, width, height, angle_deg


def verify_ellipse_vs_gaussian(cx, cy, width, height, angle_deg,
                               xs, ys, mu_x_plot, mu_y_plot,
                               sigma_x, sigma_y, rho, n_sigma=2,
                               atol=0.05):
    """
    Numeric sanity check: sample points on ellipse boundary and check they are close to
    the Gaussian level set value for n_sigma (i.e. exp(-0.5*n_sigma^2)).
    Returns (max_abs_error, passed_bool). Useful as a unit test.
    - xs, ys : meshgrid used for Gaussian (plot coords)
    - mu_x_plot, mu_y_plot : center in plot coords (must match cx,cy)
    """
    # Gaussian analytic level
    level = np.exp(-0.5 * (n_sigma**2))

    # param paramization of ellipse boundary in plot coords
    # semi-axes:
    a = width / 2.0
    b = height / 2.0
    th = np.radians(angle_deg)

    # generate points around ellipse boundary
    thetas = np.linspace(0, 2*np.pi, 180)  # dense
    x_ell = a * np.cos(thetas)
    y_ell = b * np.sin(thetas)
    # rotate by angle and translate to cx,cy
    xr = cx + (x_ell * np.cos(th) - y_ell * np.sin(th))
    yr = cy + (x_ell * np.sin(th) + y_ell * np.cos(th))

    # Evaluate gaussian (in plot coords we assume gaussian_2d uses plot coords)
    X_rel = xr - mu_x_plot
    Y_rel = yr - mu_y_plot
    denom = 2.0 * (1.0 - rho**2)
    Zr = np.exp(-(X_rel**2 / sigma_x**2 + Y_rel**2 / sigma_y**2 - 2*rho*X_rel*Y_rel/(sigma_x*sigma_y)) / denom)

    # compute error relative to analytic level
    err = np.abs(Zr - level)
    return float(err.max()), (err.max() <= atol)