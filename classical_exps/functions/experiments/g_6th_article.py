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
#####  PART VII :   #####
#####   ---------------------------------------------------------------   #####
#####          #####
#####                                     #####
#####   ---------------------------------------------------------------   #####
#####      DOI :           #####
###############################################################################

def draw_bar_on_array(img_array, x, y, angle, length, width):

    x, y = int(round(x)), int(round(y))

    dx = (length / 2) * np.cos(angle)
    dy = (length / 2) * np.sin(angle)

    wx = (width / 2) * np.sin(angle)
    wy = (width / 2) * np.cos(angle)

    pts = np.array([
        [x - dx - wx, y - dy + wy], 
        [x + dx - wx, y + dy + wy],  
        [x + dx + wx, y + dy - wy], 
        [x - dx + wx, y - dy - wy], 
    ], dtype=np.int32)

    cv2.fillPoly(img_array, [pts], color=1)
    
    return img_array

def generate_texture_pattern(img_size=93, 
                             bar_length=2, 
                             bar_width=1, 
                             spacing=5, 
                             num_layers=4, 
                             center_bar_angle=np.pi/2, 
                             surround_bars_angle=np.pi/2, 
                             layers_numenator = 6, 
                             jiggle_amount=0.1,
                             contrast=True,
                             center_x=93/2,
                             center_y=93/2
                             ):
    # Create a blank image array (white background)
    img_array = np.zeros((img_size, img_size), dtype=np.uint8)

    if center_bar_angle:
        draw_bar_on_array(img_array, center_x, center_y, angle=center_bar_angle, length=bar_length, width=bar_width)
    
    # Initialize list for the surrounding bars' positions
    surround_positions = []
    
    # Generate circular layers of bars
    for layer in range(1, num_layers + 1):
        # Each layer's radius increases with spacing
        radius = layer * spacing
        
        # We will place bars evenly spaced around the circle
        num_bars_in_layer = layers_numenator * layer  
        angle_step = 2 * np.pi / num_bars_in_layer  
        
        for i in range(num_bars_in_layer):
            angle_of_shift = i * angle_step
            # Apply jiggle in the radial direction
            jiggle_radius = np.random.uniform(-jiggle_amount * spacing, jiggle_amount * spacing)
            new_radius = radius + jiggle_radius
            
            x = center_x + new_radius * np.cos(angle_of_shift)
            y = center_y + new_radius * np.sin(angle_of_shift)
            surround_positions.append((x, y))  # Store position and angle for rotation
    
    # Draw bars surrounding the center, alternating the angle (vertical/horizontal)
    if surround_bars_angle:
        for i, (x, y) in enumerate(surround_positions):
            # Draw bars at surrounding positions
            draw_bar_on_array(img_array, x, y, surround_bars_angle, length=bar_length, width=bar_width)
    if contrast:
        img_array = 1 - img_array
    return img_array




def get_orientation_contrast_stimulus(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    contrast=1,
    orientations = np.linspace(0, np.pi, 4, endpoint=False),  
    lengths=np.linspace(0, 45),
    widths=np.linspace(0, 45),
    contrasts=[True, False],
    bar_sets=[[np.pi/4, None],[np.pi/4, np.pi/4],[np.pi/4, 3*np.pi/4],[None, np.pi/4],[3*np.pi/4, None],[3*np.pi/4, 3*np.pi/4],[3*np.pi/4, np.pi/4],[None, 3*np.pi/4]],
    spacing=10,
    num_layers=4,
    layers_numenator=6,
    jiggle_amount=0.1,
    img_res=[93, 93],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
):
    '''
    Function to first determine preferred grating parameters using find_preferred_grating_parameters_full_field
    and then apply modulator to it, and find the preffered combination of such grating with modulation.
        

        Arguments :

            - h5_file           : The HDF5 file containing the data
            - overwrite         : If set to True, will erase the data in the group "preferred_grating_params" and "modulator_params" and then fill it again. 
                                  If set to False, the function will conserve the existing data and only add the one not already present
            - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
            - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
            - others            : Explained in 'find_preferred_grating_parameters_full_field'

        Prerequisite : 

            - function 'get_all_grating_parameters'        executed for the required neurons with matching arguments

        Outputs :

            - datasets in /modulator_params : An array containing the preferred modulator parameters
                                               Format = [modulator_orientation, modulator_spatial_frequency, response]
    '''
    print(' > Get preferred orientation contrast stimulation')

    # Define paths
    orientation_contrast_group = "/orietantion_contrast"
    group_path_pos         = "/preferred_pos"

    # import center here


    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clear groups if overwrite is requested
    if overwrite:
        clear_group(h5_file, orientation_contrast_group)

    ## This will serve to create and verify the arguments of the group
    args_str = f"contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, orientation_contrast_group, args_str)

    ## If the neurons are not all in the data : 
    if check_neurons_presence(h5_file, [orientation_contrast_group], neuron_ids) : 
        return

    else:

        with h5py.File(h5_file, 'a') as file :
            ## Access the group
            ori_contrast_group = file[orientation_contrast_group]

            ## Add description
            ori_contrast_group.attrs["description"] = "datasets = [ori_max, length_max, width_max, contrast_max, bar_set, response]"

            results = []

            ## Get the model for the neuron and retrieve max exciting responses
            for neuron_id in tqdm(neuron_ids, desc="Applying Orientation Contrasts"):
                neuron = f"neuron_{neuron_id}"

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id)
                single_model.to(device)
                single_model.eval()

                max_resp = 0

                try:
                    # get center coordinates
                    x_pix = file[group_path_pos][neuron][:][0]
                    y_pix = file[group_path_pos][neuron][:][1]
                    
                except:
                    continue

                # Create the the center stimuli and choose optimal orientation, len/width and contrast

                # GET THEM HERE And make aprox func
                for ori in orientations:
                    for length in lengths:
                        for width in widths:
                            for contrast_color in contrasts:

                                texture_pattern = generate_texture_pattern(img_size=img_res[0], 
                                                    bar_length=length, 
                                                    bar_width=width, 
                                                    spacing=spacing, 
                                                    num_layers=num_layers, 
                                                    center_bar_angle=ori, 
                                                    surround_bars_angle=None, 
                                                    layers_numenator = layers_numenator, 
                                                    jiggle_amount=jiggle_amount,
                                                    contrast=contrast_color,
                                                    center_x=x_pix,
                                                    center_y=y_pix)
                                
                                
                                directory = "/project/results/texture_patterns/center_only_stimuli"  + "/" + neuron + "/"
                                os.makedirs(directory, exist_ok=True)
                                plt.imsave(directory + f"{neuron}_center_only_ori_{ori}_length_{length}_width_{width}_contrast_color_{contrast_color}.png", 
                                    texture_pattern.squeeze(), cmap='gray', format='png')

                                
                                texture_pattern = torch.tensor(texture_pattern, dtype=torch.float32)
                                texture_pattern = texture_pattern.reshape(1,*img_res).to(device)

                                ## Rescale because the output of imagen has values from 0 to 1 and we want 
                                ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                                texture_pattern = rescale(texture_pattern, 0, 1, -1, 1)*contrast
                                texture_pattern = rescale(texture_pattern, -1, 1, pixel_min, pixel_max)

                                resp = single_model(texture_pattern)

                                if resp > max_resp:
                                    max_resp = resp
                                    result = [ori, length, width, contrast_color]


                ori_max = result[0]
                length_max = result[1]
                width_max = result[2]
                contrast_max = result[3]
                results = []

                # with the optimal stimuli get results for all the different contrast orientation settings
                for bar_set in bar_sets:

                    center_bar_angle_shift = bar_set[0] 
                    surround_bars_angle_shift = bar_set[1]
                    if center_bar_angle_shift:
                        center_bar_angle = ori_max+center_bar_angle_shift
                    else:
                        center_bar_angle = None
                    if surround_bars_angle_shift: 
                        surround_bars_angle = ori_max+surround_bars_angle_shift
                    else:
                        surround_bars_angle = None

                    print(center_bar_angle, surround_bars_angle)

                    texture_pattern = generate_texture_pattern(img_size=img_res[0], 
                                                    bar_length=length_max, 
                                                    bar_width=width_max, 
                                                    spacing=spacing, 
                                                    num_layers=num_layers, 
                                                    center_bar_angle=center_bar_angle, 
                                                    surround_bars_angle=surround_bars_angle, 
                                                    layers_numenator = layers_numenator, 
                                                    jiggle_amount=jiggle_amount,
                                                    contrast=contrast_max)
                    
                    directory = "/project/results/texture_patterns/texture_patterns"  + "/" + neuron + "/"
                    os.makedirs(directory, exist_ok=True)
                    plt.imsave(directory + f"{neuron}_center_bar_angle_shift_{center_bar_angle_shift}_surround_bars_angle_shift_{surround_bars_angle_shift}.png", 
                        texture_pattern.queeze(), cmap='gray', format='png')

                    
                    texture_pattern = torch.tensor(texture_pattern, dtype=torch.float32)
                    texture_pattern = texture_pattern.reshape(1,*img_res).to(device)

                    ## Rescale because the output of imagen has values from 0 to 1 and we want 
                    ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                    texture_pattern = rescale(texture_pattern, 0, 1, -1, 1)*contrast
                    texture_pattern = rescale(texture_pattern, -1, 1, pixel_min, pixel_max)

                    resp = single_model(texture_pattern)

                    center_bar_angle_shift = center_bar_angle_shift if center_bar_angle_shift is not None else -1
                    surround_bars_angle_shift = surround_bars_angle_shift if surround_bars_angle_shift is not None else -1

                    result =  [ori_max,
                            length_max,
                            width_max,
                            contrast_max,
                            neuron_id,
                            center_bar_angle_shift,
                            surround_bars_angle_shift,
                            resp.detach().cpu().numpy() ]
                    
                    results.append(result)

                
                ## Create a dataset for the neuron if it doen't already exists
                ori_contrast_group.create_dataset(neuron, data=results)