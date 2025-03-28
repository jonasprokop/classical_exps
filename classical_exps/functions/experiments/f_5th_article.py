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
import traceback
import shutil



###############################################################################
#####  PART VI : Experiments for the fifth article, Hallum et al., 2014   #####
#####   ---------------------------------------------------------------   #####
#####       Surround suppression supports second-order feature encoding   #####
#####                     by macaque V1 and V2 neurons                    #####
#####   ---------------------------------------------------------------   #####
#####      DOI : http://dx.doi.org/10.1016/j.visres.2014.10.004           #####
###############################################################################

def get_all_grating_parameters_with_modulator(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    modulator_orientations = np.linspace(-np.pi/2, np.pi/2, 8, endpoint=False), 
    modulator_spatial_frequencies_multiplicators = [0.5],
    modulator_phases = np.linspace(0, 2*np.pi, 20, endpoint=False),
    carrier_contrast = 0.75,
    contrast=1,
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
    print(' > Get preferred grating parameters and modulator responses')

    # Define paths
    grating_group_path = "/full_field_params"
    modulator_group_path = "/modulator_params"
    group_path_pos         = "/preferred_pos"


    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clear groups if overwrite is requested
    if overwrite:
        clear_group(h5_file, modulator_group_path)

    ## This will serve to create and verify the arguments of the group
    args_str = f"modulator_orientations={modulator_orientations}/modulator_spatial_frequencies_multiplicators ={modulator_spatial_frequencies_multiplicators }/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, modulator_group_path, args_str)

    ## If the neurons are not all in the data : 
    if check_neurons_presence(h5_file, [grating_group_path, modulator_group_path], neuron_ids) : 
        return

    else:
        with h5py.File(h5_file, 'a') as file :

            ## Access the group
            mod_group = file[modulator_group_path]
            grating_group   = file[grating_group_path]

            ## Add description
            mod_group.attrs["description"] = "datasets = [relative_orientation, relative_modulator_spatial_frequency, modulator_phase, response]"


            ## Get the model for the neuron and retrieve max exciting responses
            for neuron_id in tqdm(neuron_ids, desc="Applying Modulator"):
                neuron = f"neuron_{neuron_id}"

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id)
                single_model.to(device)
                single_model.eval()

                preferred_ori   = grating_group[neuron][:][0]
                preferred_sf    = grating_group[neuron][:][1]
                preferred_phase = grating_group[neuron][:][2]

                try:
                    # get center coordinates
                    x_pix         = file[group_path_pos][neuron][:][0]
                    y_pix         = file[group_path_pos][neuron][:][1]
                    
                except:
                    continue

                carrier  = torch.Tensor(imagen.SineGrating(
                    orientation=preferred_ori,
                    frequency=preferred_sf,
                    phase=preferred_phase,
                    bounds=BoundingBox(points=((-size / 2, -size / 2), (size / 2, size / 2))),
                    offset = 0, #?
                    scale = 1,  #?
                    xdensity=img_res[1] / size,
                    ydensity=img_res[0] / size,
                    x = get_offset_in_degr(x_pix, img_res[1], size), 
                    y = -get_offset_in_degr(y_pix, img_res[0], size)
                )())

                carrier = carrier * carrier_contrast

                directory = "/project/results/modulation/carriers"  + "/" + neuron + "/"
                os.makedirs(directory, exist_ok=True)
                plt.imsave(directory + f"{neuron}_carrier.png", 
                    carrier.squeeze(), cmap='gray', format='png')
            

                for relative_ori in modulator_orientations:
                    for modulator_spatial_frequencies_multiplicator in modulator_spatial_frequencies_multiplicators:
                            for mod_phase in modulator_phases:

                                # ori is relative to max exciting 
                                # mod is half to max exciting
                                # phases are gone over
                                
                                mod_sf = preferred_sf * modulator_spatial_frequencies_multiplicator

                                modulator = torch.Tensor(imagen.SineGrating(
                                    orientation= preferred_ori + relative_ori,
                                    frequency=mod_sf,
                                    phase=mod_phase,
                                    bounds=BoundingBox(points=((-size / 2, -size / 2), (size / 2, size / 2))),
                                    offset = 0, #?
                                    scale = 1,  #?
                                    xdensity=img_res[1] / size,
                                    ydensity=img_res[0] / size,
                                x = get_offset_in_degr(x_pix, img_res[1], size), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size)
                                )())

                                grating = carrier * modulator

                                directory = "/project/results/modulation/modulators"  + "/" + neuron + "/"
                                os.makedirs(directory, exist_ok=True)
                                plt.imsave(directory + f"{neuron}_modulator_relative_ori_{relative_ori}_relative_mod_sf_{modulator_spatial_frequencies_multiplicator}_mod_phase_{mod_phase}.png", 
                                    modulator.squeeze(), cmap='gray', format='png')
                                

                                directory = "/project/results/modulation/gratings"  + "/" + neuron + "/"
                                os.makedirs(directory, exist_ok=True)
                                plt.imsave(directory + f"{neuron}_grating_relative_ori_{relative_ori}_relative_mod_sf_{modulator_spatial_frequencies_multiplicator}_mod_phase_{mod_phase}.png", 
                                    grating.squeeze(), cmap='gray', format='png')

                                grating = grating.reshape(1,1,*img_res).to(device)

                                ## Rescale because the output of imagen has values from 0 to 1 and we want 
                                ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                                grating = rescale(grating, 0, 1, -1, 1)*contrast
                                grating = rescale(grating, -1, 1, pixel_min, pixel_max)


                                # Evaluate response
                                resp = single_model(grating)
                                # Create the entry for this modulator combination
                                response_data = [relative_ori, mod_sf, mod_phase, resp.cpu().item()] 
                                try:
                                    dataset = mod_group.create_dataset(f"neuron_{neuron_id}", shape=(0, 4), maxshape=(None, 4), dtype='f4')
                                except ValueError:
                                    dataset = mod_group[f"neuron_{neuron_id}"]

                                # Resize dataset to add the new response
                                dataset.resize((dataset.shape[0] + 1, 4))

                            # Save the new response to the dataset
                                dataset[-1] = response_data
