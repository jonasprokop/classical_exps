################################################################################################
#####    IN THIS FILE, YOU WILL CONFIGURE THE EXPERIMENTS AND ANALYSES YOU WILL PERFORM    #####
#####             -----------------------------------------------------------              #####
#####                 The everything will be executed in the file "main.py"                #####
################################################################################################

## Numpy
import numpy as np

## Functions
from classical_exps.functions import *
from run_pipeline.config.variables import *

## If overwrite is set to True, this will clean the results of the performed experiments before reperforming them
overwrite = True 

## (optional) Device to perform the experiments on (default will be gpu if available, cpu else)
device=None

#####  ##   ##  #####
#        # #    #    #
###       #     #####
#        # #    #
#####  ##   ##  #

# The experiments are explained with more details in the "experiments.py" file, every arguments are explained there
#
# List of available experiments :
#
#   PRE-ANALYSES
#
#       - get_all_grating_parameters : Find the preferred orientation, spatial frequency and phase for full fields gratings (this phase value will rarely be used since it is specific to the full field image)
#
#       - get_preferred_position     : Find the receptive field center by fitting the neurons response to dots at different positions in the receptive field to a Gaussian model
#
#   EXPERIMENTS 1 
#
#       - size_tuning_experiment_all_phases           : Compute the size tuning curve of circular and annular grating stimuli and computes the Grating Summation Field (GSF), the surround extent, the Annular Minimum Response Field (AMRF) and the Suppression Index (SI)
#
#       - contrast_response_experiment                : Compute the response to grating stimuli with different center and surround contrast
#
#       - contrast_size_tuning_experiment_all_phases  : Compute the size tuning curves for different contrasts and computes the ratio of GSF at low contrast over GSF at high contrast
#
#   EXPERIMENTS 2
#
#       - orientation_tuning_experiment_all_phases        : Compute the orientation tuning curves of different stimuli : 1) Orientation of the center 2) Orientation of the surround with a fixed center orientation (fixed at -45°, 0°, +45° relative to the preferred position)
#
#       - center_contrast_surround_suppression_experiment : Compute the response of different stimuli for different fixed center contrasts : center + iso-oriented surround, center + ortho-oriented surround, center with no surround
#
#   EXPERIMENTS 3
#
#       - black_white_preference_experiment   : Compute the Signal Noise Ratio (SNR) to black and white stimuli
#
#   EXPERIMENTS 4
#
#       - texture_noise_response_experiment   : Compute the response to texture images and matched noise images


## Example Pipeline
experiments_config = [
    # ['get_all_grating_parameters', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'orientations':orientations, 'spatial_frequencies':spatial_frequencies, 'phases':phases, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}], 
    # ['get_preferred_position', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels_gauss, 'contrast':contrast, 'num_dots':num_dots, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'bs':bs, 'seed':seed}],
    # ['size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'radii':radii, 'phases':phases, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['contrast_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts':center_contrasts, 'surround_contrasts':surround_contrasts, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['contrast_size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'contrasts':contrasts, 'radii':radii, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['orientation_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'ori_shifts':ori_shifts, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['center_contrast_surround_suppression_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts_ccss':center_contrasts_ccss, 'surround_contrast':surround_contrast, 'phases':phases, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['black_white_preference_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'seed':seed}],
    # ['texture_noise_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'directory_imgs':directory_imgs, 'overwrite':overwrite, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'num_samples':num_samples, 'img_res':img_res, 'device':device}]
    # ['get_all_grating_parameters_with_modulator', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],
    # ['get_orientation_contrast_stimulus', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],
    # ['get_surround_contrast_facilitation', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],

    ]

#####   #####   ####  #   #  #   #######  ####
#    #  #      #      #   #  #      #    #
#####   ###     ###   #   #  #      #     ###     
#   #   #          #  #   #  #      #        #
#    #  #####  ####    ###   #####  #    ####

# The results are explained with more details in the TODO rename_the_file.py file, every arguments are explained there
#
# List of available analyses :
#
# FILTERING
#
# - filter_fitting_error       (1)     : This function shows the filtered neurons
# 
# - filter_no_supp_neurons     (2)
#
# - filter_low_supp_neurons    (3)
#
# - filter_SNR                 (4)
#
# EXPERIMENTS 1 

# - size_tuning_results_1            (1), (2), (3)
#
# - size_tuning_results_2            (1), (2)
#
# - contrast_response_results_1      (1), (2)
#
# - contrast_size_tuning_results_1   (1), (2)        
#
# EXPERIMENTS 2
#      
# - orientation_tuning_results_1        (1), (2)  
# 
# - orientation_tuning_results_2        (1), (2)  
#
# - ccss_results_1                      (1), (2) 
#
# - ccss_results_2                      (1), (2)  
#
# EXPERIMENTS 3
#
# - black_white_results_1               (4)
#
# EXPERIMENTS 4
#
# - texture_noise_response_results_1   
#
# - texture_noise_response_results_2
#
# - texture_noise_response_results_3

analyses_config = [
    # ['plot_size_tuning_curve', {'h5_file':h5_file, 'neuron_id':neuron_id}],
    # ['size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'supp_thresh':supp_thresh}],
    # ['size_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['contrast_response_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'sort_by_std':sort_by_std, 'spread_to_plot':spread_to_plot}],
    # ['contrast_size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'shift_to_plot':shift_to_plot, 'low_contrast_id':low_contrast_id, 'high_contrast_id':high_contrast_id}],
    # ['orientation_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['orientation_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':high_center_contrast_id, 'norm_center_contrast_id':high_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':low_center_contrast_id, 'norm_center_contrast_id':low_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['black_white_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'neuron_depths':neuron_depths, 'SNR_thresh':SNR_thresh}],
    # ['texture_noise_response_results_1',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_2',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_3',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    ['recreate_histograms_second_order_orientation',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    # ['recreate_plots_general_suppresion_index',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    # ['recreate_plots_results_article_7',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}]

    

]


def execute_function(func_name, params):
    # Get the function object by name
    func = globals().get(func_name)
    
    # Check if the function exists
    if func is None or not callable(func):
        raise ValueError(f"Function '{func_name}' not found or not callable.")
    
    # Call the function with the parameters
    func(**params)