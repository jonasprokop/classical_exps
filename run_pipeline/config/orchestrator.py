################################################################################################
#####    IN THIS FILE, YOU WILL CONFIGURE THE EXPERIMENTS AND ANALYSES YOU WILL PERFORM    #####
#####             -----------------------------------------------------------              #####
#####                 The everything will be executed in the file "main.py"                #####
################################################################################################

from __future__ import annotations

## Variables
from run_pipeline.config.variables import *


## Numpy
import numpy as np

from typing import Callable, Any

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
    # ['contrast_size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'contrasts':contrasts_article_1, 'radii':radii, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['orientation_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'ori_shifts':ori_shifts, 'contrast':experiment_2_contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['center_contrast_surround_suppression_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts_ccss':center_contrasts_ccss, 'surround_contrast':surround_contrast, 'phases':phases, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['black_white_preference_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'seed':seed}],
    # ['texture_noise_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'directory_imgs':directory_imgs, 'overwrite':overwrite, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'num_samples':num_samples, 'img_res':img_res, 'device':device}]
    # ['get_all_grating_parameters_with_modulator', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],
    # ['get_all_grating_parameters_with_modulator', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],
    # ['get_surround_contracst_facilitation', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}],

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
    [
        "perform_analysis_nature_and_interactions",
        {
            "h5_file": h5_file,
            "neuron_ids": neuron_ids,

            # axes for loaders (mandatory if you load those datasets)
            "center_contrasts": center_contrasts,
            "surround_contrasts": surround_contrasts,
            "contrasts": contrasts_article_1,   
            "radii": radii,

            # thresholds
            "fit_err_thresh": fit_err_thresh,
            "supp_thresh": supp_thresh,

            # CR spread selection
            "sort_by_std": sort_by_std,
            "spread_to_plot": spread_to_plot,

            # CST shift selection
            "shift_to_plot": shift_to_plot,
            "low_contrast_id": low_contrast_id,
            "high_contrast_id": high_contrast_id,

            # output
            "output_dir": main_dir + "/results/nature_and_interactions",

            # run flags 
            "run": {
                "check_h5": True,
                "load_size_results": True,
                "load_contrast_response": True,
                "load_contrast_size_tuning": True,

                "size_results_1": True,
                "size_results_2": True,
                "contrast_response_results_1": True,
                "contrast_size_tuning_results_1": True,
                "overwrite_size_results_1": True,  # if True, will overwrite the size_results_1 in H5 with the newly computed one (useful if you change the analysis code and want to update the results in H5)

                "export_excel": False, 
                # "excel_path": main_dir + "/results/nature_and_interactions/summary.xlsx",

                "plot_all_contrast_response":True,
                "plot_all_contrast_size_tuning":True,
                "print_loaded_inventory": True,
                "load_size_tuning_curves": True,
                "size_tuning_plots": True,
                "plot_size_tuning_curves":True,

                "fit_size_tuning": True,         # refit now + overwrite in H5
                "use_size_tuning_fits": True,    # downstream uses fits if available
                "fit_force": True,               # if True, refit even if already in H5
                "fit_strict": True,               # if True, error if missing fits when use_size_tuning_fits

                
                "fit_contrast_response": True,
                "use_contrast_response_fits": True,
                "contrast_size_tuning_fit_model":"gain" ,  # "gain" or "size" or "uniform"

                "fit_contrast_size_tuning": True,
                "use_contrast_size_tuning_fits": True,
                "fit_contrast_size_tuning_force": True,
                "fit_contrast_size_tuning_strict": True,
            },
        }
    ],

    # ['orientation_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['orientation_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':high_center_contrast_id, 'norm_center_contrast_id':high_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':low_center_contrast_id, 'norm_center_contrast_id':low_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['black_white_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'neuron_depths':neuron_depths, 'SNR_thresh':SNR_thresh}],
    # ['texture_noise_response_results_1',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_2',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_3',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    # ['recreate_histograms_second_order_orientation',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    # ['recreate_histograms_second_order_orientation',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}],
    # ['recreate_plots_results_article_7',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}]
]


# --------------------------------------------------------------------------------------
# Explicit registry (string -> callable), no globals(), no star-import magic.
# --------------------------------------------------------------------------------------

# Experiments (pre-analysis)
from classical_exps.functions.experiments.a_preanalysis import (
    get_all_grating_parameters,
    get_preferred_position,
)

# Experiments (article 1)
from classical_exps.functions.article_1.experiments.size_tuning import (
    size_tuning_experiment_all_phases,
)

from classical_exps.functions.article_1.experiments.contrast_response import (
    contrast_response_experiment,
)
from classical_exps.functions.article_1.experiments.contrast_size_tuning import (
    contrast_size_tuning_experiment_all_phases,   
)


# Experiments (article 2)
from classical_exps.functions.article_2.experiments.orientation_tuning import (
    orientation_tuning_experiment_all_phases,
    )
from classical_exps.functions.article_2.experiments.ccss import (
    center_contrast_surround_suppression_experiment,
)

# Experiments (other)
from classical_exps.functions.experiments.d_3rd_article import black_white_preference_experiment
from classical_exps.functions.experiments.e_4th_article import texture_noise_response_experiment
from classical_exps.functions.experiments.f_5th_article import get_all_grating_parameters_with_modulator
from classical_exps.functions.experiments.g_6th_article import get_surround_contrast_facilitation

# Analyses (your Nature+Interactions pipeline)
from classical_exps.functions.article_1.analysis.main import perform_analysis_nature_and_interactions

# Analyses (filters)
from classical_exps.functions.analysis.a_filtering_fucntions import (
    filter_fitting_error,
    filter_no_supp_neurons,
    filter_low_supp_neurons,
    filter_SNR,
)

# Analyses (optional results helpers that appear in comments / typical usage)
from classical_exps.functions.article_1.analysis.step_size_tuning import (
    size_tuning_results_1,
    size_tuning_results_2,
)
from classical_exps.functions.article_1.analysis.step_contrast_response import contrast_response_results_1
from classical_exps.functions.article_1.analysis.step_contrast_size import contrast_size_tuning_results_1

from classical_exps.functions.article_2.analysis.orientation_tuning import (
    orientation_tuning_results_1,
    orientation_tuning_results_2,
)
from classical_exps.functions.article_2.analysis.ccss import (
    ccss_results_1,
    ccss_results_2,
    )

FUNCTION_REGISTRY: dict[str, Callable[..., Any]] = {
    # Experiments: minimal list you said you use
    "get_all_grating_parameters": get_all_grating_parameters,
    "get_preferred_position": get_preferred_position,
    "size_tuning_experiment_all_phases": size_tuning_experiment_all_phases,
    "contrast_response_experiment": contrast_response_experiment,
    "contrast_size_tuning_experiment_all_phases": contrast_size_tuning_experiment_all_phases,
    "orientation_tuning_experiment_all_phases": orientation_tuning_experiment_all_phases,
    "center_contrast_surround_suppression_experiment": center_contrast_surround_suppression_experiment,
    "black_white_preference_experiment": black_white_preference_experiment,
    "texture_noise_response_experiment": texture_noise_response_experiment,
    "get_all_grating_parameters_with_modulator": get_all_grating_parameters_with_modulator,

    # NOTE: your config has a typo: contracst -> contrast. Keep backwards compatible alias:
    "get_surround_contracst_facilitation": get_surround_contrast_facilitation,
    # Also register the correct spelling, so new configs can be sane:
    "get_surround_contrast_facilitation": get_surround_contrast_facilitation,

    # Analyses
    "perform_analysis_nature_and_interactions": perform_analysis_nature_and_interactions,

    # Optional filters
    "filter_fitting_error": filter_fitting_error,
    "filter_no_supp_neurons": filter_no_supp_neurons,
    "filter_low_supp_neurons": filter_low_supp_neurons,
    "filter_SNR": filter_SNR,

    # Optional results helpers
    "size_tuning_results_1": size_tuning_results_1,
    "size_tuning_results_2": size_tuning_results_2,
    "contrast_response_results_1": contrast_response_results_1,
    "contrast_size_tuning_results_1": contrast_size_tuning_results_1,
    "orientation_tuning_results_1": orientation_tuning_results_1,
    "orientation_tuning_results_2": orientation_tuning_results_2,
    "ccss_results_1": ccss_results_1,
    "ccss_results_2": ccss_results_2,

}


def _validate_configs() -> None:
    """Fail fast if configs reference unknown functions."""
    missing: list[str] = []

    for item in experiments_config:
        if not item:
            continue
        name = item[0]
        if name not in FUNCTION_REGISTRY:
            missing.append(name)

    for item in analyses_config:
        if not item:
            continue
        name = item[0]
        if name not in FUNCTION_REGISTRY:
            missing.append(name)

    if missing:
        missing_unique = sorted(set(missing))
        known = sorted(FUNCTION_REGISTRY.keys())
        msg = (
            "Orchestrator config references unknown function name(s):\n"
            f"  missing = {missing_unique}\n\n"
            "Fix: add them to FUNCTION_REGISTRY or rename in configs.\n"
            f"  known (registry) = {known}\n"
        )
        raise KeyError(msg)


_validate_configs()


def execute_function(func_name: str, params: dict[str, Any]) -> None:
    func = FUNCTION_REGISTRY.get(func_name)
    if func is None:
        raise KeyError(f"Function '{func_name}' not registered in FUNCTION_REGISTRY.")
    func(**params)