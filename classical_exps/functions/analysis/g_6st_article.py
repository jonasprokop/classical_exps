############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import scipy.interpolate
import scipy.optimize
import torch
import math
## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.experiments import get_GSF_surround_AMRF
## Plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
import os
import pandas as pd
import openpyxl
import datetime

from scipy.optimize import differential_evolution, minimize, Bounds
from scipy.stats.qmc import LatinHypercube
import numpy as np, time

from scipy.optimize import basinhopping, Bounds
from scipy.stats.qmc import LatinHypercube
import numpy as np
import time


def recreate_plots_results_article_7(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2,
    size=2.67,
    device=None,
    fit = True,
    n_trials=1000,
    plot_tunning_curves=True, 
    store_intermediate_results = True
):
    
    # Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    # Groups to fill
    group_facilitation = "/surround_contrast_group_facilitation" 
    subgroup_facilitation_st_results_high_path = group_facilitation + "/results/high"
    subgroup_facilitation_st_results_low_path  = group_facilitation + "/results/low"
    subgroup_facilitation_st_curves_high_path  = group_facilitation + "/curves/high"
    subgroup_facilitation_st_curves_low_path  = group_facilitation +"/curves/low"

    subgroup_facilitation_st_curves_size_tunning_path  = group_facilitation + "/curves/size_tunning"
    subgroup_facilitation_st_results_size_tunning_path = group_facilitation + "/results/size_tunning"

    subgroup_facilitation_st_curves_HH_path  = group_facilitation +"/curves/HH"
    subgroup_facilitation_st_curves_LH_path  = group_facilitation +"/curves/LH"
    subgroup_facilitation_st_curves_LL_path  = group_facilitation +"/curves/LL"

    subgroup_facilitation_st_results_HH_path  = group_facilitation +"/results/HH"
    subgroup_facilitation_st_results_LH_path  = group_facilitation +"/results/LH"
    subgroup_facilitation_st_results_LL_path  = group_facilitation +"/results/LL"
    
    subgroup_annular_tunning_fits = group_facilitation +"/annular_tunning_fits"
    subgroup_annular_tunning_residuals = group_facilitation +"/annular_tunning_residuals"
    subgroup_patch_tunning_fits= group_facilitation +"/patch_tunning_fits"
    subgroup_patch_tunning_residuals = group_facilitation +"/patch_tunning_residuals"
    

    minimal_receptive_fields_path = group_facilitation + "/minimal_receptive_fields"
    contrast_values_path = group_facilitation + "/contrast_values"


    # if refitting delete the related groups
    if fit:
        args_str = f"percentage_of_guassian_energy=/contrasts=/radii=/pix"
        clear_group(h5_file,subgroup_annular_tunning_fits)
        clear_group(h5_file,subgroup_patch_tunning_fits)
        clear_group(h5_file,subgroup_annular_tunning_residuals)
        clear_group(h5_file,subgroup_patch_tunning_residuals)
        group_init(h5_file=h5_file, group_path=subgroup_annular_tunning_fits, group_args_str=args_str)
        group_init(h5_file=h5_file, group_path=subgroup_patch_tunning_fits, group_args_str=args_str)
        group_init(h5_file=h5_file, group_path=subgroup_annular_tunning_residuals, group_args_str=args_str)
        group_init(h5_file=h5_file, group_path=subgroup_patch_tunning_residuals, group_args_str=args_str)


    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    filtered_neuron_ids = neuron_ids
    # ## Filter on fitting error
    # filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    with h5py.File(h5_file, "a") as file:

        ## Access the groups 
        subgroup_facilitation_st_results_high = file[subgroup_facilitation_st_results_high_path]
        subgroup_facilitation_st_results_low   = file[subgroup_facilitation_st_results_low_path]

        subgroup_facilitation_st_curves_size_tunning = file[subgroup_facilitation_st_curves_size_tunning_path]
        subgroup_facilitation_st_results_size_tunning = file[subgroup_facilitation_st_results_size_tunning_path]

        subgroup_facilitation_st_curves_high = file[subgroup_facilitation_st_curves_high_path]
        subgroup_facilitation_st_curves_low = file[subgroup_facilitation_st_curves_low_path]

        subgroup_facilitation_st_curves_HH = file[subgroup_facilitation_st_curves_HH_path]
        subgroup_facilitation_st_curves_LH = file[subgroup_facilitation_st_curves_LH_path]
        subgroup_facilitation_st_curves_LL = file[subgroup_facilitation_st_curves_LL_path]

        subgroup_facilitation_st_results_HH = file[subgroup_facilitation_st_results_HH_path]
        subgroup_facilitation_st_results_LH = file[subgroup_facilitation_st_results_LH_path]
        subgroup_facilitation_st_results_LL = file[subgroup_facilitation_st_results_LL_path]
        minimal_receptive_fields = file[minimal_receptive_fields_path]
        contrast_values = file[contrast_values_path]    

        annular_tunning_fits = file[subgroup_annular_tunning_fits]
        annular_tunning_residuals = file[subgroup_annular_tunning_residuals]

        patch_tunning_fits = file[subgroup_patch_tunning_fits]
        patch_tunning_residuals = file[subgroup_patch_tunning_residuals]


        # Results collection
        data_patch_tuning_curves = {
            "data": {},
            "fit": {}
        }

        results_patch_tuning_curves = {
            "data": {},
            "fit": {}
        }

        data_annular_tuning_curves = {
            "data": {
                "HH": {},
                "LH": {},
                "LL": {}
            },
            "fit": {
                "HH": {},
                "LH": {},
                "LL": {}
            }
        }

        results_annular_tuning_curves = {
            "data": {
                "HH": {},
                "LH": {},
                "LL": {}
            },
            "fit": {
                "HH": {},
                "LH": {},
                "LL": {}
            }
        }

        # checks collection
        counters = {
            "skipped": {
                "on_curves": 0,
                "on_MRFs": 0,

            },
            "status": {
                "all": 0,
                "success": 0,
                "failed": 0,
                "success_low_contrast": 0,
                "failed_low_contrast": 0,
            }
        }

                
        for neuron_id in filtered_neuron_ids:

            if neuron_id != 111:
                continue

            neuron = f"neuron_{neuron_id}"
            counters["status"]["all"] += 1

            ensure_dict_path(data_patch_tuning_curves, "data", neuron)
            ensure_dict_path(data_patch_tuning_curves, "fit", neuron)

            ensure_dict_path(results_patch_tuning_curves, "data", neuron)
            ensure_dict_path(results_patch_tuning_curves, "fit", neuron)

            for cond in ["HH", "LH", "LL"]:
                ensure_dict_path(data_annular_tuning_curves, cond, "data", neuron)
                ensure_dict_path(data_annular_tuning_curves, cond, "fit", neuron)

            for cond in ["HH", "LH", "LL"]:
                ensure_dict_path(results_annular_tuning_curves, "data", cond, neuron)
                ensure_dict_path(results_annular_tuning_curves, "fit", cond, neuron)

            # Check if the neuron exists in the required groups+
            # load data from the groups

            try:
                # Extract responses for each condition
                radii_HH, responses_HH = subgroup_facilitation_st_curves_HH[neuron][:]
                radii_LH, responses_LH = subgroup_facilitation_st_curves_LH[neuron][:] 
                radii_LL, responses_LL = subgroup_facilitation_st_curves_LL[neuron][:]

                radii_low_contrast, responses_low_contrast =  subgroup_facilitation_st_curves_low[neuron][:]
                radii_high_contrast, responses_high_contrast =  subgroup_facilitation_st_curves_high[neuron][:]

                radii_normal, responses_normal =  subgroup_facilitation_st_curves_size_tunning[neuron][:]
                mrf_radius_size_tunning, ratio_size_tunning, expected_rf = subgroup_facilitation_st_results_size_tunning[neuron][:]
                
                data_annular_tuning_curves["HH"]["data"][neuron].update({"radii" : radii_HH, "responses" : responses_HH})
                data_annular_tuning_curves["LH"]["data"][neuron].update({"radii" : radii_LH, "responses" : responses_LH})
                data_annular_tuning_curves["LL"]["data"][neuron].update({"radii" : radii_LL, "responses" : responses_LL})

            except:
                # print(f"Skipping {neuron} due to missing data in curves.")
                counters["skipped"]["on_curves"] += 1
                continue
            
            try:
                # Extract low contrast response
                low_contrast, high_contrast = contrast_values[neuron]
                data_patch_tuning_curves["data"][neuron].update({"low_contrast" : low_contrast, "high_contrast": high_contrast})

            except:
                print(f"Skipping {neuron} due to missing contrast values.")
                counters["skipped"]["on_contrasts"] += 1
                continue

            try:
                # Extract minimal receptive field radius
                mRF_radius = minimal_receptive_fields[neuron][0]
                data_patch_tuning_curves["data"][neuron].update({"mrf": mRF_radius})
                ratio_gauss = mRF_radius / expected_rf

            except:
                print(f"Skipping {neuron} due to missing minimal receptive field data.")
                counters["skipped"]["on_MRFs"] += 1
                continue
            
            try:
                # Extract sRFhigh and sRFlow values
                sRFlow = subgroup_facilitation_st_results_HH[neuron][1]
                sRFhigh = subgroup_facilitation_st_results_HH[neuron][2]

                data_patch_tuning_curves["data"][neuron].update({"sRFlow":sRFlow})
                data_patch_tuning_curves["data"][neuron].update({"sRFhigh":sRFhigh})

            except:
                raise ValueError(f"There are srfs missing for {neuron}.")

            try:
                # extract gsf
                _, GSF_low, GSF_high, _, _ = subgroup_facilitation_st_results_HH[neuron][:]

                data_patch_tuning_curves["data"][neuron].update({"gsf_low": GSF_low,"gsf_high": GSF_high})

            except:
                raise ValueError(f"There are GSFs missing for {neuron}.")
            
            try:
                R_full_grating_H = subgroup_facilitation_st_results_HH[neuron][4]
                R_full_grating_L = subgroup_facilitation_st_results_LL[neuron][4]

                data_annular_tuning_curves["HH"]["data"][neuron].update({"full_grating":R_full_grating_H}) 
                data_annular_tuning_curves["LH"]["data"][neuron].update({"full_grating":R_full_grating_L})
                data_annular_tuning_curves["LL"]["data"][neuron].update({"full_grating":R_full_grating_L})

            except:
                raise ValueError(f"There is full grating response missing for {neuron}.")

            try:
                # Extract center alone response 
                R_ctr_HH = subgroup_facilitation_st_results_HH[neuron][3]
                R_ctr_LH = subgroup_facilitation_st_results_LH[neuron][3]
                R_ctr_LL = subgroup_facilitation_st_results_LL[neuron][3]
                
                data_annular_tuning_curves["HH"]["data"][neuron].update({"rctr":R_ctr_HH})
                data_annular_tuning_curves["LH"]["data"][neuron].update({"rctr":R_ctr_LH})
                data_annular_tuning_curves["LL"]["data"][neuron].update({"rctr":R_ctr_LL}) 
            except:
                raise ValueError(f"There are center only responses data missing for {neuron}.")


            # first we get the results of the patch size tunning experiment, 

            # we do them both with the experimental data as in per article
            # and also fit them and analyse the fit to do comparison
            
            responses_fit_low_contrast = None
            params_low_contrast_curve = None
            residual_low_contrast = None

            if responses_low_contrast is not None:
                results_low_contrast_raw_data = analyse_low_contrast_facilitation(radii_low_contrast, responses_low_contrast, data_annular_tuning_curves["LL"]["data"][neuron]["rctr"])
                
                if results_low_contrast_raw_data is not None:
                    results_patch_tuning_curves["data"][neuron].update({"facilitation_strength" :results_low_contrast_raw_data["facilitation_strength"]})

                # Patch size tunning experiment fit
                if fit:
                    params_low_contrast_curve, residual_low_contrast = fit_the_curve_one_inhibition_tdog(radii_low_contrast, responses_low_contrast, n_trials, neuron, None)

                    params_low_contrast_curve = np.asarray(params_low_contrast_curve, dtype=np.float64)
                    residual_low_contrast = float(residual_low_contrast)

                    patch_tunning_fits.create_dataset(name=neuron, data=params_low_contrast_curve)

                    patch_tunning_residuals.create_dataset(name=neuron, data=residual_low_contrast)
                
                else:
                    # Load saved fits
                    try:
                        # Extract low contrast response
                        params_low_contrast_curve = patch_tunning_fits[neuron]
                        residual_low_contrast = patch_tunning_residuals[neuron]
                        params_low_contrast_curve, residual_low_contrast = None, None
                        print(f"Loaded fit for low contrast patch tuning curve for {neuron} with params: {residual_low_contrast}")
                        residual_low_contrast = np.asarray(residual_low_contrast, dtype=np.float64)
                        residual_low_contrast = residual_low_contrast.item()
                        residual_low_contrast = float(residual_low_contrast)

                    except:
                        params_low_contrast_curve, residual_low_contrast = None, None

                if params_low_contrast_curve is None:
                    counters["status"]["failed_low_contrast"] += 1
                    # print(f"Fit low contrast failed for {neuron})")

                # if fit exists, create the fitted analyses for comparison
                else:
                    counters["status"]["success_low_contrast"] += 1
                    # print(f"Fit for low cotrast patch tunning curve done for {neuron} with params: {params_low_contrast}")
                    responses_fit_low_contrast = dog_oneinh(radii_low_contrast, *params_low_contrast_curve)
                    results_low_contrast_fit = analyse_low_contrast_facilitation(radii_low_contrast, responses_fit_low_contrast, data_annular_tuning_curves["LL"]["data"][neuron]["rctr"])

                if plot_tunning_curves:
                    plot_size_tuning_contrast_panel(
                        radii_low=radii_low_contrast,
                        responses_low=responses_low_contrast,
                        GSF_low=GSF_low,
                        low_contrast=low_contrast,
                        radii_norm=radii_normal,
                        responses_norm=responses_normal,
                        mrf_radius=mrf_radius_size_tunning,
                        ratio=ratio_size_tunning,
                        expected_rf=expected_rf,
                        radii_high=radii_high_contrast,
                        responses_high=responses_high_contrast,
                        GSF_high=GSF_high,
                        high_contrast=high_contrast,
                        neuron=neuron,
                        mrf_gauss=mRF_radius,
                        ratio_gauss=ratio_gauss,
                        y_label="Response magnitude",
                        x_label="Stimulus radius (°)",
                    )
                    
            # Then we iterate over conditions and for each condition we make the fit and save its plot
            # lets also add one final plot where we do them together
            for cond, (radii, responses) in {
                "HH": (radii_HH, responses_HH),
                "LH": (radii_LH, responses_LH),
                "LL": (radii_LL, responses_LL),
                }.items():  

                params, residual, responses_fit = None, None, None

                if fit:
                    params, residual = fit_the_curve_dog(radii, responses, n_trials, neuron, cond)
                    if params is None:
                        counters["status"]["failed"] += 1
                        # print(f"Fit failed for {neuron} in condition {cond}")
                    else:
                        counters["status"]["success"] += 1


                    data_annular_tuning_curves[cond]["fit"][neuron] = [params, residual]
                    name_neuron = neuron + "_" + cond
                                        
                    params = np.asarray(params, dtype=np.float64)
                    # residual = residual.item()
                    residual = residual


                    if params is None:
                        annular_tunning_fits.create_dataset(name=name_neuron, data=np.full((0,), np.nan))
                    else:
                        annular_tunning_fits.create_dataset(name=name_neuron, data=params)

                    if residual is None:
                        annular_tunning_residuals.create_dataset(name=name_neuron, data=np.nan)
                    else:
                        annular_tunning_residuals.create_dataset(name=name_neuron, data=residual)

                else:
                    # Load saved fits
                    try:
                        # Extract low contrast response
                        name_neuron = neuron + "_" + cond

                        params = annular_tunning_fits[name_neuron]

                        residual = annular_tunning_residuals[name_neuron]

                        params = np.asarray(params, dtype=np.float64)
                        residual = residual.item()
                        residual = float(residual)

                    except:
                        params, residual = None, None
                    
                data_annular_tuning_curves[cond]["fit"][neuron] = [params, residual]

                result_annular_tunning_fit, result_annular_tunning_raw = None, None

                # analyse facilitation on the fited annular responses
                if params is not None:
                    # fit the curve
                    responses_fit = dog(radii, *params)
                    result_annular_tunning_fit = find_facilitation_annular_tunning_curve(responses_fit, data_annular_tuning_curves[cond]["data"][neuron]["rctr"], radii)

                # raw results to check the results on them
                result_annular_tunning_raw = find_facilitation_annular_tunning_curve(responses, data_annular_tuning_curves[cond]["data"][neuron]["rctr"], radii)

                results__tuning_curves = store_annular_tuning_results(
                results_annular_tuning_curves,
                result_annular_tunning_fit,
                dtype="fit",  
                cond=cond,    
                neuron_id=neuron  
                    )
                
                results_annular_tuning_curves = store_annular_tuning_results(
                results_annular_tuning_curves,
                result_annular_tunning_raw,
                dtype="data",  
                cond=cond,    
                neuron_id=neuron  
                    )
                

                if plot_tunning_curves:
                    # plot the patch tunning curve
                    plot_tunning_curve(radii=radii,
                                        responses=responses,
                                        R_ctr=data_annular_tuning_curves[cond]["data"][neuron]["rctr"],
                                        response_full_grating=data_annular_tuning_curves[cond]["data"][neuron]["full_grating"], 
                                        params=params, 
                                        neuron=neuron, 
                                        cond=cond, 
                                        residual=residual, 
                                        type="annular"
                                        )
                    
            if plot_tunning_curves:
                curve_data = []
                for cond in ["HH", "LH", "LL"]:
                    try:
                        responses = data_annular_tuning_curves[cond]["data"][neuron]["responses"]
                        R_ctr = data_annular_tuning_curves[cond]["data"][neuron]["rctr"]
                        params, residual = data_annular_tuning_curves[cond]["fit"].get(neuron, [None, None])

                        curve_data.append({
                            'cond': cond,
                            'responses': responses,
                            'params': params,         
                            'residual': residual,     
                            'R_ctr': R_ctr
                        })

                    except KeyError as e:
                        print(f"Skipping {neuron} condition {cond} in combined plot due to missing key: {e}")
                        continue

                # Plot all available conditions
                if curve_data:
                    radii_ref = data_annular_tuning_curves["HH"]["data"][neuron]["radii"]

                    plot_tunning_curves_all_conditions(
                        radii_ref,
                        curve_data,
                        neuron,
                        type="annular"
                    )

                    plot_tunning_curves_all_conditions_panel(
                        radii_ref,
                        curve_data,
                        neuron,
                        type="annular"
                    )

                print(f"Completed processing for {neuron}.")


        print("\n--- Summary ---")
        print(f"{counters['status']['all']} total filtered neurons")
        print(f"{counters['status']['success']} annular fits successful")
        print(f"{counters['status']['failed']} annular fits failed")
        print(f"{counters['status']['success_low_contrast']} patch fits successful")
        print(f"{counters['status']['failed_low_contrast']} patch fits failed")
        print(f"{counters['status']['all'] - counters['status']['success'] - counters['status']['failed']} skipped before fitting\n")
        print(f"{counters['skipped']['on_curves']} skipped on missing tuning curves")

        # Compute sRFlow / sRFhigh ratio
        sRF_ratios = []
        for neuron, data in data_patch_tuning_curves["data"].items():
            try:
                print(data["sRFlow"])
                sRFlow = data["sRFlow"]
                sRFhigh = data["sRFhigh"]
                if sRFhigh != 0:
                    sRF_ratios.append(sRFlow / sRFhigh)
            except KeyError:
                continue
        sRF_ratios = np.array(sRF_ratios)

        # Common neurons that have values for both conditions and fit results
        shared_neurons = (
            set(results_annular_tuning_curves["fit"]["HH"].keys())
            & set(results_annular_tuning_curves["fit"]["LH"].keys())
            & set(results_annular_tuning_curves["fit"]["LL"].keys())
            & set(data_patch_tuning_curves["data"].keys())
        )

        # Prepare aligned values
        supp_radii_HH = []
        supp_radii_LH = []
        supp_radii_LL = []
        sRFhigh_vals = []

        for neuron in shared_neurons:
            try:
                R_HH = results_annular_tuning_curves["fit"]["HH"][neuron].get("suppression_radius_from_center_only")
                R_LH = results_annular_tuning_curves["fit"]["LH"][neuron].get("suppression_radius_from_center_only")
                R_LL = results_annular_tuning_curves["fit"]["LL"][neuron].get("suppression_radius_from_center_only")
                srf = data_patch_tuning_curves["data"][neuron].get("sRFhigh")

                if R_HH is not None and R_LH is not None and R_LL is not None and srf is not None:
                    supp_radii_HH.append(R_HH)
                    supp_radii_LH.append(R_LH)
                    supp_radii_LL.append(R_LL)
                    sRFhigh_vals.append(srf)
            except KeyError:
                continue

        sRFhigh_values = []
        sRFlow_values = []

        for neuron, data in data_patch_tuning_curves["data"].items():
            if "sRFhigh" in data and "sRFlow" in data:
                sRFhigh_values.append(data["sRFhigh"])
                sRFlow_values.append(data["sRFlow"])

        sRFhigh_values = np.array(sRFhigh_values)
        sRFlow_values = np.array(sRFlow_values)

        print(f"Number of shared neurons: {len(shared_neurons)}")
        print(f"Number of neurons with sRFhigh values: {len(sRFhigh_values)}")
        print(f"Number of neurons with sRFlow values: {len(sRFlow_values)}")    
        
        # Compute normalized ratios
        normalized_annular_ratio_at_suprresion_onset_LH = compute_normalized_annular_ratio(
            np.array(supp_radii_HH),
            np.array(supp_radii_LH),
            np.array(sRFhigh_vals)
        )

        normalized_annular_ratio_at_suprresion_onset_LL = compute_normalized_annular_ratio(
            np.array(supp_radii_HH),
            np.array(supp_radii_LL),
            np.array(sRFhigh_vals)
        )

        # do some filtering 
        for cond in ["HH", "LH", "LL"]:
            for neuron, res in results_annular_tuning_curves["fit"][cond].items():
                peak = res.get("R_peak")
                suppression = res.get("suppression_radius_from_center_only")
                fac_strength = res.get("facilitation_strength")

                res["non_zero_peak_radius"] = peak if peak is not None else None
                res["non_zero_suppression_radius_from_center_only"] = suppression if suppression is not None else None
                res["facilitation_strength_non_negative"] = fac_strength if fac_strength is not None and fac_strength >= 0 else None
                res["facilitation_strength_clipped_0"] = max(fac_strength, 0) if fac_strength is not None else None


        HH_width_center_only = []
        LH_center_only = []
        LL_center_only = []


        colors ={
        "HH":"#000000",    # Black
        "LH":"#8B0000",    # Dark Red
        "LL":"#708090",    # Slate Gray
            }


        for neuron in shared_neurons:
            try:
                r_hh = results_annular_tuning_curves["fit"]["HH"][neuron].get("non_zero_suppression_radius_from_center_only")
                r_lh = results_annular_tuning_curves["fit"]["LH"][neuron].get("non_zero_suppression_radius_from_center_only")
                r_ll = results_annular_tuning_curves["fit"]["LL"][neuron].get("non_zero_suppression_radius_from_center_only")

                if r_hh is not None and r_lh is not None and r_ll is not None:
                    HH_width_center_only.append(r_hh * 2)
                    LH_center_only.append(r_lh * 2)
                    LL_center_only.append(r_ll * 2)
            except KeyError:
                continue

        # Figure 5A: Histogram of near facilitation strength
        near_facilitation_strengths = []
        for neuron, res in results_patch_tuning_curves["data"].items():
            val = res.get("facilitation_strength")
            if val is not None:
                near_facilitation_strengths.append(val)
        near_facilitation_strengths = np.array(near_facilitation_strengths)

        plot_near_faci_histogram(near_facilitation_strengths)

        # Figure 5B: Far facilitation strengh
        far_facilitation_strength_H = []
        far_facilitation_strength_L = []

        for neuron in shared_neurons:
            fH = results_annular_tuning_curves["fit"]["HH"][neuron].get("facilitation_strength_clipped_0")
            fL = results_annular_tuning_curves["fit"]["LH"][neuron].get("facilitation_strength_clipped_0")
            if fH is not None and fL is not None:
                far_facilitation_strength_H.append(fH)
                far_facilitation_strength_L.append(fL)

        far_facilitation_strength_H = np.array(far_facilitation_strength_H)
        far_facilitation_strength_L = np.array(far_facilitation_strength_L)

        plot_far_faci_histogram(far_facilitation_strength_H, far_facilitation_strength_L, colors)

        # Figure 5C: Annulus inner radius at peak response in low contrast
        peaks_L = []
        cond = "LL"  # low contrast condition

        for neuron in shared_neurons:
            peak_val = results_annular_tuning_curves["fit"][cond][neuron].get("R_peak")
            if peak_val is not None:
                peaks_L.append(peak_val)

        peaks_L = np.array(peaks_L)

        plot_inner_radius_peak_response_histogram(peaks_L, colors, cond)

        # Figure 5D: Scatter plot of suppression vs. facilitation distances
        facilitation_onset_from_peak = []
        facilitation_onset_from_center = []
        suppression_onset = []

        for neuron in shared_neurons:
            fit_res = results_annular_tuning_curves["fit"]["LL"][neuron]
            f_peak = fit_res.get("facilitation_radius_from_peak")
            f_center = fit_res.get("facilitation_radius_from_center_only")
            sup = fit_res.get("suppression_radius_from_center_only")
            if f_peak is not None and f_center is not None and sup is not None:
                facilitation_onset_from_peak.append(f_peak)
                facilitation_onset_from_center.append(f_center)
                suppression_onset.append(sup)

        facilitation_onset_from_peak = np.array(facilitation_onset_from_peak)
        facilitation_onset_from_center = np.array(facilitation_onset_from_center)
        suppression_onset = np.array(suppression_onset)

        plot_suppresion_facilitation_distances_scatter(facilitation_onset_from_peak, facilitation_onset_from_center, suppression_onset)

        # Figure 6A Scatter plot of sRFhigh vs. sRFlow 
        plot_srfhigh_srflow_scatter(sRFhigh_values, sRFlow_values)

        # Figure 6B: Histogram of sRFlow/sRFhigh ratio
        plot_srfhigh_srflow_histogram(sRF_ratios)

        # Figure 6C:  Scatter plot of suppression onset (high vs. low contrast)
        plot_suppresion_onset_scatter_plot(HH_width_center_only, LH_center_only, LL_center_only)

        # Figure 6D: Histogram of normalized annulus width at suprresion onset
        plot_normalised_annulus_widths_suppresion_onset_histogram(
            normalized_annular_ratio_at_suprresion_onset_LH,
            normalized_annular_ratio_at_suprresion_onset_LL,
            colors
        )

        # # At last we saved intermediate data into easily accesible form
        # if store_intermediate_results:
        #     minimal_receptive_fields_dataframe = pd.DataFrame.from_dict(checking_data_MRFs, orient='index', columns=['mRF_radius']).reset_index()
        #     checking_data_contrasts_df = pd.DataFrame.from_dict(checking_data_contrasts, orient='index', columns=['low_contrast', "high_contrast"]).reset_index()
        #     gsf = pd.DataFrame.from_dict(checking_data_GFSs, orient='index', columns=['GSF_low', "GSF_high"]).reset_index()

        #     joined_df = pd.merge(minimal_receptive_fields_dataframe, checking_data_contrasts_df, on="index", suffixes=('_x', '_y'))
        #     joined_df = pd.merge(joined_df, gsf, on="index", suffixes=('_x', '_y'))
        #     joined_df.to_excel("/project/results/facilitation/contrasts_and_mrf.xls", index=False, engine='openpyxl')

        #     directory = "/project/results/facilitation" + "/"
        #     os.makedirs(directory, exist_ok=True)
        #     np.savetxt(directory + "srfratios.txt", sRF_ratios)



def ensure_dict_path(d, *keys):
    """
    Ensures that nested keys in a dict `d` exist, initializing them as empty dicts if needed.
    Returns the dict at the deepest level.
    """
    for key in keys:
        if key not in d:
            d[key] = {}
        d = d[key]
    return d

# ---------- model: two-inh, one sigma, hard-threshold ----------
def dog(s, A_E, A_I1, A_I2, sigma, T_I1, T_I2, baseline=0.0):
    g = np.exp(-s**2 / (2.0 * sigma**2))
    E  = A_E  * g
    I1 = np.maximum(A_I1 * g - T_I1, 0.0)
    I2 = np.maximum(A_I2 * g - T_I2, 0.0)
    return E - I1 - I2 + baseline

# # ---------- model: two-inh, central + inhibitory sigma, hard-threshold ----------
# def dog(s, A_E, A_I1, A_I2, sigma, sigma_surround, T_I1, T_I2, baseline=0.0):
#     # Center Gaussian
#     g = np.exp(-s**2 / (2.0 * sigma**2))
#     E  = A_E * g
#     I1 = np.maximum(A_I1 * g - T_I1, 0.0)
    
#     # Surround Gaussian for I2
#     g_surround = np.exp(-s**2 / (2.0 * sigma_surround**2))
#     I2 = np.maximum(A_I2 * g_surround - T_I2, 0.0)
    
#     return E - I1 - I2 + baseline

# ---------- robust + tail-weighted + variance-normalized ----------
def _huber(r, delta):
    a = np.abs(r)
    q = np.minimum(a, delta)
    return 0.5*q*q + delta*(a - q)

def _objective_factory(s, y, bounds_list, tail_power=2, huber_delta=None, normalize=True):
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    y_var  = float(np.nanvar(y)) if normalize else 1.0
    if y_var <= 0: y_var = 1.0

    smax   = max(1e-6, float(np.nanmax(np.abs(s))))
    w      = (np.abs(s)/smax)**tail_power if tail_power>0 else np.ones_like(s)
    w     /= w.mean()

    y_span = max(1e-6, np.nanmax(y) - np.nanmin(y))
    delta  = huber_delta if huber_delta is not None else 0.5*y_span

    lows  = np.array([b[0] for b in bounds_list])
    highs = np.array([b[1] for b in bounds_list])

    def objective(theta):
        if np.any(theta < lows) or np.any(theta > highs):
            return np.inf
        pred = dog(s, *theta)
        if not np.all(np.isfinite(pred)):
            return np.inf
        r = y - pred
        loss = _huber(r, delta) * w
        return float(np.mean(loss)) / y_var
    return objective

def fit_dog(
    radii, responses,
    target_residual=1e-3,
    max_de_time=800,          #  time for DE exploration
    de_maxiter=100000,       #  DE search
    popsize=100,              #  diversity
    tail_power=0.2,           #  tail-weighting, to fit the later responses with more weight (prevents underfitting the far surround faci)
    n_polish_perturb=15,      # local starts
    n_polish_lhs=15,          # LHS samples
    polish_maxiter=30000      # polish iterations
):

    s = np.asarray(radii, float)
    y = np.asarray(responses, float)

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_span = max(1e-3, y_max - y_min)
    s_max  = max(1e-3, float(np.nanmax(np.abs(s))))

    # bounds
    A_hi   = 50.0 * y_span
    sig_hi = 10.0 * s_max

    bounds_list = [
        (0, A_hi),   # A_E
        (0, A_hi),   # A_I1
        (0, A_hi),   # A_I2
        (1e-5, 50*sig_hi), # sigma
        (0, A_hi),   # T_I1
        (0, A_hi),   # T_I2
        (y_min - 1.5*y_span, y_max + 1.5*y_span)  # baseline
    ]

    # bounds_list = [
    # (0, A_hi),               # A_E
    # (0, A_hi),               # A_I1
    # (0, A_hi),               # A_I2
    # (1e-5, 50*sig_hi),       # sigma
    # (1e-5, 50*sig_hi),       # sigma_surround
    # (0, A_hi),               # T_I1
    # (0, A_hi),               # T_I2
    # (y_min - 0.5*y_span, y_max)  # baseline
    # ]

    lows  = np.array([b[0] for b in bounds_list])
    highs = np.array([b[1] for b in bounds_list])
    bounds = Bounds(lows, highs)

    # auto-huber
    huber_delta = 0.5 * y_span

    obj = _objective_factory(s, y, bounds_list, tail_power, huber_delta)

    # --- DE global search
    t0 = time.time()
    def de_cb(xk, conv):
        return (obj(xk) <= target_residual) or ((time.time()-t0) > max_de_time)

    de = differential_evolution(
        obj, bounds_list, maxiter=de_maxiter, popsize=popsize,
        tol=1e-6, mutation=(0.7,1.2), recombination=0.7,  
        updating='deferred', polish=False, seed=42,
        callback=de_cb, workers=1
    )
    x_best, best_loss = de.x, float(de.fun)

    # --- multi-start polish
    starts = [x_best]
    rng = np.random.default_rng(123)

    for _ in range(n_polish_perturb):
        pert = x_best + rng.normal(0, 0.1*(highs-lows), size=len(x_best))
        starts.append(np.clip(pert, lows, highs))

    if n_polish_lhs > 0:
        sampler = LatinHypercube(d=len(bounds_list), seed=7)
        lhs = sampler.random(n=n_polish_lhs)
        lhs = np.array([low+(high-low)*lhs[:,i] for i,(low,high) in enumerate(bounds_list)]).T
        for row in lhs:
            starts.append(np.clip(row, lows, highs))

    for x0 in starts:
        loc = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": polish_maxiter, "ftol":1e-9})
        if loc.success and float(loc.fun) < best_loss:
            best_loss = float(loc.fun)
            x_best = loc.x
        if best_loss <= target_residual:
            break

    # --- diagnostics
    yhat = dog(s, *x_best)
    mse = float(np.mean((y-yhat)**2))
    r2  = 1.0 - mse/(np.var(y) if np.var(y)>0 else 1.0)

    return {"success": True, "params": x_best,
            "residual_norm": best_loss, "mse": mse, "r2": r2}



# ---------- model: one-sigma, one-inhibition, hard-thresholded ----------
def dog_oneinh(s, A_E, A_I1, sigma, T_I1, baseline=0.0):
    g = np.exp(-s**2 / (2.0 * sigma**2))
    E  = A_E  * g
    I1 = np.maximum(A_I1 * g - T_I1, 0.0)
    return E - I1 + baseline


# ---------- robust + tail-weighted loss ----------
def _huber_oneinh(r, delta):
    a = np.abs(r)
    q = np.minimum(a, delta)
    return 0.5*q*q + delta*(a - q)

def _objective_factory_oneinh(s, y, bounds_list, tail_power=2):
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    y_span = max(1e-6, np.nanmax(y) - np.nanmin(y))
    delta  = 0.5 * y_span
    smax   = max(1e-6, float(np.nanmax(np.abs(s))))
    w      = (np.abs(s)/smax)**tail_power
    w     /= w.mean()

    lows  = np.array([b[0] for b in bounds_list])
    highs = np.array([b[1] for b in bounds_list])

    def objective(theta):
        if np.any(theta < lows) or np.any(theta > highs):
            return np.inf
        try:
            pred = dog_oneinh(s, *theta)
            if not np.all(np.isfinite(pred)): 
                return np.inf
            r = y - pred
            loss = _huber_oneinh(r, delta) * w
            return float(np.mean(loss))
        except Exception:
            return np.inf
    return objective


def fit_dog_oneinh(
    radii, responses,
    target_residual=1e-3,
    max_de_time=1000,
    de_maxiter=1000000,
    popsize=50,
    tail_power=1,
    polish=True,
    seed=42
):
    s = np.asarray(radii, float)
    y = np.asarray(responses, float)

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_span = max(1e-6, y_max - y_min)
    s_max  = max(1e-6, float(np.nanmax(np.abs(s))))

    # --- data-adaptive bounds ---
    A_hi    = 50.0 * y_span
    A_lo    = 0.0
    sig_lo  = 1e-3
    sig_hi  = 10.0 * s_max
    T_lo, T_hi = 0.0, A_hi
    base_lo, base_hi = (y_min - 5*y_span, y_max + 5*y_span)

    bounds_list = [
        (A_lo, A_hi),       # A_E
        (A_lo, A_hi),       # A_I1
        (sig_lo, sig_hi),   # sigma
        (T_lo, T_hi),       # T_I1
        (base_lo, base_hi)  # baseline
    ]
    lows  = np.array([b[0] for b in bounds_list])
    highs = np.array([b[1] for b in bounds_list])
    bounds = Bounds(lows, highs)

    # --- objective with tail weighting ---
    obj = _objective_factory_oneinh(s, y, bounds_list, tail_power=tail_power)

    # --- DE global search ---
    import time
    t0 = time.time()
    def de_cb(xk, conv):
        # early stop if good fit or time exhausted
        return (obj(xk) <= target_residual) or ((time.time()-t0) > max_de_time)

    de = differential_evolution(
        obj, bounds_list,
        maxiter=de_maxiter, popsize=popsize,
        tol=1e-6, mutation=(0.5, 1.0), recombination=0.5,
        updating='deferred', polish=False, seed=seed,
        callback=de_cb, workers=1
    )

    x0 = np.clip(de.x, lows, highs)
    best_res = float(de.fun)

    # --- local polish with L-BFGS-B ---
    if polish:
        loc = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 20000, "ftol": 1e-12})
        if loc.success and float(loc.fun) < best_res:
            return {"success": True, "params": loc.x, "residual": float(loc.fun)}

    return {"success": True, "params": x0, "residual": best_res}


def fit_the_curve_dog(radii, responses, n_trials, neuron, cond, threshold=0.2):
    try:
        fit_result = fit_dog(radii, responses)
        if fit_result['success']:
            params = fit_result['params']
            residual = fit_result.get('residual_norm', fit_result.get('residual', None))
            mse = fit_result.get('mse', None)

            # --- RMSE filter ---
            y = np.asarray(responses, float)
            y_range = np.nanmax(y) - np.nanmin(y)
            if y_range <= 0:
                nrmse = np.inf
            else:
                nrmse = np.sqrt(mse) / y_range if mse is not None else np.inf

            if nrmse <= threshold:
                msg = f"Fit OK for neuron {neuron}"
                if cond: msg += f" condition {cond}"
                print(f"{msg}: nRMSE={nrmse:.3f}, params={params}")
                residual = nrmse
            else:
                msg = f"Fit rejected (bad nRMSE={nrmse:.3f}) for neuron {neuron}"
                if cond: msg += f" condition {cond}"
                print(msg)
                params, residual = params, nrmse
        else:
            params, residual = None, None

    except Exception as e:
        if cond:
            print(f"Fit failed for neuron {neuron}, condition {cond}: {e}")
        else:
            print(f"Fit failed for neuron {neuron}: {e}")
        params, residual = None, None

    return params, residual

def fit_the_curve_one_inhibition_tdog(radii, responses, n_trials, neuron, cond, threshold=0.2):
    try:
        fit_result = fit_dog_oneinh(radii, responses)
        if fit_result['success']:
            params = fit_result['params']
            residual = fit_result['residual']

            # --- RMSE filter ---
            y = np.asarray(responses, float)
            y_span = max(1e-6, np.nanmax(y) - np.nanmin(y))
            nrmse = np.sqrt(residual) / y_span

            if nrmse <= threshold:
                if cond:
                    print(f"Fit successful for neuron {neuron} condition {cond}: "
                          f"{params}, residual={residual:.4f}, nRMSE={nrmse:.3f}")
                else:
                    print(f"Fit successful for neuron {neuron}: "
                          f"{params}, residual={residual:.4f}, nRMSE={nrmse:.3f}")
            else:
                if cond:
                    print(f"Fit rejected (nRMSE={nrmse:.3f} > 0.2) for neuron {neuron}, condition {cond}")
                else:
                    print(f"Fit rejected (nRMSE={nrmse:.3f} > 0.2) for neuron {neuron}")
                params = None
                residual = None
        else:
            params = None
            residual = None

    except Exception as e:
        if cond:
            print(f"Fit failed for neuron {neuron}, condition {cond}: {e}")
        else:
            print(f"Fit failed for neuron {neuron}: {e}")
        params = None
        residual = None

    return params, residual

def plot_size_tuning_contrast_panel(
    radii_low, responses_low, GSF_low, low_contrast,
    radii_norm, responses_norm, mrf_radius, ratio, expected_rf,
    radii_high, responses_high, GSF_high, high_contrast,
    neuron, mrf_gauss, ratio_gauss,
    y_label="Response magnitude", x_label="Stimulus radius (°)", 
    params_norm=None
):
    """
    Make a 3-panel plot comparing size tuning across low, normal, and high contrast.
    Includes GSF_low and GSF_high vertical lines and the mRF fit at normal contrast.
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # determine axis limits shared across panels
    all_radii = np.concatenate([radii_low, radii_norm, radii_high])
    all_responses = np.concatenate([responses_low, responses_norm, responses_high])
    x_min, x_max = np.min(all_radii), np.max(all_radii)
    y_min, y_max = np.min(all_responses), np.max(all_responses)
    y_margin = 0.2 * (y_max - y_min)

    # low
    ax = axs[0]
    ax.scatter(radii_low, responses_low, color="#0072B2", label="Responses", s=25)

    ax.axvline(GSF_low, color="black", linestyle="--", label=f"GSF low = {GSF_low:.2f}°")
    ax.axvline(expected_rf, color="green", linestyle="--", label=f"Expected RF = {expected_rf:.2f}°")

    ax.set_title(f"Low contrast size tunning, contrast = {low_contrast*100:.1f}%")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='y', which='both', labelleft=True)


    # high
    ax = axs[1]
    ax.scatter(radii_high, responses_high, color="#0072B2", label="Responses", s=25)
    ax.axvline(GSF_high, color="black", linestyle="--", label=f"GSF high = {GSF_high:.2f}°")
    ax.axvline(expected_rf, color="green", linestyle="--", label=f"Expected RF = {expected_rf:.2f}°")

    ax.set_title(f"High contrast size tunning, contrast = {high_contrast*100:.1f}%")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='y', which='both', labelleft=True)


    # classical
    ax = axs[2]
    ax.scatter(radii_norm, responses_norm, color="#0072B2", s=25, label="Responses")
    if params_norm is not None:
        radii_fit = np.linspace(x_min, x_max, 300)
        from scipy.stats import norm
        fit_response = dog(radii_fit, *params_norm)
        ax.plot(radii_fit, fit_response, color="orange", label="Fit")

    # Lines
    ax.axvline(mrf_radius, color="black", linestyle="--",
            label=f"mRF radius - size tuning: {float(mrf_radius):.2f}°")
    ax.axvline(float(mrf_gauss), color="red", linestyle="--",
            label=f"mRF radius - Gaussian fit: {float(mrf_gauss):.2f}°")
    ax.axvline(expected_rf, color="green", linestyle="--",
            label=f"Expected RF: {float(expected_rf):.2f}°")

    # Add ratio info as text-only entry in legend
    extra = plt.Line2D([], [], color='none', label=(
        f"SF ratio - size tuning: {float(ratio):.2f}\n"
        f"SF ratio - Gaussian fit: {float(ratio_gauss):.2f}"
    ))

    # Combine all entries
    handles, labels = ax.get_legend_handles_labels()
    handles.append(extra)
    labels.append(extra.get_label())

    ax.legend(handles, labels, frameon=True, facecolor="white", framealpha=0.8)
    ax.set_title("Full contrast size tunning")
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='y', which='both', labelleft=True)



    fig.suptitle(f"Size tuning across contrast levels", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save figure
    out_dir = f"/project/results/facilitation/contrast_panels/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{neuron}_contrast_tuning_panel.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_tunning_curve(
    radii,
    responses,
    R_ctr,
    response_full_grating,
    params,
    neuron,
    cond,
    residual,
    type="annular",
    contrast=None,
    gsf=None
):
    """Plot the size tuning curve and save it to a file."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # --- Base scatter ---
    plt.plot(radii, responses, 'o', label=f"{type.capitalize()} Size Tuning Curve", color='b')

    # --- Fitted curve ---
    y_values = list(responses)  # collect all values for auto-scaling
    if params is not None:
        radii_fit = np.linspace(min(radii), max(radii), 300)
        if type == "annular":
            fit_response = dog(radii_fit, *params)
        else:
            fit_response = dog_oneinh(radii_fit, *params)
        plt.plot(radii_fit, fit_response, '-', color='g', label="Fit")
        y_values.extend(fit_response)

    # --- Add reference lines ---
    if gsf is not None and contrast is not None:
        plt.axvline(x=gsf, color='green', linestyle='-.', label=f"GSF {contrast} = {gsf:.2f}°")
    else:
        plt.axhline(y=R_ctr, color='red', linestyle='--', label=f"R_center = {R_ctr:.2f}")
        plt.axhline(y=response_full_grating, color='blue', linestyle=':', label=f"Whole grating = {response_full_grating:.2f}")
        y_values.extend([R_ctr, response_full_grating])

    # --- Dynamic y-axis scaling ---
    ymin, ymax = min(y_values), max(y_values)
    yrange = ymax - ymin
    plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

    # --- Add residual info ---
    if residual is not None:
        plt.text(
            0.95, 0.05,
            f"Fit Residual: {residual:.3g}",
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=plt.gca().transAxes,
            fontsize=9,
            color='gray'
        )

    # --- Labels and title ---
    plt.xlabel(f"{type.capitalize()} width (°)")
    plt.ylabel("Neuronal Response")
    if cond is not None:
        plt.title(f"{type.capitalize()} Size Tuning Curve - {neuron} - {cond}")
    elif contrast is not None:
        plt.title(f"{type.capitalize()} Size Tuning Curve - {neuron} - {contrast} contrast")
    else:
        plt.title(f"{type.capitalize()} Size Tuning Curve - {neuron}")

    plt.legend()
    plt.grid(True)

    # --- Directory and save path ---
    if cond is not None:
        base = f"/project/results/facilitation/{cond}_{type}_size_tunning_curves/{neuron}/"
        if contrast is not None:
            plot_name = f"{cond}_{type}_size_tunning_curve_{neuron}_{contrast}_contrast.png"
        else:
            plot_name = f"{cond}_{type}_size_tunning_curve_{neuron}.png"
    else:
        base = f"/project/results/facilitation/{type}_size_tunning_curves/{neuron}/"
        if contrast is not None:
            plot_name = f"{type}_size_tunning_curve_{neuron}_{contrast}_contrast.png"
        else:
            plot_name = f"{type}_size_tunning_curve_{neuron}.png"

    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, plot_name))
    plt.close()


def plot_tunning_curves_all_conditions(
    radii,
    curves_data,
    neuron,
    type="annular"
):
    """
    Plot size tuning curves for multiple conditions on a single plot.
    """
    colors = [
        "#000000",    # Black
        "#8B0000",    # Dark Red
        "#708090",    # Slate Gray
    ]

    plt.figure(figsize=(8, 5))

    for i, curve in enumerate(curves_data):
        responses = curve['responses']
        params = curve.get('params')
        residual = curve.get('residual')
        cond = curve.get('cond')
        R_ctr = curve.get('R_ctr', None)

        label = f"{cond} Response"
        plt.plot(radii, responses, 'o', label=label, color=colors[i])

        if params is not None:
            radii_fit = np.linspace(min(radii), max(radii), 300)
            fit_response = dog(radii_fit, *params)
            plt.plot(radii_fit, fit_response, '-', color=colors[i], alpha=0.7, label=f"{cond} Fit")
        
        # if R_ctr is not None:
        #     plt.axhline(y=R_ctr, color=colors[i], linestyle='--', linewidth=1.2, alpha=0.5, label=f"{cond} R_ctr")
        
        if residual is not None:
            plt.text(
                0.95, 0.9 - 0.1 * i,
                f"{cond} Residual: {residual}",
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=plt.gca().transAxes,
                fontsize=5,
                color=colors[i]
            )

    plt.xlabel(f"{type.capitalize()} width (°)")
    plt.ylabel("Neuronal Response")
    plt.title(f"{type.capitalize()} Size Tuning Curves - {neuron}")
    plt.legend()
    plt.grid(True)

    directory = f"/project/results/facilitation/combined_{type}_curves/{neuron}/"
    os.makedirs(directory, exist_ok=True)
    plot_name = directory + f"{type}_combined_size_tunning_{neuron}.png"

    plt.savefig(plot_name)
    plt.close()

def plot_tunning_curves_all_conditions_panel(radii, curve_data, neuron, type="annular"):
    """
    Plot size tuning curves for multiple conditions side-by-side (1xN panel)
    with normalized y-limits that include R_ctr.
    """
    n = len(curve_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # --- Collect all y-values for normalization (include R_ctr!) ---
    all_y = []
    for cd in curve_data:
        all_y.extend(cd["responses"])
        if cd["params"] is not None:
            radii_fit = np.linspace(min(radii), max(radii), 300)
            fit_response = dog(radii_fit, *cd["params"]) if type == "annular" else dog_oneinh(radii_fit, *cd["params"])
            all_y.extend(fit_response)
        if cd.get("R_ctr") is not None:
            all_y.append(cd["R_ctr"])

    # --- Compute consistent y-limits ---
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_margin = 0.2 * (y_max - y_min)
    y_lim = (y_min - y_margin, y_max + y_margin)

    # --- Plot each subplot ---
    for ax, cd in zip(axes, curve_data):
        cond = cd["cond"]
        responses = cd["responses"]
        params = cd["params"]
        residual = cd["residual"]
        R_ctr = cd["R_ctr"]

        ax.plot(radii, responses, 'o', label=f"Response magnitude", color='b')

        if params is not None:
            radii_fit = np.linspace(min(radii), max(radii), 300)
            fit_response = dog(radii_fit, *params) if type == "annular" else dog_oneinh(radii_fit, *params)
            ax.plot(radii_fit, fit_response, '-', color='g', label="Fit")

        if R_ctr is not None:
            ax.axhline(y=R_ctr, color='r', linestyle='--', label=f"R_ctr = {R_ctr:.2f}")

        if residual is not None:
            ax.text(
                0.95, 0.05,
                f"Residual: {residual:.2f}",
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=9, color='gray'
            )

        ax.set_ylim(y_lim)
        ax.set_title(f"{type.capitalize()} tuning - {cond}")
        ax.set_xlabel(f"{type.capitalize()} width (°)")
        ax.grid(True)
        ax.legend()
        ax.tick_params(axis='y', which='both', labelleft=True)

    axes[0].set_ylabel("Response magnitude")
    fig.suptitle(f"{type.capitalize()} tuning curves - {neuron}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    directory = f"/project/results/facilitation/{type}_curves_panel/"
    os.makedirs(directory, exist_ok=True)
    plot_name = os.path.join(directory, f"{neuron}_{type}_curves_panel.png")
    plt.savefig(plot_name)
    plt.close(fig)

def find_facilitation_annular_tunning_curve(responses, 
                                            R_ctr,
                                            radii, 
    ):

    # Global peak and minimum
    R_peak = np.max(responses)
    peak_idx = np.argmax(responses)
    global_min_idx = np.argmin(responses)

    # Suppression: drop below 90% of R_ctr or R_peak
    suppression_idx_from_center = np.where(responses < 0.9 * R_ctr)[0]
    suppression_idx_from_peak = np.where(responses < 0.9 * R_peak)[0]

    facilitation_peak_idx = None
    facilitation_onset_from_center = None
    facilitation_offset = None

    # Find peak before global minimum
    pre_min_indices = np.arange(global_min_idx)
    if pre_min_indices.size == 0:
        return None
    
    pre_min_responses = responses[pre_min_indices]
    peak_idx_relative = np.argmax(pre_min_responses)
    peak_response = pre_min_responses[peak_idx_relative]

    # print(f"DEBUG: Peak response before global minimum: {peak_response} at index {peak_idx_relative} (relative to pre-min indices)")

    if peak_response > R_ctr:
        # If peak is above R_ctr, we consider it as a facilitation peak
        facilitation_peak_idx = pre_min_indices[peak_idx_relative]

        # Onset: look backward from peak for last point <= R_ctr
        for i in range(facilitation_peak_idx - 1, -1, -1):
            if responses[i] <= R_ctr:
                facilitation_onset_from_center = radii[i]
                # print(f"DEBUG: Facilitation onset from center: {facilitation_onset_from_center} at index {i}")
                break

        # look forward from peak for first point <= R_ctr
        for i in range(facilitation_peak_idx + 1, len(responses)):
            if responses[i] <= R_ctr:
                facilitation_offset = radii[i]
                # print(f"DEBUG: Facilitation offset: {facilitation_offset} at index {i}")
                break

    # Suppression onset
    if suppression_idx_from_center.size > 0:
        suppression_onset = radii[suppression_idx_from_center[0]]
        # print(f"DEBUG: Suppression onset: {suppression_onset} at index {suppression_idx_from_center[0]}")

    # Suppression radius (from center and peak)
    if suppression_idx_from_center.size > 0:
        suppression_radii_from_center_only = radii[suppression_idx_from_center[0]]
        # print(f"DEBUG: Suppression radius from center: {suppression_radii_from_center_only} at index {suppression_idx_from_center[0]}")

    if suppression_idx_from_peak.size > 0:
        suppression_radii_from_peak = radii[suppression_idx_from_peak[0]]
        # print(f"DEBUG: Suppression radius from peak: {suppression_radii_from_peak} at index {suppression_idx_from_peak[0]}")

    # Facilitation strength
    facilitation_strength = ((R_peak - R_ctr) / R_ctr) * 100


    return {
        "R_peak": R_peak,
        "peak_idx": peak_idx,
        "global_min_idx": global_min_idx,
        "suppression_radii_from_center_only": suppression_radii_from_center_only if 'suppression_radii_from_center_only' in locals() else None,
        "suppression_radii_from_peak": suppression_radii_from_peak if 'suppression_radii_from_peak' in locals() else None,
        "suppression_onset": suppression_onset if 'suppression_onset' in locals() else None,
        "facilitation_onset_from_center": facilitation_onset_from_center if 'facilitation_onset_from_center' in locals() else None,
        "facilitation_strength": facilitation_strength,
    }
    
def analyse_low_contrast_facilitation(radii, responses, R_ctr):
    radii = np.array(radii)
    responses = np.array(responses)

    R_peak = np.max(responses)
    peak_idx = np.argmax(responses)

    facilitation_strength = ((R_peak - R_ctr) / R_ctr) * 100

    # Onset: look backward for last point ≤ R_ctr before the peak
    facilitation_onset = None
    for i in range(peak_idx - 1, -1, -1):
        if responses[i] <= R_ctr:
            facilitation_onset = radii[i]
            break

    # Offset: look forward for first point ≤ R_ctr after the peak
    facilitation_offset = None
    for i in range(peak_idx + 1, len(responses)):
        if responses[i] <= R_ctr:
            facilitation_offset = radii[i]
            break

    return {
        "R_peak": R_peak,
        "peak_idx": peak_idx,
        "facilitation_strength": facilitation_strength,
        "facilitation_onset": facilitation_onset,
        "facilitation_offset": facilitation_offset
    }

def compute_normalized_annular_ratio(R_outer, R_inner, sRF_high):
    R_outer = np.array(R_outer, dtype=np.float64)
    R_inner = np.array(R_inner, dtype=np.float64)
    sRF_high = np.array(sRF_high, dtype=np.float64)

    print("DEBUG: Input arrays:")
    print(f"  R_outer (shape {R_outer.shape}): {R_outer}")
    print(f"  R_inner (shape {R_inner.shape}): {R_inner}")
    print(f"  sRF_high (shape {sRF_high.shape}): {sRF_high}")

    # Check for NaNs explicitly and print counts
    print(f"  R_outer NaNs: {np.isnan(R_outer).sum()}")
    print(f"  R_inner NaNs: {np.isnan(R_inner).sum()}")
    print(f"  sRF_high NaNs: {np.isnan(sRF_high).sum()}")

    # Check if all input arrays have the same shape
    if not (R_outer.shape == R_inner.shape == sRF_high.shape):
        print(f"ERROR: Input arrays have different shapes!")
        print(f"  R_outer shape: {R_outer.shape}")
        print(f"  R_inner shape: {R_inner.shape}")
        print(f"  sRF_high shape: {sRF_high.shape}")
        raise ValueError("Input arrays must have the same shape.")

    valid_mask = (~np.isnan(R_outer)) & (~np.isnan(R_inner)) & (~np.isnan(sRF_high)) & (R_outer > R_inner)
    print(f"  Valid mask sum (number of valid elements): {valid_mask.sum()}")

    sRF_ratio = np.full_like(R_outer, np.nan)

    sRF_ratio[valid_mask] = np.sqrt(R_outer[valid_mask]**2 - R_inner[valid_mask]**2) / sRF_high[valid_mask]

    print(f"  Resulting sRF_ratio (non-NaN values): {sRF_ratio[~np.isnan(sRF_ratio)]}")

    return sRF_ratio[~np.isnan(sRF_ratio)]

# Figure 5A: Histogram of near facilitation strength
def plot_near_faci_histogram(near_facilitation_strengths):

    # Figure out how to best filter outliers - maybe just mention them?
    near_facilitation_strengths = near_facilitation_strengths[near_facilitation_strengths < 200]  # Filter out outliers

    plt.figure(figsize=(10, 5))
    plt.hist(near_facilitation_strengths, bins=100, color="gray", edgecolor="black", 
             weights=np.ones(len(near_facilitation_strengths)) / len(near_facilitation_strengths))

    plt.xlabel("Facilitation Strength (%)")
    plt.ylabel("Percentage of Neurons")
    plt.title("Near Facilitation Strength")
    plt.axvline(np.mean(near_facilitation_strengths), color="r", linestyle="dashed", label="Mean")
    plt.legend()
    plt.savefig("/project/results/facilitation/75A.png")
    plt.close

# Figure 5B: Far facilitation strengh
def plot_far_faci_histogram(far_facilitation_strength_H, far_facilitation_strength_L, colors):

    print(far_facilitation_strength_H)
    print(far_facilitation_strength_L)

    num_bins = 8
    bins = np.linspace(
        min(min(far_facilitation_strength_L), min(far_facilitation_strength_H)), 
        max(max(far_facilitation_strength_L), max(far_facilitation_strength_H)), 
        num_bins + 1 
    )

    bin_width = bins[1] - bins[0]
    bar_width = bin_width / 2

    plt.figure(figsize=(10, 8))
    plt.hist(
        [far_facilitation_strength_H, 
            far_facilitation_strength_L], 
        bins=bins, color=[colors["HH"], 
                            colors["LL"]],
        weights=[np.ones(len(far_facilitation_strength_H)) * 100 / len(far_facilitation_strength_H),
                    np.ones(len(far_facilitation_strength_L)) * 100 / len(far_facilitation_strength_L)], 
        width=bar_width, edgecolor="black", label=["HH", "LH, LL",],  alpha=0.8,
    )

    plt.yscale("log")
    plt.xlabel("Facilitation Strength")
    plt.ylabel("Percentage of Neurons")
    plt.title("Far Facilitation Strength")
    plt.legend()
    plt.savefig("/project/results/facilitation/75B.png")
    plt.close()


# Figure 5C: Annulus inner radius at peak response in low contrast
def plot_inner_radius_peak_response_histogram(peaks_L, colors, cond):

    plt.figure(figsize=(10, 8))
    plt.hist(peaks_L, bins=50, color=colors[cond], alpha=0.5, label=cond, edgecolor="black", 
                weights=np.ones(len(peaks_L)) / len(peaks_L))

    plt.xlabel("Annulus Inner Radius at peak Response (°)")
    plt.ylabel("Percentage of Neurons")
    plt.title("Annulus Inner Radius at peak Response (°) in low contrast")
    plt.axvline(np.mean(peaks_L), color="blue", linestyle="dashed", label="Low contrast Mean")        
    plt.legend()
    plt.savefig("/project/results/facilitation/75C.png")
    plt.close()


# Figure 5D: Scatter plot of suppression vs. facilitation distances
def plot_suppresion_facilitation_distances_scatter(facilitation_onset_from_peak, facilitation_onset_from_center, suppression_onset):

    plt.figure(figsize=(10, 8))
    plt.scatter(facilitation_onset_from_peak, suppression_onset,
                    color="red", alpha=0.6, label="from peak")
    plt.scatter(facilitation_onset_from_center, suppression_onset,
                    color= "white", alpha=0.6, label="from center", edgecolor="black")
    plt.xlabel("Annulus Inner Radius at Suppression onset")
    plt.ylabel("Annulus Inner Radius at Facilitation onset")
    plt.title("Suppression vs. Facilitation Distances")
    plt.plot([0, 2.0], [0, 2.0], 'k--') 
    plt.legend()
    plt.savefig("/project/results/facilitation/75D.png")
    plt.close()

# Figure 6A Scatter plot of sRFhigh vs. sRFlow 
def plot_srfhigh_srflow_scatter(sRFhigh_values, sRFlow_values):
    plt.figure(figsize=(10, 8))
    plt.scatter(sRFhigh_values, sRFlow_values, color="b", alpha=0.6, edgecolor="black")
    plt.plot([0, 2.0], [0, 2.0], 'k--') 
    plt.xlabel("sRFhigh (High Contrast RF size)")
    plt.ylabel("sRFlow (Low Contrast RF size)")
    plt.title("RF Size Shift at Low Contrast")
    plt.savefig("/project/results/facilitation/76A.png")
    plt.close()


# Figure 6B: Histogram of sRFlow/sRFhigh ratio
def plot_srfhigh_srflow_histogram(sRF_ratios):

    plt.figure(figsize=(10, 5))
    plt.hist(sRF_ratios, bins=6, edgecolor="black", label="sRFlow / sRFhigh", color="gray",
                weights=np.ones(len(sRF_ratios)) / len(sRF_ratios))
    
    plt.axvline(np.mean(sRF_ratios), color="blue", linestyle="dashed", label="SRF ratios Mean")        

    plt.xlabel("sRFlow / sRFhigh")
    plt.xticks([1,2,3,4,5,7,9])
    plt.ylabel("Percentage of Neurons")
    plt.title("Contrast-Dependent RF Expansion")
    plt.savefig("/project/results/facilitation/76B.png")
    plt.close()

# Figure 6C:  Scatter plot of suppression onset (high vs. low contrast)
def plot_suppresion_onset_scatter_plot(HH_width_center_only, LH_center_only, LL_center_only):

    plt.figure(figsize=(10, 5))
    plt.scatter(HH_width_center_only, LH_center_only, color="red", label="HH vs LH")
    plt.scatter(HH_width_center_only, LL_center_only, color="white", label="HH vs LL", edgecolor="black")
    plt.plot([0, 4.0], [0, 4.0], 'k--') 
    plt.xlabel("Annulus Width at Suppression (High Contrast)")
    plt.ylabel("Annulus Width at Suppression (Low Contrast)")
    plt.title("Suppression Onset Shift")
    plt.legend()
    plt.savefig("/project/results/facilitation/76C.png")
    plt.close()

# Figure 6D: Histogram of normalized annulus width at suprresion onset
def plot_normalised_annulus_widths_suppresion_onset_histogram(normalized_annular_ratio_at_suprresion_onset_LH,
                                                              normalized_annular_ratio_at_suprresion_onset_LL,
                                                              colors):

    plt.figure(figsize=(10, 5))
    plt.hist([normalized_annular_ratio_at_suprresion_onset_LH, normalized_annular_ratio_at_suprresion_onset_LL], bins=10, color=[colors["LH"], colors["LL"]], edgecolor="black", label=["LH", "LL"],
                weights=[np.ones(len(normalized_annular_ratio_at_suprresion_onset_LH)) / len(normalized_annular_ratio_at_suprresion_onset_LH),
                    np.ones(len(normalized_annular_ratio_at_suprresion_onset_LL)) / len(normalized_annular_ratio_at_suprresion_onset_LL),])        
    plt.xlabel("Far surround / sRFhigh")
    plt.ylabel("Percentage of Neurons")
    plt.title("Normalized Suppression Onset")
    plt.legend()
    plt.savefig("/project/results/facilitation/76D.png")
    plt.close()

def store_annular_tuning_results(results_dict, result_annular_tunning, dtype, cond, neuron_id):
    if result_annular_tunning is None:
        return results_dict

    # Get neuron entry or create if missing
    neuron_results = results_dict[dtype][cond].setdefault(neuron_id, {})

    # Add basic metrics

    neuron_results["facilitation_onset_from_center"] = result_annular_tunning.get("facilitation_onset_from_center")
    neuron_results["facilitation_onset_from_peak"] = result_annular_tunning.get("facilitation_onset_from_peak")
    neuron_results["suppression_onset"] = result_annular_tunning.get("suppression_onset")
    neuron_results["facilitation_strength"] = result_annular_tunning.get("facilitation_strength")
    neuron_results["peak_idx"] = result_annular_tunning.get("peak_idx")

    R_peak = result_annular_tunning.get("R_peak")
    if R_peak is not None:
        neuron_results["R_peak"] = R_peak
        neuron_results["suppression_radius_from_center_only"] = result_annular_tunning.get("suppression_radius_from_center_only")
        neuron_results["suppression_radius_from_peak"] = result_annular_tunning.get("suppression_radius_from_peak")

    return results_dict











        # # Curve analysis - optional


        # condition_colors = {
        #     'HH': 'tab:blue',
        #     'LH': 'tab:green',
        #     'LL': 'tab:red'
        # }

        # # plt.figure(figsize=(18, 12))

        # totals = {
        #     'HH': len(collected_data['HH']),
        #     'LH': len(collected_data['LH']),
        #     'LL': len(collected_data['LL'])
        # }

        # radii_after_min_index_list = []

        # monotonically_decreasing = {k: 0 for k in totals}
        # facilitation_before_minimum = {k: 0 for k in totals}
        # facilitation_after_minimum = {k: 0 for k in totals}

        # for cond, neurons in collected_data.items():
        #     color = condition_colors.get(cond, 'gray')
    
        #     for neuron, (radii, response) in neurons.items():
        #         # Monotonically Decreasing
        #         if np.all(np.diff(response) <= 0):
        #             monotonically_decreasing[cond] += 1
        #             continue  # skip others, already classified

        #         # One Peak Before Minimum == surround facilitation
        #         peaks, _ = scipy.signal.find_peaks(response, prominence=0.0001)

        #         min_index = np.argmin(response)

        #         if len(peaks) > 0:
        #             peak_index = peaks[0]

        #             is_min_after_peak = min_index > peak_index
        #             has_single_peak = np.sum(np.diff(np.sign(np.diff(response))) == -2) == 1

        #             if is_min_after_peak and has_single_peak:
        #                 facilitation_before_minimum[cond] += 1
        #                 continue

        #         # Type 3: Rising After Minimum  == ??

        #         rising_after_min = np.all(np.diff(response[min_index:]) >= 0)
        #         if rising_after_min:
        #             radii_after_min_index_list.append(radii[min_index])
        #             facilitation_after_minimum[cond] += 1

        # print("=== Curve Classification Summary ===")

        # # Sum across all conditions
        # total_all = sum(totals.values())
        # mono_all = sum(monotonically_decreasing.values())
        # premin_all = sum(facilitation_before_minimum.values())
        # postmin_all = sum(facilitation_after_minimum.values())
        # mean_radii_after_min = np.mean(radii_after_min_index_list) if radii_after_min_index_list else None

        # print(mean_radii_after_min)

        # def pct(n): return f"{(n / total_all * 100):.1f}%" if total_all > 0 else "n/a"

        # others_all = total_all - (mono_all + premin_all + postmin_all)

        # print(f"  Monotonically decreasing: {mono_all} ({pct(mono_all)})")
        # print(f"  Peak before minimum     : {premin_all} ({pct(premin_all)})")
        # print(f"  Rise after minimum      : {postmin_all} ({pct(postmin_all)})")
        # print(f"  None of the above: {others_all} ({pct(others_all)})")
        # print(f"  Mean radii after minimum: {mean_radii_after_min:.2f}°")

        # for cond in totals:
        #     total = totals[cond]
        #     mono = monotonically_decreasing[cond]
        #     premin = facilitation_before_minimum[cond]
        #     postmin = facilitation_after_minimum[cond]
        #     others = total - (mono + premin + postmin)

        #     def pct(n): return f"{(n / total * 100):.1f}%" if total > 0 else "n/a"

        #     print(f"\nCondition: {cond}")
        #     print(f"  Monotonically decreasing: {mono} ({pct(mono)})")
        #     print(f"  Peak before minimum     : {premin} ({pct(premin)})")
        #     print(f"  Rise after minimum      : {postmin} ({pct(postmin)})")
        #     print(f"  Others (no clear pattern): {others} ({pct(others)})")

