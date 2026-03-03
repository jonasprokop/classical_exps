############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import scipy.interpolate
import scipy.optimize
import torch
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

from scipy.optimize import differential_evolution, minimize, Bounds
from scipy.stats.qmc import LatinHypercube

from scipy.optimize import basinhopping, Bounds
from scipy.stats.qmc import LatinHypercube
import time


## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF ## Plots
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
import os
import pandas as pd
import openpyxl
import datetime



def recreate_plots_results_article_7(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2,
    size=2.67,
    device=None,
    fit = False,
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

            # allowed = {2, 3, 6, 8, 9, 11}      
            # if neuron_id not in allowed:
            #     continue

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
                        ds_params = patch_tunning_fits[neuron]
                        ds_resid  = patch_tunning_residuals[neuron]

                        params_low_contrast_curve = np.asarray(ds_params, dtype=np.float64)
                        residual_low_contrast     = np.asarray(ds_resid,  dtype=np.float64)

                        # Handle scalar / NaN placeholders
                        if params_low_contrast_curve.ndim == 0:
                            if np.isnan(params_low_contrast_curve):
                                params_low_contrast_curve = None
                        if residual_low_contrast.ndim == 0:
                            residual_low_contrast = float(residual_low_contrast)

                        print(
                            f"Loaded fit for low contrast patch tuning curve for {neuron}: "
                            f"params={params_low_contrast_curve}, residual={residual_low_contrast}"
                        )

                    except Exception as e:
                        print(f"Could not load low contrast patch fit for {neuron}: {e}")
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
                        name_neuron = neuron + "_" + cond

                        # HDF5 datasets
                        ds_params   = annular_tunning_fits[name_neuron]
                        ds_residual = annular_tunning_residuals[name_neuron]

                        # Convert to numpy arrays
                        params_arr   = np.asarray(ds_params,   dtype=np.float64)
                        residual_arr = np.asarray(ds_residual, dtype=np.float64)

                        # Detect "no fit" markers: empty array or all-NaN
                        if params_arr.size == 0 or np.all(np.isnan(params_arr)):
                            params, residual = None, None
                            print(f"No valid annular fit for {neuron} in condition {cond} (empty/NaN params).")
                        else:
                            params = params_arr
                            # residual should be scalar; handle shape safely
                            if residual_arr.size == 0 or np.all(np.isnan(residual_arr)):
                                residual = None
                                print(f"Loaded params but residual NaN for {neuron} in condition {cond}.")
                            else:
                                residual = float(residual_arr.reshape(-1)[0])
                                print(
                                    f"Loaded fit for annular tuning curve for {neuron} in condition {cond} "
                                    f"with params={params} and residual={residual:.4g}"
                                )

                    except Exception as e:
                        print(f"Could not load annular fit for {neuron} in condition {cond}: {e}")
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

                    print(params, residual) 
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

        colors ={
        "HH":"#000000",    # Black
        "LH":"#8B0000",    # Dark Red
        "LL":"#708090",    # Slate Gray
            }

        HH_width_center_only = []
        LH_center_only = []
        LL_center_only = []

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



        # 5D: Summary statistics and 5D scatter plot

        def build_fac_supp_table(results_annular_tuning_curves):
            rows = []
            all_neurons = set()
            for cond in ["HH", "LH", "LL"]:
                all_neurons |= set(results_annular_tuning_curves["fit"][cond].keys())

            for neuron in sorted(all_neurons):
                for cond in ["HH", "LH", "LL"]:
                    res = results_annular_tuning_curves["fit"][cond].get(neuron, None)

                    if not isinstance(res, dict):
                        rows.append({
                            "neuron": neuron,
                            "cond": cond,
                            "fac_strength": None,
                            "f_center": None,
                            "s_center": None,
                            "category": "no_data",
                            "ratio_center": None,
                        })
                        continue

                    fac = res.get("facilitation_strength_clipped_0")
                    f_center = res.get("facilitation_radius_from_center_only")
                    s_center = res.get("suppression_radius_from_center_only")

                    has_fac = fac is not None and np.isfinite(fac) and fac >= 10
                    has_supp = s_center is not None and np.isfinite(s_center)

                    if not has_fac and not has_supp:
                        category = "none"
                    elif has_fac and not has_supp:
                        category = "fac_only"
                    elif not has_fac and has_supp:
                        category = "supp_only"
                    else:
                        category = "both"

                    ratio_center = None
                    if (
                        f_center is not None and np.isfinite(f_center) and f_center > 0 and
                        s_center is not None and np.isfinite(s_center)
                    ):
                        ratio_center = float(s_center) / float(f_center)

                    rows.append({
                        "neuron": neuron,
                        "cond": cond,
                        "fac_strength": fac,
                        "f_center": f_center,
                        "s_center": s_center,
                        "category": category,
                        "ratio_center": ratio_center,
                    })

            return pd.DataFrame(rows)


        def summarize_fac_supp_df(df):
            print("\n========== PER-NEURON SUMMARY ==========")

            for cond in ["HH", "LH", "LL"]:
                sub = df[df.cond == cond]
                print(f"\n--- Condition {cond} ---")
                print(f"Total neurons: {len(sub)}")

                # distribution of regimes
                vc = sub["category"].value_counts()
                for cat in ["none", "fac_only", "supp_only", "both"]:
                    print(f"  {cat:10s}: {vc.get(cat, 0)}")

                # ratio statistics
                ratios = sub["ratio_center"].dropna().values
                if ratios.size > 0:
                    print(f"  ratio_center: min={ratios.min():.3g}, median={np.median(ratios):.3g}, max={ratios.max():.3g}")
                    print(f"    frac(<1.0): {np.mean(ratios < 1.0):.2%}")
                    print(f"    frac(0.8–1.25): {np.mean((ratios >= 0.8) & (ratios <= 1.25)):.2%}")
                    print(f"    frac(>1.0): {np.mean(ratios > 1.0):.2%}")
                else:
                    print("  ratio_center: no usable values")


        def neuron_to_neuron_pairwise(df):
            print("\n========== PAIRWISE NEURON–NEURON SUMMARY ==========")

            for cond in ["HH", "LH", "LL"]:
                sub = df[df.cond == cond].copy()

                # numeric subset
                valid = sub.dropna(subset=["f_center", "s_center"])
                if len(valid) < 3:
                    print(f"\n--- {cond}: insufficient data for pairwise stats ---")
                    continue

                print(f"\n--- Condition {cond} ---")
                print(f"Usable neurons: {len(valid)}")

                f_vals = valid["f_center"].values
                s_vals = valid["s_center"].values

                # correlations
                if len(f_vals) > 2:
                    corr_fs = np.corrcoef(f_vals, s_vals)[0,1]
                    print(f"  corr(f_center, s_center) = {corr_fs:.3f}")
                else:
                    print("  correlation unavailable")

                # regime overlap
                cats = valid["category"]
                frac_both = np.mean(cats == "both")
                frac_fac_only = np.mean(cats == "fac_only")
                frac_supp_only = np.mean(cats == "supp_only")

                print(f"  frac(both)     = {frac_both:.2%}")
                print(f"  frac(fac_only) = {frac_fac_only:.2%}")
                print(f"  frac(supp_only)= {frac_supp_only:.2%}")



        fac_supp_df = build_fac_supp_table(results_annular_tuning_curves)
        fac_supp_df.to_excel("/project/results/facilitation/fac_supp_table.xlsx", index=False)

        summarize_fac_supp_df(fac_supp_df)
        neuron_to_neuron_pairwise(fac_supp_df)



        fac_center, supp_center, fac_peak, supp_peak = build_5D_data(
            results_annular_tuning_curves, shared_neurons
        )

        # classify whole population for ghost lines (fac-only / supp-only / none)
        summary_5d = summarize_5D_population(
            results_annular_tuning_curves, shared_neurons, fac_thresh=10.0
        )

        fac_center, supp_center, fac_peak, supp_peak = build_5D_data(
            results_annular_tuning_curves,
            shared_neurons,
        )

        plot_suppresion_facilitation_distances_scatter_panel(
            fac_center,
            supp_center,
            fac_peak=fac_peak,
            supp_peak=supp_peak,
            summary_5d=summary_5d,
        )

        plot_5d_category_counts(summary_5d)
        plot_fac_strength_histogram(results_annular_tuning_curves, shared_neurons)
        plot_suppression_onset_histogram(results_annular_tuning_curves, shared_neurons)


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
def dog(s, A_E, A_I1, sigma, T_I1, baseline=0.0):
    g = np.exp(-s**2 / (2.0 * sigma**2))
    E  = A_E  * g
    I1 = np.maximum(A_I1 * g - T_I1, 0.0)
    return E - I1 + baseline

# # ---------- model: two-inh, one sigma, hard-threshold ----------
# def dog(s, A_E, A_I1, A_I2, sigma, T_I1, T_I2, baseline=0.0):
#     g = np.exp(-s**2 / (2.0 * sigma**2))
#     E  = A_E  * g
#     I1 = np.maximum(A_I1 * g - T_I1, 0.0)
#     I2 = np.maximum(A_I2 * g - T_I2, 0.0)
#     return E - I1 - I2 + baseline

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
    max_de_time=1000,          #  time for DE exploration
    de_maxiter=50000,       #  DE search
    popsize=50,              #  diversity
    tail_power=0.2,           #  tail-weighting, to fit the later responses with more weight (prevents underfitting the far surround faci)
    n_polish_perturb=15,      # local starts
    n_polish_lhs=15,          # LHS samples
    polish_maxiter=10000      # polish iterations
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
        (0, A_hi),               # A_E
        (0, A_hi),               # A_I1
        (1e-5, 50*sig_hi),       # sigma
        (0, A_hi),               # T_I1
        (y_min - 0.5*y_span, y_max)  # baseline
    ]
    
    # bounds_list = [
    #     (0, A_hi),   # A_E
    #     (0, A_hi),   # A_I1
    #     (0, A_hi),   # A_I2
    #     (1e-5, 50*sig_hi), # sigma
    #     (0, A_hi),   # T_I1
    #     (0, A_hi),   # T_I2
    #     (y_min - 1.5*y_span, y_max + 1.5*y_span)  # baseline
    # ]

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
    max_de_time=200,
    de_maxiter=10000,
    popsize=20,
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


def find_facilitation_annular_tunning_curve(resp_ann, R_center, radii, perc_thr=10.0):
    """
    Analyse annular tuning curve in the sense used in the article.

    resp_ann : 1D array-like
        Annular responses (raw or fit).
    R_center : float
        Response to center-only stimulation in the same contrast condition.
    radii    : 1D array-like
        Inner radii of annulus (same shape as resp_ann).

    perc_thr : float
        Percentage threshold for facilitation/suppression (default 10%).
    """
    resp_ann = np.asarray(resp_ann, dtype=float)
    radii    = np.asarray(radii,    dtype=float)

    # --- sanity checks ---
    if resp_ann.ndim == 0 or radii.ndim == 0:
        raise ValueError(
            f"find_facilitation_annular_tunning_curve: got scalar(s); "
            f"resp_ann.ndim={resp_ann.ndim}, radii.ndim={radii.ndim}"
        )
    if resp_ann.shape != radii.shape:
        raise ValueError(
            f"find_facilitation_annular_tunning_curve: shape mismatch: "
            f"resp_ann.shape={resp_ann.shape}, radii.shape={radii.shape}"
        )

    result = {}

    # ---------- PEAK (R_peak) ----------
    peak_idx    = int(np.nanargmax(resp_ann))
    peak_resp   = float(resp_ann[peak_idx])   # R_peak (response)
    peak_radius = float(radii[peak_idx])      # radius at peak

    # Downstream code uses R_peak as radius, so keep both
    result["R_peak"]   = peak_radius
    result["peak_idx"] = peak_idx

    # ---------- FACILITATION VS CENTER-ONLY ----------
    if R_center is None or not np.isfinite(R_center) or R_center <= 0:
        result["facilitation_strength"] = None
        result["facilitation_radius_from_center_only"] = None
    else:
        # strength exactly as in the article: from R_peak and R_ctr
        fac_strength = 100.0 * (peak_resp - R_center) / R_center
        result["facilitation_strength"] = float(fac_strength)

        # onset: first radius where resp >= R_center * (1 + perc_thr/100)
        fac_level = R_center * (1.0 + perc_thr / 100.0)
        idx_fac = np.where(resp_ann >= fac_level)[0]
        if idx_fac.size > 0:
            result["facilitation_radius_from_center_only"] = float(radii[idx_fac[0]])
        else:
            result["facilitation_radius_from_center_only"] = None

    # ---------- SUPPRESSION VS CENTER-ONLY ----------
    if R_center is None or not np.isfinite(R_center) or R_center <= 0:
        result["suppression_radius_from_center_only"] = None
        result["suppression_onset"] = None
    else:
        # onset: first radius where resp <= R_center * (1 - perc_thr/100)
        supp_level_ctr = R_center * (1.0 - perc_thr / 100.0)
        idx_supp_ctr = np.where(resp_ann <= supp_level_ctr)[0]
        if idx_supp_ctr.size > 0:
            supp_radius_ctr = float(radii[idx_supp_ctr[0]])
            result["suppression_radius_from_center_only"] = supp_radius_ctr
            result["suppression_onset"] = supp_radius_ctr
        else:
            result["suppression_radius_from_center_only"] = None
            result["suppression_onset"] = None

    # ---------- SUPPRESSION VS PEAK ----------
    if not np.isfinite(peak_resp) or peak_resp <= 0:
        result["suppression_radius_from_peak"] = None
    else:
        # only look on the far side of the peak for suppression
        supp_level_peak = peak_resp * (1.0 - perc_thr / 100.0)
        mask_far = (radii > peak_radius)
        idx_supp_peak = np.where(mask_far & (resp_ann <= supp_level_peak))[0]
        if idx_supp_peak.size > 0:
            result["suppression_radius_from_peak"] = float(radii[idx_supp_peak[0]])
        else:
            result["suppression_radius_from_peak"] = None

    # ---------- FAR-SURROUND FACILITATION VS PEAK ----------
    # The article does not define a separate "re-facilitation" onset relative to the peak.
    # To stay faithful to the methods, we do NOT invent one here.
    result["facilitation_radius_from_peak"] = None

    # ---------- GLOBAL MIN ----------
    try:
        global_min_idx = int(np.nanargmin(resp_ann))
    except ValueError:
        global_min_idx = None
    result["global_min_idx"] = global_min_idx

    return result


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
    near_facilitation_strengths = near_facilitation_strengths[near_facilitation_strengths > 0]  # Filter out non-facilitating neurons

    plt.figure(figsize=(10, 5))
    plt.hist(near_facilitation_strengths, bins=10, color="gray", edgecolor="black", 
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

    num_bins = 10
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

    # plt.yscale("log")
    plt.xlabel("Facilitation Strength")
    plt.ylabel("Percentage of Neurons")
    plt.title("Far Facilitation Strength")
    plt.legend()
    plt.savefig("/project/results/facilitation/75B.png")
    plt.close()


# Figure 5C: Annulus inner radius at peak response in low contrast

def plot_inner_radius_peak_response_histogram(peaks_L, colors, cond):
    """
    Histogram of annulus inner radius at peak response in low contrast.
    `peaks_L` should already be the R_peak radii for the chosen low-contrast condition.
    """
    plt.figure(figsize=(8, 6))

    weights = np.ones(len(peaks_L)) / len(peaks_L) if len(peaks_L) > 0 else None
    plt.hist(
        peaks_L,
        bins=10,
        color=colors[cond],
        alpha=0.5,
        edgecolor="black",
        weights=weights,
        label=cond,
    )

    if len(peaks_L) > 0:
        plt.axvline(
            np.mean(peaks_L),
            color="blue",
            linestyle="dashed",
            label="Low-contrast mean",
        )

    plt.xlabel("Annulus inner radius at peak response (°)")
    plt.ylabel("Percentage of neurons")
    plt.title("Annulus inner radius at peak response in low contrast")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/project/results/facilitation/75C.png")
    plt.close()


def _plot_center_based_5D(
    ax,
    fac_center,
    supp_center,
    summary_5d=None,
    title="Suppression vs. facilitation distances (center-based)",
    fac_thresh=10.0,
):
    fac_center = np.asarray(fac_center, dtype=float)
    supp_center = np.asarray(supp_center, dtype=float)

    fac_only = np.array([], dtype=float)
    supp_only = np.array([], dtype=float)
    n_tot = n_both = 0

    if summary_5d is not None:
        fac_only = np.asarray(summary_5d.get("fac_only_center", []), dtype=float)
        supp_only = np.asarray(summary_5d.get("supp_only_center", []), dtype=float)
        n_tot = int(summary_5d.get("n_total", 0))
        n_both = int(summary_5d.get("n_with_both", 0))

    xs_all, ys_all = [], []

    # main points: both fac & supp
    if fac_center.size > 0:
        ax.scatter(
            fac_center,
            supp_center,
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
            s=40,
            label="Fac & supp (center-only)",
        )
        xs_all.append(fac_center)
        ys_all.append(supp_center)

    # include rugs in limits
    if fac_only.size > 0:
        xs_all.append(fac_only)
    if supp_only.size > 0:
        ys_all.append(supp_only)

    if xs_all and ys_all:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        lo = float(min(xs_all.min(), ys_all.min()))
        hi = float(max(xs_all.max(), ys_all.max()))
    else:
        lo, hi = 0.0, 1.0

    if hi <= lo:
        hi = lo + 1.0

    margin = 0.1 * (hi - lo)
    lo -= margin
    hi += margin

    # diagonal
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    span = hi - lo
    y0 = lo
    y1 = lo + 0.04 * span
    x0 = lo
    x1 = lo + 0.04 * span

    # fac-only rugs
    if fac_only.size > 0:
        for x in fac_only:
            ax.vlines(x, y0, y1, colors="darkgray", alpha=0.7, linewidth=1)

    # supp-only rugs
    if supp_only.size > 0:
        for y in supp_only:
            ax.hlines(y, x0, x1, colors="darkgray", alpha=0.7, linewidth=1)

    # legend
    legend_handles, legend_labels = [], []

    if fac_center.size > 0:
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="None",
                   markerfacecolor="white", markeredgecolor="black")
        )
        legend_labels.append("Fac & supp (center-only)")

    if fac_only.size > 0:
        legend_handles.append(Line2D([], [], linestyle="-", color="darkgray"))
        legend_labels.append(f"Fac only (no supp), n={fac_only.size}")

    if supp_only.size > 0:
        legend_handles.append(Line2D([], [], linestyle="-", color="darkgray"))
        legend_labels.append(f"Supp only (no fac ≥10%), n={supp_only.size}")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)

    # text summary – recompute "none" from the categories
    if summary_5d is not None and n_tot > 0:
        n_fac_only = fac_only.size
        n_supp_only = supp_only.size
        n_none_eff = max(n_tot - (n_both + n_fac_only + n_supp_only), 0)

        txt = (
            f"n_total={n_tot}\n"
            f"both={n_both}\n"
            f"fac-only={n_fac_only}\n"
            f"supp-only={n_supp_only}\n"
            f"none={n_none_eff}"
        )
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_xlabel("Facilitation onset radius (°)")
    ax.set_ylabel("Suppression onset radius (°)")
    ax.set_title(title)
    ax.grid(True)

def plot_suppresion_facilitation_distances_scatter_panel(
    fac_center,
    supp_center,
    fac_peak=None,
    supp_peak=None,
    summary_5d=None,
    fac_thresh=10.0,
):
    """
    Make a 1x2 panel:

      Left:  center-based onsets (Fig. 5D-style)
             - uses fac_center, supp_center and summary_5d
      Right: peak-based far-surround onsets
             - uses fac_peak, supp_peak (subset of neurons where both are defined)

    All radii are absolute (degrees); what differs is how onsets were *defined*
    (relative to center vs relative to peak).
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # ---- left: center-based (with rugs & counts) ----
    _plot_center_based_5D(
        axes[0],
        fac_center=fac_center,
        supp_center=supp_center,
        summary_5d=summary_5d,
        fac_thresh=fac_thresh,
        title="Far-surround onsets\ndefined relative to center",
    )

    # ---- right: peak-based ----
    ax = axes[1]

    fac_peak = np.asarray([] if fac_peak is None else fac_peak, dtype=float)
    supp_peak = np.asarray([] if supp_peak is None else supp_peak, dtype=float)

    if fac_peak.size > 0 and supp_peak.size > 0:
        ax.scatter(
            fac_peak,
            supp_peak,
            color="gray",
            alpha=0.7,
            s=40,
            label="Fac & supp (peak-based)",
        )

        # share limits with left subplot by using same min/max
        xs_all = np.concatenate(
            [np.asarray(fac_center, float), fac_peak]
        )
        ys_all = np.concatenate(
            [np.asarray(supp_center, float), supp_peak]
        )
        lo = float(min(xs_all.min(), ys_all.min()))
        hi = float(max(xs_all.max(), ys_all.max()))
        if hi <= lo:
            hi = lo + 1.0
        margin = 0.1 * (hi - lo)
        lo -= margin
        hi += margin
        axes[0].set_xlim(lo, hi)
        axes[0].set_ylim(lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # diagonal
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Facilitation onset radius (°)")
    ax.set_ylabel("Suppression onset radius (°)")
    ax.set_title("Far-surround onsets\ndefined relative to peak")
    ax.grid(True)

    fig.suptitle("Suppression vs. facilitation distances\ncenter-based vs peak-based", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("/project/results/facilitation/75D_center_vs_peak_panel.png")
    plt.close(fig)


# =========================================================
# 5D scatter – center-based + peak-based panel
# =========================================================

def find_facilitation_annular_tunning_curve(resp_ann, R_center, radii, perc_thr=10.0):
    """
    Analyse annular tuning curve.

    resp_ann : 1D array-like
        Annular responses (raw or fit).
    R_center : float
        Center-only response (same contrast).
    radii    : 1D array-like
        Inner radii of annulus (same shape as resp_ann).

    perc_thr : float
        Percentage threshold for facilitation/suppression (default 10%).
    """
    resp_ann = np.asarray(resp_ann, dtype=float)
    radii    = np.asarray(radii,    dtype=float)

    # Sanity checks
    if resp_ann.ndim == 0 or radii.ndim == 0:
        raise ValueError(
            f"find_facilitation_annular_tunning_curve: got scalar(s); "
            f"resp_ann.ndim={resp_ann.ndim}, radii.ndim={radii.ndim}"
        )
    if resp_ann.shape != radii.shape:
        raise ValueError(
            f"find_facilitation_annular_tunning_curve: shape mismatch: "
            f"resp_ann.shape={resp_ann.shape}, radii.shape={radii.shape}"
        )

    result = {}

    # ---------- PEAK (absolute maximum of annular response) ----------
    peak_idx    = int(np.nanargmax(resp_ann))
    peak_resp   = float(resp_ann[peak_idx])     # response at peak
    peak_radius = float(radii[peak_idx])        # radius at peak

    # Downstream code historically uses R_peak as *radius*
    result["R_peak"]   = peak_radius
    result["peak_idx"] = peak_idx

    # ================================================================
    # CENTER-BASED DEFINITIONS (match text in article)
    # ================================================================
    if R_center is None or not np.isfinite(R_center) or R_center <= 0:
        result["facilitation_strength"] = None
        result["facilitation_radius_from_center_only"] = None
        result["suppression_radius_from_center_only"] = None
        result["suppression_onset"] = None
    else:
        # --- facilitation strength: 100 * (R_peak - R_ctr) / R_ctr
        fac_strength = 100.0 * (peak_resp - R_center) / R_center
        result["facilitation_strength"] = float(fac_strength)

        # --- facilitation onset: first radius with resp >= R_ctr*(1 + thr)
        rel_change_center = 100.0 * (resp_ann - R_center) / R_center
        idx_fac = np.where(rel_change_center >= perc_thr)[0]
        if idx_fac.size > 0:
            result["facilitation_radius_from_center_only"] = float(radii[idx_fac[0]])
        else:
            result["facilitation_radius_from_center_only"] = None

        # --- suppression onset vs center: first radius with resp <= R_ctr*(1 - thr)
        rel_supp_center = 100.0 * (R_center - resp_ann) / R_center  # positive = suppression
        idx_supp = np.where(rel_supp_center >= perc_thr)[0]
        if idx_supp.size > 0:
            supp_radius = float(radii[idx_supp[0]])
            result["suppression_radius_from_center_only"] = supp_radius
            result["suppression_onset"] = supp_radius
        else:
            result["suppression_radius_from_center_only"] = None
            result["suppression_onset"] = None

    # ================================================================
    # PEAK-BASED DEFINITIONS (our extra analysis)
    # ================================================================
    # We look only *after* the peak radius.
    if not np.isfinite(peak_resp) or peak_resp <= 0:
        result["suppression_radius_from_peak"] = None
        result["facilitation_radius_from_peak"] = None
    else:
        # Drop from peak in %
        rel_supp_peak = 100.0 * (peak_resp - resp_ann) / peak_resp

        # --- suppression onset vs peak: first radius > peak_radius with drop ≥ perc_thr
        idx_supp_peak = np.where(
            (radii > peak_radius) & (rel_supp_peak >= perc_thr)
        )[0]

        if idx_supp_peak.size > 0:
            first_supp_idx = idx_supp_peak[0]
            result["suppression_radius_from_peak"] = float(radii[first_supp_idx])

            # --- re-fac vs peak: first radius *after suppression onset*
            # where drop from peak is <= perc_thr again (i.e. recovered within thr% of peak)
            idx_refac = np.where(
                (np.arange(resp_ann.size) > first_supp_idx) &
                (rel_supp_peak <= perc_thr)
            )[0]

            if idx_refac.size > 0:
                result["facilitation_radius_from_peak"] = float(radii[idx_refac[0]])
            else:
                result["facilitation_radius_from_peak"] = None
        else:
            result["suppression_radius_from_peak"] = None
            result["facilitation_radius_from_peak"] = None

    # ---------- GLOBAL MIN (just for bookkeeping) ----------
    try:
        global_min_idx = int(np.nanargmin(resp_ann))
    except ValueError:
        global_min_idx = None
    result["global_min_idx"] = global_min_idx

    return result

def summarize_5D_population(results_annular_tuning_curves, shared_neurons, fac_thresh=10.0):
    """
    Classify neurons according to whether they show >= fac_thresh % far-surround
    facilitation and/or have a suppression radius, based on the *best* low-contrast
    annulus condition (LH vs LL).
    """

    fac_only_center = []
    supp_only_center = []
    n_both = 0
    n_none = 0

    # debug počítadla
    n_no_dict_LH_LL = 0
    n_fac_nan_or_missing = 0
    n_fac_lt_0 = 0
    n_fac_0_to_thr = 0
    n_fac_ge_thr_missing_fcenter = 0
    n_fac_ge_thr_missing_scenter = 0

    all_fac_strengths = []

    for neuron in shared_neurons:
        # pick best low-contrast condition (LH / LL) by facilitation_strength
        best_res = None
        best_score = -np.inf

        for cond in ["LH", "LL"]:
            res = results_annular_tuning_curves["fit"][cond].get(neuron, None)
            if not isinstance(res, dict):
                continue

            fac_val = res.get("facilitation_strength")
            if fac_val is None or not np.isfinite(fac_val):
                fac_val = res.get("facilitation_strength_clipped_0")

            if fac_val is None or not np.isfinite(fac_val):
                continue

            fac_val = float(fac_val)
            if best_res is None or fac_val > best_score:
                best_res = res
                best_score = fac_val

        if best_res is None:
            n_no_dict_LH_LL += 1
            n_none += 1
            continue

        fac_val = best_res.get("facilitation_strength")
        if fac_val is None or not np.isfinite(fac_val):
            fac_val = best_res.get("facilitation_strength_clipped_0")

        if fac_val is None or not np.isfinite(fac_val):
            n_fac_nan_or_missing += 1
            n_none += 1
            continue

        fac_val = float(fac_val)
        all_fac_strengths.append(fac_val)

        f_center = best_res.get("facilitation_radius_from_center_only")
        s_center = best_res.get("suppression_radius_from_center_only")

        if fac_val < 0:
            n_fac_lt_0 += 1
            # negativní „facilitace“ – považujme jako žádná facil.
            has_fac = False
        elif fac_val < fac_thresh:
            n_fac_0_to_thr += 1
            has_fac = False
        else:
            # fac >= threshold, teď záleží na radiích
            if f_center is None or not np.isfinite(f_center):
                n_fac_ge_thr_missing_fcenter += 1
                has_fac = False
            else:
                has_fac = True

        has_supp = (s_center is not None and np.isfinite(s_center))

        if has_fac and has_supp:
            n_both += 1
        elif has_fac and not has_supp:
            fac_only_center.append(float(f_center))
            n_fac_ge_thr_missing_scenter += 1
        elif has_supp and not has_fac:
            supp_only_center.append(float(s_center))
            # tady důvod: buď fac < thr, nebo chybí f_center
            n_none += 1  # započítáme do none+supp-only statistiky
        else:
            n_none += 1

    summary = {
        "fac_only_center": np.array(fac_only_center, dtype=np.float64),
        "supp_only_center": np.array(supp_only_center, dtype=np.float64),
        "n_with_both": n_both,
        "n_none": n_none,
        "n_total": len(shared_neurons),
    }

    print(f"\n[5D SUMMARY] neuron categories (center-based, fac ≥ {fac_thresh:.1f} %):")
    print(f"  total shared_neurons          : {summary['n_total']}")
    print(f"  both fac & supp               : {summary['n_with_both']}")
    print(f"  fac-only (fac≥thr, no supp)   : {summary['fac_only_center'].size}")
    print(f"  supp-only (supp, fac<thr)     : {summary['supp_only_center'].size}")
    print(f"  none                          : {summary['n_none']}")

    # detailní debug: proč jsou kde
    print("\n[5D DEBUG] breakdown of reasons:")
    print(f"  no usable LH/LL dict          : {n_no_dict_LH_LL}")
    print(f"  fac NaN / missing             : {n_fac_nan_or_missing}")
    print(f"  fac < 0                       : {n_fac_lt_0}")
    print(f"  0 <= fac < {fac_thresh:.1f}          : {n_fac_0_to_thr}")
    print(f"  fac ≥ {fac_thresh:.1f} but missing f_center : {n_fac_ge_thr_missing_fcenter}")
    print(f"  fac ≥ {fac_thresh:.1f} with f_center but no s_center "
          f"(=> fac-only)     : {n_fac_ge_thr_missing_scenter}")

    if all_fac_strengths:
        all_fac_strengths = np.array(all_fac_strengths, float)
        print("\n[5D DEBUG] facilitation_strength distribution (all usable LH/LL):")
        print(f"  n={all_fac_strengths.size}, "
              f"min={all_fac_strengths.min():.2f}, "
              f"median={np.median(all_fac_strengths):.2f}, "
              f"max={all_fac_strengths.max():.2f}")
    else:
        print("\n[5D DEBUG] no usable facilitation_strength values at all.")

    return summary

def build_5D_data(results_annular_tuning_curves, shared_neurons, fac_thresh=10.0):
    """
    Extract per-neuron radii for 5D-style plots.

    Returns:
        fac_center : array of facilitation-onset radii relative to center
        supp_center: array of suppression-onset radii relative to center
        fac_peak   : array of re-facilitation-onset radii relative to peak
        supp_peak  : array of suppression-onset radii relative to peak
    """
    fac_center = []
    supp_center = []
    fac_peak = []
    supp_peak = []

    # debug počítadla
    n_considered = 0
    n_with_best_cond = 0
    n_pass_fac_thresh = 0
    n_pass_center_both = 0

    n_any_supp_peak = 0
    n_any_fac_peak = 0
    n_peak_both = 0

    for neuron in shared_neurons:
        n_considered += 1

        best_res = None
        best_fac = -np.inf

        for cond in ["LH", "LL"]:
            res = results_annular_tuning_curves["fit"][cond].get(neuron, None)
            if not isinstance(res, dict):
                continue

            fac_val = res.get("facilitation_strength")
            if fac_val is None or not np.isfinite(fac_val):
                fac_val = res.get("facilitation_strength_clipped_0")

            if fac_val is None or not np.isfinite(fac_val):
                continue

            fac_val = float(fac_val)
            if fac_val > best_fac:
                best_fac = fac_val
                best_res = res

        if best_res is None:
            continue

        n_with_best_cond += 1

        fac_strength = best_res.get("facilitation_strength")
        if fac_strength is None or not np.isfinite(fac_strength):
            fac_strength = best_res.get("facilitation_strength_clipped_0")

        if fac_strength is None or not np.isfinite(fac_strength):
            continue

        fac_strength = float(fac_strength)

        f_center = best_res.get("facilitation_radius_from_center_only")
        s_center = best_res.get("suppression_radius_from_center_only")
        f_pk     = best_res.get("facilitation_radius_from_peak")
        s_pk     = best_res.get("suppression_radius_from_peak")

        # center-based inclusion
        if fac_strength >= fac_thresh:
            n_pass_fac_thresh += 1
            if (f_center is not None and np.isfinite(f_center) and
                s_center is not None and np.isfinite(s_center)):
                fac_center.append(float(f_center))
                supp_center.append(float(s_center))
                n_pass_center_both += 1

        # peak-based debug (nezávisle na thresholdu, čistě přítomnost)
        if s_pk is not None and np.isfinite(s_pk):
            n_any_supp_peak += 1
        if f_pk is not None and np.isfinite(f_pk):
            n_any_fac_peak += 1
        if (s_pk is not None and np.isfinite(s_pk) and
            f_pk is not None and np.isfinite(f_pk)):
            supp_peak.append(float(s_pk))
            fac_peak.append(float(f_pk))
            n_peak_both += 1

    fac_center = np.array(fac_center, dtype=np.float64)
    supp_center = np.array(supp_center, dtype=np.float64)
    fac_peak = np.array(fac_peak, dtype=np.float64)
    supp_peak = np.array(supp_peak, dtype=np.float64)

    print("\n[5D BUILD] summary:")
    print(f"  neurons considered               : {n_considered}")
    print(f"  with usable LH/LL & fac_strength : {n_with_best_cond}")
    print(f"  fac_strength ≥ {fac_thresh:.1f}             : {n_pass_fac_thresh}")
    print(f"  (fac≥thr & f_center & s_center)  : {n_pass_center_both}")
    print(f"  supp_radius_from_peak present    : {n_any_supp_peak}")
    print(f"  fac_radius_from_peak present     : {n_any_fac_peak}")
    print(f"  BOTH peak-based radii present    : {n_peak_both}")
    print(f"  fac_center array size            : {fac_center.size}")
    print(f"  supp_center array size           : {supp_center.size}")
    print(f"  fac_peak array size              : {fac_peak.size}")
    print(f"  supp_peak array size             : {supp_peak.size}")

    return fac_center, supp_center, fac_peak, supp_peak


def collect_best_lowcontrast_fac_strengths(results_annular_tuning_curves, shared_neurons):
    fac_vals = []

    for neuron in shared_neurons:
        best_fac = None
        for cond in ["LH", "LL"]:
            res = results_annular_tuning_curves["fit"][cond].get(neuron, None)
            if not isinstance(res, dict):
                continue
            fac = res.get("facilitation_strength_clipped_0")
            if fac is None or not np.isfinite(fac):
                continue
            if best_fac is None or fac > best_fac:
                best_fac = fac
        if best_fac is not None:
            fac_vals.append(float(best_fac))

    return np.array(fac_vals, dtype=np.float64)


def plot_fac_strength_histogram(results_annular_tuning_curves, shared_neurons):
    fac_vals = collect_best_lowcontrast_fac_strengths(
        results_annular_tuning_curves, shared_neurons
    )

    if fac_vals.size == 0:
        print("[FAC HIST] No facilitation strengths to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(fac_vals, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(10.0, color="red", linestyle="--", label="10% threshold")

    plt.xlabel("Best low-contrast facilitation strength (% vs center)")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of far-surround facilitation strengths")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/project/results/facilitation/fac_strength_hist.png")
    plt.close()

    print(
        f"[FAC HIST] n={fac_vals.size}, "
        f"min={fac_vals.min():.2f}, median={np.median(fac_vals):.2f}, max={fac_vals.max():.2f}"
    )

def collect_suppression_onsets_center(results_annular_tuning_curves, shared_neurons):
    supp_radii = []

    for neuron in shared_neurons:
        # vezmeme zase best low-contrast LH/LL
        best_res = None
        best_score = -np.inf

        for cond in ["LH", "LL"]:
            res = results_annular_tuning_curves["fit"][cond].get(neuron, None)
            if not isinstance(res, dict):
                continue
            fac = res.get("facilitation_strength_clipped_0")
            score = fac if (fac is not None and np.isfinite(fac)) else -np.inf
            if best_res is None or score > best_score:
                best_res = res
                best_score = score

        if best_res is None:
            continue

        s_center = best_res.get("suppression_radius_from_center_only")
        if s_center is not None and np.isfinite(s_center):
            supp_radii.append(float(s_center))

    return np.array(supp_radii, dtype=np.float64)


def plot_suppression_onset_histogram(results_annular_tuning_curves, shared_neurons):
    supp_radii = collect_suppression_onsets_center(
        results_annular_tuning_curves, shared_neurons
    )

    if supp_radii.size == 0:
        print("[SUPP HIST] No suppression radii to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(supp_radii, bins=30, edgecolor="black", alpha=0.7)

    plt.xlabel("Suppression onset radius vs center (°)")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of suppression onset radii (center-based)")

    plt.axvline(0.05, color="red", linestyle="--", label="0.05° bin")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/project/results/facilitation/supp_onset_hist.png")
    plt.close()

    print(
        f"[SUPP HIST] n={supp_radii.size}, "
        f"min={supp_radii.min():.4f}, median={np.median(supp_radii):.4f}, "
        f"max={supp_radii.max():.4f}"
    )

def plot_5d_category_counts(summary_5d):
    counts = {
        "both": summary_5d["n_with_both"],
        "fac-only": summary_5d["fac_only_center"].size,
        "supp-only": summary_5d["supp_only_center"].size,
        "none": summary_5d["n_none"],
    }

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, edgecolor="black", alpha=0.7)

    plt.ylabel("Number of neurons")
    plt.title("Neuron categories (best low-contrast annulus)")

    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("/project/results/facilitation/5D_categories_bar.png")
    plt.close()

    print("[5D CATEGORIES] counts:", counts)


# Figure 5D: Scatter plot of suppression vs. facilitation distances


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
    plt.xticks([1,2,3,4])
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
def plot_normalised_annulus_widths_suppresion_onset_histogram(
    ratios_LH,
    ratios_LL,
    colors,
):
    ratios_LH = np.asarray(ratios_LH, float)
    ratios_LL = np.asarray(ratios_LL, float)

    bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8])

    plt.figure(figsize=(10, 5))

    plt.hist(
        ratios_LH,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color=colors["LH"],
        label=f"LH (n={len(ratios_LH)})",
        density=True, 
    )

    plt.hist(
        ratios_LL,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color=colors["LL"],
        label=f"LL (n={len(ratios_LL)})",
        density=True,
    )

    plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
    plt.text(1.02, plt.ylim()[1]*0.9, "sRF_high", fontsize=9)

    plt.xlabel("Far surround / sRF_high")
    plt.ylabel("Probability density")
    plt.title("Normalized Suppression Onset")
    plt.legend()
    plt.grid(True)

    plt.savefig("/project/results/facilitation/76D.png")
    plt.close()


def store_annular_tuning_results(results_dict, result_annular_tunning, dtype, cond, neuron_id):
    if result_annular_tunning is None:
        return results_dict

    neuron_results = results_dict[dtype][cond].setdefault(neuron_id, {})

    for key in [
        "R_peak",
        "peak_idx",
        "global_min_idx",
        "suppression_radius_from_center_only",
        "suppression_radius_from_peak",
        "suppression_onset",
        "facilitation_radius_from_center_only",
        "facilitation_radius_from_peak",
        "facilitation_strength",
    ]:
        neuron_results[key] = result_annular_tunning.get(key)

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

