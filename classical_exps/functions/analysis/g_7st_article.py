############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.experiments import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
import os


def compute_normalized_annular_ratio(R_outer, R_inner, sRF_high):
    R_outer = np.array(R_outer, dtype=np.float64)
    R_inner = np.array(R_inner, dtype=np.float64)
    sRF_high = np.array(sRF_high, dtype=np.float64)

    R_outer = np.where(np.isnan(R_outer), np.nan, R_outer)
    R_inner = np.where(np.isnan(R_inner), np.nan, R_inner)
    sRF_high = np.where(np.isnan(sRF_high), np.nan, sRF_high)

    valid_mask = (~np.isnan(R_outer)) & (~np.isnan(R_inner)) & (~np.isnan(sRF_high)) & (R_outer > R_inner)

    sRF_ratio = np.full_like(R_outer, np.nan)

    sRF_ratio[valid_mask] = np.sqrt(R_outer[valid_mask]**2 - R_inner[valid_mask]**2) / sRF_high[valid_mask]

    return sRF_ratio[~np.isnan(sRF_ratio)]

def recreate_plots_results_article_7(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2,
    size=2.67,
    device=None,
    plot_tunning_curves=False
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

    subgroup_facilitation_st_curves_HH_path  = group_facilitation +"/curves/HH"
    subgroup_facilitation_st_curves_LH_path  = group_facilitation +"/curves/LH"
    subgroup_facilitation_st_curves_LL_path  = group_facilitation +"/curves/LL"

    subgroup_facilitation_st_results_HH_path  = group_facilitation +"/results/HH"
    subgroup_facilitation_st_results_LH_path  = group_facilitation +"/results/LH"
    subgroup_facilitation_st_results_LL_path  = group_facilitation +"/results/LL"


    minimal_receptive_fields_path = group_facilitation + "/minimal_receptive_fields"

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    with h5py.File(h5_file, "a") as file:

        ## Access the groups 
        subgroup_facilitation_st_results_high = file[subgroup_facilitation_st_results_high_path]
        subgroup_facilitation_st_results_low   = file[subgroup_facilitation_st_results_low_path]
        subgroup_facilitation_st_curves_high = file[subgroup_facilitation_st_curves_high_path]
        subgroup_facilitation_st_curves_low = file[subgroup_facilitation_st_curves_low_path]

        subgroup_facilitation_st_curves_HH = file[subgroup_facilitation_st_curves_HH_path]
        subgroup_facilitation_st_curves_LH = file[subgroup_facilitation_st_curves_LH_path]
        subgroup_facilitation_st_curves_LL = file[subgroup_facilitation_st_curves_LL_path]

        subgroup_facilitation_st_results_HH = file[subgroup_facilitation_st_results_HH_path]
        subgroup_facilitation_st_results_LH = file[subgroup_facilitation_st_results_LH_path]
        subgroup_facilitation_st_results_LL = file[subgroup_facilitation_st_results_LL_path]
        minimal_receptive_fields = file[minimal_receptive_fields_path]

        facilitation_strengths_low_contrast = []
        facilitation_strengths_annular_tunning = {"HH": [], "LH": [], "LL": []}
        peak_radii = {"HH": [], "LH": [], "LL": []}
        suppression_radii_from_center_only = {"HH": [], "LH": [], "LL": []}
        suppression_radii_from_peak = {"HH": [], "LH": [], "LL": []}
        sRFhigh_values = []
        sRFlow_values = []
        
        for neuron_id in filtered_neuron_ids:

            neuron = f"neuron_{neuron_id}"

            try:
                # Extract responses for each condition
                radii_HH, responses_HH = subgroup_facilitation_st_curves_HH[neuron][:]
                radii_LH, responses_LH = subgroup_facilitation_st_curves_LH[neuron][:]  
                radii_LL, responses_LL = subgroup_facilitation_st_curves_LL[neuron][:]
                radii_low_contrast, responses_low_contrast =  subgroup_facilitation_st_curves_low[neuron][:]
            except:
                continue
        
            # Store conditions
            conditions = {
                "HH": (radii_HH, responses_HH),
                "LH": (radii_LH, responses_LH),
                "LL": (radii_LL, responses_LL),
            }

            try:
                # Extract center alone response 
                R_ctr_HH = subgroup_facilitation_st_results_HH[neuron][3]
                R_ctr_LH = subgroup_facilitation_st_results_LH[neuron][3]
                R_ctr_LL = subgroup_facilitation_st_results_LL[neuron][3]
            except:
                continue

            peak_index_near_facilitation = np.argmax(responses_low_contrast)  
            response_peak = responses_low_contrast[peak_index_near_facilitation]  
            stim_at_peak = radii_low_contrast[peak_index_near_facilitation]  
            facilitation_strength = ((response_peak - R_ctr_LL) / R_ctr_LL) * 100
            facilitation_strengths_low_contrast.append(facilitation_strength)
            facilitation_strengths_low_contrast_non_negative = [r for r in facilitation_strengths_low_contrast if r >= 0]
            diamter_low_contrast = 2 * radii_low_contrast

            
            # Extract sRFhigh and sRFlow values
            sRFlow = subgroup_facilitation_st_results_HH[neuron][1]
            sRFhigh = subgroup_facilitation_st_results_HH[neuron][2]

            sRFhigh_values.append(sRFhigh)
            sRFlow_values.append(sRFlow)


            if plot_tunning_curves:
                plt.figure(figsize=(6, 4)) 
                plt.plot(diamter_low_contrast, responses_low_contrast, 'o', label="Size Tuning Curve", color='b')  
                plt.xlabel("Stimulus Size (°)")
                plt.ylabel("Neuronal Response")
                plt.title("Size Tuning Curve")
                plt.axhline(y=R_ctr_LL, color='r', linestyle='--', label=f"R_center = {R_ctr_LL}")
                plt.legend()
                plt.grid(True)  

                directory = "/project/results/facilitation/annular_size_tunning_curves"  + "/" + neuron + "/"
                os.makedirs(directory, exist_ok=True)
                plt.savefig(directory + f"annular_size_tunning_curve_{neuron}.png")
            
            R_ctrs = {"HH": R_ctr_HH, "LH": R_ctr_LH, "LL": R_ctr_LL}
            supression_onset = {"HH": None, "LH": None, "LL":None}

            for cond, (radii, responses) in conditions.items():
                R_ctr = R_ctrs[cond]
                R_peak = np.max(responses)
                peak_idx = np.argmax(responses)
                suppression_idx = np.where(responses < 0.9 * R_ctr)[0]
                suppression_radii_from_peak_idx = np.where(responses < 0.9 * R_peak)[0]

                facilitation_strength = ((R_peak - R_ctr) / R_ctr) * 100
                if facilitation_strength > 1000:
                    print(neuron_id)
                    print(R_peak, R_ctr, facilitation_strength)
                facilitation_strengths_annular_tunning[cond].append(facilitation_strength)

                peak_radii[cond].append(radii[peak_idx])

                if suppression_idx.size > 0:
                    suppression_radii_from_center_only[cond].append(radii[suppression_idx[0]])

                else:
                    suppression_radii_from_center_only[cond].append(None)  
                
                if suppression_radii_from_peak_idx.size > 0:
                    suppression_radii_from_peak[cond].append(radii[suppression_radii_from_peak_idx[0]])
                else:
                    suppression_radii_from_peak[cond].append(None)  

                if plot_tunning_curves:
                    plt.figure(figsize=(6, 4)) 
                    plt.plot(radii, responses, 'o', label="Size Tuning Curve", color='b') 
                    plt.xlabel("Stimulus Size (°)")
                    plt.ylabel("Neuronal Response")
                    plt.title("Size Tuning Curve")
                    plt.axhline(y=R_ctr, color='r', linestyle='--', label=f"R_center = {R_ctr}")
                    plt.legend()
                    plt.grid(True) 

                    directory = "/project/results/facilitation/low_contrast_size_tunning_curves"  + "/" + neuron + "/"
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(directory + f"low_contrast_size_tunning_curve_{neuron}.png")
                    

                    plt.close


        # Compute sRFlow / sRFhigh ratio for each condition
        sRF_ratios = np.array(sRFlow_values) / np.array(sRFhigh_values)

        normalized_annular_ratio_at_suprresion_onset_LH = compute_normalized_annular_ratio(
                            suppression_radii_from_center_only["HH"],
                            suppression_radii_from_center_only["LH"],
                            sRFhigh_values

        )

        normalized_annular_ratio_at_suprresion_onset_LL = compute_normalized_annular_ratio(                           
                            suppression_radii_from_center_only["HH"],
                            suppression_radii_from_center_only["LL"],
                            sRFhigh_values)

        
        non_zero_peak_radii = {}
        non_zero_suppression_radii_from_center_only = {}
        facilitation_strengths_non_negative = {}
        larger_then_0_peak_radii = {}
        larger_then_0_facilitation_strengths = {}

        for condition in conditions:
            non_zero_peak_radii[condition] = [r for r in peak_radii[condition] if r is not None]
            non_zero_suppression_radii_from_center_only[condition] = [r for r in suppression_radii_from_center_only[condition] if r is not None]
            facilitation_strengths_non_negative[condition] = [r for r in facilitation_strengths_annular_tunning[condition] if r >= 0]
            larger_then_0_peak_radii[condition]= [r for r in peak_radii[condition] if r >= 0.1]
            larger_then_0_facilitation_strengths[condition] =  [r if r >= 0 else 0 for r in facilitation_strengths_annular_tunning[condition]] 

        larger_then_0_peaks_L = larger_then_0_peak_radii["LH"] + larger_then_0_peak_radii["LL"]
        larger_then_0_facilitation_strengths_L = larger_then_0_facilitation_strengths["LH"] + larger_then_0_facilitation_strengths["LL"]
        larger_then_0_facilitation_strengths_H = larger_then_0_facilitation_strengths["HH"]

        conditions = ["HH", "LH", "LL"]
        colors = {"HH": "black", "LH": "red", "LL": "gray"}

        # Figure 5A: Histogram of near facilitation strength
        plt.figure(figsize=(10, 5))
        plt.hist(facilitation_strengths_low_contrast_non_negative, bins=15, color="gray", edgecolor="black", 
                 weights=np.ones(len(facilitation_strengths_low_contrast_non_negative)) / len(facilitation_strengths_low_contrast_non_negative))

        plt.xlabel("Facilitation Strength (%)")
        plt.ylabel("Percentage of Neurons")
        plt.title("Near Facilitation Strength")
        # plt.axvline(np.mean(facilitation_strengths_annular_tunning), color="r", linestyle="dashed", label="Mean")
        plt.legend()
        plt.savefig("/project/results/facilitation/75A.png")
        plt.close

        # Figure 6B: Far facilitation strengh
        num_bins = 30
        bins = np.linspace(
            min(min(larger_then_0_facilitation_strengths_L), min(larger_then_0_facilitation_strengths_H)), 
            max(max(larger_then_0_facilitation_strengths_L), max(larger_then_0_facilitation_strengths_H)), 
            num_bins + 1 
        )

        bin_width = bins[1] - bins[0]
        bar_width = bin_width / 2

        plt.figure(figsize=(10, 8))
        plt.hist(
            [larger_then_0_facilitation_strengths_H, 
                larger_then_0_facilitation_strengths_L], 
            bins=bins, color=[colors["HH"], 
                              colors["LL"]],
            weights=[np.ones(len(larger_then_0_facilitation_strengths_H)) * 100 / len(larger_then_0_facilitation_strengths_H),
                        np.ones(len(larger_then_0_facilitation_strengths_L)) * 100 / len(larger_then_0_facilitation_strengths_L)], 
            width=bar_width, edgecolor="black", label=["HH", "LH, LL",],  alpha=0.8,
        )

        plt.yscale("log")
        plt.xlabel("Facilitation Strength")
        plt.ylabel("Percentage of Neurons")
        plt.title("Far Facilitation Strength")
        plt.legend()
        plt.savefig("/project/results/facilitation/75B.png")
        plt.close()

        # Figure 5C: Annulus inner radius at peak response
        plt.figure(figsize=(10, 8))
        plt.hist(larger_then_0_peaks_L, bins=50, color=colors[cond], alpha=0.5, label=cond, edgecolor="black", 
                 weights=np.ones(len(larger_then_0_peaks_L)) / len(larger_then_0_peaks_L))

        plt.xlabel("Annulus Inner Radius at peak Response (°)")
        plt.ylabel("Percentage of Neurons")
        plt.title("Annulus Inner Radius at peak Response (°) in low contrast")
        plt.axvline(np.mean(larger_then_0_peaks_L), color="blue", linestyle="dashed", label="Low contrast Mean")        
        plt.legend()
        plt.savefig("/project/results/facilitation/75C.png")
        plt.close()

        suppression_radii_from_peak_across_conditions = sum(suppression_radii_from_peak.values(), [])

        suppression_radii_from_center_only_across_conditions = sum(suppression_radii_from_center_only.values(), [])

        peak_radii_across_conditions = sum(peak_radii.values(), [])

        # Figure 5D: Scatter plot of suppression vs. facilitation distances
        plt.figure(figsize=(10, 8))
        plt.scatter(suppression_radii_from_peak_across_conditions, peak_radii_across_conditions,
                     color="red", alpha=0.6, label="from peak")
        plt.scatter(suppression_radii_from_center_only_across_conditions, peak_radii_across_conditions,
                     color= "white", alpha=0.6, label="from center", edgecolor="black")
        plt.xlabel("Annulus Inner Radius at Suppression")
        plt.ylabel("Annulus Inner Radius at Peak Facilitation")
        plt.title("Suppression vs. Facilitation Distances")
        plt.plot([0, 2.0], [0, 2.0], 'k--') 
        plt.legend()
        plt.savefig("/project/results/facilitation/75D.png")
        plt.close()

        # Figure 6A Scatter plot of sRFhigh vs. sRFlow 
        plt.figure(figsize=(10, 8))
        plt.scatter(sRFhigh_values, sRFlow_values, color="b", alpha=0.6, edgecolor="black")
        plt.plot([0, 2.0], [0, 2.0], 'k--') 
        plt.xlabel("sRFhigh (High Contrast RF size)")
        plt.ylabel("sRFlow (Low Contrast RF size)")
        plt.title("RF Size Shift at Low Contrast")
        plt.savefig("/project/results/facilitation/76A.png")
        plt.close()

        # Figure 6B: Histogram of sRFlow/sRFhigh ratio
        plt.figure(figsize=(10, 5))
        plt.hist(sRF_ratios, bins=20, edgecolor="black", label="sRFlow / sRFhigh", color="gray",
                 weights=np.ones(len(sRF_ratios)) / len(sRF_ratios))
        plt.xlabel("sRFlow / sRFhigh")
        plt.ylabel("Percentage of Neurons")
        plt.title("Contrast-Dependent RF Expansion")
        plt.savefig("/project/results/facilitation/76B.png")
        plt.close()
        
        # Figure 6C:  Scatter plot of suppression onset (high vs. low contrast)
        plt.figure(figsize=(10, 5))
        plt.scatter(suppression_radii_from_center_only["HH"], suppression_radii_from_center_only["LH"], color="r", label="HH vs LH")
        plt.scatter(suppression_radii_from_center_only["HH"], suppression_radii_from_center_only["LL"], color="g", label="HH vs LL")
        plt.plot([0, 2.0], [0, 2.0], 'k--') 
        plt.xlabel("Annulus Width at Suppression (High Contrast)")
        plt.ylabel("Annulus Width at Suppression (Low Contrast)")
        plt.title("Suppression Onset Shift")
        plt.legend()
        plt.savefig("/project/results/facilitation/76C.png")
        plt.close()


        # Figure 6D: Histogram of normalized annulus width at suprresion onset
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


