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
import pandas as pd
from tqdm import tqdm
import openpyxl


def compute_sosi(orientations, responses):
    """
    Compute Second Order Orientation Selectivity Index (SOSI) using Ringach's circular variance method.
    
    Parameters:
        orientations (array): Array of orientation angles (in radians).
        responses (array): Array of responses for each orientation.
    
    Returns:
        float: OSI value (0 to 1).
    """

    averages = []

    # Go over unique orientations and average all phases (spatial frequency is constant)
    unique_orientations = np.unique(orientations)
    
    for unique_orientation in unique_orientations:
        indices_orientation = tuple(np.argwhere(orientations == unique_orientation)[0])
        resposes_orientation = np.take(responses, indices_orientation)
        average_response = np.average(resposes_orientation)
        averages.append(average_response)


    # Average reponses
    averaged_responses = np.array(averages)

    # Normalize responses
    total_response = np.sum(averaged_responses)
    
    # Compute circular variance
    vector_sum = np.sum(averaged_responses * np.exp(2j * (unique_orientations)))
    R = np.abs(vector_sum) / total_response
    cv = 1 - R

    # OSI is 1 - CV
    return 1 - cv


def compute_sosi_and_asses_by_permutation_test(relative_orientations, responses, num_permutations=1000, alpha=0.05):
    """
    Perform a Bonferroni-corrected permutation test to assess the significance of SOSI.
    
    Args:
        relative_orientations (array): Stimulus orientations.
        mod_sfs (array): Spatial frequencies.
        mod_phases (array): Phases.
        responses (array): Neural responses.
        compute_sosi (function): Function to compute OSI.
        num_permutations (int): Number of shuffles.
        alpha (float): Significance level.
    
    Returns:
        p_value (float): p-value from the permutation test.
        significant (bool): Whether OSI is statistically significant after correction.
    """

    # Compute sample Second Order Orientation Selectivity Index
    observed_sosi = compute_sosi(relative_orientations, responses)

    # Create list of shuffled osi values
    shuffled_ssosi_values = []

    # In range on num permutations first shuffle the responses and then compute SOSI for each random iter
    for _ in range(num_permutations):
        shuffled_responses = np.random.permutation(responses)
        shuffled_sosi = compute_sosi(relative_orientations, shuffled_responses)
        shuffled_ssosi_values.append(shuffled_sosi)


    shuffled_ssosi_values = np.array(shuffled_ssosi_values)
    
    # Get p value by comparing the sample with shuffled data
    p_value = np.mean(shuffled_ssosi_values >= observed_sosi)

    # Adjust the significance treshold based on the number of used comparisons 
    corrected_alpha = alpha / len(responses)

    # returns computed sosi, p value and significance
    return observed_sosi, p_value, p_value < corrected_alpha


def plot_second_order_orientation_preferences(second_order_preferences, results_sosi, bins=5):
    """
    Plot distribution of second-order orientation preferences counted based on sosi significance of neurons.
    
    Parameters:
    - second_order_preferences: Array of orientation preferences
    - results_sosi: Array containing significance information
    - bins: Number of bins to use for histogram (default 5)
    """
    

    # Convert second-order preferences to degrees for visualization
    second_order_preferences = np.degrees(second_order_preferences)

    # Separete each of the values
    sosi_values = results_sosi[:, 0]
    p_values = results_sosi[:, 1]
    is_significants = results_sosi[:, 2]

    # Sort by significance
    second_order_preferences_significant = second_order_preferences[is_significants == 1]
    second_order_preferences_insignificant = second_order_preferences[is_significants == 0]
    
    # edges = [-90, 90]

    # # Create bin edges (2x for circular histogram)
    # bin_edges = np.linspace(edges[0], edges[1], 2 * (bins - 1))

    # def compute_histogram(data, bin_edges):
    #     min_val, max_val = data.min(), data.max()
    #     hist, _ = np.histogram(data, bins=bin_edges, range=(min_val, max_val), density=False)
    #     hist[0] += hist[-1]  # Merge first and last bin
    #     hist[-1] = hist[0]
    #     return hist / len(second_order_preferences)  # Normalize

    # # Compute histograms in parallel
    # hist_significant = compute_histogram(second_order_preferences_significant, bin_edges) 
    # hist_insignificant = compute_histogram(second_order_preferences_insignificant, bin_edges) 
    
    # # # Merge symmetric bins
    # merged_edges = []
    # merged_hist_significant = []
    # merged_hist_insignificant = []

    # step = (len(bin_edges) - 2) // (bins - 1)

    # merged_hist_significant.append(hist_significant[0])
    # merged_hist_insignificant.append(hist_insignificant[0])

    # for i in range(0, bins):
    #     left_index = 1 + i * step
    #     right_index = len(bin_edges) - 2 - i * step
    #     merged_edges.append(bin_edges[left_index])
    # for i in range(1, bins - 1):
    #     merged_hist_significant.append(hist_significant[left_index] + hist_significant[right_index])
    #     merged_hist_insignificant.append(hist_insignificant[left_index] + hist_insignificant[right_index])

    # merged_hist_significant.append(hist_significant[-1])
    # merged_hist_insignificant.append(hist_insignificant[-1])

    # # Convert to numpy arrays
    # merged_hist_significant = np.array(merged_hist_significant)
    # merged_hist_insignificant = np.array(merged_hist_insignificant)
    # merged_edges = np.array(merged_edges)

    #     # Compute bin widths
    # bar_widths = np.diff(bin_edges)

    # # Initialize total_width list
    # total_width = []

    # # Keep the first width
    # total_width.append(bar_widths[0])

    # # Iterate through the middle widths and sum each pair
    # i = 1  # Start from the second element
    # while i < len(bar_widths) - 1:
    #     total_width.append(bar_widths[i] + bar_widths[i + 1])
    #     i += 2  # Move two steps forward

    # merged_hist_significant = [
    #     hist_significant[0],  # First bin
    #     hist_significant[1] + hist_significant[2],  # Merging two middle bins
    #     hist_significant[3] + hist_significant[4],  # Merging two middle bins
    #     hist_significant[-1]  # Last bin
    # ]

    # merged_hist_insignificant = [
    #     hist_insignificant[0],  
    #     hist_insignificant[1] + hist_insignificant[2],  
    #     hist_insignificant[3] + hist_insignificant[4],  
    #     hist_insignificant[-1]
    # ]

    # # Convert lists to numpy arrays
    # merged_hist_significant = np.array(merged_hist_significant)
    # merged_hist_insignificant = np.array(merged_hist_insignificant)

    # This aint ideal 
    merged_edges = np.array([-90, -67.5, -22.5, 22.5, 67.5, 90])
    total_width = np.diff(merged_edges)

    hist_significant, edges = np.histogram(second_order_preferences_significant, bins=merged_edges)
    hist_insignificant, _ = np.histogram(second_order_preferences_insignificant, bins=merged_edges)

    hist_significant = (hist_significant / len(second_order_preferences))
    hist_insignificant = (hist_insignificant / len(second_order_preferences))


    print(hist_significant, hist_insignificant, merged_edges, total_width)
    print(sum(total_width))

    # Plot stacked histogram
    plt.figure(figsize=(8, 6))
    plt.bar(merged_edges[:-1], hist_significant, width=total_width, color='black', edgecolor='black', label='Significant', align='edge')
    plt.bar(merged_edges[:-1], hist_insignificant, width=total_width, color='white', edgecolor='black', bottom=hist_significant, label='Insignificant', align='edge')

    
    
    plt.title("Figure 3B: Distribution of Second-Order Orientation Preferences")
    plt.xlabel("Orientation Preference (degrees)")
    plt.ylabel("Proportion of Cells")
    plt.xticks(np.linspace(-90, 90, 5))
    plt.savefig("/project/results/modulation/second_order_preferences.png")
    plt.close()

def plot_sosi_histogram(results_sosi, bins=8):


    # Separete each of the values
    sosi_values = results_sosi[:, 0]
    p_values = results_sosi[:, 1]
    is_significants = results_sosi[:, 2]

    # Sort by significance
    sosi_values_significant = sosi_values[is_significants == 1]
    sosi_values_insignificant = sosi_values[is_significants == 0]

    # Define the range of the histogram
    min_val, max_val = sosi_values.min(), sosi_values.max()

    # Create histograms with the same binning
    significant_hist, bin_edges = np.histogram(sosi_values_significant, bins=bins, range=(min_val, max_val), density=False)
    insignificant_hist, _ = np.histogram(sosi_values_insignificant, bins=bins, range=(min_val, max_val), density=False)

    significant_hist = significant_hist / len(sosi_values)
    insignificant_hist = insignificant_hist / len(sosi_values)
    plt.figure(figsize=(8, 6))

    # Stack them together
    plt.bar(bin_edges[:-1], significant_hist, width=np.diff(bin_edges), color='black', edgecolor='black', label='Significant', align='edge')
    plt.bar(bin_edges[:-1], insignificant_hist, width=np.diff(bin_edges), color='white', edgecolor='black', bottom=significant_hist, label='Insignificant', align='edge')

    # Plot
    plt.xlabel("SOSI")
    plt.ylabel('Proportion of Cells')
    plt.title("Figure 3A: Distribution of Second Orientation Selectivity Index (SOSI)")
    plt.savefig("/project/results/modulation/osi_distribution.png")
    plt.close() 

def plot_circular_tunning_curves(raw_data, scaling_factor=100, neuron="0", show_only_averaged_phase=False):

    # For each neuron plots its circlular tunning

    relative_orientations, responses = raw_data

    if show_only_averaged_phase:
        unique_orientations = np.unique(relative_orientations)
        averaged_responses = []
        averaged_relative_ori = []
        for unique_orientation in unique_orientations:
            indices_orientation = tuple(np.argwhere(relative_orientations == unique_orientation)[0])
            resposes_orientation = np.take(responses, indices_orientation)
            average_response = np.average(resposes_orientation)
            averaged_responses.append(average_response)
            averaged_relative_ori.append(unique_orientation)
        averaged_responses = np.array(averaged_responses)
        averaged_relative_ori = np.array(averaged_relative_ori)
        responses = averaged_responses
        relative_orientations = averaged_relative_ori

    # Scale the responses for better vsibility
    area = scaling_factor * responses
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='polar')

    # Scatter plot with linear scaled area
    c = ax.scatter(relative_orientations, responses, 
                c=responses, s=area, alpha=0.75)

    # Add a colorbar to show response intensity
    plt.colorbar(c, label='Neuron Response')

    # Title and labels
    plt.title('Neuron Responses in Polar Coordinates')
    plt.tight_layout()

    directory = "/project/results/modulation/circular_tunning"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"{neuron}_circular_tunning_curve.png")
    plt.close()


def save_underlaying_data(neuron_ids, second_order_preferences, results_sosi, all_responses, all_oris):

    # For the untrusting we create whole results and stimulation overview file

    observed_sosi, p_value, is_significant = results_sosi.T
    
    df_raw_data = pd.DataFrame({
        'neuron_ids': list(neuron_ids),
        'relative_orientations': list(all_responses),
        'responses': list(all_oris)
    })

    df_results_sosi = pd.DataFrame({
        'neuron_ids': list(neuron_ids),
        'observed_sosi': list(observed_sosi),
        'p_value': list(p_value),
        'is_significant': list(is_significant)                    
    })

    df_second_order = pd.DataFrame({
        'neuron_ids': list(neuron_ids),
        'second_order_preferences': list(second_order_preferences)
    })

    df_joined_data = df_raw_data.merge(df_results_sosi, on='neuron_ids').merge(df_second_order, on='neuron_ids')

    # Create an Excel writer
    with pd.ExcelWriter('/project/results/modulation/modulation_results.xlsx') as writer:
        df_joined_data.to_excel(writer, sheet_name='Results', index=False)


def recreate_histograms_second_order_orientation(
        h5_file, 
        neuron_ids,
        num_permutations=10*4,
        alpha=0.05,
        plot_circular_tunning_curves_flag=True,
        save_underlaying_data_flag=True,
        ):
    
    # Paths to the saved data
    grating_group_path = '/full_field_params'
    modulator_group_path = '/modulator_params'

    results_sosi = []
    second_order_preferences = []
    all_responses = []
    all_oris = []


    # Extract data from HDF5
    with h5py.File(h5_file, 'r') as file:
        mod_group = file[modulator_group_path]
        
        print("Modulator group attributes:", dict(mod_group.attrs))
        
        for neuron_id in tqdm(neuron_ids, desc="Applying Modulator"):

            neuron = f"neuron_{neuron_id}"

            mod_group_data = mod_group[neuron]

            relative_orientations = mod_group_data[:, 0]
            mod_sfs = mod_group_data[:, 1]
            mod_phases = mod_group_data[:, 2]
            responses = mod_group_data[:, 3]

            max_response_idx = np.argmax(responses)
            max_orientation = relative_orientations[max_response_idx]
            max_response = responses[max_response_idx]

            all_responses.append(responses)
            all_oris.append(relative_orientations)

            observed_sosi, p_value, is_significant = compute_sosi_and_asses_by_permutation_test(relative_orientations, responses, num_permutations=num_permutations, alpha=alpha)
            results_sosi.append([observed_sosi, p_value, is_significant])

            second_order_preferences.append(max_orientation)

            if plot_circular_tunning_curves_flag:
                plot_circular_tunning_curves(raw_data=[relative_orientations, responses], neuron=neuron)



    # Move results into np array
    results_sosi = np.array(results_sosi)
    second_order_preferences = np.array(second_order_preferences)
    all_responses = np.array(all_responses)
    all_oris = np.array(all_oris)


    if save_underlaying_data_flag:
        save_underlaying_data(neuron_ids, second_order_preferences, results_sosi, all_responses, all_oris)


    # Move results into np array
    results_sosi = np.array(results_sosi)
    second_order_preferences = np.array(second_order_preferences)

    plot_second_order_orientation_preferences(second_order_preferences, results_sosi)
    plot_sosi_histogram(results_sosi)


