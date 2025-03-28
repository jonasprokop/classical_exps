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


def calculate_general_suppresion_index(neuron_results):
    C = neuron_results[0]
    CequalS = neuron_results[1]
    CnotequalS = neuron_results[2]

    Capo = neuron_results[4]
    CequalSapo = neuron_results[5]
    CnotequalSapo = neuron_results[6]

    ASI = 1-(np.mean([CequalS, CnotequalS])/C)

    ASIapo = 1-(np.mean([CequalSapo, CnotequalSapo])/Capo)
    
    GSI = - ((C*ASI + Capo*ASIapo)/(C+Capo))

    return GSI

def bin_the_neurons_based_on_selectivity(neuronal_data):
    
    means = [np.mean(experiment) for experiment in neuronal_data]
    std_dev_pop = np.std(means, ddof=0)
    neuron_se = std_dev_pop / np.sqrt(len(neuronal_data))

    def is_significant(a, b, a_se, b_se):
        return abs(a - b) > (a_se + b_se)
    
    def is_larger(a, b, a_se, b_se):
        return a - b > (a_se + b_se)
    
    classifications = []
        
    for neuron in neuronal_data:

        bin = None
        C = neuron[0]
        CequalS = neuron[1]
        CnotequalS = neuron[2]
        S = neuron[3]

        Capo = neuron[4]
        CequalSapo = neuron[5]
        CnotequalSapo = neuron[6]
        Sapo = neuron[7]

        if 1 - (Capo/C) > 0.7:
            if is_larger(CnotequalS, CequalS, neuron_se, neuron_se):
                classifications.append("Orientation Contrast")

                if is_larger(CequalSapo, CnotequalSapo, neuron_se, neuron_se):
                    classifications.append("Surround-Dependent Suppression")

            elif is_larger(CequalS, CnotequalS, neuron_se, neuron_se):
                classifications.append("Uniform Orientation")

                if is_larger(CequalSapo, CnotequalSapo, neuron_se, neuron_se):
                    classifications.append("Surround-Dependent Suppression")

            elif is_significant(CequalS, C, neuron_se, neuron_se) and is_significant(CnotequalS, C, neuron_se, neuron_se):
                classifications.append("General Suppression")
    
            elif not (is_significant(CequalS, C, neuron_se, neuron_se) or is_significant(CnotequalS, C, neuron_se, neuron_se)):
                classifications.append("No Effect")
            
            else:
                if is_significant((CnotequalS + CequalS)/2,(C + S)/2, neuron_se, neuron_se):
                    classifications.append("General Suppression")
                else:
                    classifications.append("No Effect")
        else: 

            if is_larger(CnotequalS, CequalS, neuron_se, neuron_se) or is_larger(CnotequalSapo, CequalSapo, neuron_se, neuron_se):
                classifications.append("Orientation Contrast")

                if is_larger(CequalSapo, CnotequalSapo, neuron_se, neuron_se):
                    classifications.append("Surround-Dependent Suppression")

            elif is_larger(CequalS, CnotequalS, neuron_se, neuron_se) or is_larger(CequalSapo, CnotequalSapo, neuron_se, neuron_se):
                classifications.append("Uniform Orientation")

                if is_larger(CequalSapo, CnotequalSapo, neuron_se, neuron_se):
                    classifications.append("Surround-Dependent Suppression")

            elif is_significant(CequalS, C, neuron_se, neuron_se) and is_significant(CnotequalS, C, neuron_se, neuron_se):
                classifications.append("General Suppression")
            
                if (is_significant(C, CequalS, neuron_se, neuron_se) and is_significant(C, CnotequalS, neuron_se, neuron_se)) and \
                    not (is_significant(Capo, CequalSapo, neuron_se, neuron_se) or is_significant(Capo, CnotequalSapo, neuron_se, neuron_se)):
                    classifications.append("Center-Dependent Suppression")
    
            elif not (is_significant(CequalS, C, neuron_se, neuron_se) or is_significant(CnotequalS, C, neuron_se, neuron_se)):
                classifications.append("No Effect")
            
            else:
                if is_significant((CnotequalS + CequalS)/2,(C + S)/2, neuron_se, neuron_se):
                    classifications.append("General Suppression")
                else:
                    classifications.append("No Effect")
    
    return classifications

def plot_pie_chart(neuron_binning):

    # Count occurrences of each value
    plt.figure(figsize=(10, 10))
    count_dict = Counter(neuron_binning)
    proportions = [v / sum(count_dict.values()) for v in count_dict.values()]
    print(count_dict)
    print(proportions)

    labels = count_dict.keys()

    plt.pie(proportions, labels = labels, autopct='%1.1f%%', startangle=140)

    plt.savefig("/project/results/texture_patterns/GSI_pie_chart.png")
    plt.close()

def plot_GSI_histogram(results):

    gsi_values = np.array(list(results.values()), dtype=np.float32)

    # Plot histogram
    plt.figure(figsize=(10, 8))
    plt.hist(gsi_values, bins=100, color='gray', edgecolor='black', alpha=0.7, range=[-1.5, 1.5])

    # Add mean GSI line
    mean_GSI = np.mean(gsi_values)
    plt.axvline(mean_GSI, color='red', linestyle='dashed', linewidth=2)

    # Labels and title
    plt.xlabel(f"General Suppression Index (GSI), Mean GSI = {mean_GSI:.2f}")
    plt.ylabel("Number of Neurons")
    plt.title("Distribution of General Suppression Index (GSI)")
    plt.savefig("/project/results/texture_patterns/GSI_hist.png")
    plt.close()

def recreate_plots_general_suppresion_index(
        h5_file, 
        neuron_ids
        ):
    
    # Paths to the saved data
    orientation_contrast_group = "/orietantion_contrast"

    GSI_results = {}
    neuron_binning = {}
    neuronal_data = []

    # Extract data from HDF5
    with h5py.File(h5_file, 'r') as file:
        ori_contrast_group = file[orientation_contrast_group]
        
        print("Modulator group attributes:", dict(ori_contrast_group.attrs))
        
        for neuron_id in neuron_ids:

            # if neuron_id >= 1:
            #     continue
            
            ori_contrast_data_neuron = ori_contrast_group[f'neuron_{neuron_id}'][()]
            neuron_results = {}

            bar_sets= [[np.pi/4, -1],
                       [np.pi/4, np.pi/4],
                       [np.pi/4, 3*np.pi/4],
                       [-1, np.pi/4],
                       [3*np.pi/4, -1],
                       [3*np.pi/4, 3*np.pi/4],
                       [3*np.pi/4, np.pi/4],
                       [-1, 3*np.pi/4]]

            for i, bar_set in enumerate(bar_sets):
                matching_row = None
                for row in ori_contrast_data_neuron:
                    center_bar_angle_shift = row[5]
                    surround_bars_angle_shift = row[6]  
                    # print(bar_set[0], bar_set[1], "///",  center_bar_angle_shift, surround_bars_angle_shift)
                    if np.all(np.isclose(center_bar_angle_shift, bar_set[0], atol=1e-2)) and np.all(np.isclose(surround_bars_angle_shift, bar_set[1], atol=1e-2)):
                        matching_row = row
                    if matching_row is not None:
                        neuron_results[i] = matching_row[7] 
                        matching_row = None

            GSI = calculate_general_suppresion_index(neuron_results)
            GSI_results.update({neuron_id:GSI})

            trial_data = np.array(list(neuron_results.values()))
            neuronal_data.append(trial_data)

        neuron_binning = bin_the_neurons_based_on_selectivity(neuronal_data)



    plot_GSI_histogram(GSI_results)
    plot_pie_chart(neuron_binning)




                
