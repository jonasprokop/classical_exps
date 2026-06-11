############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.core.tools.utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
## Data storage
import h5py
from collections import Counter
import scipy
from tqdm import tqdm

from classical_exps.core.simulations.hallum2014_second_order_surround.analysis.tools import (
    compute_sosi, 
    compute_sosi_and_asses_by_permutation_test, 
    save_underlaying_data,
)


from classical_exps.core.simulations.hallum2014_second_order_surround.analysis.plotter import (
    plot_circular_tunning_curves,
    plot_second_order_orientation_preferences,
    plot_second_order_orientation_preferences_scraped_comparison,
    plot_sosi_histogram,
    plot_sosi_histogram_scraped_comparison,
    save_phase_vector_circle,
    save_orientation_vector_circle,
)

###############################################################################
#####  PART V : Experiments for the fifth article, Hallum et al., 2014   #####
#####   ---------------------------------------------------------------   #####
#####       Surround suppression supports second-order feature encoding   #####
#####                     by macaque V1 and V2 neurons                    #####
#####   ---------------------------------------------------------------   #####
#####      DOI : http://dx.doi.org/10.1016/j.visres.2014.10.004           #####
###############################################################################


def recreate_histograms_second_order_orientation(
        h5_file, 
        neuron_ids,
        num_permutations=10*4,
        alpha=0.05,
        plot_circular_tunning_curves_flag=True,
        save_underlaying_data_flag=True,
        save_vector_debug_plots=True,
        debug_plot_max_oris=12,
        scraped_second_order_config= None,
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

            unique_orientations = np.unique(relative_orientations)
            vector_sums = []
            averaged_relative_ori = []

            if save_vector_debug_plots:
                if len(unique_orientations) > debug_plot_max_oris:
                    # evenly sample orientations for sanity visuals
                    ori_plot_subset = unique_orientations[
                        np.linspace(0, len(unique_orientations)-1, debug_plot_max_oris).astype(int)
                    ]
                else:
                    ori_plot_subset = unique_orientations

            for unique_orientation in unique_orientations:
                indices = np.where(relative_orientations == unique_orientation)[0]

                phases = mod_phases[indices]
                ori_responses = responses[indices]

                # ---- (NEW) save phase-circle BEFORE summing (only for subset)
                if save_vector_debug_plots and (unique_orientation in ori_plot_subset):
                    outpath = f"/project/results/modulation/vector_debug/{neuron}/phase_circle_relori_{np.rad2deg(unique_orientation):.0f}.png"
                    save_phase_vector_circle(
                        phases=phases,
                        responses=ori_responses,
                        outpath=outpath,
                        neuron=neuron,
                        rel_ori=unique_orientation,   # ok: this is the orientation at which we're showing the phase circle
                    )
                # Compute vector in the complex plane
                vectors = np.exp(1j * phases)
                
                # Multiply unit vectors by response strengths (weighting)
                weighted_vectors = ori_responses * vectors
                vector_sum = np.sum(weighted_vectors)

                # Save the magnitude of the sum (phase consistency weighted by strength)
                vector_sums.append(np.abs(vector_sum))
                averaged_relative_ori.append(unique_orientation)

            # Output vectors as numpy arrays
            responses = np.array(vector_sums)
            relative_orientations = np.array(averaged_relative_ori)


            if save_vector_debug_plots:
                outpath = f"/project/results/modulation/vector_debug/{neuron}/orientation_circle_2theta.png"
                save_orientation_vector_circle(
                    rel_oris=relative_orientations,          # θ grid (radians)
                    phase_invariant_amplitudes=responses,    # A(θ)
                    outpath=outpath,
                    neuron=neuron,
                )

            all_responses.append(responses)
            all_oris.append(relative_orientations)

            observed_sosi, p_value, is_significant = compute_sosi_and_asses_by_permutation_test(relative_orientations =relative_orientations, 
                                                                                                 responses=responses,
                                                                                                  num_permutations=num_permutations, alpha=alpha)           
           
            results_sosi.append([observed_sosi, p_value, is_significant])

            pref_idx = int(np.argmax(responses))
            max_orientation = relative_orientations[pref_idx]
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

    if scraped_second_order_config is not None:
        plot_second_order_orientation_preferences_scraped_comparison(
            second_order_preferences,
            results_sosi,
            scraped_config=scraped_second_order_config,
            save_path="/project/results/modulation/second_order_preferences_scraped_comparison.svg",
        )

        plot_sosi_histogram_scraped_comparison(
            results_sosi,
            scraped_config=scraped_second_order_config,
            save_path="/project/results/modulation/sosi_distribution_scraped_comparison.svg",
        )