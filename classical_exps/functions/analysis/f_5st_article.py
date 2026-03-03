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
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF## Plots
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



def recreate_histograms_second_order_orientation(
        h5_file, 
        neuron_ids,
        num_permutations=10*4,
        alpha=0.05,
        plot_circular_tunning_curves_flag=True,
        save_underlaying_data_flag=True,
        save_vector_debug_plots=True
        ,debug_plot_max_oris=12
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


def compute_sosi(orientations, responses):
    """
    Compute Second Order Orientation Selectivity Index (SOSI) using Ringach's circular variance method.
    
    Parameters:
        orientations (array): Array of orientation angles (in radians).
        responses (array): Array of responses for each orientation.
    
    Returns:
        float: OSI value (0 to 1).
    """

    # We will normalise by total response energy
    total_response = np.sum(responses)

    # Compute mean vector in the complex plane with doubled angles (2h as its the second harmonic of furrier transform, eg. direction inselective)
    R = np.abs(np.sum(responses * np.exp(2j * orientations))) / total_response

    # Compute second-order circular variance CV_{2h}, which is 1 - R
    cv2h = 1 - R

    # SOSI = 1 - CV_{2h}
    sosi = 1 - cv2h

    return sosi

def compute_sosi_and_asses_by_permutation_test(relative_orientations, responses, num_permutations=2000, alpha=0.05, num_tests=1):
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
    # We use Bonferroni correction here

    corrected_alpha = alpha / num_tests  # Only one test here, but kept for consistency if multiple tests are added

    # returns computed sosi, p value and significance
    return observed_sosi, p_value, p_value < corrected_alpha


def plot_second_order_orientation_preferences(second_order_preferences, results_sosi, bins=6):
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
    
    edges = [-90, 90]

    new_bin_count=bins*2-1

    hist=np.linspace(*edges,new_bin_count)
    merged_edges = np.array(list(hist[0:2]) + [hist[i]  for i in range(3, new_bin_count-2, 2)] + list(hist[new_bin_count-2:new_bin_count]))


    merged_edges = np.array([-90, -67.5, -22.5, 22.5, 67.5, 90])
    total_width = np.diff(merged_edges)

    hist_significant, _ = np.histogram(second_order_preferences_significant, bins=merged_edges)
    hist_insignificant, _ = np.histogram(second_order_preferences_insignificant, bins=merged_edges)

    hist_significant[0] += hist_significant[-1] 
    hist_significant[-1] = hist_significant[0]

    hist_insignificant[0] += hist_insignificant[-1] 
    hist_insignificant[-1] = hist_insignificant[0]

    hist_significant = (hist_significant / len(second_order_preferences))
    hist_insignificant = (hist_insignificant / len(second_order_preferences))

    np.save("/project/results/modulation/second_order_preferences.npy", second_order_preferences)
    np.save("/project/results/modulation/results_sosi.npy", results_sosi)


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

def plot_circular_tunning_curves(raw_data, scaling_factor=100, neuron="0", show_only_averaged_phase=True):

    # For each neuron plots its circlular tunning

    relative_orientations, responses = raw_data

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

    # Create whole stim and results file

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
import os
import numpy as np
import matplotlib.pyplot as plt

def wrap_deg(x, period):
    """Wrap degrees into [0, period)."""
    return float(x) % float(period)

def _unit_circle_ax(ax, title=None):
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, linewidth=0.8, alpha=0.6)
    ax.axvline(0, linewidth=0.8, alpha=0.6)
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), linewidth=1.0)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)

def draw_degree_spokes(ax, degrees, label_radius=1.10, spoke_radius=1.0, fontsize=9):
    """
    Draw reference spokes with: 0° at TOP, degrees increase CCW.
    """
    for d in degrees:
        d = float(d)
        ang = np.deg2rad(90.0 - d)   # <-- THIS is the fix
        x = np.cos(ang) * spoke_radius
        y = np.sin(ang) * spoke_radius
        ax.plot([0, x], [0, y], linewidth=0.8, alpha=0.25)

        lx = np.cos(ang) * label_radius
        ly = np.sin(ang) * label_radius
        ax.text(lx, ly, f"{d:.0f}°", ha="center", va="center", fontsize=fontsize, alpha=0.9)


def vec_from_deg_top0(deg):
    a = np.deg2rad(deg)
    return np.sin(a), np.cos(a)   # 0° -> (0,1)


def save_phase_vector_circle(phases, responses, outpath, neuron, rel_ori, max_points=120, show_weighted=True):
    phases = np.asarray(phases)
    responses = np.asarray(responses)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rel_ori_deg = wrap_deg(np.rad2deg(rel_ori), 180)
    _unit_circle_ax(ax, title=f"{neuron} | phase vectors (ϕ) @ rel ori={rel_ori_deg:.0f}°")

    draw_degree_spokes(ax, [0, 90, 180, 270])  # now correct

    w_norm = np.zeros_like(responses) if np.all(responses == 0) else responses / (np.max(responses) + 1e-12)

    V = 0 + 0j
    for phi, r, rn in zip(phases, responses, w_norm):
        phi_deg = wrap_deg(np.rad2deg(phi), 360)
        ux, uy = vec_from_deg_top0(phi_deg)

        ax.scatter([ux], [uy], s=25, alpha=0.75)
        ax.arrow(0, 0, ux*rn, uy*rn,
                 length_includes_head=True,
                 head_width=0.03, head_length=0.04,
                 linewidth=0.9, alpha=0.35)

        V += r * np.exp(1j * phi)

    mag = np.abs(V)
    phi_pref = np.angle(V)
    phi_pref_deg = wrap_deg(np.rad2deg(phi_pref), 360)

    rx, ry = vec_from_deg_top0(phi_pref_deg)
    ax.arrow(0, 0, rx, ry,
             length_includes_head=True,
             head_width=0.06, head_length=0.08,
             linewidth=2.6, alpha=0.95, label="resultant")

    ax.text(-1.15, -1.17, f"|Σ r·e^{{iϕ}}| = {mag:.3g} | ϕ_pref = {phi_pref_deg:.0f}°", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def _semi_circle_ax(ax, title=None):
    """Semicircle for orientation (θ in [0,180)), with 0° at top."""
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, linewidth=0.8, alpha=0.6)  # horizontal reference
    ax.axvline(0, linewidth=0.8, alpha=0.6)  # vertical reference

    # Semicircle: from top (0°) -> right (90°) -> bottom (180°)
    # Parameterize by θ in degrees: x=sin(θ), y=cos(θ)
    th = np.linspace(0, 180, 512)
    x = np.sin(np.deg2rad(th))
    y = np.cos(np.deg2rad(th))
    ax.plot(x, y, linewidth=1.2)

    # Only show right half-plane so it's visually "half a circle"
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)


def save_orientation_vector_circle(rel_oris, phase_invariant_amplitudes, outpath, neuron):
    rel_oris = np.asarray(rel_oris).astype(float)
    A = np.asarray(phase_invariant_amplitudes).astype(float)

    m = np.isfinite(rel_oris) & np.isfinite(A) & (A >= 0)
    rel_oris, A = rel_oris[m], A[m]
    if rel_oris.size == 0:
        raise ValueError("No valid orientation samples.")

    # θ ∈ [0, π)
    rel_oris = np.mod(rel_oris, np.pi)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _unit_circle_ax(ax, title=f"{neuron} | orientation tuning (θ), math uses 2θ")

    # Label θ only
    draw_degree_spokes(ax, [0, 45, 90, 135, 180])

    # Normalize for visuals
    A_norm = np.zeros_like(A) if np.all(A == 0) else A / (np.max(A) + 1e-12)

    # ---- computation in 2θ (unchanged, correct)
    V = np.sum(A * np.exp(2j * rel_oris))
    denom = np.sum(A) + 1e-12
    sosi = np.abs(V) / denom

    # ---- display in θ (unidirectional)
    xs = np.sin(rel_oris)
    ys = np.cos(rel_oris)

    ax.scatter(xs, ys, s=45, alpha=0.85, label=r"samples at $\theta$")

    for x, y, wn in zip(xs, ys, A_norm):
        ax.arrow(
            0, 0, x*wn, y*wn,
            length_includes_head=True,
            head_width=0.03, head_length=0.04,
            linewidth=1.2, alpha=0.45
        )

    # Resultant: show DIRECTION on the unit circle + show MAGNITUDE as a radial bar
    if np.abs(V) > 0:
        theta_pref = 0.5 * np.angle(V)
        theta_pref = np.mod(theta_pref, np.pi)
        theta_pref_deg = wrap_deg(np.rad2deg(theta_pref), 180)

        rx, ry = vec_from_deg_top0(theta_pref_deg)

        # 1) magnitude bar (0 -> sOSI)
        ax.arrow(
            0, 0, rx*sosi, ry*sosi,
            length_includes_head=True,
            head_width=0.05, head_length=0.07,
            linewidth=3.2, alpha=0.95, label="resultant magnitude"
        )

        # 2) direction arrow to the unit circle (always ends on the circle)
        # ax.arrow(
        #     0, 0, rx*1.0, ry*1.0,
        #     length_includes_head=True,
        #     head_width=0.03, head_length=0.05,
        #     linewidth=1.2, alpha=0.55, label="resultant direction"
        # )

        # # optional: mark the unit-circle tip so it's obvious
        # ax.scatter([rx], [ry], s=55, alpha=0.85)
    else:
        theta_pref_deg = np.nan

    ax.text(
        -1.15, -1.17,
        f"SOSI = {sosi:.3f} | θ_pref = {theta_pref_deg:.1f}°",
        fontsize=10
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
