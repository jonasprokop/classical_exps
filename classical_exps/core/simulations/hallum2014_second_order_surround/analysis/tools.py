import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import scipy
from tqdm import tqdm


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




