############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.core.tools.filtering_functions import *
from classical_exps.core.tools.utils import *
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy

import os
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import colors

from classical_exps.core.simulations.yeh2009_black_dominance.analysis.loaders import (
    load_black_white_results_bulk,
    load_black_white_maps_bulk,
    load_model_black_white_histogram_dataset,
    load_model_black_white_scatter_dataset,
    load_scraped_black_white_histogram_dataset,
    load_scraped_black_white_scatter_dataset,
    )

from classical_exps.core.simulations.yeh2009_black_dominance.analysis.plotters import (
    plot_black_white_histogram_dataset,
    plot_black_white_histogram_comparison,
    plot_black_white_scatter_dataset,
    plot_black_white_scatter_comparison_panel,
    plot_black_white_scatter_comparison_overlay,
    plot_black_white_energy_cumulative,
    plot_black_white_maps,
    plot_black_white_difference_panels,
)

###################################################################################
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------   #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1      #####
#####   -------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009       #####
###################################################################################



BLACK = "black"
DARK = "0.20"
MID = "0.45"
LIGHT = "0.85"
WHITE = "white"

ARTICLE_COLOR = BLACK
MODEL_COLOR = "#2F5D8C"


def black_white_results_1(
    h5_file,
    neuron_ids,
    neuron_depths=None,
    energy_thresh=None,
    plot_response_maps=True,
    maps_group_path="/black_white_preference/position_response_img",
    maps_save_dir="/project/results/black_and_white_experiment/summary_maps/",
    maps_prefix="bw",
    annotate_small_maps=True,
    convolution_demo=False,
    scraped_snr_config=None,
):
    """
    Visualise model black/white preference and, if provided, compare to scraped Yeh data.

    Expected model H5 result format:
        results[0] = E_black
        results[1] = E_white
        results[2] = log10(E_white / E_black)
    """

    summary_dir = "/project/results/black_and_white_experiment/summary_plots/"
    scraped_dir = "/project/results/black_and_white_experiment/scraped_snr/"
    comparison_dir = "/project/results/black_and_white_experiment/comparison/"

    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(scraped_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    loaded_results = load_black_white_results_bulk(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        group_path="/black_white_preference/results",
        strict=True,
    )

    if len(loaded_results["present_ids"]) == 0:
        print("--------------------------------------")
        print("Visualisation of black/white spatial-energy preference:")
        print("    > No valid model summary results available.")
        print("--------------------------------------")
        print()
        return

    valid_result_ids = loaded_results["neuron_ids"]

    energy_black = loaded_results["energy_b"]
    energy_white = loaded_results["energy_w"]
    log_energy_wb = loaded_results["log_energy_wb"]

    print("--------------------------------------")
    print("Visualisation of black/white spatial-energy preference:")
    print(f"    > Analysis requested on {len(neuron_ids)} neurons")
    print(f"    > {len(valid_result_ids)} neurons with valid model results")
    print(f"    > Mean spatial energy for black stimuli: {np.mean(energy_black):.3f}")
    print(f"    > Mean spatial energy for white stimuli: {np.mean(energy_white):.3f}")
    print(f"    > Mean log10(E_white / E_black): {np.mean(log_energy_wb):.3f}")
    print(f"    > Black-dominant model neurons: {np.mean(log_energy_wb < 0) * 100:.1f}%")

    # -------------------------------------------------------------------------
    # Build unified model datasets
    # -------------------------------------------------------------------------

    if scraped_snr_config is not None and "histogram" in scraped_snr_config:
        scraped_hist_dataset = load_scraped_black_white_histogram_dataset(
            scraped_snr_config,
            histogram_key="histogram",
            layers=None,
            label="Yeh experiment",
            hatch="///",
        )
        model_hist_bin_edges = scraped_hist_dataset["bin_edges"]
    else:
        scraped_hist_dataset = None
        model_hist_bin_edges = np.linspace(-1.8, 1.8, 13)

    model_hist_dataset = load_model_black_white_histogram_dataset(
        loaded_results,
        bin_edges=model_hist_bin_edges,
        label="Model",
        hatch=None,
    )

    model_scatter_dataset = load_model_black_white_scatter_dataset(
        loaded_results,
        label="Model",
    )

    # -------------------------------------------------------------------------
    # Model-only plots
    # -------------------------------------------------------------------------

    plot_black_white_histogram_dataset(
        model_hist_dataset,
        save_path=os.path.join(summary_dir, "log_ratio_histogram_model.png"),
        title="Model black/white bias",
        use_percent=False,
    )

    plot_black_white_scatter_dataset(
        model_scatter_dataset,
        save_path=os.path.join(summary_dir, "E_white_vs_black_scatter_model.png"),
        title="Model black/white response scatter",
        scale_model=False,
    )

    plot_black_white_energy_cumulative(
        energy_black,
        energy_white,
        save_path=os.path.join(summary_dir, "E_black_white_cumulative.png"),
        title="Cumulative black/white spatial energy distributions",
    )

    # -------------------------------------------------------------------------
    # Scraped-only and comparison plots
    # -------------------------------------------------------------------------

    if scraped_snr_config is not None:
        if "histogram" in scraped_snr_config:
            plot_black_white_histogram_dataset(
                scraped_hist_dataset,
                save_path=os.path.join(scraped_dir, "log_ratio_histogram_yeh_all_layers.svg"),
                title="Yeh experiment black/white bias",
                use_percent=False,
            )

            plot_black_white_histogram_comparison(
                [model_hist_dataset, scraped_hist_dataset],
                save_path=os.path.join(comparison_dir, "hist_model_vs_yeh_all_layers.svg"),
                title="Black/white bias: model vs Yeh experiment",
                use_proportion=True,
            )

        if "scatter" in scraped_snr_config:
            scraped_scatter_dataset = load_scraped_black_white_scatter_dataset(
                scraped_snr_config,
                scatter_key="scatter",
                layers=None,
                label="Yeh experiment",
            )

            plot_black_white_scatter_dataset(
                scraped_scatter_dataset,
                save_path=os.path.join(scraped_dir, "scatter_yeh_all_layers.svg"),
                title="Yeh experiment black/white SNR scatter",
                scale_model=False,
            )

            plot_black_white_scatter_comparison_panel(
                [model_scatter_dataset, scraped_scatter_dataset],
                save_path=os.path.join(comparison_dir, "scatter_model_vs_yeh_panel.svg"),
            )

            plot_black_white_scatter_comparison_overlay(
                [model_scatter_dataset, scraped_scatter_dataset],
                save_path=os.path.join(comparison_dir, "scatter_model_vs_yeh_overlay.svg"),
            )

    print(f"    > Model summary plots saved to: {summary_dir}")
    print(f"    > Scraped plots saved to: {scraped_dir}")
    print(f"    > Comparison plots saved to: {comparison_dir}")

    # -------------------------------------------------------------------------
    # Response maps + per-neuron black/white/difference panels
    # -------------------------------------------------------------------------

    if plot_response_maps:
        loaded_true = load_black_white_maps_bulk(
            h5_file=h5_file,
            neuron_ids=valid_result_ids,
            group_path=maps_group_path,
            strict=True,
        )

        if len(loaded_true["present_ids"]) == 0:
            print("    > No response maps available for filtered neurons.")
        else:
            plot_black_white_maps(
                maps=loaded_true["maps"],
                neuron_ids=loaded_true["neuron_ids"],
                save_dir=maps_save_dir,
                prefix=maps_prefix,
                annotate_small=annotate_small_maps,
                make_per_neuron=True,
                make_population_mean=True,
                make_convolution_demo=convolution_demo,
            )

            print(f"    > Response-map plots saved to: {maps_save_dir}")

            result_by_id = {
                int(nid): (
                    loaded_results["energy_b"][i],
                    loaded_results["energy_w"][i],
                    loaded_results["log_energy_wb"][i],
                )
                for i, nid in enumerate(loaded_results["neuron_ids"])
            }

            true_by_id = {
                int(nid): loaded_true["maps"][i]
                for i, nid in enumerate(loaded_true["neuron_ids"])
            }

            common_ids = [
                int(nid)
                for nid in loaded_true["neuron_ids"]
                if int(nid) in result_by_id
            ]

            if len(common_ids) == 0:
                print("    > No neurons with aligned response maps and energy results.")
            else:
                example_dir = "/project/results/black_and_white_experiment/example_maps/"
                os.makedirs(example_dir, exist_ok=True)

                true_maps = np.stack([true_by_id[nid] for nid in common_ids], axis=0)

                energy_black_common = np.asarray(
                    [result_by_id[nid][0] for nid in common_ids],
                    dtype=float,
                )
                energy_white_common = np.asarray(
                    [result_by_id[nid][1] for nid in common_ids],
                    dtype=float,
                )
                log_energy_wb_common = np.asarray(
                    [result_by_id[nid][2] for nid in common_ids],
                    dtype=float,
                )

                plot_black_white_difference_panels(
                    true_maps=true_maps,
                    neuron_ids=np.asarray(common_ids, dtype=int),
                    energy_b=energy_black_common,
                    energy_w=energy_white_common,
                    log_energy_wb=log_energy_wb_common,
                    save_dir=example_dir,
                    prefix=maps_prefix,
                )

                print(f"    > Per-neuron black/white/difference panels saved to: {example_dir}")

    print("--------------------------------------")
    print()

    return


def black_white_scraped_snr_results(
    scraped_config,
    *,
    save_dir="/project/results/black_and_white_experiment/scraped_snr/",
):
    """
    Standalone touchpoint for scraped Yeh 2009 black/white data.

    Produces:
        - summed histogram across all histogram layers, if available
        - summed scatter across all scatter layers, if available
    """

    os.makedirs(save_dir, exist_ok=True)

    print("--------------------------------------")
    print("Scraped Yeh 2009 black/white data")

    if "histogram" in scraped_config:
        hist_dataset = load_scraped_black_white_histogram_dataset(
            scraped_config,
            histogram_key="histogram",
            layers=None,
            label="Yeh experiment",
            hatch="///",
        )

        print("    > Histogram")
        print(f"        layers = {hist_dataset['layers']}")
        print(f"        n = {hist_dataset['n']}")
        print(f"        n reported = {hist_dataset.get('n_reported', 'NA')}")

        plot_black_white_histogram_dataset(
            hist_dataset,
            save_path=os.path.join(save_dir, "log_ratio_histogram_yeh_all_layers.png"),
            title="Yeh experiment black/white bias",
            use_percent=False,
        )

    if "scatter" in scraped_config:
        scatter_dataset = load_scraped_black_white_scatter_dataset(
            scraped_config,
            scatter_key="scatter",
            layers=None,
            label="Yeh experiment",
        )

        print("    > Scatter")
        print(f"        layers = {scatter_dataset['layers']}")
        print(f"        n = {scatter_dataset['n']}")
        print(
            f"        black > white = "
            f"{np.mean(scatter_dataset['black'] > scatter_dataset['white']) * 100:.1f}%"
        )

        plot_black_white_scatter_dataset(
            scatter_dataset,
            save_path=os.path.join(save_dir, "scatter_yeh_all_layers.png"),
            title="Yeh experiment black/white SNR scatter",
            scale_model=False,
        )

    print(f"    > saved to: {save_dir}")
    print("--------------------------------------")
    print()

    return




# def black_white_results_1(
#     h5_file,
#     neuron_ids,
#     neuron_depths = None,
#     SNR_thresh = 2
# ):  
#     ''' This function aims to visualise whether there is a black or white preference for the selected neurons

#         Prerequisite :
            
#             - function 'black_white_preference_experiment' executed for the required neurons

#         Filtering :

#             - Exclude the neurons with neither SNRw nor SNRb that are > SNR_thresh. function 'filter_SNR'

#         Arguments :

#             - neuron_depths : If not None, will plot the results including depth
#                               IMPORTANT : neuron_depths should contain ALL NEURONS DEPTHS, so that the neuron_depths[neuron_id] correspond to the depth of that specific neuron

#     '''

#     ## This function verify if the prerequisite function has been performed for the requested neurons
#     filtered_neuron_ids = filter_SNR(h5_file=h5_file, neuron_ids=neuron_ids, SNR_thresh=SNR_thresh, print_results=False)

#     group_path = '/black_white_preference'
    

#     ## Get the results
#     with h5py.File(h5_file, 'r') as f:
        
#         subgroup_results = f[group_path + '/results']
        
#         ## Get the log_SNRwb value of every neuron, and sort them whether this value is negative (black preference = OFF) or positive (white preference = ON)
#         logSNR_OFF = []
#         logSNR_ON  = []

#         if neuron_depths is not None :
#             neuron_depths_OFF = []
#             neuron_depths_ON = []

#         all_SNRb   = []
#         all_SNRw   = []

#         for neuron_id in filtered_neuron_ids :

#             neuron = f"neuron_{neuron_id}"
            
#             results_neuron = subgroup_results[neuron][:]

#             SNR_b    = results_neuron[0]
#             SNR_w    = results_neuron[1]
#             logSNRwb = results_neuron[2]

#             if logSNRwb < 0 : 
#                 logSNR_OFF.append(logSNRwb)
#                 if neuron_depths is not None :
#                     neuron_depths_OFF.append(neuron_depths[neuron_id])

#             else :
#                 logSNR_ON.append(logSNRwb)
#                 if neuron_depths is not None :
#                     neuron_depths_ON.append(neuron_depths[neuron_id])

#             all_SNRb.append(SNR_b)
#             all_SNRw.append(SNR_w)
    
#     ## Informations to print
#     n = len(neuron_ids)
#     n_new = len(filtered_neuron_ids)
#     mean_SNRb   = round(np.mean(all_SNRb),3)
#     mean_SNRw   = round(np.mean(all_SNRw),3)
#     mean_logSNR = round(np.mean(logSNR_OFF + logSNR_ON),3)

#     ## Show the results
#     print("--------------------------------------")
#     print("Visualisation of the neurons Black or white preference :")
#     print(f"    > Analysis made on {n} neurons")
#     print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
#     print(f"    > The mean SNR for black stimuli is {mean_SNRb}")
#     print(f"    > The mean SNR for white stimuli is {mean_SNRw}")
#     print(f"    > The mean log SNR white over black is {mean_logSNR}")
#     print(f"    > Plot :")
#     if neuron_depths is not None :
#         plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON, neuron_depths_OFF=neuron_depths_OFF, neuron_depths_ON=neuron_depths_ON)
#     else :
#         plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON)

#     print("--------------------------------------")
#     print()
        
#     return