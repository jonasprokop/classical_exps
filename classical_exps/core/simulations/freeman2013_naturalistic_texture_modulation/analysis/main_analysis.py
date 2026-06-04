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

from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.tools import (
    compute_freeman_modulation_index,

)

from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.audit import (
    texture_noise_response_validation_plots,
    texture_noise_image_statistics_vs_model_mi,
    texture_noise_lowlevel_family_backtest,
)


from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.plotters import (
    plot_family_modulation_index_comparison,
    plot_neuron_mi_distribution_comparison,
    plot_texture_noise_ratio_comparison,
    plot_texture_modulation_sign_summary,
    plot_family_modulation_significance_comparison
)




def texture_noise_response_results_1(
    h5_file, 
    neuron_ids,
    wanted_fam_order=None,
    plot_all_neurons=False,
    scraped_texture_noise_config=None,
):  
    """
    Visualise mean responses to texture and noise images.

    Stored response shape per neuron:
        [n_family, n_sample]

    Loaded response shape:
        all_tex_resp   [n_neuron, n_family, n_sample]
        all_noise_resp [n_neuron, n_family, n_sample]

    Plots:
        1. Population mean response by texture family.
        2. Per-neuron response-by-family plots for all selected neurons.
    """

    group_path = '/texture_noise_response'
    subgroup_tex_path = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)

    check_neurons_presence_error(
        h5_file=h5_file,
        list_group_path=[subgroup_noise_path, subgroup_tex_path],
        neuron_ids=neuron_ids,
    )

    neuron_ids = np.asarray(neuron_ids, dtype=int)

    print("--------------------------------------")
    print("Visualisation of neurons response to texture and noise images:")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot:")

    all_tex_resp = []
    all_noise_resp = []

    with h5py.File(h5_file, 'r') as file:
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        description = subgroup_tex.attrs["description"]
        family_ids = np.asarray(description.split('-'))

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"

            neuron_results_tex = np.asarray(subgroup_tex[neuron][:], dtype=float)
            neuron_results_noise = np.asarray(subgroup_noise[neuron][:], dtype=float)

            if neuron_results_tex.shape != neuron_results_noise.shape:
                raise ValueError(
                    f"Texture/noise shape mismatch for {neuron}: "
                    f"tex={neuron_results_tex.shape}, noise={neuron_results_noise.shape}"
                )

            if neuron_results_tex.ndim != 2:
                raise ValueError(
                    f"Expected [family, sample] for {neuron}, got {neuron_results_tex.shape}"
                )

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    # [n_neuron, n_family, n_sample]
    all_tex_resp = np.stack(all_tex_resp, axis=0)
    all_noise_resp = np.stack(all_noise_resp, axis=0)

    n_neuron, n_family, n_sample = all_tex_resp.shape

    if len(family_ids) != n_family:
        raise ValueError(
            f"Family ID count does not match response shape: "
            f"{len(family_ids)} vs {n_family}"
        )

    # Optional family ordering
    if wanted_fam_order is not None:
        dict_old_order = {fam: pos for pos, fam in enumerate(family_ids)}
        wanted_fam_order = np.asarray(wanted_fam_order).astype(str)

        new_order = np.array([dict_old_order[fam] for fam in wanted_fam_order])

        family_ids = family_ids[new_order]
        all_tex_resp = all_tex_resp[:, new_order, :]
        all_noise_resp = all_noise_resp[:, new_order, :]

    directory = "/project/results/texture_noise_response/"
    os.makedirs(directory, exist_ok=True)

    per_neuron_dir = os.path.join(directory, "per_neuron_response_curves")
    os.makedirs(per_neuron_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Population mean plot
    # ------------------------------------------------------------------
    # First average samples within each neuron/family, then average neurons.
    tex_neuron_family = np.mean(all_tex_resp, axis=2)      # [n_neuron, n_family]
    noise_neuron_family = np.mean(all_noise_resp, axis=2)  # [n_neuron, n_family]

    mean_tex_resp = np.mean(tex_neuron_family, axis=0)
    mean_noise_resp = np.mean(noise_neuron_family, axis=0)

    # SEM across neurons, not across samples of already-averaged population.
    sem_tex_resp = np.std(tex_neuron_family, axis=0, ddof=1) / math.sqrt(n_neuron)
    sem_noise_resp = np.std(noise_neuron_family, axis=0, ddof=1) / math.sqrt(n_neuron)

    fig, ax = plt.subplots(figsize=(8.5, 5))

    x = np.arange(n_family)

    ax.errorbar(
        x,
        mean_tex_resp,
        yerr=sem_tex_resp,
        fmt='o',
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        markersize=8,
        color='darkgoldenrod',
        label='Texture',
    )

    ax.errorbar(
        x,
        mean_noise_resp,
        yerr=sem_noise_resp,
        fmt='o',
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        markersize=8,
        color='orange',
        label='Noise',
        alpha=0.75,
    )

    max_plot = max(
        np.max(mean_tex_resp + sem_tex_resp),
        np.max(mean_noise_resp + sem_noise_resp),
        1e-6,
    )
    ax.set_ylim(0, max_plot * 1.08)

    ax.set_title('Mean responses to texture and noise images')
    ax.set_xlabel('Texture family')
    ax.set_ylabel('Mean response')
    ax.set_xticks(x)
    ax.set_xticklabels(family_ids, rotation=0)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(directory, "mean_response_to_noise_and_images.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # 1b. Texture/noise response-ratio comparison
    # ------------------------------------------------------------------
    # The current scraped article config contains MI, not raw texture/noise
    # responses. The ratio is recoverable from MI:
    #     T/N = (1 + MI) / (1 - MI)
    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
    )

    mean_mi_family = np.mean(mi_neuron_family, axis=0)

    scraped_mean_mi_family = None
    scraped_yerr = None

    if scraped_texture_noise_config is not None:
        scraped_block = scraped_texture_noise_config["texture_family_modulation"]["V1"]

        scraped_mean_mi_family = np.asarray(
            scraped_block["modulation_index"],
            dtype=float,
        )

        scraped_low = np.asarray(
            scraped_block["errorbar_endpoint_low"],
            dtype=float,
        )

        scraped_high = np.asarray(
            scraped_block["errorbar_endpoint_high"],
            dtype=float,
        )

        scraped_yerr = np.vstack([
            np.abs(scraped_mean_mi_family - scraped_low),
            np.abs(scraped_high - scraped_mean_mi_family),
        ])

    plot_texture_noise_ratio_comparison(
    family_ids=family_ids,
    model_mean_mi_family=mean_mi_family,
    save_path=os.path.join(directory, "mean_response_to_noise_ratio.svg"),
    scraped_mean_mi_family=scraped_mean_mi_family,
    scraped_yerr=scraped_yerr,
    scraped_label="Experimental",
    )

    # ------------------------------------------------------------------
    # 2. Per-neuron plots
    # ------------------------------------------------------------------
    if plot_all_neurons:
        print(f"    > Saving per-neuron response curves for {n_neuron} neurons")

        for idx, neuron_id in enumerate(neuron_ids):
            tex_family_sample = all_tex_resp[idx]      # [n_family, n_sample]
            noise_family_sample = all_noise_resp[idx]  # [n_family, n_sample]

            mean_tex_resp = np.mean(tex_family_sample, axis=1)
            mean_noise_resp = np.mean(noise_family_sample, axis=1)

            sem_tex_resp = np.std(tex_family_sample, axis=1, ddof=1) / math.sqrt(n_sample)
            sem_noise_resp = np.std(noise_family_sample, axis=1, ddof=1) / math.sqrt(n_sample)

            fig, ax = plt.subplots(figsize=(8.5, 5))

            ax.errorbar(
                x,
                mean_tex_resp,
                yerr=sem_tex_resp,
                fmt='o',
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                color='darkgreen',
                label='Texture',
            )

            ax.errorbar(
                x,
                mean_noise_resp,
                yerr=sem_noise_resp,
                fmt='o',
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                color='limegreen',
                label='Noise',
                alpha=0.75,
            )

            max_plot = max(
                np.max(mean_tex_resp + sem_tex_resp),
                np.max(mean_noise_resp + sem_noise_resp),
                1e-6,
            )
            ax.set_ylim(0, max_plot * 1.08)

            ax.set_title(f'Responses to texture and noise images for neuron {neuron_id}')
            ax.set_xlabel('Texture family')
            ax.set_ylabel('Response')
            ax.set_xticks(x)
            ax.set_xticklabels(family_ids, rotation=0)
            ax.legend(frameon=False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(per_neuron_dir, f"responses_neuron_{neuron_id}.png"),
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"    > Population plot saved to: {directory}")
    if plot_all_neurons:
        print(f"    > Per-neuron plots saved to: {per_neuron_dir}")
    print("--------------------------------------")


    texture_noise_response_validation_plots(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        wanted_fam_order=wanted_fam_order,
    )

    texture_noise_image_statistics_vs_model_mi(
    h5_file=h5_file,
    neuron_ids=neuron_ids,
    wanted_fam_order=wanted_fam_order,
    )

    texture_noise_lowlevel_family_backtest(
    h5_file=h5_file,
    neuron_ids=neuron_ids,
    wanted_fam_order=wanted_fam_order,
    )


def texture_noise_response_results_2(
    h5_file, 
    neuron_ids,
    wanted_fam_order,
    scraped_texture_noise_config=None,
):  
    """
    Visualise average modulation index across neurons for each texture family.

    MI = (response_texture - response_noise) / (response_texture + response_noise)

    Model statistics:
        mean_mi_family = mean over model neurons
        sem_mi_family  = SEM over model neurons
        significance   = family-wise MI vs 0, approximate t-test from mean ± SEM

    Article statistics:
        scraped means and SEMs from digitized Freeman V1 panel
        significance = approximate family-wise MI vs 0 from scraped mean ± SEM
    """

    group_path = '/texture_noise_response'
    subgroup_tex_path = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    check_neurons_presence_error(
        h5_file=h5_file,
        list_group_path=[subgroup_noise_path, subgroup_tex_path],
        neuron_ids=neuron_ids,
    )

    print("--------------------------------------")
    print("Visualisation of the average modulation index for each texture family:")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot:")

    all_tex_resp = []
    all_noise_resp = []

    with h5py.File(h5_file, 'r') as file:
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        description = subgroup_tex.attrs["description"]
        family_ids = np.asarray(description.split('-'))

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"

            neuron_results_tex = np.asarray(subgroup_tex[neuron][:], dtype=float)
            neuron_results_noise = np.asarray(subgroup_noise[neuron][:], dtype=float)

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    # Shape: [n_neuron, n_family, n_sample]
    all_tex_resp = np.stack(all_tex_resp, axis=0)
    all_noise_resp = np.stack(all_noise_resp, axis=0)

    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
    )

    n_neuron = mi_neuron_family.shape[0]

    # Mean across neurons -> one MI value per texture family
    mean_mi_family = np.mean(mi_neuron_family, axis=0)

    if n_neuron > 1:
        sem_mi_family = (
            np.std(mi_neuron_family, axis=0, ddof=1)
            / np.sqrt(n_neuron)
        )
    else:
        sem_mi_family = np.zeros_like(mean_mi_family)

    # ------------------------------------------------------------------
    # Optional family order
    # ------------------------------------------------------------------
    if wanted_fam_order is not None:
        dict_old_order = {fam: pos for pos, fam in enumerate(family_ids)}

        wanted_fam_order = np.asarray(wanted_fam_order)

        new_order = np.array([
            dict_old_order[str(fam)] for fam in wanted_fam_order
        ])

        family_ids = family_ids[new_order]
        mean_mi_family = mean_mi_family[new_order]
        sem_mi_family = sem_mi_family[new_order]
        mi_neuron_family = mi_neuron_family[:, new_order]

    directory = "/project/results/texture_noise_response/"
    os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------------
    # Scraped experimental V1 data
    # ------------------------------------------------------------------
    scraped_mean_mi_family = None
    scraped_yerr = None
    scraped_n = None

    if scraped_texture_noise_config is not None:
        scraped_block = scraped_texture_noise_config["texture_family_modulation"]["V1"]

        scraped_mean_mi_family = np.asarray(
            scraped_block["modulation_index"],
            dtype=float,
        )

        scraped_low = np.asarray(
            scraped_block["errorbar_endpoint_low"],
            dtype=float,
        )

        scraped_high = np.asarray(
            scraped_block["errorbar_endpoint_high"],
            dtype=float,
        )

        scraped_yerr = np.vstack([
            np.abs(scraped_mean_mi_family - scraped_low),
            np.abs(scraped_high - scraped_mean_mi_family),
        ])

        scraped_n = scraped_texture_noise_config[
            "v1_modulation_index_distribution"
        ]["n_cells_reported"]

        # Important:
        # Model family_ids are actual image-family IDs, e.g. "60".
        # Scraped article families are ordinal labels 1..15.
        # These are not the same namespace.
        #
        # If wanted_fam_order was chosen to put model families into the same
        # visual/order positions as the article bars, the scraped arrays should
        # stay in their native article order.

    # ------------------------------------------------------------------
    # Main family modulation figure
    # ------------------------------------------------------------------
    plot_family_modulation_index_comparison(
        family_ids=family_ids,
        model_mean_mi_family=mean_mi_family,
        model_yerr=sem_mi_family,
        model_n=n_neuron,
        save_path=os.path.join(directory, "average_modulation_index.svg"),
        scraped_mean_mi_family=scraped_mean_mi_family,
        scraped_yerr=scraped_yerr,
        scraped_n=scraped_n,
        scraped_label="Experimental",
        show_significance=True,
    )

    # ------------------------------------------------------------------
    # Dedicated family-wise significance comparison
    # ------------------------------------------------------------------
    if scraped_texture_noise_config is not None:
        plot_family_modulation_significance_comparison(
            family_ids=family_ids,
            model_mi_neuron_family=mi_neuron_family,
            scraped_config=scraped_texture_noise_config,
            save_path=os.path.join(
                directory,
                "family_modulation_significance_comparison.svg",
            ),
            article_area="V1",
            article_n=scraped_texture_noise_config[
                "v1_modulation_index_distribution"
            ]["n_cells_reported"],
            model_label="Model",
            article_label="Experimental V1",
        )

    print(f"    > Mean MI across families: {np.mean(mean_mi_family):.4f}")
    print(f"    > Family MI plot saved to: {directory}average_modulation_index.png")

    if scraped_texture_noise_config is not None:
        print(
            "    > Family significance comparison saved to: "
            f"{directory}family_modulation_significance_comparison.svg"
        )

    print("--------------------------------------")

def texture_noise_response_results_3(
    h5_file, 
    neuron_ids,
    scraped_texture_noise_config=None,

):  
    ''' This function aims to visualise the distribution of the modulation index averaged accross every texture family and every sample
        The modulation index is defined as so : (response_texture - response_noise) / (response_texture + response_noise)

        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None
    '''


    group_path = '/texture_noise_response'
    subgroup_tex_path = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    check_neurons_presence_error(
        h5_file=h5_file,
        list_group_path=[subgroup_noise_path, subgroup_tex_path],
        neuron_ids=neuron_ids,
    )

    print("--------------------------------------")
    print("Visualisation of distribution of mean modulation index across neurons:")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot:")

    all_tex_resp = []
    all_noise_resp = []

    with h5py.File(h5_file, 'r') as file:
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        for neuron_id in neuron_ids:
            neuron = f"neuron_{neuron_id}"

            neuron_results_tex = np.asarray(subgroup_tex[neuron][:], dtype=float)
            neuron_results_noise = np.asarray(subgroup_noise[neuron][:], dtype=float)

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    # Shape: [n_neuron, n_family, n_sample]
    all_tex_resp = np.stack(all_tex_resp, axis=0)
    all_noise_resp = np.stack(all_noise_resp, axis=0)

    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
    )

    # Mean across texture families -> one MI value per neuron
    mean_mi_neuron = np.mean(mi_neuron_family, axis=1)

    meanval = float(np.mean(mean_mi_neuron))

    plt.figure(figsize=(7, 5))

    weights = np.ones_like(mean_mi_neuron, dtype=float) / len(mean_mi_neuron)

    n_bins = 31

    max_abs = float(np.max(np.abs(mean_mi_neuron)))
    max_abs = max(max_abs, 1e-6)

    # Add padding, then create bin centers symmetric around zero.
    plot_abs = max(0.5, max_abs * 1.10)

    bin_width = (2 * plot_abs) / n_bins
    bin_centers = np.linspace(
        -plot_abs + bin_width / 2,
        plot_abs - bin_width / 2,
        n_bins,
    )
    bin_edges = np.concatenate([
        [bin_centers[0] - bin_width / 2],
        bin_centers + bin_width / 2,
    ])

    values, bins, _ = plt.hist(
        mean_mi_neuron,
        bins=bin_edges,
        edgecolor="black",
        linewidth=0.8,
        density=False,
        weights=weights,
        color="0.55",
    )

    max_hist = float(max(values)) if len(values) > 0 else 1.0

    plt.axvline(0, color='black', linestyle='-', linewidth=1, label='zero')
    plt.axvline(meanval, color='darkgoldenrod', linestyle="--", linewidth=1.5, label=f'mean = {meanval:.3f}')

    minval = float(min(bins))
    maxval = float(max(bins))

    minlim = min(-0.6, minval - abs(0.1 * minval))
    maxlim = max(0.6, maxval + abs(0.1 * maxval))

    plt.xlim([minlim, maxlim])
    plt.xlabel('Mean modulation index')
    plt.ylabel('Fraction of neurons')
    plt.xticks(labels=[-0.5, 0, 0.5], ticks=[-0.5, 0, 0.5])
    plt.title('Distribution of neuron-level modulation index')
    plt.legend()

    directory = "/project/results/texture_noise_response/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + "distribution_of_modulation_index.svg", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"    > Mean neuron MI: {meanval:.4f}")
    print("--------------------------------------")

    plot_neuron_mi_distribution_comparison(
        model_mean_mi_neuron=mean_mi_neuron,
        save_path=os.path.join(directory, "distribution_of_modulation_index.svg"),
        scraped_config=scraped_texture_noise_config,
    )

    if scraped_texture_noise_config is not None:
        plot_texture_modulation_sign_summary(
            model_mean_mi_neuron=mean_mi_neuron,
            scraped_config=scraped_texture_noise_config,
            save_path=os.path.join(
                directory,
                "texture_modulation_sign_summary.svg",
            ),
            model_label="Model",
            article_label="Experimental V1",
        )