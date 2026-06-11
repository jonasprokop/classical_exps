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
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.analysis.shared import plot_scatter_hist, neuron_key, ensure_dir

from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.loaders import (
    load_ccss_curves_bulk,
    neuron_key,
)
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.cohorts import (
    get_selectivity_filtered_neuron_ids,
)

## Plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
import os
from matplotlib.ticker import FormatStrFormatter 


##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "font.size": 10,
    "legend.frameon": False,
})

BLACK = "black"
DARK = "0.20"
MID = "0.45"
LIGHT = "0.85"
WHITE = "white"

ARTICLE_COLOR = BLACK
MODEL_COLOR = "#2F5D8C"



def ccss_results(
    h5_file,
    neuron_ids,
    *,
    fit_err_thresh=0.2,
    supp_thresh=0.1,
    apply_fit_error_filter=True,
    apply_no_supp_filter=True,
    apply_low_supp_filter=False,
    apply_valid_gsf_amrf_filter=True,
    do_mean_curves=True,
    do_single_neuron_curves=True,
    save_plots=True,
    show_mean_plot=True,
    show_example_neurons=0,
    random_seed=0,
    scraped_ccss_data=None,
    group_path="/center_contrast_surround_suppression",
    out_dir="/project/results/selectivity_and_spatial_distribution/ccss",
):
    """
    Main touchpoint for article-faithful CCSS contrast-family analysis.

    Loads the saved experiment config and neuron response matrices,
    applies cohort filtering, and generates article-style contrast-family plots.

    Saved data convention
    ---------------------
    Per neuron matrix shape:
        (n_surround_contrasts, n_center_contrasts)

    Axis convention:
        axis 0 -> surround contrast
        axis 1 -> center contrast

    Normalization conventions
    -------------------------
    1) joint-max normalization
       Used for overview / family plots.

    2) center-high-alone normalization
       Denominator = center-alone response at high center contrast.
       Used for Figure 5B / 5C scatter values via compute_ccss_figure_5_data.

    3) Figure 5A population curves
       Prepared by compute_ccss_figure5A_population_curves, which returns
       center/same/orth curves already normalized in the Figure-5A style
       expected by plot_ccss_figure5A.
    """
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    check_group_exists_error(h5_file=h5_file, group_path=group_path)

    filtered_neuron_ids = get_selectivity_filtered_neuron_ids(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        apply_fit_error_filter=apply_fit_error_filter,
        apply_no_supp_filter=apply_no_supp_filter,
        apply_low_supp_filter=apply_low_supp_filter,
        apply_valid_gsf_amrf_filter=apply_valid_gsf_amrf_filter,
        verbose=False,
    )
    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)

    # load scraped data

    if len(filtered_neuron_ids) == 0:
        raise ValueError("No neurons left after filtering.")

    load_result = load_ccss_curves_bulk(
        h5_file=h5_file,
        neuron_ids=filtered_neuron_ids,
        group_path=group_path,
        strict=True,
    )

    present_ids = np.asarray(load_result["present_ids"], dtype=int)
    missing_ids = np.asarray(load_result["missing_ids"], dtype=int)
    center_contrasts = np.asarray(load_result["center_contrasts"], dtype=float)
    surround_contrasts = np.asarray(load_result["surround_contrasts"], dtype=float)
    config = load_result["config"]

    if len(present_ids) == 0:
        raise ValueError("No filtered neurons were found in the CCSS results group.")

    responses_same = np.stack(
        [load_result["responses_same_by_id"][nid] for nid in present_ids],
        axis=0,
    )  # shape: (N, S, C)

    responses_orth = np.stack(
        [load_result["responses_orth_by_id"][nid] for nid in present_ids],
        axis=0,
    )  # shape: (N, S, C)

    def normalize_per_neuron(arr, norm_factors=None):
        """
        Normalize arr of shape (N, S, C) by one scalar per neuron.
        """
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 3:
            raise ValueError(f"Expected arr shape (N, S, C), got {arr.shape}")

        if norm_factors is None:
            norm_factors = np.max(arr.reshape(arr.shape[0], -1), axis=1)

        norm_factors = np.asarray(norm_factors, dtype=float)
        if norm_factors.ndim != 1 or norm_factors.shape[0] != arr.shape[0]:
            raise ValueError(
                f"norm_factors must have shape ({arr.shape[0]},), got {norm_factors.shape}"
            )

        norm_factors = norm_factors.copy()
        norm_factors[~np.isfinite(norm_factors)] = 1.0
        norm_factors[np.isclose(norm_factors, 0.0)] = 1.0

        return arr / norm_factors[:, None, None], norm_factors

    def sem_over_neurons(arr):
        """
        SEM across neuron axis for arr of shape (N, S, C).
        Returns shape (S, C).
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected arr shape (N, S, C), got {arr.shape}")

        n = arr.shape[0]
        if n < 2:
            return np.zeros(arr.shape[1:], dtype=float)

        return np.std(arr, axis=0, ddof=1) / np.sqrt(n)
    
    scraped_css_data = None

    # --------------------------------------------------
    # Common indices
    # --------------------------------------------------
    s0_idx = int(np.argmin(np.abs(surround_contrasts - 0.0)))
    s_hi_idx = int(np.argmin(np.abs(surround_contrasts - 0.5)))
    c_hi_idx = int(np.argmin(np.abs(center_contrasts - 0.5)))

    # --------------------------------------------------
    # 1) JOINT-MAX normalization for overview/family plots
    # --------------------------------------------------
    joint_max_per_neuron = np.maximum(
        np.max(responses_same.reshape(responses_same.shape[0], -1), axis=1),
        np.max(responses_orth.reshape(responses_orth.shape[0], -1), axis=1),
    )
    joint_max_per_neuron[~np.isfinite(joint_max_per_neuron)] = 1.0
    joint_max_per_neuron[np.isclose(joint_max_per_neuron, 0.0)] = 1.0

    responses_same_norm_joint_max, norm_factors_joint_max = normalize_per_neuron(
        responses_same,
        norm_factors=joint_max_per_neuron,
    )
    responses_orth_norm_joint_max, _ = normalize_per_neuron(
        responses_orth,
        norm_factors=joint_max_per_neuron,
    )

    mean_responses_same_joint_max = np.mean(responses_same_norm_joint_max, axis=0)
    mean_responses_orth_joint_max = np.mean(responses_orth_norm_joint_max, axis=0)
    sem_responses_same_joint_max = sem_over_neurons(responses_same_norm_joint_max)
    sem_responses_orth_joint_max = sem_over_neurons(responses_orth_norm_joint_max)

    # --------------------------------------------------
    # 2) CENTER-HIGH-ALONE normalization for Figure 5B/C
    # denominator = center-alone response at high center contrast
    # --------------------------------------------------
    center_hi_alone_per_neuron = responses_same[:, s0_idx, c_hi_idx].astype(float, copy=True)
    center_hi_alone_per_neuron[~np.isfinite(center_hi_alone_per_neuron)] = 1.0
    center_hi_alone_per_neuron[np.isclose(center_hi_alone_per_neuron, 0.0)] = 1.0

    responses_same_norm_center_hi, norm_factors_center_hi = normalize_per_neuron(
        responses_same,
        norm_factors=center_hi_alone_per_neuron,
    )
    responses_orth_norm_center_hi, _ = normalize_per_neuron(
        responses_orth,
        norm_factors=center_hi_alone_per_neuron,
    )

    mean_responses_same_center_hi = np.mean(responses_same_norm_center_hi, axis=0)
    mean_responses_orth_center_hi = np.mean(responses_orth_norm_center_hi, axis=0)
    sem_responses_same_center_hi = sem_over_neurons(responses_same_norm_center_hi)
    sem_responses_orth_center_hi = sem_over_neurons(responses_orth_norm_center_hi)

    print("--------------------------------------")
    print("CCSS analysis")
    print(f"Input neurons:                    {len(neuron_ids)}")
    print(f"After filtering:                  {len(filtered_neuron_ids)}")
    print(f"Present in CCSS results:          {len(present_ids)}")
    print(f"Missing after filtering/loading:  {len(missing_ids)}")
    if len(neuron_ids) > 0:
        print(f"Final retained fraction:          {100 * len(present_ids) / len(neuron_ids):.2f}%")
    print(f"Center contrasts:                 {center_contrasts.tolist()}")
    print(f"Surround contrasts:               {surround_contrasts.tolist()}")
    print("--------------------------------------")

    mean_dir = os.path.join(out_dir, "population")
    single_dir = os.path.join(out_dir, "single_neurons")

    mean_family_plot_path = None
    figure_5A_mean_plot_path = None
    figure_5B_plot_path = None
    figure_5B_marginal_plot_path = None
    figure_5C_plot_path = None
    figure_5C_marginal_plot_path = None

    single_curve_plot_paths = []
    single_fig5_plot_paths = []

    # --------------------------------------------------
    # Figure 5A population curves
    # --------------------------------------------------
    fig5A_mean_data = compute_ccss_figure5A_population_curves(
        center_contrasts=center_contrasts,
        surround_contrasts=surround_contrasts,
        responses_same=responses_same,
        responses_orth=responses_orth,
        high_center_contrast=0.5,
        high_surround_contrast=0.5,
    )

    # --------------------------------------------------
    # Mean family curves
    # --------------------------------------------------
    if do_mean_curves:
        if save_plots:
            mean_family_plot_path = os.path.join(
                mean_dir,
                "ccss_mean_family_curves_same_vs_orth.png",
            )
            figure_5A_mean_plot_path = os.path.join(
                mean_dir,
                "ccss_figure5A_population_mean_relative.png",
            )

        # Overview family plot uses JOINT-MAX normalization
        plot_mean_ccss_family_curves_dual(
            center_contrasts=center_contrasts,
            surround_contrasts=surround_contrasts,
            mean_responses_same=mean_responses_same_joint_max,
            mean_responses_orth=mean_responses_orth_joint_max,
            sem_same=sem_responses_same_joint_max,
            sem_orth=sem_responses_orth_joint_max,
            draw_sem=True,
            n_neurons=len(present_ids),
            save_path=mean_family_plot_path,
            show=show_mean_plot,
            ylabel="Relative response",
            close=True,
            title=f"Mean normalised CCSS contrast family (n={len(present_ids)})",
        )

        # double up and inject the scraped data

        # Figure 5A population plot:
        # plot_ccss_figure5A expects (S, C) matrices, so build pseudo-matrices
        # from the already-prepared Figure-5A population curves.
        s_len = len(surround_contrasts)
        c_len = len(center_contrasts)

        fig5A_same_mat = np.zeros((s_len, c_len), dtype=float)
        fig5A_orth_mat = np.zeros((s_len, c_len), dtype=float)

        fig5A_same_mat[s0_idx] = fig5A_mean_data["mean_center"]
        fig5A_same_mat[s_hi_idx] = fig5A_mean_data["mean_same"]
        fig5A_orth_mat[s_hi_idx] = fig5A_mean_data["mean_orth"]

        plot_ccss_figure5A(
            center_contrasts=center_contrasts,
            surround_contrasts=surround_contrasts,
            responses_same=fig5A_same_mat,
            responses_orth=fig5A_orth_mat,
            high_center_contrast=0.5,
            nominal_low_center_contrast=0.12,
            high_surround_contrast=0.5,
            low_fraction=0.20,
            sem_center=fig5A_mean_data["sem_center"],
            sem_same=fig5A_mean_data["sem_same"],
            sem_orth=fig5A_mean_data["sem_orth"],
            draw_sem=True,
            print_values=False,
            title=f"Population Figure 5A-style curves (n={len(present_ids)})",
            ylabel="Relative response",
            legend=True,
            save_path=figure_5A_mean_plot_path,
            show=show_mean_plot,
            close=True,
        )

    # --------------------------------------------------
    # Joint mean tunning curves for same vs orth family, with SEM shading
    # --------------------------------------------------
    joint_figure_5A_path = None

    if save_plots and scraped_ccss_data is not None:
        joint_figure_5A_path = os.path.join(
            mean_dir,
            "ccss_figure5A_model_article_overlay.svg",
        )

        plot_ccss_figure5A_model_article_overlay(
            fig5A_mean_data=fig5A_mean_data,
            figure_5A_data=scraped_ccss_data["figure_5A_data"],
            center_contrasts=center_contrasts,
            ylabel="Relative response",
            save_path=joint_figure_5A_path,
            show=False,
            close=True,
        )

        print(f"    > Saved joint Figure 5A overlay to {joint_figure_5A_path}")

    # --------------------------------------------------
    # Per-neuron plots
    # --------------------------------------------------
    if do_single_neuron_curves and save_plots:
        for nid in present_ids:
            family_save_path = os.path.join(
                single_dir,
                "ccss_family_curves",
                f"ccss_family_curves_same_vs_orth_{neuron_key(nid)}.png",
            )

            plot_ccss_family_curves_dual(
                center_contrasts=center_contrasts,
                surround_contrasts=surround_contrasts,
                responses_same=load_result["responses_same_by_id"][nid],
                responses_orth=load_result["responses_orth_by_id"][nid],
                neuron_id=int(nid),
                save_path=family_save_path,
                show=False,
                close=True,
            )
            single_curve_plot_paths.append(family_save_path)

            fig5_single_save_path = os.path.join(
                single_dir,
                "figure_5A",
                f"ccss_figure5A_{neuron_key(nid)}.png",
            )

            plot_ccss_figure5A(
                center_contrasts=center_contrasts,
                surround_contrasts=surround_contrasts,
                responses_same=load_result["responses_same_by_id"][nid],
                responses_orth=load_result["responses_orth_by_id"][nid],
                neuron_id=int(nid),
                high_center_contrast=0.5,
                nominal_low_center_contrast=0.12,
                high_surround_contrast=0.5,
                low_fraction=0.20,
                draw_sem=False,
                print_values=False,
                save_path=fig5_single_save_path,
                show=False,
                close=True,
            )
            single_fig5_plot_paths.append(fig5_single_save_path)

    # --------------------------------------------------
    # Article-faithful Figure 5 B and C scatter data
    # --------------------------------------------------
    fig5_scatter_data = compute_ccss_figure_5_data(
        present_ids=present_ids,
        center_contrasts=center_contrasts,
        surround_contrasts=surround_contrasts,
        responses_same=responses_same,
        responses_orth=responses_orth,
        responses_same_norm_max=responses_same_norm_joint_max,
        responses_orth_norm_max=responses_orth_norm_joint_max,
        responses_same_norm_center=responses_same_norm_center_hi,
        responses_orth_norm_center=responses_orth_norm_center_hi,
        scatter_norm_mode="panel",
        nominal_low_center_contrast=0.12,
        high_center_contrast=0.5,
        high_surround_contrast=0.5,
        low_response_fraction=0.20,
    )

    if save_plots:
        figure_5B_plot_path = plot_ccss_figure_5B_high_contrast_scatter(
            fig5_scatter_data,
            out_path=mean_dir,
            filename="ccss_figure_5B_high_contrast.png",
            with_marginals=False,
            show=False,
            close=True,
        )

        figure_5B_marginal_plot_path = plot_ccss_figure_5B_high_contrast_scatter(
            fig5_scatter_data,
            out_path=mean_dir,
            filename="ccss_figure_5B_high_contrast_with_marginals.png",
            with_marginals=True,
            show=False,
            close=True,
        )

        figure_5C_plot_path = plot_ccss_figure_5C_low_contrast_scatter(
            fig5_scatter_data,
            out_path=mean_dir,
            filename="ccss_figure_5C_low_contrast.png",
            with_marginals=False,
            show=False,
            close=True,
        )

        figure_5C_marginal_plot_path = plot_ccss_figure_5C_low_contrast_scatter(
            fig5_scatter_data,
            out_path=mean_dir,
            filename="ccss_figure_5C_low_contrast_with_marginals.png",
            with_marginals=True,
            show=False,
            close=True,
        )

        print("Saved Figure 5 plots:")
        print("    B:", figure_5B_plot_path)
        print("    B + marginals:", figure_5B_marginal_plot_path)
        print("    C:", figure_5C_plot_path)
        print("    C + marginals:", figure_5C_marginal_plot_path)

    scraped_plot_paths = {}

    if scraped_ccss_data is not None and save_plots:
        scraped_out_dir = os.path.join(out_dir, "scraped_article")

        scraped_result = run_scraped_ccss_figure_5(
            figure_5A_data=scraped_ccss_data["figure_5A_data"],
            figure_5B_data=scraped_ccss_data["figure_5B_data"],
            figure_5C_data=scraped_ccss_data["figure_5C_data"],
            out_path=scraped_out_dir,
            with_marginals=True,
        )
        scraped_plot_paths = scraped_result["plot_paths"]

    joint_figure_5B_plot_path = None
    joint_figure_5B_marginal_plot_path = None
    joint_figure_5C_plot_path = None
    joint_figure_5C_marginal_plot_path = None

    if scraped_ccss_data is not None and save_plots:
        joint_figure_5B_plot_path = plot_ccss_figure_5B_model_article_scatter(
            fig5_scatter_data,
            scraped_ccss_data["figure_5B_data"],
            out_path=mean_dir,
            filename="ccss_figure_5B_high_contrast_model_article.svg",
            with_marginals=False,
            show=False,
            close=True,
        )

        joint_figure_5B_marginal_plot_path = plot_ccss_figure_5B_model_article_scatter(
            fig5_scatter_data,
            scraped_ccss_data["figure_5B_data"],
            out_path=mean_dir,
            filename="ccss_figure_5B_high_contrast_model_article_with_boxplots.svg",
            with_marginals=True,
            show=False,
            close=True,
        )

        joint_figure_5C_plot_path = plot_ccss_figure_5C_model_article_scatter(
            fig5_scatter_data,
            scraped_ccss_data["figure_5C_data"],
            out_path=mean_dir,
            filename="ccss_figure_5C_low_contrast_model_article.svg",
            with_marginals=False,
            show=False,
            close=True,
        )

        joint_figure_5C_marginal_plot_path = plot_ccss_figure_5C_model_article_scatter(
            fig5_scatter_data,
            scraped_ccss_data["figure_5C_data"],
            out_path=mean_dir,
            filename="ccss_figure_5C_low_contrast_model_article_with_boxplots.svg",
            with_marginals=True,
            show=False,
            close=True,
        )

        print("Saved joint Figure 5 scatter overlays:")
        print("    B:", joint_figure_5B_plot_path)
        print("    B + boxplots:", joint_figure_5B_marginal_plot_path)
        print("    C:", joint_figure_5C_plot_path)
        print("    C + boxplots:", joint_figure_5C_marginal_plot_path)

    return {
        # ids / config
        "present_ids": present_ids,
        "missing_ids": missing_ids,
        "config": config,

        # axes / raw responses
        "center_contrasts": center_contrasts,
        "surround_contrasts": surround_contrasts,
        "responses_same": responses_same,
        "responses_orth": responses_orth,

        # common indices
        "s0_idx": s0_idx,
        "s_hi_idx": s_hi_idx,
        "c_hi_idx": c_hi_idx,

        # joint-max normalization branch
        "norm_factors_joint_max": norm_factors_joint_max,
        "responses_same_norm_joint_max": responses_same_norm_joint_max,
        "responses_orth_norm_joint_max": responses_orth_norm_joint_max,
        "mean_responses_same_joint_max": mean_responses_same_joint_max,
        "mean_responses_orth_joint_max": mean_responses_orth_joint_max,
        "sem_responses_same_joint_max": sem_responses_same_joint_max,
        "sem_responses_orth_joint_max": sem_responses_orth_joint_max,

        # center-high-alone normalization branch
        "norm_factors_center_hi": norm_factors_center_hi,
        "responses_same_norm_center_hi": responses_same_norm_center_hi,
        "responses_orth_norm_center_hi": responses_orth_norm_center_hi,
        "mean_responses_same_center_hi": mean_responses_same_center_hi,
        "mean_responses_orth_center_hi": mean_responses_orth_center_hi,
        "sem_responses_same_center_hi": sem_responses_same_center_hi,
        "sem_responses_orth_center_hi": sem_responses_orth_center_hi,

        # figure 5 derived data
        "fig5A_mean_data": fig5A_mean_data,
        "fig5_scatter_data": fig5_scatter_data,

        # plot paths
        "mean_family_plot_path": mean_family_plot_path,
        "figure_5A_mean_plot_path": figure_5A_mean_plot_path,
        "single_curve_plot_paths": single_curve_plot_paths,
        "single_fig5_plot_paths": single_fig5_plot_paths,
        "figure_5B_plot_path": figure_5B_plot_path,
        "figure_5B_marginal_plot_path": figure_5B_marginal_plot_path,
        "figure_5C_plot_path": figure_5C_plot_path,
        "figure_5C_marginal_plot_path": figure_5C_marginal_plot_path,

        # joint model-article overlay plot paths
        "joint_figure_5A_path": joint_figure_5A_path,
        "joint_figure_5B_plot_path": joint_figure_5B_plot_path,
        "joint_figure_5B_marginal_plot_path": joint_figure_5B_marginal_plot_path,
        "joint_figure_5C_plot_path": joint_figure_5C_plot_path,
        "joint_figure_5C_marginal_plot_path": joint_figure_5C_marginal_plot_path,

    }


def _round_up_to_step(x, step):
    if not np.isfinite(x) or x <= 0:
        return step
    return step * np.ceil(x / step)

def _safe_sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    valid = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = sd / np.sqrt(valid)
    sem = np.where(valid > 1, sem, np.nan)
    return sem

def _get_ccss_fig5A_styles():
    return {
        "center": dict(
            color="black",
            linewidth=1.4,
            linestyle="-",
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Center alone",
        ),
        "same": dict(
            color="black",
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=4.2,
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Same surround",
        ),
        "orth": dict(
            color="black",
            linewidth=1.6,
            linestyle="--",
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Orthogonal surround",
        ),
    }


def _prepare_ccss_fig5A_axes(x, curves):
    """
    Round tiny scrape noise away and produce sane shared-looking axes.
    """
    x = np.asarray(x, dtype=float)
    x = np.round(x, 2)

    pos_x = x[x > 0]
    if pos_x.size == 0:
        raise ValueError("Need positive center contrasts for log-scale plotting.")

    x_ticks = np.unique(pos_x)
    xlim = (pos_x.min() * 0.95, pos_x.max() * 1.05)

    ymax = 0.0
    for y in curves:
        y = np.asarray(y, dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size:
            ymax = max(ymax, float(np.max(finite)))

    if ymax <= 1.0:
        y_step = 0.05
    elif ymax <= 2.0:
        y_step = 0.10
    else:
        y_step = 0.20

    ylim = (0.0, _round_up_to_step(1.06 * ymax, y_step))
    return x, x_ticks, xlim, ylim

def compute_ccss_center_contrast_tuning_summary(
    loaded_ccss,
    *,
    surround_index=-1,
    normalize=True,
):
    """
    Build model center-contrast tuning curves from loaded CCSS data.

    Curves:
        center_alone:
            taken from surround contrast index 0

        same_surround:
            taken from surround_index in responses_same

        orthogonal_surround:
            taken from surround_index in responses_orth

    Responses are averaged across neurons and optionally normalized by
    the peak center-alone response.
    """
    present_ids = np.asarray(loaded_ccss["present_ids"], dtype=int)
    center_contrasts = np.asarray(loaded_ccss["center_contrasts"], dtype=float)
    surround_contrasts = np.asarray(loaded_ccss["surround_contrasts"], dtype=float)

    responses_same_by_id = loaded_ccss["responses_same_by_id"]
    responses_orth_by_id = loaded_ccss["responses_orth_by_id"]

    if present_ids.size == 0:
        raise ValueError("No present neurons in loaded CCSS data")

    n_surround = len(surround_contrasts)

    if n_surround == 0:
        raise ValueError("No surround contrasts in loaded CCSS data")

    if surround_index < 0:
        surround_index = n_surround + surround_index

    if surround_index < 0 or surround_index >= n_surround:
        raise ValueError(
            f"Invalid surround_index={surround_index}; "
            f"available range is 0..{n_surround - 1}"
        )

    center_alone_curves = []
    same_curves = []
    orth_curves = []

    for nid in present_ids:
        nid = int(nid)

        same = np.asarray(responses_same_by_id[nid], dtype=float)
        orth = np.asarray(responses_orth_by_id[nid], dtype=float)

        # Shape convention from loader:
        # responses[surround_contrast_idx, center_contrast_idx]
        if same.ndim != 2 or orth.ndim != 2:
            continue

        if same.shape != orth.shape:
            continue

        if same.shape[0] <= surround_index or same.shape[0] == 0:
            continue

        # No-surround condition. Same/orth should be equivalent at surround contrast 0.
        # Use their average to avoid arbitrary branch choice.
        center_alone = 0.5 * (same[0, :] + orth[0, :])

        center_alone_curves.append(center_alone)
        same_curves.append(same[surround_index, :])
        orth_curves.append(orth[surround_index, :])

    if len(center_alone_curves) == 0:
        raise ValueError("No valid CCSS curves found")

    center_alone_curves = np.asarray(center_alone_curves, dtype=float)
    same_curves = np.asarray(same_curves, dtype=float)
    orth_curves = np.asarray(orth_curves, dtype=float)

    mean_center = np.nanmean(center_alone_curves, axis=0)
    mean_same = np.nanmean(same_curves, axis=0)
    mean_orth = np.nanmean(orth_curves, axis=0)

    sem_center = _safe_sem(center_alone_curves, axis=0)
    sem_same = _safe_sem(same_curves, axis=0)
    sem_orth = _safe_sem(orth_curves, axis=0)

    norm = 1.0

    if normalize:
        norm = float(np.nanmax(mean_center))

        if not np.isfinite(norm) or np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize CCSS curves: invalid center-alone peak")

        mean_center = mean_center / norm
        mean_same = mean_same / norm
        mean_orth = mean_orth / norm

        sem_center = sem_center / norm
        sem_same = sem_same / norm
        sem_orth = sem_orth / norm

    return {
        "center_contrasts": center_contrasts,
        "surround_contrasts": surround_contrasts,
        "surround_index": int(surround_index),
        "surround_contrast": float(surround_contrasts[surround_index]),
        "mean_center": mean_center,
        "mean_same": mean_same,
        "mean_orth": mean_orth,
        "sem_center": sem_center,
        "sem_same": sem_same,
        "sem_orth": sem_orth,
        "normalization": norm,
        "n_neurons": int(len(center_alone_curves)),
    }



    


def _style_ccss_axis(ax, center_contrasts, ylabel="Response"):
    LABELSIZE = 20
    TICKSIZE = 15
    TICKLEN = 6
    TICKWIDTH = 1.4

    ax.set_xscale("log")
    ax.set_xticks(center_contrasts)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Center contrast", fontsize=LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=LABELSIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        direction="out",
        length=TICKLEN,
        width=TICKWIDTH,
        colors=BLACK,
        labelsize=TICKSIZE,
    )

def plot_ccss_family_curves_dual(
    center_contrasts,
    surround_contrasts,
    responses_same,
    responses_orth,
    *,
    ylabel="Response",
    sem_same=None,
    sem_orth=None,
    draw_sem=False,
    neuron_id=None,
    title=None,
    save_path=None,
    show=True,
    close=True,
):
    """
    Plot CCSS contrast families for:
        - same-orientation surround
        - orthogonal surround
    """
    center_contrasts = np.asarray(center_contrasts, dtype=float)
    surround_contrasts = np.asarray(surround_contrasts, dtype=float)
    responses_same = np.asarray(responses_same, dtype=float)
    responses_orth = np.asarray(responses_orth, dtype=float)

    expected_shape = (len(surround_contrasts), len(center_contrasts))
    if responses_same.shape != expected_shape:
        raise ValueError(
            f"responses_same shape {responses_same.shape} does not match {expected_shape}"
        )
    if responses_orth.shape != expected_shape:
        raise ValueError(
            f"responses_orth shape {responses_orth.shape} does not match {expected_shape}"
        )

    if draw_sem:
        if sem_same is None or sem_orth is None:
            raise ValueError("When draw_sem=True, both sem_same and sem_orth must be provided.")

        sem_same = np.asarray(sem_same, dtype=float)
        sem_orth = np.asarray(sem_orth, dtype=float)

        if sem_same.shape != expected_shape:
            raise ValueError(
                f"sem_same shape {sem_same.shape} does not match {expected_shape}"
            )
        if sem_orth.shape != expected_shape:
            raise ValueError(
                f"sem_orth shape {sem_orth.shape} does not match {expected_shape}"
            )

    pos_mask = center_contrasts > 0
    if not np.any(pos_mask):
        raise ValueError("Need at least one positive center contrast for log-scale plotting.")

    center_contrasts_pos = center_contrasts[pos_mask]
    responses_same_pos = responses_same[:, pos_mask]
    responses_orth_pos = responses_orth[:, pos_mask]

    if draw_sem:
        sem_same_pos = sem_same[:, pos_mask]
        sem_orth_pos = sem_orth[:, pos_mask]

    FIGSIZE = (13.0, 5.4)
    LEGENDSIZE = 14
    TITLESIZE = 18
    LINEWIDTH = 1.8
    MARKERSIZE = 6.0
    MARKEREDGEWIDTH = 1.2
    ERR_LW = 1.1

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    ax_same, ax_orth = axes

    s_min = np.min(surround_contrasts)
    s_max = np.max(surround_contrasts)

    if np.isclose(s_min, s_max):
        gray_levels = np.full(len(surround_contrasts), 0.40)
    else:
        norm = (surround_contrasts - s_min) / (s_max - s_min)
        gray_levels = 0.75 - 0.55 * norm

    for s_idx, surround_contrast in enumerate(surround_contrasts):
        color = str(gray_levels[s_idx])

        y_same = responses_same_pos[s_idx]
        y_orth = responses_orth_pos[s_idx]

        ax_same.plot(
            center_contrasts_pos,
            y_same,
            marker="o",
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=MARKEREDGEWIDTH,
            label=f"{surround_contrast:.2f}",
        )
        ax_orth.plot(
            center_contrasts_pos,
            y_orth,
            marker="o",
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=MARKEREDGEWIDTH,
            label=f"{surround_contrast:.2f}",
        )

        if draw_sem:
            yerr_same = sem_same_pos[s_idx]
            yerr_orth = sem_orth_pos[s_idx]

            ax_same.errorbar(
                center_contrasts_pos,
                y_same,
                yerr=yerr_same,
                fmt="none",
                ecolor=color,
                elinewidth=ERR_LW,
                capsize=0,
            )
            ax_orth.errorbar(
                center_contrasts_pos,
                y_orth,
                yerr=yerr_orth,
                fmt="none",
                ecolor=color,
                elinewidth=ERR_LW,
                capsize=0,
            )

    if title is None:
        title = (
            f"CCSS contrast family for neuron {neuron_id}"
            if neuron_id is not None else
            "CCSS contrast family"
        )

    _style_ccss_axis(ax_same, center_contrasts_pos, ylabel=ylabel)
    _style_ccss_axis(ax_orth, center_contrasts_pos, ylabel=ylabel)

    ax_orth.legend(
        title="Surround contrast",
        frameon=False,
        fontsize=LEGENDSIZE,
        title_fontsize=LEGENDSIZE,
        handlelength=2.0,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.4,
    )

    fig.subplots_adjust(wspace=0.12)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig, axes

def plot_mean_ccss_family_curves_dual(
    center_contrasts,
    surround_contrasts,
    mean_responses_same,
    mean_responses_orth,
    *,
    ylabel="Response",
    sem_same=None,
    sem_orth=None,
    draw_sem=False,
    n_neurons=None,
    title=None,
    save_path=None,
    show=True,
    close=True,
):
    """
    Plot mean CCSS contrast families across neurons for:
        - same-orientation surround
        - orthogonal surround
    """
    if title is None:
        if n_neurons is None:
            title = "Mean normalised CCSS contrast family"
        else:
            title = f"Mean normalised CCSS contrast family (n={n_neurons})"

    return plot_ccss_family_curves_dual(
        center_contrasts=center_contrasts,
        surround_contrasts=surround_contrasts,
        responses_same=mean_responses_same,
        responses_orth=mean_responses_orth,
        ylabel=ylabel,
        sem_same=sem_same,
        sem_orth=sem_orth,
        draw_sem=draw_sem,
        neuron_id=None,
        title=title,
        save_path=save_path,
        show=show,
        close=close,
    )


def compute_ccss_figure_5_data(
    present_ids,
    center_contrasts,
    surround_contrasts,
    responses_same,
    responses_orth,
    *,
    responses_same_norm_max=None,
    responses_orth_norm_max=None,
    responses_same_norm_center=None,
    responses_orth_norm_center=None,
    scatter_norm_mode="panel",
    nominal_low_center_contrast=0.12,
    high_center_contrast=0.5,
    high_surround_contrast=0.5,
    low_response_fraction=0.20,
):
    """
    Compute Figure 5 data.

    Parameters
    ----------
    scatter_norm_mode : {"panel", "center", "max"}
        How to compute Figure 5B/5C scatter values.

    - "panel":
        article-faithful per-panel normalization
        high -> divide by center-alone at high contrast
        low  -> divide by center-alone at selected low contrast

        - "center":
            read values from responses_*_norm_center
            expected to be normalized by center-alone at high contrast

        - "max":
            read values from responses_*_norm_max
            expected to be normalized by shared per-neuron maximum

    Notes
    -----
    For Figure 5C, article-faithful behavior is only guaranteed with
    scatter_norm_mode="panel".
    """
    present_ids = np.asarray(present_ids, dtype=int)
    center_contrasts = np.asarray(center_contrasts, dtype=float)
    surround_contrasts = np.asarray(surround_contrasts, dtype=float)
    responses_same = np.asarray(responses_same, dtype=float)
    responses_orth = np.asarray(responses_orth, dtype=float)

    if responses_same.shape != responses_orth.shape:
        raise ValueError(
            f"responses_same shape {responses_same.shape} != responses_orth shape {responses_orth.shape}"
        )

    if responses_same.ndim != 3:
        raise ValueError(f"Expected responses shape (N, S, C), got {responses_same.shape}")

    n_neurons, _, _ = responses_same.shape
    if len(present_ids) != n_neurons:
        raise ValueError(
            f"present_ids length {len(present_ids)} does not match number of neurons {n_neurons}"
        )

    s0_idx = int(np.argmin(np.abs(surround_contrasts - 0.0)))
    s_hi_idx = int(np.argmin(np.abs(surround_contrasts - high_surround_contrast)))
    c_hi_idx = int(np.argmin(np.abs(center_contrasts - high_center_contrast)))
    c_nominal_lo_idx = int(np.argmin(np.abs(center_contrasts - nominal_low_center_contrast)))

    # sanity check: surround=0 should collapse same and orth conditions
    center_alone_mismatch = np.max(
        np.abs(responses_same[:, s0_idx, :] - responses_orth[:, s0_idx, :])
    )
    print("Max center-alone mismatch between same and orth conditions:", center_alone_mismatch)

    def _validate_optional_norm(name, arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.shape != responses_same.shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match raw response shape {responses_same.shape}"
            )
        return arr

    responses_same_norm_max = _validate_optional_norm(
        "responses_same_norm_max", responses_same_norm_max
    )
    responses_orth_norm_max = _validate_optional_norm(
        "responses_orth_norm_max", responses_orth_norm_max
    )
    responses_same_norm_center = _validate_optional_norm(
        "responses_same_norm_center", responses_same_norm_center
    )
    responses_orth_norm_center = _validate_optional_norm(
        "responses_orth_norm_center", responses_orth_norm_center
    )

    if scatter_norm_mode not in {"panel", "center", "max"}:
        raise ValueError(
            f"scatter_norm_mode must be one of 'panel', 'center', 'max', got {scatter_norm_mode!r}"
        )

    if scatter_norm_mode == "center":
        if responses_same_norm_center is None or responses_orth_norm_center is None:
            raise ValueError(
                "scatter_norm_mode='center' requires responses_same_norm_center and "
                "responses_orth_norm_center."
            )

    if scatter_norm_mode == "max":
        if responses_same_norm_max is None or responses_orth_norm_max is None:
            raise ValueError(
                "scatter_norm_mode='max' requires responses_same_norm_max and "
                "responses_orth_norm_max."
            )

    def choose_low_contrast_idx(center_alone_curve):
        """
        Paper-faithful per-neuron low contrast selection.
        """
        r_hi = center_alone_curve[c_hi_idx]
        if not np.isfinite(r_hi) or np.isclose(r_hi, 0.0):
            return None

        threshold = low_response_fraction * r_hi

        # first try nominal 0.12
        r_nominal = center_alone_curve[c_nominal_lo_idx]
        if np.isfinite(r_nominal) and r_nominal >= threshold:
            return c_nominal_lo_idx

        # otherwise choose the lowest positive center contrast that reaches threshold
        positive_idxs = np.where(center_contrasts > 0)[0]
        for idx in positive_idxs:
            r = center_alone_curve[idx]
            if np.isfinite(r) and r >= threshold:
                return int(idx)

        return None

    example_idx = None
    best_score = -np.inf

    high_same_norm = []
    high_orth_norm = []
    low_same_norm = []
    low_orth_norm = []

    low_center_contrasts_used = []

    valid_high_ids = []
    valid_low_ids = []
    low_idx_per_neuron = np.full(n_neurons, -1, dtype=int)

    for i, nid in enumerate(present_ids):
        center_alone_curve = responses_same[i, s0_idx, :]
        same_curve = responses_same[i, s_hi_idx, :]
        orth_curve = responses_orth[i, s_hi_idx, :]

        c_lo_idx = choose_low_contrast_idx(center_alone_curve)

        # -------------------------
        # High contrast panel
        # -------------------------
        if scatter_norm_mode == "panel":
            r_center_hi = center_alone_curve[c_hi_idx]
            if np.isfinite(r_center_hi) and not np.isclose(r_center_hi, 0.0):
                rs_hi = same_curve[c_hi_idx] / r_center_hi
                ro_hi = orth_curve[c_hi_idx] / r_center_hi
            else:
                rs_hi = np.nan
                ro_hi = np.nan

        elif scatter_norm_mode == "center":
            rs_hi = responses_same_norm_center[i, s_hi_idx, c_hi_idx]
            ro_hi = responses_orth_norm_center[i, s_hi_idx, c_hi_idx]

        elif scatter_norm_mode == "max":
            rs_hi = responses_same_norm_max[i, s_hi_idx, c_hi_idx]
            ro_hi = responses_orth_norm_max[i, s_hi_idx, c_hi_idx]

        if np.isfinite(rs_hi) and np.isfinite(ro_hi):
            high_same_norm.append(rs_hi)
            high_orth_norm.append(ro_hi)
            valid_high_ids.append(int(nid))

            score = ro_hi - rs_hi
            if score > best_score:
                best_score = score
                example_idx = i

        # -------------------------
        # Low contrast panel
        # -------------------------
        if c_lo_idx is not None:
            low_idx_per_neuron[i] = int(c_lo_idx)

            if scatter_norm_mode == "panel":
                r_center_hi = center_alone_curve[c_hi_idx]
                if np.isfinite(r_center_hi) and not np.isclose(r_center_hi, 0.0):
                    rs_lo = same_curve[c_lo_idx] / r_center_hi
                    ro_lo = orth_curve[c_lo_idx] / r_center_hi
                else:
                    rs_lo = np.nan
                    ro_lo = np.nan

            elif scatter_norm_mode == "center":
                rs_lo = responses_same_norm_center[i, s_hi_idx, c_lo_idx]
                ro_lo = responses_orth_norm_center[i, s_hi_idx, c_lo_idx]

            elif scatter_norm_mode == "max":
                rs_lo = responses_same_norm_max[i, s_hi_idx, c_lo_idx]
                ro_lo = responses_orth_norm_max[i, s_hi_idx, c_lo_idx]

            if np.isfinite(rs_lo) and np.isfinite(ro_lo):
                low_same_norm.append(rs_lo)
                low_orth_norm.append(ro_lo)
                low_center_contrasts_used.append(center_contrasts[c_lo_idx])
                valid_low_ids.append(int(nid))

    example_data = None
    if example_idx is not None:
        example_data = {
            "neuron_id": int(present_ids[example_idx]),
            "center_alone_curve": responses_same[example_idx, s0_idx, :].copy(),
            "same_curve": responses_same[example_idx, s_hi_idx, :].copy(),
            "orth_curve": responses_orth[example_idx, s_hi_idx, :].copy(),
        }

    return {
        "center_contrasts": center_contrasts,
        "surround_contrasts": surround_contrasts,
        "s0_idx": s0_idx,
        "s_hi_idx": s_hi_idx,
        "c_hi_idx": c_hi_idx,
        "c_nominal_lo_idx": c_nominal_lo_idx,
        "nominal_low_center_contrast": float(center_contrasts[c_nominal_lo_idx]),
        "high_center_contrast": float(center_contrasts[c_hi_idx]),
        "high_surround_contrast": float(surround_contrasts[s_hi_idx]),
        "low_response_fraction": float(low_response_fraction),
        "scatter_norm_mode": scatter_norm_mode,
        "example_data": example_data,
        "high_same_norm": np.asarray(high_same_norm, dtype=float),
        "high_orth_norm": np.asarray(high_orth_norm, dtype=float),
        "low_same_norm": np.asarray(low_same_norm, dtype=float),
        "low_orth_norm": np.asarray(low_orth_norm, dtype=float),
        "low_center_contrasts_used": np.asarray(low_center_contrasts_used, dtype=float),
        "valid_high_ids": np.asarray(valid_high_ids, dtype=int),
        "valid_low_ids": np.asarray(valid_low_ids, dtype=int),
        "low_idx_per_neuron": low_idx_per_neuron,
    }


def _plot_ccss_scatter_panel(
    x,
    y,
    *,
    title,
    xlabel,
    ylabel,
    save_path,
    with_marginals=False,
    axis_limits=(0.0, 2.0),
    show_unity_guides=True,
    bins=12,
    show=False,
    close=True,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo, hi = map(float, axis_limits)
    span = hi - lo if hi > lo else 1.0
    eps = 1e-6 * span

    LABELSIZE = 20
    BORDER_LABELSIZE = 16
    TICKSIZE = 13
    TICKLEN = 6
    TICKWIDTH = 1.4
    SCATTER_SIZE_MAIN = 34
    SCATTER_SIZE_SIMPLE = 38
    DIAG_LW = 1.6
    GUIDE_LW = 1.3
    HIST_LW = 1.4

    # shared major ticks for main x/y axes
    tick_step = 0.5
    main_ticks = np.arange(lo, hi + 0.5 * tick_step, tick_step)

    # unified marginal proportion axes
    PROP_MAX = 0.7
    PROP_TICKS = [0.2, 0.4, 0.6]
    PROP_TICKLABELS = [f"{t:.1f}" for t in PROP_TICKS]

    if with_marginals:
        fig = plt.figure(figsize=(8.4, 7.2))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4.3, 1.25],
            height_ratios=[1.0, 4.5],
            wspace=0.18,
            hspace=0.10,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        hist_bins = np.linspace(lo, hi + eps, bins + 1)

        x_weights = np.ones_like(x, dtype=float) / len(x) if len(x) > 0 else None
        y_weights = np.ones_like(y, dtype=float) / len(y) if len(y) > 0 else None

        ax_main.scatter(
            x,
            y,
            s=SCATTER_SIZE_MAIN,
            color=BLACK,
            linewidths=0,
            alpha=0.8,
        )
        ax_main.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=DIAG_LW)
        ax_main.set_xlim(lo, hi + eps)
        ax_main.set_ylim(lo, hi + eps)
        ax_main.set_xlabel(xlabel, fontsize=LABELSIZE)
        ax_main.set_ylabel(ylabel, fontsize=LABELSIZE)
        ax_main.set_xticks(main_ticks)
        ax_main.set_yticks(main_ticks)

        if show_unity_guides:
            ax_main.axvline(1.0, linestyle="-.", color="black", linewidth=GUIDE_LW)
            ax_main.axhline(1.0, linestyle="-.", color="black", linewidth=GUIDE_LW)

        _, _, p_top = ax_top.hist(
            x,
            bins=hist_bins,
            weights=x_weights,
            edgecolor=BLACK,
            facecolor=LIGHT,
            linewidth=HIST_LW,
        )
        for p in p_top:
            p.set_clip_on(False)

        ax_top.set_xlim(lo, hi + eps)
        ax_top.set_ylim(0.0, PROP_MAX)
        ax_top.set_yticks(PROP_TICKS)
        ax_top.set_yticklabels(PROP_TICKLABELS)
        ax_top.set_ylabel("Proportion", fontsize=BORDER_LABELSIZE)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        _, _, p_right = ax_right.hist(
            y,
            bins=hist_bins,
            weights=y_weights,
            orientation="horizontal",
            edgecolor=BLACK,
            facecolor=LIGHT,
            linewidth=HIST_LW,
        )
        for p in p_right:
            p.set_clip_on(False)

        ax_right.set_ylim(lo, hi + eps)
        ax_right.set_xlim(0.0, PROP_MAX)
        ax_right.set_xticks(PROP_TICKS)
        ax_right.set_xticklabels(PROP_TICKLABELS)
        ax_right.set_xlabel("Proportion", fontsize=BORDER_LABELSIZE)
        ax_right.tick_params(axis="y", labelleft=False, pad=8)
        ax_right.tick_params(axis="x", labelsize=TICKSIZE, pad=4, length=TICKLEN, width=TICKWIDTH)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["left"].set_position(("outward", 12))

        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        ax_main.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)

        fig.subplots_adjust(wspace=0.18, hspace=0.10)

    else:
        fig, ax_main = plt.subplots(figsize=(6.6, 5.8))
        ax_main.scatter(x, y, s=SCATTER_SIZE_SIMPLE, alpha=0.8, color=BLACK, linewidths=0)
        ax_main.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=DIAG_LW)
        ax_main.set_xlim(lo, hi + eps)
        ax_main.set_ylim(lo, hi + eps)
        ax_main.set_xlabel(xlabel, fontsize=LABELSIZE)
        ax_main.set_ylabel(ylabel, fontsize=LABELSIZE)
        ax_main.set_xticks(main_ticks)
        ax_main.set_yticks(main_ticks)

        if show_unity_guides:
            ax_main.axvline(1.0, linestyle="-.", color="black", linewidth=GUIDE_LW)
            ax_main.axhline(1.0, linestyle="-.", color="black", linewidth=GUIDE_LW)

        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        ax_main.tick_params(axis="both", which="both", labelsize=TICKSIZE, length=TICKLEN, width=TICKWIDTH)
        fig.tight_layout()

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return save_path


def compute_ccss_figure5A_population_curves(
    center_contrasts,
    surround_contrasts,
    responses_same,
    responses_orth,
    *,
    high_center_contrast=0.5,
    high_surround_contrast=0.5,
):
    """
    Compute normalized mean and SEM for Figure-5A-style curves.
    Here the plots show raw responses, not normalised to the center-alone response at high contrast, 
    = so this function does not perform any normalization.

    Parameters
    ----------
    responses_same : array, shape (N, S, C)
    responses_orth : array, shape (N, S, C)

    Returns
    -------
    dict with mean/sem curves of shape (C,)
    """
    center_contrasts = np.asarray(center_contrasts, dtype=float)
    surround_contrasts = np.asarray(surround_contrasts, dtype=float)
    responses_same = np.asarray(responses_same, dtype=float)
    responses_orth = np.asarray(responses_orth, dtype=float)

    if responses_same.shape != responses_orth.shape:
        raise ValueError("responses_same and responses_orth must have the same shape.")
    if responses_same.ndim != 3:
        raise ValueError("responses_same and responses_orth must have shape (N, S, C).")

    n_neurons, _, n_center = responses_same.shape

    s0_idx = int(np.argmin(np.abs(surround_contrasts - 0.0)))
    s_hi_idx = int(np.argmin(np.abs(surround_contrasts - high_surround_contrast)))
    c_hi_idx = int(np.argmin(np.abs(center_contrasts - high_center_contrast)))

    center_curves = np.zeros((n_neurons, n_center), dtype=float)
    same_curves = np.zeros((n_neurons, n_center), dtype=float)
    orth_curves = np.zeros((n_neurons, n_center), dtype=float)

    for i in range(n_neurons):
        center_curve = responses_same[i, s0_idx].astype(float, copy=True)
        same_curve = responses_same[i, s_hi_idx].astype(float, copy=True)
        orth_curve = responses_orth[i, s_hi_idx].astype(float, copy=True)

        denom = center_curve[c_hi_idx]
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0

        center_curves[i] = center_curve / denom
        same_curves[i] = same_curve / denom
        orth_curves[i] = orth_curve / denom

    def _sem(arr):
        if arr.shape[0] < 2:
            return np.zeros(arr.shape[1], dtype=float)
        return np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

    return {
        "center_curves": center_curves,
        "same_curves": same_curves,
        "orth_curves": orth_curves,
        "mean_center": np.mean(center_curves, axis=0),
        "mean_same": np.mean(same_curves, axis=0),
        "mean_orth": np.mean(orth_curves, axis=0),
        "sem_center": _sem(center_curves),
        "sem_same": _sem(same_curves),
        "sem_orth": _sem(orth_curves),
        "high_idx": c_hi_idx,
        "s0_idx": s0_idx,
        "s_hi_idx": s_hi_idx,
        "n_neurons": n_neurons,
    }



def plot_ccss_figure5A(
    center_contrasts,
    surround_contrasts,
    responses_same,
    responses_orth,
    *,
    neuron_id=None,
    high_center_contrast=0.5,
    nominal_low_center_contrast=0.12,
    high_surround_contrast=0.5,
    low_fraction=0.20,
    sem_center=None,
    sem_same=None,
    sem_orth=None,
    draw_sem=False,
    print_values=False,
    title=None,
    ylabel="Response magnitude",
    legend=True,
    save_path=None,
    show=False,
    close=True,
):
    """
    Figure-5A-style CCSS plot.
    """
    center_contrasts = np.asarray(center_contrasts, dtype=float)
    surround_contrasts = np.asarray(surround_contrasts, dtype=float)

    responses_same = np.asarray(responses_same, dtype=float)
    responses_orth = np.asarray(responses_orth, dtype=float)

    expected_shape = (len(surround_contrasts), len(center_contrasts))
    if responses_same.shape != expected_shape:
        raise ValueError(
            f"responses_same shape {responses_same.shape} does not match {expected_shape}"
        )
    if responses_orth.shape != expected_shape:
        raise ValueError(
            f"responses_orth shape {responses_orth.shape} does not match {expected_shape}"
        )

    if draw_sem:
        if sem_center is None or sem_same is None or sem_orth is None:
            raise ValueError(
                "When draw_sem=True, sem_center, sem_same, and sem_orth must all be provided."
            )

        sem_center = np.asarray(sem_center, dtype=float)
        sem_same = np.asarray(sem_same, dtype=float)
        sem_orth = np.asarray(sem_orth, dtype=float)

        for name, arr in [
            ("sem_center", sem_center),
            ("sem_same", sem_same),
            ("sem_orth", sem_orth),
        ]:
            if arr.shape != center_contrasts.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match {(len(center_contrasts),)}"
                )

    s0_idx = int(np.argmin(np.abs(surround_contrasts - 0.0)))
    s_hi_idx = int(np.argmin(np.abs(surround_contrasts - high_surround_contrast)))
    c_hi_idx = int(np.argmin(np.abs(center_contrasts - high_center_contrast)))
    c_nom_lo_idx = int(np.argmin(np.abs(center_contrasts - nominal_low_center_contrast)))

    center_alone = responses_same[s0_idx]
    same_curve = responses_same[s_hi_idx]
    orth_curve = responses_orth[s_hi_idx]

    if draw_sem:
        center_sem = sem_center
        same_sem = sem_same
        orth_sem = sem_orth

    r_hi = center_alone[c_hi_idx]

    threshold = low_fraction * r_hi
    if center_alone[c_nom_lo_idx] >= threshold:
        c_lo_idx = c_nom_lo_idx
    else:
        c_lo_idx = None
        for idx in np.where(center_contrasts > 0)[0]:
            if center_alone[idx] >= threshold:
                c_lo_idx = idx
                break

    r_hi_norm_same = same_curve[c_hi_idx] / r_hi if r_hi > 0 else np.nan
    r_hi_norm_orth = orth_curve[c_hi_idx] / r_hi if r_hi > 0 else np.nan

    r_lo_norm_same = np.nan
    r_lo_norm_orth = np.nan
    if c_lo_idx is not None:
        r_lo = center_alone[c_lo_idx]
        if r_lo > 0:
            r_lo_norm_same = same_curve[c_lo_idx] / r_lo
            r_lo_norm_orth = orth_curve[c_lo_idx] / r_lo

    if print_values:
        label = f"Neuron {neuron_id}" if neuron_id is not None else "Population mean"
        print(label)
        print(
            f"  High contrast ({center_contrasts[c_hi_idx]:.3f}): "
            f"same={r_hi_norm_same:.3f}, orth={r_hi_norm_orth:.3f}"
        )
        if c_lo_idx is None:
            print("  Low contrast: not found")
        else:
            print(
                f"  Low contrast ({center_contrasts[c_lo_idx]:.3f}): "
                f"same={r_lo_norm_same:.3f}, orth={r_lo_norm_orth:.3f}"
            )

    pos_mask = center_contrasts > 0
    x_raw = center_contrasts[pos_mask]

    FIGSIZE = (6.8, 5.6)
    LEGENDSIZE = 17
    LINEWIDTH_ERR = 1.1
    TICKLEN = 6
    TICKWIDTH = 1.4

    fig, ax = plt.subplots(figsize=FIGSIZE)

    styles = _get_ccss_fig5A_styles()
    x_plot, x_ticks, xlim, ylim = _prepare_ccss_fig5A_axes(
        x_raw,
        [
            center_alone[pos_mask],
            same_curve[pos_mask],
            orth_curve[pos_mask],
        ],
    )

    # inflate style locally, keep logic intact
    styles = {
        "center": {**styles["center"], "linewidth": 1.8, "markersize": 6.0, "markeredgewidth": 1.2},
        "same": {**styles["same"], "linewidth": 2.6, "markersize": 6.0, "markeredgewidth": 1.2},
        "orth": {**styles["orth"], "linewidth": 2.0, "markersize": 6.0, "markeredgewidth": 1.2},
    }

    ax.plot(x_plot, center_alone[pos_mask], **styles["center"])
    ax.plot(x_plot, same_curve[pos_mask], **styles["same"])
    ax.plot(x_plot, orth_curve[pos_mask], **styles["orth"])

    if draw_sem:
        ax.errorbar(
            x_plot, center_alone[pos_mask], yerr=center_sem[pos_mask],
            fmt="none", ecolor="black", elinewidth=LINEWIDTH_ERR, capsize=0
        )
        ax.errorbar(
            x_plot, same_curve[pos_mask], yerr=same_sem[pos_mask],
            fmt="none", ecolor="black", elinewidth=LINEWIDTH_ERR, capsize=0
        )
        ax.errorbar(
            x_plot, orth_curve[pos_mask], yerr=orth_sem[pos_mask],
            fmt="none", ecolor="black", elinewidth=LINEWIDTH_ERR, capsize=0
        )

    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Center contrast", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    if legend:
        ax.legend(
            frameon=False,
            loc="best",
            fontsize=LEGENDSIZE,
            handlelength=2.0,
            handletextpad=0.5,
            labelspacing=0.4,
            borderpad=0.2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=TICKLEN, width=TICKWIDTH, labelsize=15)

    fig.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return {
        "high_same": r_hi_norm_same,
        "high_orth": r_hi_norm_orth,
        "low_same": r_lo_norm_same,
        "low_orth": r_lo_norm_orth,
        "low_idx": c_lo_idx,
        "high_idx": c_hi_idx,
    }



def _get_scatter_limits(x, y):
    vals = np.concatenate([
        x[np.isfinite(x)] if x.size else np.array([]),
        y[np.isfinite(y)] if y.size else np.array([]),
    ])
    if vals.size == 0:
        return 0.0, 1.2
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return 0.0, 1.2
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_ccss_figure_5B_high_contrast_scatter(
    fig5_data,
    *,
    out_path,
    filename="ccss_figure_5B_high_contrast.png",
    with_marginals=False,
    show=False,
    close=True,
):
    """
    Figure 5B:
    Population scatter at high center contrast.
    x = orthogonal surround / center alone
    y = same-orientation surround / center alone
    """
    ensure_dir(out_path)

    high_same = np.asarray(fig5_data["high_same_norm"], dtype=float)
    high_orth = np.asarray(fig5_data["high_orth_norm"], dtype=float)
    c_hi = float(fig5_data["high_center_contrast"])

    save_path = os.path.join(out_path, filename)

    return _plot_ccss_scatter_panel(
        high_orth,
        high_same,
        title=f"Figure 5B. High center contrast",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        show=show,
        close=close,
    )


def plot_ccss_figure_5C_low_contrast_scatter(
    fig5_data,
    *,
    out_path,
    filename="ccss_figure_5C_low_contrast.png",
    with_marginals=False,
    show=False,
    close=True,
):
    ensure_dir(out_path)

    low_same = np.asarray(fig5_data["low_same_norm"], dtype=float)
    low_orth = np.asarray(fig5_data["low_orth_norm"], dtype=float)

    save_path = os.path.join(out_path, filename)

    path = _plot_ccss_scatter_panel(
        low_orth,
        low_same,
        title="Figure 5C. Low center contrast",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        show=show,
        close=close,
    )

    return path


def prepare_scraped_ccss_figure_5A(figure_5A_data):
    data = figure_5A_data

    center_contrast = np.asarray(data["center_contrast"], dtype=float)
    center_alone = np.asarray(data["center_alone_response"], dtype=float)
    orth = np.asarray(data["orthogonal_surround_response"], dtype=float)
    same = np.asarray(data["same_surround_response"], dtype=float)

    expected_shape = center_contrast.shape
    for name, arr in [
        ("center_alone_response", center_alone),
        ("orthogonal_surround_response", orth),
        ("same_surround_response", same),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")

    order = np.argsort(center_contrast)

    return {
        "source": data.get("source", "scraped_article"),
        "panel": data.get("panel", "5A"),
        "center_contrast": np.round(center_contrast[order], 2),
        "center_alone_response": np.round(center_alone[order], 3),
        "orthogonal_surround_response": np.round(orth[order], 3),
        "same_surround_response": np.round(same[order], 3),
    }


def prepare_scraped_ccss_scatter(fig5_scatter_data, *, clamp_x_to_zero=False, clamp_y_to_zero=False):
    data = fig5_scatter_data

    x = np.asarray(data["orthogonal_surround_response"], dtype=float)
    y = np.asarray(data["same_surround_response"], dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} != y shape {y.shape}")

    if clamp_x_to_zero:
        x = np.maximum(x, 0.0)
    if clamp_y_to_zero:
        y = np.maximum(y, 0.0)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    return {
        "source": data.get("source", "scraped_article"),
        "panel": data.get("panel", None),
        "n_cells_reported": int(data.get("n_cells_reported", len(x))),
        "n_points_scraped": int(len(x)),
        "orthogonal_surround_response": x,
        "same_surround_response": y,
    }



def plot_scraped_ccss_figure_5A(
    fig5A_scraped_prep,
    *,
    title="",
    ylabel="Response magnitude",
    save_path=None,
    show=False,
    close=True,
):
    x = np.asarray(fig5A_scraped_prep["center_contrast"], dtype=float)
    center_alone = np.asarray(fig5A_scraped_prep["center_alone_response"], dtype=float)
    orth = np.asarray(fig5A_scraped_prep["orthogonal_surround_response"], dtype=float)
    same = np.asarray(fig5A_scraped_prep["same_surround_response"], dtype=float)

    x = np.round(x, 2)
    center_alone = np.round(center_alone, 3)
    orth = np.round(orth, 3)
    same = np.round(same, 3)

    FIGSIZE = (6.8, 5.6)
    LEGENDSIZE = 17
    TICKLEN = 6
    TICKWIDTH = 1.4

    fig, ax = plt.subplots(figsize=FIGSIZE)

    styles = _get_ccss_fig5A_styles()
    styles = {
        "center": {**styles["center"], "linewidth": 1.8, "markersize": 6.0, "markeredgewidth": 1.2},
        "same": {**styles["same"], "linewidth": 2.6, "markersize": 6.0, "markeredgewidth": 1.2},
        "orth": {**styles["orth"], "linewidth": 2.0, "markersize": 6.0, "markeredgewidth": 1.2},
    }

    x_plot, x_ticks, xlim, ylim = _prepare_ccss_fig5A_axes(
        x,
        [center_alone, same, orth],
    )

    ax.plot(x_plot, center_alone, **styles["center"])
    ax.plot(x_plot, same, **styles["same"])
    ax.plot(x_plot, orth, **styles["orth"])

    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Center contrast", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.legend(
        frameon=False,
        loc="best",
        fontsize=LEGENDSIZE,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=TICKLEN, width=TICKWIDTH, labelsize=15)

    fig.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return save_path




def plot_scraped_ccss_figure_5B(
    fig5B_scraped_prep,
    *,
    out_path,
    filename="ccss_figure_5B_scraped.png",
    with_marginals=False,
    show=False,
    close=True,
):
    ensure_dir(out_path)

    x = np.asarray(fig5B_scraped_prep["orthogonal_surround_response"], dtype=float)
    y = np.asarray(fig5B_scraped_prep["same_surround_response"], dtype=float)

    save_path = os.path.join(out_path, filename)

    return _plot_ccss_scatter_panel(
        x,
        y,
        title="Figure 5B. High center contrast (scraped)",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        show=show,
        close=close,
    )


def plot_scraped_ccss_figure_5C(
    fig5C_scraped_prep,
    *,
    out_path,
    filename="ccss_figure_5C_scraped.png",
    with_marginals=False,
    show=False,
    close=True,
):
    ensure_dir(out_path)

    x = np.asarray(fig5C_scraped_prep["orthogonal_surround_response"], dtype=float)
    y = np.asarray(fig5C_scraped_prep["same_surround_response"], dtype=float)

    save_path = os.path.join(out_path, filename)

    return _plot_ccss_scatter_panel(
        x,
        y,
        title="Figure 5C. Low center contrast (scraped)",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        show=show,
        close=close,
    )


def run_scraped_ccss_figure_5(
    figure_5A_data,
    figure_5B_data,
    figure_5C_data,
    out_path,
    with_marginals=True,
):
    """
    Plot scraped article Figure 5 A-C.
    """
    ensure_dir(out_path)

    fig5A_prep = prepare_scraped_ccss_figure_5A(figure_5A_data)
    fig5B_prep = prepare_scraped_ccss_scatter(figure_5B_data)
    fig5C_prep = prepare_scraped_ccss_scatter(
        figure_5C_data,
        clamp_x_to_zero=True,   # scrape noise
    )

    fig5A_path = plot_scraped_ccss_figure_5A(
        fig5A_prep,
        title="Figure 5A. Scraped article curve",
        save_path=os.path.join(out_path, "ccss_figure_5A_scraped.png"),
        show=False,
        close=True,
    )

    fig5B_path = plot_scraped_ccss_figure_5B(
        fig5B_prep,
        out_path=out_path,
        filename="ccss_figure_5B_scraped.png",
        with_marginals=False,
        show=False,
        close=True,
    )

    fig5B_marginal_path = plot_scraped_ccss_figure_5B(
        fig5B_prep,
        out_path=out_path,
        filename="ccss_figure_5B_scraped_with_marginals.png",
        with_marginals=with_marginals,
        show=False,
        close=True,
    )

    fig5C_path = plot_scraped_ccss_figure_5C(
        fig5C_prep,
        out_path=out_path,
        filename="ccss_figure_5C_scraped.png",
        with_marginals=False,
        show=False,
        close=True,
    )

    fig5C_marginal_path = plot_scraped_ccss_figure_5C(
        fig5C_prep,
        out_path=out_path,
        filename="ccss_figure_5C_scraped_with_marginals.png",
        with_marginals=with_marginals,
        show=False,
        close=True,
    )

    return {
        "figure_5A_prep": fig5A_prep,
        "figure_5B_prep": fig5B_prep,
        "figure_5C_prep": fig5C_prep,
        "plot_paths": {
            "figure_5A": fig5A_path,
            "figure_5B": fig5B_path,
            "figure_5B_with_marginals": fig5B_marginal_path,
            "figure_5C": fig5C_path,
            "figure_5C_with_marginals": fig5C_marginal_path,
        },
    }

def plot_ccss_figure5A_model_article_overlay(
    fig5A_mean_data,
    figure_5A_data,
    center_contrasts,
    *,
    title="",
    ylabel="Relative response",
    save_path=None,
    show=False,
    close=True,
):
    """
    Joint Figure 5A:
    experimental scraped CCSS center-contrast tuning + model population mean.

    Experimental: ARTICLE_COLOR
    Model: MODEL_COLOR

    Both are normalized by their own center-alone high-contrast response.
    """
    if figure_5A_data is None:
        raise ValueError("Missing scraped Figure 5A data")

    # ------------------------------------------------------------------
    # Article
    # ------------------------------------------------------------------
    article = prepare_scraped_ccss_figure_5A(figure_5A_data)

    article_x = np.asarray(article["center_contrast"], dtype=float)
    article_center = np.asarray(article["center_alone_response"], dtype=float)
    article_orth = np.asarray(article["orthogonal_surround_response"], dtype=float)
    article_same = np.asarray(article["same_surround_response"], dtype=float)

    article_pos_mask = article_x > 0

    article_x = article_x[article_pos_mask]
    article_center = article_center[article_pos_mask]
    article_orth = article_orth[article_pos_mask]
    article_same = article_same[article_pos_mask]

    article_norm = float(np.nanmax(article_center))
    if not np.isfinite(article_norm) or np.isclose(article_norm, 0.0):
        raise ValueError("Cannot normalize scraped Figure 5A: invalid center-alone peak")

    article_center = article_center / article_norm
    article_orth = article_orth / article_norm
    article_same = article_same / article_norm

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_x = np.asarray(center_contrasts, dtype=float)
    model_center = np.asarray(fig5A_mean_data["mean_center"], dtype=float)
    model_same = np.asarray(fig5A_mean_data["mean_same"], dtype=float)
    model_orth = np.asarray(fig5A_mean_data["mean_orth"], dtype=float)

    model_sem_center = np.asarray(fig5A_mean_data["sem_center"], dtype=float)
    model_sem_same = np.asarray(fig5A_mean_data["sem_same"], dtype=float)
    model_sem_orth = np.asarray(fig5A_mean_data["sem_orth"], dtype=float)

    model_pos_mask = model_x > 0

    model_x = model_x[model_pos_mask]
    model_center = model_center[model_pos_mask]
    model_same = model_same[model_pos_mask]
    model_orth = model_orth[model_pos_mask]

    model_sem_center = model_sem_center[model_pos_mask]
    model_sem_same = model_sem_same[model_pos_mask]
    model_sem_orth = model_sem_orth[model_pos_mask]

    # compute_ccss_figure5A_population_curves already normalizes by
    # center-alone high contrast, but normalize again defensively.
    model_norm = float(np.nanmax(model_center))
    if not np.isfinite(model_norm) or np.isclose(model_norm, 0.0):
        raise ValueError("Cannot normalize model Figure 5A: invalid center-alone peak")

    model_center = model_center / model_norm
    model_same = model_same / model_norm
    model_orth = model_orth / model_norm

    model_sem_center = model_sem_center / model_norm
    model_sem_same = model_sem_same / model_norm
    model_sem_orth = model_sem_orth / model_norm

    # ------------------------------------------------------------------
    # Axis prep
    # ------------------------------------------------------------------
    x_all = np.concatenate([article_x, model_x])
    x_all = np.round(x_all, 2)

    x_ticks = np.unique(x_all)
    xlim = (np.min(x_ticks) * 0.95, np.max(x_ticks) * 1.05)

    y_max = np.nanmax([
        np.nanmax(article_center),
        np.nanmax(article_orth),
        np.nanmax(article_same),
        np.nanmax(model_center),
        np.nanmax(model_orth),
        np.nanmax(model_same),
    ])

    y_min = np.nanmin([
        np.nanmin(article_center),
        np.nanmin(article_orth),
        np.nanmin(article_same),
        np.nanmin(model_center),
        np.nanmin(model_orth),
        np.nanmin(model_same),
    ])

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    FIGSIZE = (6.8, 5.4)
    LABELSIZE = 14
    TICKSIZE = 10
    LEGENDSIZE = 10

    TICKLEN = 5
    TICKWIDTH = 1.3

    ERR_LW = 0.8

    CENTER_LS = "-"
    ORTH_LS = "--"
    SAME_LS = (0, (1.0, 1.8))

    ARTICLE_LW = 2.0
    MODEL_LW = 1.8

    ARTICLE_MS = 4.0
    MODEL_MS = 4.0

    ARTICLE_ALPHA = 0.95
    MODEL_ALPHA = 0.85

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # ------------------------------------------------------------------
    # Experimental curves: black, square markers
    # ------------------------------------------------------------------
    ax.plot(
        article_x,
        article_center,
        color=ARTICLE_COLOR,
        linestyle=CENTER_LS,
        linewidth=ARTICLE_LW,
        marker="s",
        markersize=ARTICLE_MS,
        markerfacecolor=ARTICLE_COLOR,
        markeredgecolor=ARTICLE_COLOR,
        markeredgewidth=0.8,
        alpha=ARTICLE_ALPHA,
        label="_nolegend_",
        zorder=4,
    )

    ax.plot(
        article_x,
        article_orth,
        color=ARTICLE_COLOR,
        linestyle=ORTH_LS,
        linewidth=ARTICLE_LW,
        marker="s",
        markersize=ARTICLE_MS,
        markerfacecolor=ARTICLE_COLOR,
        markeredgecolor=ARTICLE_COLOR,
        markeredgewidth=0.8,
        alpha=ARTICLE_ALPHA,
        label="_nolegend_",
        zorder=4,
    )

    ax.plot(
        article_x,
        article_same,
        color=ARTICLE_COLOR,
        linestyle=SAME_LS,
        linewidth=ARTICLE_LW,
        marker="s",
        markersize=ARTICLE_MS,
        markerfacecolor=ARTICLE_COLOR,
        markeredgecolor=ARTICLE_COLOR,
        markeredgewidth=0.8,
        alpha=ARTICLE_ALPHA,
        label="_nolegend_",
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Model curves: blue, circle markers
    # ------------------------------------------------------------------
    ax.plot(
        model_x,
        model_center,
        color=MODEL_COLOR,
        linestyle=CENTER_LS,
        linewidth=MODEL_LW,
        marker="o",
        markersize=MODEL_MS,
        markerfacecolor=MODEL_COLOR,
        markeredgecolor=MODEL_COLOR,
        markeredgewidth=0.7,
        alpha=MODEL_ALPHA,
        label="_nolegend_",
        zorder=3,
    )

    ax.plot(
        model_x,
        model_orth,
        color=MODEL_COLOR,
        linestyle=ORTH_LS,
        linewidth=MODEL_LW,
        marker="o",
        markersize=MODEL_MS,
        markerfacecolor=MODEL_COLOR,
        markeredgecolor=MODEL_COLOR,
        markeredgewidth=0.7,
        alpha=MODEL_ALPHA,
        label="_nolegend_",
        zorder=3,
    )

    ax.plot(
        model_x,
        model_same,
        color=MODEL_COLOR,
        linestyle=SAME_LS,
        linewidth=MODEL_LW,
        marker="o",
        markersize=MODEL_MS,
        markerfacecolor=MODEL_COLOR,
        markeredgecolor=MODEL_COLOR,
        markeredgewidth=0.7,
        alpha=MODEL_ALPHA,
        label="_nolegend_",
        zorder=3,
    )

    ax.errorbar(
        model_x,
        model_center,
        yerr=model_sem_center,
        fmt="none",
        ecolor=MODEL_COLOR,
        elinewidth=ERR_LW,
        capsize=0,
        alpha=0.55,
        zorder=1,
    )

    ax.errorbar(
        model_x,
        model_orth,
        yerr=model_sem_orth,
        fmt="none",
        ecolor=MODEL_COLOR,
        elinewidth=ERR_LW,
        capsize=0,
        alpha=0.55,
        zorder=1,
    )

    ax.errorbar(
        model_x,
        model_same,
        yerr=model_sem_same,
        fmt="none",
        ecolor=MODEL_COLOR,
        elinewidth=ERR_LW,
        capsize=0,
        alpha=0.55,
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(*xlim)

    ax.set_ylim(
        min(0.0, y_min - 0.05),
        max(1.1, 1.08 * y_max),
    )

    ax.set_xlabel("Center contrast", fontsize=LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=LABELSIZE)

    source_handles = [
        Line2D(
            [], [],
            color=ARTICLE_COLOR,
            linestyle="-",
            marker="s",
            markerfacecolor=ARTICLE_COLOR,
            markeredgecolor=ARTICLE_COLOR,
            linewidth=ARTICLE_LW,
            markersize=ARTICLE_MS,
            label="Experimental",
        ),
        Line2D(
            [], [],
            color=MODEL_COLOR,
            linestyle="-",
            marker="o",
            markerfacecolor=MODEL_COLOR,
            markeredgecolor=MODEL_COLOR,
            linewidth=MODEL_LW,
            markersize=MODEL_MS,
            label="Model",
        ),
    ]

    condition_handles = [
        Line2D([], [], color="0.25", linestyle=CENTER_LS, linewidth=1.8, label="Center"),
        Line2D([], [], color="0.25", linestyle=ORTH_LS, linewidth=1.8, label="Orthogonal"),
        Line2D([], [], color="0.25", linestyle=SAME_LS, linewidth=1.8, label="Same"),
    ]

    legend_source = ax.legend(
        handles=source_handles,
        loc="upper left",
        frameon=False,
        fontsize=LEGENDSIZE,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.2,
    )

    ax.add_artist(legend_source)

    ax.legend(
        handles=condition_handles,
        loc="upper left",
        bbox_to_anchor=(0.32, 1.0),
        frameon=False,
        fontsize=LEGENDSIZE,
        handlelength=2.2,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=TICKLEN,
        width=TICKWIDTH,
        labelsize=TICKSIZE,
    )

    fig.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return save_path


def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _get_joint_axis_limits(*arrays, default=(0.0, 2.0), hard_zero=True):
    vals = []

    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)

    if len(vals) == 0:
        return default

    vals = np.concatenate(vals)

    lo = float(np.min(vals))
    hi = float(np.max(vals))

    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return default

    pad = 0.07 * (hi - lo)

    if hard_zero:
        lo = min(0.0, lo - pad)
    else:
        lo = lo - pad

    hi = hi + pad

    return lo, hi

def _plot_joint_marginal_histograms(
    ax_top,
    ax_right,
    *,
    model_x,
    model_y,
    article_x,
    article_y,
    hist_bins,
    axis_limits,
):
    """
    Joint marginal histograms aligned to the certified scatter layout.

    top:
        x-distribution

    right:
        y-distribution

    Histograms are plotted as proportions, side-by-side:
        article/experiment = black
        model = MODEL_COLOR
    """
    lo, hi = axis_limits

    BORDER_LABELSIZE = 16
    TICKSIZE = 13
    TICKLEN = 6
    TICKWIDTH = 1.4
    HIST_LW = 0.8

    ARTICLE_COLOR = BLACK

    bin_edges = np.asarray(hist_bins, dtype=float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])

    # same logic as the certified Fig. 3B plot:
    # slightly narrower bars, side-by-side
    bar_width = 0.38 * bin_width
    bar_offset = 0.22 * bin_width

    def _prop_hist(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        counts, _ = np.histogram(values, bins=bin_edges)

        if values.size == 0:
            return counts.astype(float)

        return counts.astype(float) / float(values.size)

    model_x_hist = _prop_hist(model_x)
    article_x_hist = _prop_hist(article_x)

    model_y_hist = _prop_hist(model_y)
    article_y_hist = _prop_hist(article_y)

    max_prop = max(
        np.max(model_x_hist) if model_x_hist.size else 0.0,
        np.max(article_x_hist) if article_x_hist.size else 0.0,
        np.max(model_y_hist) if model_y_hist.size else 0.0,
        np.max(article_y_hist) if article_y_hist.size else 0.0,
    )

    PROP_MAX = max(0.1, 1.15 * max_prop)

    if PROP_MAX <= 0.3:
        PROP_TICKS = [0.1, 0.2, 0.3]
    elif PROP_MAX <= 0.5:
        PROP_TICKS = [0.2, 0.4]
    else:
        PROP_TICKS = [0.2, 0.4, 0.6]

    PROP_TICKS = [t for t in PROP_TICKS if t <= PROP_MAX]
    PROP_TICKLABELS = [f"{t:.1f}" for t in PROP_TICKS]

    # --------------------------------------------------
    # Top marginal: x-distribution
    # --------------------------------------------------
    ax_top.bar(
        bin_centers - bar_offset,
        article_x_hist,
        width=bar_width,
        color=ARTICLE_COLOR,
        edgecolor=ARTICLE_COLOR,
        linewidth=HIST_LW,
        alpha=0.85,
        align="center",
        label="Experiment",
    )

    ax_top.bar(
        bin_centers + bar_offset,
        model_x_hist,
        width=bar_width,
        color=MODEL_COLOR,
        edgecolor=MODEL_COLOR,
        linewidth=HIST_LW,
        alpha=0.85,
        align="center",
        label="Model",
    )

    ax_top.set_xlim(lo, hi)
    ax_top.set_ylim(0.0, PROP_MAX)
    ax_top.set_yticks(PROP_TICKS)
    ax_top.set_yticklabels(PROP_TICKLABELS)
    ax_top.set_ylabel("Prop.", fontsize=BORDER_LABELSIZE)

    ax_top.tick_params(
        axis="x",
        labelbottom=False,
        length=TICKLEN,
        width=TICKWIDTH,
    )
    ax_top.tick_params(
        axis="y",
        labelsize=TICKSIZE,
        length=TICKLEN,
        width=TICKWIDTH,
    )

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --------------------------------------------------
    # Right marginal: y-distribution
    # --------------------------------------------------
    ax_right.barh(
        bin_centers - bar_offset,
        article_y_hist,
        height=bar_width,
        color=ARTICLE_COLOR,
        edgecolor=ARTICLE_COLOR,
        linewidth=HIST_LW,
        alpha=0.85,
        align="center",
        label="Experiment",
    )

    ax_right.barh(
        bin_centers + bar_offset,
        model_y_hist,
        height=bar_width,
        color=MODEL_COLOR,
        edgecolor=MODEL_COLOR,
        linewidth=HIST_LW,
        alpha=0.85,
        align="center",
        label="Model",
    )

    ax_right.set_ylim(lo, hi)
    ax_right.set_xlim(0.0, PROP_MAX)
    ax_right.set_xticks(PROP_TICKS)
    ax_right.set_xticklabels(PROP_TICKLABELS)
    ax_right.set_xlabel("Prop.", fontsize=BORDER_LABELSIZE)

    ax_right.tick_params(
        axis="y",
        labelleft=False,
        length=TICKLEN,
        width=TICKWIDTH,
    )
    ax_right.tick_params(
        axis="x",
        labelsize=TICKSIZE,
        pad=4,
        length=TICKLEN,
        width=TICKWIDTH,
    )

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

def _plot_ccss_model_article_scatter_panel(
    *,
    model_x,
    model_y,
    article_x,
    article_y,
    title,
    xlabel,
    ylabel,
    save_path,
    with_marginals=False,
    axis_limits=(0.0, 2.0),
    show_unity_guides=True,
    bins=12,
    show=False,
    close=True,
):
    """
    Shared CCSS scatter plotter.

    Keeps the certified Figure 5 scatter geometry:
        x = orthogonal surround / center alone
        y = same-orientation surround / center alone

    Adds:
        model + article overlay
        optional marginal boxplots
    """
    model_x, model_y = _finite_xy(model_x, model_y)
    article_x, article_y = _finite_xy(article_x, article_y)

    lo, hi = map(float, axis_limits)
    span = hi - lo if hi > lo else 1.0
    eps = 1e-6 * span

    # Match old certified scatter panel geometry/style.
    LABELSIZE = 20
    TICKSIZE = 13
    TICKLEN = 6
    TICKWIDTH = 1.4

    SCATTER_SIZE_MAIN = 34
    SCATTER_SIZE_SIMPLE = 38

    DIAG_LW = 1.6
    GUIDE_LW = 1.3

    LEGENDSIZE = 12

    MODEL_ALPHA = 0.72
    ARTICLE_ALPHA = 0.72

    tick_step = 0.5
    main_ticks = np.arange(lo, hi + 0.5 * tick_step, tick_step)

    if with_marginals:
        fig = plt.figure(figsize=(8.4, 7.2))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4.3, 1.25],
            height_ratios=[1.0, 4.5],
            wspace=0.11,
            hspace=0.11,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        model_size = SCATTER_SIZE_MAIN
        article_size = SCATTER_SIZE_MAIN

    else:
        fig, ax_main = plt.subplots(figsize=(6.6, 5.8))
        ax_top = None
        ax_right = None

        model_size = SCATTER_SIZE_SIMPLE
        article_size = SCATTER_SIZE_SIMPLE

    # --------------------------------------------------
    # Main scatter
    # --------------------------------------------------

    ax_main.scatter(
        article_x,
        article_y,
        s=article_size,
        facecolors=ARTICLE_COLOR,
        edgecolors=WHITE,
        linewidths=0.6,
        alpha=1.0,
        marker="o",
        label="Experiment",
        zorder=3,
    )

    ax_main.scatter(
        model_x,
        model_y,
        s=model_size,
        facecolors=MODEL_COLOR,
        edgecolors=MODEL_COLOR,
        linewidths=0.4,
        alpha=0.9,
        marker="o",
        label="Model",
        zorder=2,
    )

    ax_main.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="black",
        linewidth=DIAG_LW,
        zorder=1,
    )

    if show_unity_guides:
        ax_main.axvline(
            1.0,
            linestyle="-.",
            color="black",
            linewidth=GUIDE_LW,
            zorder=1,
        )
        ax_main.axhline(
            1.0,
            linestyle="-.",
            color="black",
            linewidth=GUIDE_LW,
            zorder=1,
        )

    ax_main.set_xlim(lo, hi + eps)
    ax_main.set_ylim(lo, hi + eps)

    ax_main.set_xlabel(xlabel, fontsize=LABELSIZE)
    ax_main.set_ylabel(ylabel, fontsize=LABELSIZE)

    ax_main.set_xticks(main_ticks)
    ax_main.set_yticks(main_ticks)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    ax_main.tick_params(
        axis="both",
        which="both",
        direction="out",
        labelsize=TICKSIZE,
        length=TICKLEN,
        width=TICKWIDTH,
    )

    ax_main.legend(
        frameon=False,
        loc="upper left",
        fontsize=LEGENDSIZE,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.2,
    )


    # --------------------------------------------------
    # Marginal model/article histograms
    # --------------------------------------------------
    if with_marginals:
        hist_bins = np.linspace(lo, hi + eps, bins + 1)
        _plot_joint_marginal_histograms(
            ax_top,
            ax_right,
            model_x=model_x,
            model_y=model_y,
            article_x=article_x,
            article_y=article_y,
            hist_bins=hist_bins,
            axis_limits=(lo, hi),
        )

        fig.subplots_adjust(wspace=0.18, hspace=0.10)
    else:
        fig.tight_layout()

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)


def plot_ccss_figure_5B_model_article_scatter(
    fig5_data,
    figure_5B_data,
    *,
    out_path,
    filename="ccss_figure_5B_high_contrast_model_article.png",
    with_marginals=False,
    show=False,
    close=True,
):
    """
    Joint Figure 5B:
        high contrast model + high contrast experiment.
    """
    ensure_dir(out_path)

    article = prepare_scraped_ccss_scatter(figure_5B_data)

    model_x = np.asarray(fig5_data["high_orth_norm"], dtype=float)
    model_y = np.asarray(fig5_data["high_same_norm"], dtype=float)

    article_x = np.asarray(article["orthogonal_surround_response"], dtype=float)
    article_y = np.asarray(article["same_surround_response"], dtype=float)

    save_path = os.path.join(out_path, filename)

    return _plot_ccss_model_article_scatter_panel(
        model_x=model_x,
        model_y=model_y,
        article_x=article_x,
        article_y=article_y,
        title="High center contrast",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        axis_limits=(0.0, 2.0),
        show=show,
        close=close,
    )


def plot_ccss_figure_5C_model_article_scatter(
    fig5_data,
    figure_5C_data,
    *,
    out_path,
    filename="ccss_figure_5C_low_contrast_model_article.png",
    with_marginals=False,
    show=False,
    close=True,
):
    """
    Joint Figure 5C:
        low contrast model + low contrast experiment.
    """
    ensure_dir(out_path)

    article = prepare_scraped_ccss_scatter(
        figure_5C_data,
        clamp_x_to_zero=True,
    )

    model_x = np.asarray(fig5_data["low_orth_norm"], dtype=float)
    model_y = np.asarray(fig5_data["low_same_norm"], dtype=float)

    article_x = np.asarray(article["orthogonal_surround_response"], dtype=float)
    article_y = np.asarray(article["same_surround_response"], dtype=float)

    save_path = os.path.join(out_path, filename)

    return _plot_ccss_model_article_scatter_panel(
        model_x=model_x,
        model_y=model_y,
        article_x=article_x,
        article_y=article_y,
        title="Low center contrast",
        xlabel="Orthogonal surround / center alone",
        ylabel="Same surround / center alone",
        save_path=save_path,
        with_marginals=with_marginals,
        axis_limits=(0.0, 2.0),
        show=show,
        close=close,
    )
