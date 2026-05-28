############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from .shared import ensure_dir, plot_scatter_hist
from .cohorts import compute_filter_sets_nature_and_interactions

from .data_common import load_data
from .fits_size_tuning import fit_size_tuning
from .fits_contrast_response import fit_contrast_response_joint, add_fitted_contrast_response_to_scraped_config, build_loaded_contrast_response_from_scraped_config
from .fits_contrast_size import (
    fit_contrast_size_tuning,
    add_fitted_contrast_size_tuning_to_scraped_config,
    build_loaded_contrast_size_tuning_from_scraped_config,
)

from .step_size_tuning import size_tuning_results_1, size_tuning_results_2
from .step_contrast_response import contrast_response_results_bundle
from .step_contrast_size import contrast_size_tuning_results_1

from .export import export_nature_and_interactions_excel
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
from datetime import datetime
import json
import pandas as pd
import os
import openpyxl
from matplotlib.ticker import ScalarFormatter

##############################################################################
#####  PART II : Analyses for the first article, Cavanaugh et al., 2002  #####
#####   --------------------------------------------------------------   #####
#####        Nature and Interaction of Signals From the Receptive        #####
#####          Field Center and Surround in Macaque V1 Neurons           #####
#####   --------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00692.2001           #####
##################################################################################


# -------------------------
# Defaults
# -------------------------

DEFAULT_RUN = {
    # quick check
    "check_h5": False,
    "check_h5_mode": "sample",    
    "check_h5_sample_n": 20,

    # bulk loads
    "load_size_results": True,      
    "load_contrast_response": True,
    "load_contrast_size_tuning": True,

    # analysis calls
    "size_results_1": True,
    "size_results_2": True,
    "contrast_response_results_1": True,
    "contrast_size_tuning_results_1": True,

    # export
    "export_excel": True,
    "excel_path": None,   

    "plot_all_contrast_response":True,
    "plot_all_contrast_size_tuning":True,
    "print_loaded_inventory": True,
}

H5_PATHS = {
    "size_results": "/size_tuning/results",
    "contrast_response_curves": "/contrast_response/curves",
    "contrast_size_tuning_curves": "/contrast_size_tuning/curves",
}



# -------------------------
# Relay orchestrator
# -------------------------

def perform_analysis_nature_and_interactions(
    h5_file,
    neuron_ids,
    run=None,
    nature_and_interactions_scraped_data=None,

    # thresholds
    fit_err_thresh=0.2,
    supp_thresh=0.1,

    # contrast_response_results_1 args
    sort_by_std=True,
    spread_to_plot=(15, 50, 85),

    # contrast_size_tuning_results_1 args
    shift_to_plot=(15, 50, 85),
    low_contrast_id=0,
    high_contrast_id=-1,

    # axes for loading
    center_contrasts=None,
    surround_contrasts=None,
    contrasts=None,
    radii=None,

    # output
    output_dir="/project/results/nature_and_interactions",
):
    run = {**DEFAULT_RUN, **(run or {})}
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    params = dict(
        fit_err_thresh=float(fit_err_thresh),
        supp_thresh=float(supp_thresh),
        sort_by_std=bool(sort_by_std),
        spread_to_plot=list(spread_to_plot),
        shift_to_plot=list(shift_to_plot),
        low_contrast_id=int(low_contrast_id) if low_contrast_id is not None else None,
        high_contrast_id=int(high_contrast_id) if high_contrast_id is not None else None,
        output_dir=str(output_dir),
    )

    ensure_dir(output_dir)

    print(">> Performing Nature & Interactions analysis")
    print(f"   h5_file: {h5_file}")
    print(f"   neurons: {len(neuron_ids)}")
    print(f"   run: " + ", ".join([k for k, v in run.items() if isinstance(v, bool) and v]))

    # ---- BULK LOADS (ONCE) ----
    loaded = load_data(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        run=run,
        center_contrasts=center_contrasts,
        surround_contrasts=surround_contrasts,
        contrasts=contrasts,
        radii=radii,
        strict=False,
        print_inventory=bool(run.get("print_loaded_inventory", True)),
    )

    # ---- FILTERS: PRE-SIZE ----
    # article-nearest first gate: RF fit only
    size_filter_sets = compute_filter_sets_nature_and_interactions(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        mode="size_pre",
        verbose=True,
    )

    size_pre_ids = size_filter_sets["selected"]


    # ---- POST-LOAD PROCESSING ----
    loaded = fit_size_tuning(
        loaded,
        h5_file=h5_file,
        run=run,
        strict=True,
    )

    loaded = fit_contrast_response_joint(
        loaded,
        h5_file=h5_file,
        run=run,
        strict=True,
    )

    if nature_and_interactions_scraped_data is not None and run.get("fit_scraped_contrast_response", True):
        nature_and_interactions_scraped_data = add_fitted_contrast_response_to_scraped_config(
            nature_and_interactions_scraped_data,
            run=run,
            strict=True,
        )

    loaded = fit_contrast_size_tuning(
        loaded,
        h5_file=h5_file,
        run=run,
        strict=True,
    )

    if nature_and_interactions_scraped_data is not None and run.get("fit_scraped_contrast_size_tuning", True):
        nature_and_interactions_scraped_data = add_fitted_contrast_size_tuning_to_scraped_config(
            nature_and_interactions_scraped_data,
            run=run,
            strict=True,
        )

    # ---- ANALYSES ----
    if run.get("size_results_1", False):
        print("  - size_tuning_results_1")
        loaded = size_tuning_results_1(
            loaded=loaded,
            h5_file=h5_file,
            neuron_ids=size_pre_ids,
            experimental_data_scraped=nature_and_interactions_scraped_data,
            overwrite=bool(run.get("overwrite_size_results_1", False)),
            filtered_neuron_ids=size_pre_ids,
            plot_mode=run.get("size_plot_mode", "both"),
            plot_dir=os.path.join(output_dir, "size_tuning", "plots"),
            max_plot_neurons=run.get("size_plot_max_neurons", None),
            plot_only_filtered=bool(run.get("size_plot_only_filtered", False)),
            show_suppression=bool(run.get("size_show_suppression", True)),
        )
    else:
        print("  - size_tuning_results_1 skipped")
        print("    WARNING: downstream contrast filters assume /size_tuning/results already exist")

    # ---- FILTERS: POST-SIZE FOR CONTRAST ----
    contrast_filter_sets = compute_filter_sets_nature_and_interactions(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        mode="contrast",
        verbose=True,
    )

    # ---- FILTERS: POST-SIZE FOR CONTRAST-SIZE ----
    contrast_size_filter_sets = compute_filter_sets_nature_and_interactions(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        mode="contrast_size",
        verbose=True,
    )

    if run.get("size_results_2", False):
        print("  - size_tuning_results_2")
        size_tuning_results_2(
            neuron_ids=size_pre_ids,
            filtered_neuron_ids=size_pre_ids,
            size_results=loaded["size_results"],
        )

    if run.get("contrast_response_results_1", False):
        print("  - contrast_response_results_1")

        bundles = []

        # scraped paper data, if present
        loaded_scraped_contrast = build_loaded_contrast_response_from_scraped_config(
            nature_and_interactions_scraped_data,
            raw_key="contrast_tunning",
            fit_key="contrast_tunning_fit",
        )

        if loaded_scraped_contrast is not None:
            print("    > scraped center contrasts:",
                loaded_scraped_contrast["contrast_response"]["center_contrasts"])
            print("    > scraped surround contrasts:",
                loaded_scraped_contrast["contrast_response"]["surround_contrasts"])

        if loaded_scraped_contrast is not None:
            scraped_neuron_ids = list(loaded_scraped_contrast["contrast_response"]["by_id"].keys())
            bundles.append({
                "label": "scraped contrast-response",
                "neuron_ids": scraped_neuron_ids,
                "filtered_neuron_ids": scraped_neuron_ids,
                "loaded": loaded_scraped_contrast,
                "sort_by_std": bool(run.get("scraped_contrast_sort_by_std", False)),
                "spread_to_plot": list(run.get("scraped_contrast_spread_to_plot", [15, 50, 85])),
                "plot_all": bool(run.get("plot_all_scraped_contrast_response", True)),
                "xscale": str(run.get("scraped_contrast_response_xscale", "linear")),
                "print_fit_summary": bool(run.get("print_scraped_contrast_response_fit_summary", True)),
            })

        # model data
        bundles.append({
            "label": "model contrast-response",
            "neuron_ids": neuron_ids,
            "filtered_neuron_ids": contrast_filter_sets["selected"],
            "loaded": loaded,
            "sort_by_std": sort_by_std,
            "spread_to_plot": list(spread_to_plot),
            "plot_all": bool(run.get("plot_all_contrast_response", False)),
            "xscale": str(run.get("contrast_response_xscale", "symlog")),
            "print_fit_summary": bool(run.get("print_contrast_response_fit_summary", True)),
        })

        contrast_response_results_bundle(
            bundles,
            default_sort_by_std=sort_by_std,
            default_spread_to_plot=spread_to_plot,
            default_xscale=str(run.get("contrast_response_xscale", "symlog")),
            default_print_fit_summary=bool(run.get("print_contrast_response_fit_summary", True)),
        )

    if run.get("contrast_size_tuning_results_1", False):
        print("  - contrast_size_tuning_results_1")
        loaded_for_cst = loaded

        loaded_scraped_cst = build_loaded_contrast_size_tuning_from_scraped_config(
            nature_and_interactions_scraped_data,
            raw_key="contrast_size_tuning",
            fit_key="contrast_size_tuning_fit",
        )

        # if scraped exists, run a separate pass for it too
        if loaded_scraped_cst is not None:
            print("    > scraped contrast-size radii:",
                  loaded_scraped_cst["contrast_size_tuning"]["radii"])
            print("    > scraped contrast-size contrasts:",
                  loaded_scraped_cst["contrast_size_tuning"]["contrasts"])

            scraped_neuron_ids = list(loaded_scraped_cst["contrast_size_tuning"]["by_id"].keys())

            contrast_size_tuning_results_1(
                neuron_ids=scraped_neuron_ids,
                shift_to_plot=list(run.get("scraped_contrast_size_shift_to_plot", shift_to_plot)),
                low_contrast_id=low_contrast_id,
                high_contrast_id=high_contrast_id,
                filtered_neuron_ids=scraped_neuron_ids,
                loaded=loaded_scraped_cst,
                plot_all=bool(run.get("plot_all_scraped_contrast_size_tuning", True)),
            )

        contrast_size_tuning_results_1(
            neuron_ids=neuron_ids,
            shift_to_plot=list(shift_to_plot),
            low_contrast_id=low_contrast_id,
            high_contrast_id=high_contrast_id,
            filtered_neuron_ids=contrast_size_filter_sets["selected"],
            loaded=loaded_for_cst,
            plot_all=bool(run.get("plot_all_contrast_size_tuning", False)),
        )

    # ---- EXCEL EXPORT ----
    if run.get("export_excel", True):
        excel_path = run.get("excel_path", None)
        if excel_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(output_dir, f"nature_and_interactions_summary_{stamp}.xlsx")

        export_nature_and_interactions_excel(
            excel_path=excel_path,
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            run=run,
            params=params,
            loaded=loaded
        )

    print(">> Analysis complete")

    filter_sets = {
        "size_pre": size_filter_sets,
        "contrast": contrast_filter_sets,
        "contrast_size": contrast_size_filter_sets,
    }

    return loaded, filter_sets