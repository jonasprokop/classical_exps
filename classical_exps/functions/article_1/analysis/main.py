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
from .fits_contrast_response import fit_contrast_response
from .fits_contrast_size import fit_contrast_size_tuning

from .step_size_tuning import size_tuning_results_1, size_tuning_results_2
from .step_contrast_response import contrast_response_results_1
from .step_contrast_size import contrast_size_tuning_results_1
from .data_size_tuning import plot_size_tuning_curves_bulk

from .export import export_nature_and_interactions_excel
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
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
    """
    """
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
    # load_data uses `run` to decide what to load.
    loaded = load_data(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        run=run,
        center_contrasts=center_contrasts,
        surround_contrasts=surround_contrasts,
        contrasts=contrasts,
        radii=radii,
        strict=True,
        print_inventory=bool(run.get("print_loaded_inventory", True)),
    )

    # ---- FILTERS (ONCE) ----
    filter_sets = compute_filter_sets_nature_and_interactions(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        supp_thresh=supp_thresh,
        verbose=True,
    )

    # ---- POST-LOAD PROCESSING ----
    loaded = fit_size_tuning(
    loaded,
    h5_file=h5_file,
    run=run,
    strict=True,
        )
    
    loaded = fit_contrast_response(
        loaded,
        h5_file=h5_file,
        run=run,
        strict=True,
    )    
    loaded = fit_contrast_size_tuning(
        loaded,
        h5_file=h5_file,
        run=run,
        strict=True,
    )



    # ---- ANALYSES ----
    if run.get("size_results_1", False):
        print("  - size_tuning_results_1")
        loaded = size_tuning_results_1(
            loaded=loaded,
            h5_file=h5_file,
            neuron_ids=neuron_ids,
            overwrite=bool(run.get("overwrite_size_results_1", False)),
            filtered_neuron_ids=filter_sets["fit_ok_no_supp_low_supp"],
        )

    if run.get("plot_size_tuning_curves", False):
        print("  - plot_size_tuning_curves")
        plot_size_tuning_curves_bulk(
            loaded,
            save_dir=os.path.join(output_dir, "size_tuning", "plots"),
            max_neurons=run.get("plot_size_tuning_neuron_max", None),
            use_diameter=bool(run.get("plot_size_tuning_use_diameter", True)),
            h5_file=h5_file,
            load_if_missing=True,  
        )

    if run.get("size_results_2", False):
        print("  - size_tuning_results_2")
        size_tuning_results_2(
            neuron_ids=neuron_ids,
            filtered_neuron_ids=filter_sets["fit_ok_no_supp"],
            size_results=loaded["size_results"],
        )

    if run.get("contrast_response_results_1", False):
        print("  - contrast_response_results_1")
        contrast_response_results_1(
            neuron_ids=neuron_ids,
            sort_by_std=sort_by_std,
            spread_to_plot=list(spread_to_plot),
            filtered_neuron_ids=filter_sets["fit_ok_no_supp"],
            loaded=loaded,
            plot_all=bool(run.get("plot_all_contrast_response", False)),
        )

    if run.get("contrast_size_tuning_results_1", False):
        print("  - contrast_size_tuning_results_1")
        contrast_size_tuning_results_1(
            neuron_ids=neuron_ids,
            shift_to_plot=list(shift_to_plot),
            low_contrast_id=low_contrast_id,
            high_contrast_id=high_contrast_id,
            filtered_neuron_ids=filter_sets["fit_ok_no_supp"],
            loaded=loaded,
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
    return loaded, filter_sets
