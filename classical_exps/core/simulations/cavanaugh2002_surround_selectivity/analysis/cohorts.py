import numpy as np
from classical_exps.core.tools.filtering_functions import filter_fitting_error, filter_no_supp_neurons, filter_low_supp_neurons, filter_missing_gsf_amrf


def get_selectivity_filtered_neuron_ids(
    h5_file,
    neuron_ids,
    fit_err_thresh=0.2,
    supp_thresh=0.1,
    apply_fit_error_filter=True,
    apply_no_supp_filter=True,
    apply_low_supp_filter=True,
    apply_valid_gsf_amrf_filter=True,
    verbose=True,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    filtered_neuron_ids = neuron_ids.copy()

    if apply_fit_error_filter:
        filtered_neuron_ids = filter_fitting_error(
            h5_file=h5_file,
            neuron_ids=filtered_neuron_ids,
            fit_err_thresh=fit_err_thresh,
            print_results=False,
        )
        if verbose:
            print(f"Number of neurons after fitting error filter: {len(filtered_neuron_ids)}")

    if apply_no_supp_filter:
        filtered_neuron_ids = filter_no_supp_neurons(
            h5_file=h5_file,
            neuron_ids=filtered_neuron_ids,
            print_results=False,
        )
        if verbose:
            print(f"Number of neurons after no suppression filter: {len(filtered_neuron_ids)}")

    if apply_valid_gsf_amrf_filter:
        filtered_neuron_ids = filter_missing_gsf_amrf(
            h5_file=h5_file,
            neuron_ids=filtered_neuron_ids,
            print_results=False,
        )
        if verbose:
            print(f"Number of neurons after valid GSF/AMRF filter: {len(filtered_neuron_ids)}")

    filtered_neuron_ids_low = np.asarray(filtered_neuron_ids, dtype=int)

    if apply_low_supp_filter:
        filtered_neuron_ids_low = filter_low_supp_neurons(
            h5_file=h5_file,
            neuron_ids=filtered_neuron_ids,
            supp_thresh=supp_thresh,
            print_results=False,
        )
        if verbose:
            print(f"Number of neurons after low suppression filter: {len(filtered_neuron_ids_low)}")

    return filtered_neuron_ids_low