import numpy as np
from classical_exps.functions.analysis.a_filtering_fucntions import filter_fitting_error, filter_no_supp_neurons, filter_low_supp_neurons

def compute_filter_sets_nature_and_interactions(
    h5_file,
    neuron_ids,
    fit_err_thresh=0.2,
    supp_thresh=0.1,
    verbose=True,
):
    filtered_neuron_ids = filter_fitting_error(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        print_results=False,
    )
    if verbose:
        print(f"Number of neurons after fitting error filter: {len(filtered_neuron_ids)}")

    # filtered_neuron_ids = filter_no_supp_neurons(
    #     h5_file=h5_file,
    #     neuron_ids=filtered_neuron_ids,
    #     print_results=False,
    # )
    # if verbose:
    #     print(f"Number of neurons after no suppression filter: {len(filtered_neuron_ids)}")

    # filtered_neuron_ids_low = filter_low_supp_neurons(
    #     h5_file=h5_file,
    #     neuron_ids=filtered_neuron_ids,
    #     supp_thresh=supp_thresh,
    #     print_results=False,
    # )
    # if verbose:
    #     print(f"Number of neurons after low suppression filter: {len(filtered_neuron_ids_low)}")

    return {
        "fit_ok_no_supp": np.asarray(filtered_neuron_ids, dtype=int),
        "fit_ok_no_supp_low_supp": np.asarray(filtered_neuron_ids, dtype=int),
    }
