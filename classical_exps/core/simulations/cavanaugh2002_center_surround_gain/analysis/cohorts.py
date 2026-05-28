import numpy as np
from classical_exps.core.tools.filtering_functions import filter_fitting_error, filter_no_supp_neurons, filter_low_supp_neurons, filter_missing_gsf_amrf

def compute_filter_sets_nature_and_interactions(
    h5_file,
    neuron_ids,
    fit_err_thresh=0.2,
    supp_thresh=0.1,
    mode="size_pre",   # "size_pre" | "contrast" | "contrast_size"
    verbose=True,
):
    """
    Small orchestration wrapper over existing reusable filter functions.

    Modes
    -----
    size_pre:
        RF-fit filtering only.
        Use before size_tuning_results_1, because /size_tuning/results may not exist yet.

    contrast:
        RF-fit -> valid GSF/AMRF
        Use after size_tuning_results_1, for article-like center/surround placement.

    contrast_size:
        RF-fit -> valid GSF/AMRF -> measurable suppression
        Use after size_tuning_results_1, for stricter contrast-size analyses.
    """
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    out = {}

    # --------------------------------------
    # 1) RF fit gate (shared first step)
    # --------------------------------------
    fit_ok = filter_fitting_error(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
        fit_err_thresh=fit_err_thresh,
        print_results=False,
    )
    fit_ok = np.asarray(fit_ok, dtype=int)
    out["fit_ok"] = fit_ok

    if verbose:
        print(f"[filters:{mode}] after fitting error filter: {len(fit_ok)}")

    # --------------------------------------
    # 2) Mode-specific continuation
    # --------------------------------------
    if mode == "size_pre":
        # before size_tuning_results_1
        out["selected"] = fit_ok
        return out

    elif mode == "contrast":
        # after size_tuning_results_1
        valid_gsf_amrf = filter_missing_gsf_amrf(
            h5_file=h5_file,
            neuron_ids=fit_ok,
            print_results=False,
        )
        valid_gsf_amrf = np.asarray(valid_gsf_amrf, dtype=int)

        out["valid_gsf_amrf"] = valid_gsf_amrf
        out["selected"] = valid_gsf_amrf

        if verbose:
            print(f"[filters:{mode}] after valid GSF/AMRF filter: {len(valid_gsf_amrf)}")

        return out

    elif mode == "contrast_size":
        low_supp_ok = filter_low_supp_neurons(
            h5_file=h5_file,
            neuron_ids=fit_ok,
            supp_thresh=supp_thresh,
            print_results=False,
        )
        low_supp_ok = np.asarray(low_supp_ok, dtype=int)

        out["low_supp_ok"] = low_supp_ok
        out["selected"] = low_supp_ok

        if verbose:
            print(f"[filters:{mode}] after low suppression filter: {len(low_supp_ok)}")

        return out

    else:
        raise ValueError(
            f"Unknown mode={mode!r}. Expected 'size_pre', 'contrast', or 'contrast_size'."
        )