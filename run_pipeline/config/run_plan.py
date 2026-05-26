from __future__ import annotations
from difflib import get_close_matches


from run_pipeline.config.variables import *

from run_pipeline.config.experimental_scraped_data import (
    cavanaugh2002_center_surround_gain_scraped_data,
    cavanaugh2002_surround_selectivity_scraped_data,
    yeh2009_black_dominance_scraped_data,
    freeman2013_naturalistic_texture_modulation_scraped_data,
    hallum2014_second_order_surround_scraped_data,
)



# =============================================================================
# Run selection
# =============================================================================
#
# Selection is by presence:
# - put a touchpoint key in RUN_EXPERIMENTS to run its experiment step(s)
# - put a touchpoint key in RUN_ANALYSES to run its analysis step(s)
#
# Commenting out a key disables that part. Missing keys are not run.
#
# The global run_experiments / run_analyses switches in main.py are still safety
# gates: they can block all experiments or all analyses even if selected here.

RUN_EXPERIMENTS = {
    # -------------------------------------------------------------------------
    # 1. Pre-experimental / pre-analysis
    # -------------------------------------------------------------------------
    # "pre_experimental.grating_parameters",
    # "pre_experimental.preferred_position",

    # -------------------------------------------------------------------------
    # 2. Cavanaugh 2002 — center-surround gain / Nature and Interaction
    # -------------------------------------------------------------------------
    # "cavanaugh2002_center_surround_gain.size_tuning",
    # "cavanaugh2002_center_surround_gain.contrast_response",
    # "cavanaugh2002_center_surround_gain.contrast_size_tuning",

    # full_analysis is analysis-only

    # -------------------------------------------------------------------------
    # 3. Cavanaugh 2002 — surround selectivity
    # -------------------------------------------------------------------------
    # "cavanaugh2002_surround_selectivity.orientation_tuning",
    # "cavanaugh2002_surround_selectivity.ccss",

    # -------------------------------------------------------------------------
    # 4. Yeh 2009 — black/white dominance
    # -------------------------------------------------------------------------
    # "yeh2009_black_dominance.black_white_snr",

    # -------------------------------------------------------------------------
    # 5. Freeman 2013 — texture/noise modulation
    # -------------------------------------------------------------------------
    # "freeman2013_texture_noise.texture_noise_response",

    # -------------------------------------------------------------------------
    # 6. Hallum 2014 — second-order surround orientation
    # -------------------------------------------------------------------------
    # "hallum2014_second_order_orientation.second_order_orientation",
}


RUN_ANALYSES = {
    # -------------------------------------------------------------------------
    # 1. Pre-experimental / pre-analysis
    # -------------------------------------------------------------------------
    # Usually no standalone analysis here.

    # -------------------------------------------------------------------------
    # 2. Cavanaugh 2002 — center-surround gain / Nature and Interaction
    # -------------------------------------------------------------------------
    # "cavanaugh2002_center_surround_gain.size_tuning",
    # "cavanaugh2002_center_surround_gain.contrast_response",
    # "cavanaugh2002_center_surround_gain.contrast_size_tuning",
    # "cavanaugh2002_center_surround_gain.full_analysis",

    # -------------------------------------------------------------------------
    # 3. Cavanaugh 2002 — surround selectivity
    # -------------------------------------------------------------------------
    # "cavanaugh2002_surround_selectivity.orientation_tuning",
    # "cavanaugh2002_surround_selectivity.ccss",

    # -------------------------------------------------------------------------
    # 4. Yeh 2009 — black/white dominance
    # -------------------------------------------------------------------------
    "yeh2009_black_dominance.black_white_snr",

    # -------------------------------------------------------------------------
    # 5. Freeman 2013 — texture/noise modulation
    # -------------------------------------------------------------------------
    # "freeman2013_texture_noise.texture_noise_response",

    # -------------------------------------------------------------------------
    # 6. Hallum 2014 — second-order surround orientation
    # -------------------------------------------------------------------------
    # "hallum2014_second_order_orientation.second_order_orientation",
}


# =============================================================================
# Shared argument blocks
# =============================================================================

MODEL_ARGS = {
    "h5_file": h5_file,
    "all_neurons_model": all_neurons_model,
    "neuron_ids": neuron_ids,
    "overwrite": overwrite,
    "contrast": contrast,
    "img_res": img_res,
    "pixel_min": pixel_min,
    "pixel_max": pixel_max,
    "device": device,
    "size": size,
}

H5_ARGS = {
    "h5_file": h5_file,
    "neuron_ids": neuron_ids,
}

# =============================================================================
# Touchpoint / step construction
# =============================================================================

def exp(name: str, params: dict):
    return {
        "kind": "experiment",
        "name": name,
        "params": params,
    }


def ana(name: str, params: dict):
    return {
        "kind": "analysis",
        "name": name,
        "params": params,
    }


def touchpoint(
    key: str,
    *,
    label: str,
    experiments: list[dict] | None = None,
    analyses: list[dict] | None = None,
):
    return {
        "key": key,
        "label": label,
        "run_experiments": key in RUN_EXPERIMENTS,
        "run_analyses": key in RUN_ANALYSES,
        "experiments": experiments or [],
        "analyses": analyses or [],
    }


# =============================================================================
# 1. Pre-experimental / pre-analysis
# =============================================================================

PRE_EXPERIMENTAL = [
    touchpoint(
        "pre_experimental.grating_parameters",
        label="Preferred grating parameters",
        experiments=[
            exp(
                "get_all_grating_parameters",
                {
                    **MODEL_ARGS,
                    "orientations": orientations,
                    "spatial_frequencies": spatial_frequencies,
                    "phases": phases,
                },
            ),
        ],
    ),
    touchpoint(
        "pre_experimental.preferred_position",
        label="Preferred RF position",
        experiments=[
            exp(
                "get_preferred_position",
                {
                    **MODEL_ARGS,
                    "dot_size_in_pixels": dot_size_in_pixels_gauss,
                    "num_dots": num_dots,
                    "bs": bs,
                    "seed": seed,
                },
            ),
        ],
    ),
]


# =============================================================================
# 2. Cavanaugh 2002 — center-surround gain / Nature and Interaction
# =============================================================================

CAVANAUGH_CENTER_SURROUND_GAIN = [
    touchpoint(
        "cavanaugh2002_center_surround_gain.size_tuning",
        label="Size tuning / GSF / SI / AMRF",
        experiments=[
            exp(
                "size_tuning_experiment_all_phases",
                {
                    **MODEL_ARGS,
                    "radii": radii,
                    "phases": phases,
                    "neg_val": neg_val,
                },
            ),
        ],
        analyses=[
            ana(
                "size_tuning_results_1",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "supp_thresh": supp_thresh,
                },
            ),
            ana(
                "size_tuning_results_2",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "sort_by_std": sort_by_std,
                    "spread_to_plot": spread_to_plot,
                },
            ),
        ],
    ),

    touchpoint(
        "cavanaugh2002_center_surround_gain.contrast_response",
        label="Contrast response",
        experiments=[
            exp(
                "contrast_response_experiment",
                {
                    **MODEL_ARGS,
                    "center_contrasts": center_contrasts,
                    "surround_contrasts": surround_contrasts,
                    "neg_val": neg_val,
                },
            ),
        ],
        analyses=[
            ana(
                "contrast_response_results_1",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "sort_by_std": sort_by_std,
                    "spread_to_plot": spread_to_plot,
                },
            ),
        ],
    ),

    touchpoint(
        "cavanaugh2002_center_surround_gain.contrast_size_tuning",
        label="Contrast-size tuning",
        experiments=[
            exp(
                "contrast_size_tuning_experiment_all_phases",
                {
                    **MODEL_ARGS,
                    "phases": phases,
                    "contrasts": contrasts_article_1,
                    "radii": radii,
                    "neg_val": neg_val,
                },
            ),
        ],
        analyses=[
            ana(
                "contrast_size_tuning_results_1",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "shift_to_plot": shift_to_plot,
                    "low_contrast_id": low_contrast_id,
                    "high_contrast_id": high_contrast_id,
                },
            ),
        ],
    ),

    touchpoint(
        "cavanaugh2002_center_surround_gain.full_analysis",
        label="Full Nature and Interaction analysis",
        analyses=[
            ana(
                "perform_analysis_nature_and_interactions",
                {
                    **H5_ARGS,
                    "nature_and_interactions_scraped_data": cavanaugh2002_center_surround_gain_scraped_data,

                    "center_contrasts": center_contrasts,
                    "surround_contrasts": surround_contrasts,
                    "contrasts": contrasts_article_1,
                    "radii": radii,

                    "fit_err_thresh": fit_err_thresh,
                    "supp_thresh": supp_thresh,

                    "sort_by_std": sort_by_std,
                    "spread_to_plot": spread_to_plot,
                    "shift_to_plot": shift_to_plot,
                    "low_contrast_id": low_contrast_id,
                    "high_contrast_id": high_contrast_id,

                    "output_dir": main_dir + "/results/nature_and_interactions",

                    "run": {
                        "check_h5": True,
                        "load_size_results": True,
                        "load_contrast_response": True,
                        "load_contrast_size_tuning": True,

                        "size_results_1": True,
                        "size_results_2": True,
                        "contrast_response_results_1": True,
                        "contrast_size_tuning_results_1": True,

                        "overwrite_size_results_1": True,

                        "export_excel": False,

                        "plot_all_contrast_response": True,
                        "plot_all_contrast_size_tuning": True,
                        "print_loaded_inventory": True,
                        "load_size_tuning_curves": True,
                        "size_tuning_plots": True,
                        "plot_size_tuning_curves": True,

                        "fit_size_tuning": False,
                        "use_size_tuning_fits": True,
                        "fit_force": False,
                        "fit_strict": False,

                        "fit_contrast_response": False,
                        "use_contrast_response_fits": True,

                        "contrast_size_tuning_fit_model": "gain",
                        "fit_contrast_size_tuning": True,
                        "use_contrast_size_tuning_fits": True,
                        "fit_contrast_size_tuning_force": True,
                        "fit_contrast_size_tuning_strict": True,
                    },
                },
            ),
        ],
    ),
]


# =============================================================================
# 3. Cavanaugh 2002 — surround selectivity
# =============================================================================

CAVANAUGH_SURROUND_SELECTIVITY = [
    touchpoint(
        "cavanaugh2002_surround_selectivity.orientation_tuning",
        label="Orientation tuning / surround orientation selectivity",
        experiments=[
            exp(
                "orientation_tuning_experiment_all_phases",
                {
                    **MODEL_ARGS,
                    "phases": phases,
                    "ori_shifts": ori_shifts,
                    "contrast": experiment_2_contrast,
                },
            ),
        ],
        analyses=[
            ana(
                "orientation_tuning_results",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "scraped_orietation_tuning_data": cavanaugh2002_surround_selectivity_scraped_data,
                },
            ),
        ],
    ),

    touchpoint(
        "cavanaugh2002_surround_selectivity.ccss",
        label="Center-contrast surround suppression",
        experiments=[
            exp(
                "center_contrast_surround_suppression_experiment",
                {
                    **MODEL_ARGS,
                    "center_contrasts_ccss": center_contrasts_ccss,
                    "surround_contrasts_ccss": surround_contrasts_ccss,
                    "phases": phases,
                },
            ),
        ],
        analyses=[
            ana(
                "ccss_results",
                {
                    **H5_ARGS,
                    "fit_err_thresh": fit_err_thresh,
                    "scraped_ccss_data": cavanaugh2002_surround_selectivity_scraped_data,
                },
            ),
        ],
    ),
]


# =============================================================================
# 4. Yeh 2009 — black/white dominance
# =============================================================================

BLACK_WHITE_DOMINANCE = [
    touchpoint(
        "yeh2009_black_dominance.black_white_snr",
        label="Black/white SNR dominance",
        experiments=[
            exp(
                "black_white_preference_experiment",
                {
                    **MODEL_ARGS,
                    "dot_size_in_pixels": dot_size_in_pixels,
                    "seed": seed,
                },
            ),
            exp(
                "black_white_preference_experiment_strict_paper_version",
                {
                    **MODEL_ARGS,
                    "seed": seed,
                },
            ),
            exp(
                "black_white_preference_experiment_paper_energy_version",
                {
                    **MODEL_ARGS,
                    "seed": seed,
                },
            ),
        ],
        analyses=[
            ana(
                "black_white_results_1",
                {
                    **H5_ARGS,
                    "neuron_depths": neuron_depths,
                    "energy_thresh": SNR_thresh,
                    "scraped_snr_config": yeh2009_black_dominance_scraped_data,
                },
            ),
        ],
    ),
]


# =============================================================================
# 5. Freeman 2013 — texture/noise modulation
# =============================================================================

TEXTURE_NOISE_MODULATION = [
    touchpoint(
        "freeman2013_texture_noise.texture_noise_response",
        label="Texture/noise response modulation",
        experiments=[
            exp(
                "texture_noise_response_experiment",
                {
                    **MODEL_ARGS,
                    "directory_imgs": directory_imgs,
                    "num_samples": num_samples,
                },
            ),
        ],
        analyses=[
            ana(
                "texture_noise_response_results_1",
                {
                    **H5_ARGS,
                    "wanted_fam_order": wanted_fam_order,
                },
            ),
            ana(
                "texture_noise_response_results_2",
                {
                    **H5_ARGS,
                    "wanted_fam_order": wanted_fam_order,
                },
            ),
            ana(
                "texture_noise_response_results_3",
                {
                    **H5_ARGS,
                },
            ),
        ],
    ),
]


# =============================================================================
# 6. Hallum 2014 — second-order surround orientation
# =============================================================================

SECOND_ORDER_ORIENTATION = [
    touchpoint(
        "hallum2014_second_order_orientation.second_order_orientation",
        label="Second-order surround orientation",
        experiments=[
            exp(
                "get_all_grating_parameters_with_modulator",
                {
                    **MODEL_ARGS,
                },
            ),
        ],
        analyses=[
            ana(
                "recreate_histograms_second_order_orientation",
                {
                    **H5_ARGS,
                    "scraped_second_order_config": hallum2014_second_order_surround_scraped_data,
                },
            ),
        ],
    ),
]


# =============================================================================
# Final ordered touchpoint plan
# =============================================================================

TOUCHPOINT_PLAN = [
    *PRE_EXPERIMENTAL,
    *CAVANAUGH_CENTER_SURROUND_GAIN,
    *CAVANAUGH_SURROUND_SELECTIVITY,
    *BLACK_WHITE_DOMINANCE,
    *TEXTURE_NOISE_MODULATION,
    *SECOND_ORDER_ORIENTATION,
]

# =============================================================
# Validation of run selection
# =============================================================================

def _format_bullets(items: list[str]) -> str:
    if not items:
        return "  <none>"
    return "\n".join(f"  - {item}" for item in items)


def _format_unknown_with_suggestions(
    unknown: list[str],
    known: set[str],
) -> str:
    if not unknown:
        return "  <none>"

    lines = []
    known_sorted = sorted(known)

    for key in unknown:
        matches = get_close_matches(key, known_sorted, n=3, cutoff=0.65)

        lines.append(f"  - {key}")

        if matches:
            lines.append("    did you mean:")
            for match in matches:
                lines.append(f"      - {match}")

    return "\n".join(lines)


def _validate_selected_touchpoints() -> None:
    known = {tp["key"] for tp in TOUCHPOINT_PLAN}

    unknown_experiments = sorted(set(RUN_EXPERIMENTS) - known)
    unknown_analyses = sorted(set(RUN_ANALYSES) - known)

    if unknown_experiments or unknown_analyses:
        raise ValueError(
            "\n"
            "Run selection contains unknown touchpoint key(s).\n\n"
            "Unknown experiment selections:\n"
            f"{_format_unknown_with_suggestions(unknown_experiments, known)}\n\n"
            "Unknown analysis selections:\n"
            f"{_format_unknown_with_suggestions(unknown_analyses, known)}\n\n"
            "Known touchpoints:\n"
            f"{_format_bullets(sorted(known))}"
        )

    selected_for_experiments_but_empty = sorted(
        tp["key"]
        for tp in TOUCHPOINT_PLAN
        if tp["key"] in RUN_EXPERIMENTS and not tp.get("experiments")
    )

    selected_for_analyses_but_empty = sorted(
        tp["key"]
        for tp in TOUCHPOINT_PLAN
        if tp["key"] in RUN_ANALYSES and not tp.get("analyses")
    )

    if selected_for_experiments_but_empty or selected_for_analyses_but_empty:
        raise ValueError(
            "\n"
            "Run selection requests step type(s) that do not exist.\n\n"
            "Selected for experiments but has no experiment steps:\n"
            f"{_format_bullets(selected_for_experiments_but_empty)}\n\n"
            "Selected for analyses but has no analysis steps:\n"
            f"{_format_bullets(selected_for_analyses_but_empty)}"
        )


_validate_selected_touchpoints()