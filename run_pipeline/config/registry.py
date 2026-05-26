from __future__ import annotations

from typing import Any, Callable


# =============================================================================
# 1. Pre-experimental / pre-analysis
# =============================================================================

from classical_exps.core.tools.preanalysis import (
    get_all_grating_parameters,
    get_preferred_position,
)


# =============================================================================
# 2. Cavanaugh 2002 — center-surround gain / Nature and Interaction
# =============================================================================

from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.experiments.size_tuning import (
    size_tuning_experiment_all_phases,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.experiments.contrast_response import (
    contrast_response_experiment,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.experiments.contrast_size_tuning import (
    contrast_size_tuning_experiment_all_phases,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.analysis.main_analysis import (
    perform_analysis_nature_and_interactions,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.analysis.step_size_tuning import (
    size_tuning_results_1,
    size_tuning_results_2,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.analysis.step_contrast_response import (
    contrast_response_results_1,
)
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.analysis.step_contrast_size import (
    contrast_size_tuning_results_1,
)


# =============================================================================
# 3. Cavanaugh 2002 — surround selectivity / orientation + CCSS
# =============================================================================

from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.experiments.orientation_tuning import (
    orientation_tuning_experiment_all_phases,
)
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.experiments.ccss import (
    center_contrast_surround_suppression_experiment,
)
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.analysis_orientation_tuning import (
    orientation_tuning_results,
)
from classical_exps.core.simulations.cavanaugh2002_surround_selectivity.analysis.analysis_ccss import (
    ccss_results,
)


# =============================================================================
# 4. Yeh 2009 — black/white dominance
# =============================================================================

from classical_exps.core.simulations.yeh2009_black_dominance.experiment.black_and_white_experiment import (
    black_white_preference_experiment,
    black_white_preference_experiment_strict_paper_version_with_variance,
    black_white_preference_experiment_paper_energy_version,
)
from classical_exps.core.simulations.yeh2009_black_dominance.analysis.main_analysis import (
    black_white_results_1,
)


# =============================================================================
# 5. Freeman 2013 — texture/noise modulation
# =============================================================================

from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.experiment.main_experiment import (
    texture_noise_response_experiment,
)
from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.main_analysis import (
    texture_noise_response_results_1,
    texture_noise_response_results_2,
    texture_noise_response_results_3,
)


# =============================================================================
# 6. Hallum 2014 / second-order surround orientation
# =============================================================================


from classical_exps.core.simulations.hallum2014_second_order_surround.experiment.main_experiment import (
    get_all_grating_parameters_with_modulator
)

from classical_exps.core.simulations.hallum2014_second_order_surround.analysis.main_analysis import (
    recreate_histograms_second_order_orientation,
)

# =============================================================================
# Optional shared filters
# =============================================================================

from classical_exps.core.tools.filtering_functions import (
    filter_fitting_error,
    filter_no_supp_neurons,
    filter_low_supp_neurons,
    filter_SNR,
)


FUNCTION_REGISTRY: dict[str, Callable[..., Any]] = {
    # -------------------------------------------------------------------------
    # 1. Pre-experimental / pre-analysis
    # -------------------------------------------------------------------------
    "get_all_grating_parameters": get_all_grating_parameters,
    "get_preferred_position": get_preferred_position,

    # -------------------------------------------------------------------------
    # 2. Cavanaugh 2002 — center-surround gain / Nature and Interaction
    # -------------------------------------------------------------------------
    "size_tuning_experiment_all_phases": size_tuning_experiment_all_phases,
    "contrast_response_experiment": contrast_response_experiment,
    "contrast_size_tuning_experiment_all_phases": contrast_size_tuning_experiment_all_phases,

    "perform_analysis_nature_and_interactions": perform_analysis_nature_and_interactions,
    "size_tuning_results_1": size_tuning_results_1,
    "size_tuning_results_2": size_tuning_results_2,
    "contrast_response_results_1": contrast_response_results_1,
    "contrast_size_tuning_results_1": contrast_size_tuning_results_1,

    # -------------------------------------------------------------------------
    # 3. Cavanaugh 2002 — surround selectivity / orientation + CCSS
    # -------------------------------------------------------------------------
    "orientation_tuning_experiment_all_phases": orientation_tuning_experiment_all_phases,
    "center_contrast_surround_suppression_experiment": center_contrast_surround_suppression_experiment,

    "orientation_tuning_results": orientation_tuning_results,
    "ccss_results": ccss_results,

    # -------------------------------------------------------------------------
    # 4. Yeh 2009 — black/white dominance
    # -------------------------------------------------------------------------
    "black_white_preference_experiment": black_white_preference_experiment,

    # Alias, because your old config used this shorter name.
    "black_white_preference_experiment_strict_paper_version": (
        black_white_preference_experiment_strict_paper_version_with_variance
    ),
    "black_white_preference_experiment_strict_paper_version_with_variance": (
        black_white_preference_experiment_strict_paper_version_with_variance
    ),
    "black_white_preference_experiment_paper_energy_version": (
        black_white_preference_experiment_paper_energy_version
    ),

    "black_white_results_1": black_white_results_1,

    # -------------------------------------------------------------------------
    # 5. Freeman 2013 — texture/noise modulation
    # -------------------------------------------------------------------------
    "texture_noise_response_experiment": texture_noise_response_experiment,

    "texture_noise_response_results_1": texture_noise_response_results_1,
    "texture_noise_response_results_2": texture_noise_response_results_2,
    "texture_noise_response_results_3": texture_noise_response_results_3,

    # -------------------------------------------------------------------------
    # 6. Hallum 2014 / second-order surround orientation
    # -------------------------------------------------------------------------
    "get_all_grating_parameters_with_modulator": get_all_grating_parameters_with_modulator,
    "recreate_histograms_second_order_orientation": recreate_histograms_second_order_orientation,

    # -------------------------------------------------------------------------
    # Optional shared filters
    # -------------------------------------------------------------------------
    "filter_fitting_error": filter_fitting_error,
    "filter_no_supp_neurons": filter_no_supp_neurons,
    "filter_low_supp_neurons": filter_low_supp_neurons,
    "filter_SNR": filter_SNR,
}


# Remove missing optional imports cleanly.
FUNCTION_REGISTRY = {
    name: func
    for name, func in FUNCTION_REGISTRY.items()
    if func is not None
}