from __future__ import annotations

import numpy as np

# model import
from classical_exps.core.tools.utils import pickleread
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble

# additional models for import can be setted up under core.models
# and imported here, e.g.:
# from core.models import my_model


# =============================================================================
# Runtime switches
# =============================================================================

# If True, selected experiments overwrite existing results.
overwrite = True

# None lets the downstream code choose GPU if available, CPU otherwise.
device = None


# =============================================================================
# Paths
# =============================================================================

main_dir = "/project"
run_dir = main_dir + "/run_pipeline"
objects_dir = run_dir + "/objects"

h5_file = run_dir + "/results_convnext_model.h5"


# =============================================================================
# Model / neuron selection
# =============================================================================

all_neurons_model = v1_convnext_ensemble

n_neurons = 458
neuron_ids = np.arange(n_neurons)

# Correlation score of model neurons.
corrs = pickleread(objects_dir + "/avg_corr.pkl")

# Useful local filters:
# neuron_ids = neuron_ids[neuron_ids < 20]
# neuron_ids = neuron_ids[corrs > 0.75]


# =============================================================================
# Shared stimulus / image parameters
# =============================================================================

contrast = 1

img_res = [93, 93]
target_res = img_res

# Training-data pixel range. Used as black/white reference values.
pixel_min = -1.7876
pixel_max = 2.1919

# Visual field size used by the model.
size = 2.67

# Keep True: responses are represented as stimulus response minus gray-screen response.
neg_val = True

seed = 0


# =============================================================================
# Pre-experimental / pre-analysis parameters
# =============================================================================

orientations = np.linspace(0, np.pi, 37)[:-1]
spatial_frequencies = np.linspace(1, 7, 25)
phases = np.linspace(0, 2 * np.pi, 37)[:-1]

dot_size_in_pixels_gauss = 4
num_dots = 200_000
bs = 40


# =============================================================================
# Cavanaugh 2002 — center-surround gain / Nature and Interaction
# =============================================================================

# Model field is bounded at 2.67 deg; radii cover the effective tested annulus range.
radii = np.logspace(np.log10(0.15), np.log10(2.67), 8)

center_contrasts = np.logspace(np.log10(0.06), np.log10(1), 18)
surround_contrasts = np.logspace(np.log10(0.06), np.log10(1), 6)

# Article-style contrast grid, using fractional contrast.
contrasts_article_1 = np.array([0.06, 0.13, 0.25, 0.5, 1.0])


# =============================================================================
# Cavanaugh 2002 — surround selectivity / orientation + CCSS
# =============================================================================

ori_shifts = np.linspace(-np.pi / 2, np.pi / 2, 4, endpoint=False)
experiment_2_contrast = 0.5

center_contrasts_ccss = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5])
surround_contrasts_ccss = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5])

# Contrast indices used in CCSS summary plots.
high_center_contrast_id = -3
high_norm_center_contrast_id = high_center_contrast_id

low_center_contrast_id = 1
low_norm_center_contrast_id = low_center_contrast_id


# =============================================================================
# Yeh 2009 — black/white dominance
# =============================================================================

contrasts_article_3 = np.logspace(np.log10(0.06), np.log10(1), 5)
dot_size_in_pixels = 5

neuron_depths = pickleread(objects_dir + "/depth_info.pickle")


# =============================================================================
# Freeman 2013 — texture/noise modulation
# =============================================================================

directory_imgs = objects_dir + "/shareStim_NN13"
num_samples = 15

wanted_fam_order = [
    "60", "56", "13", "48", "71",
    "18", "327", "336", "402", "38",
    "23", "52", "99", "393", "30",
]


# =============================================================================
# Hallum 2014 — second-order surround orientation
# =============================================================================

# =============================================================================

# =============================================================================
# Shared analysis parameters
# =============================================================================

fit_err_thresh = 0.2
supp_thresh = 0.1
SNR_thresh = 2

sort_by_std = False
spread_to_plot = [0, 15, 50, 85, 100]

shift_to_plot = spread_to_plot
low_contrast_id = 0
high_contrast_id = -1


# =============================================================================
# Misc / examples
# =============================================================================

neuron_id = 0