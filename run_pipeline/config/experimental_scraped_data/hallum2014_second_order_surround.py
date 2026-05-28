hallum2014_second_order_surround_scraped_data = {
    "preferred_second_order_orientation_distribution": {
        "x_label": "second-order orientation preference relative to carrier orientation (deg)",
        "y_label": "proportion of cells",

        "circular_orientation_space": True,
        "orientation_period_deg": 180,

        "wrapped_bins_deg": [-90, -45, 0, 45],
        "wrapped_proportion_of_cells": [
            0.2049,
            0.2706,
            0.3039,
            0.2549,
        ],

        "wrapped_proportion_of_cells_evaluated": [
            0.4098,
            0.2706,
            0.3039,
            0.2549,
        ],

        "notes": [
            "The -90 and +90 deg bins represent the same orientation class in 180-deg orientation space.",
            "The original figure displays the wraparound bin split across both plot edges.",
            "Only the wrapped orientation-space representation is stored here.",
            "Use wrapped_proportion_of_cells_evaluated for plotting/analysis unless expression provenance is needed.",
        ],
    },
    "sosi_distribution": {
        "x_label": "second-order orientation selectivity index",
        "y_label": "proportion of cells",

        "area": "V1",
        "n": 61,

        # Digitized from article panel A, V1.
        # This is already normal histogram data, no circular edge weirdness.
        "bin_centers": [
            0.05,
            0.14,
            0.23,
            0.32,
            0.41,
            0.50,
            0.58,
        ],
        "proportion_of_cells": [
            0.37450980392156863,
            0.2666666666666666,
            0.15098039215686274,
            0.08431372549019601,
            0.0686274509803921,
            0.03725490196078422,
            0.03725490196078422,
        ],

        "xlim": [0.0, 0.70],

        "notes": [
            "Hallum et al. Fig. 3A V1 article panel.",
            "Digitized article bar heights; raw cell-level SOSI values are not available.",
            "Use only as scraped experimental histogram comparison.",
        ],
    },

}