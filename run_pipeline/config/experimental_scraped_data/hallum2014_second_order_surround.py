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
}