yeh2009_black_dominance_scraped_data = {
    "histogram": {
        "source": "scraped_article",
        "panel": "C",
        "x_label": "log(SNR_white / SNR_black)",
        "y_label": "number of cells",

        "bin_edges": [
            -1.8, -1.5, -1.2, -0.9, -0.6, -0.3,
             0.0,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8,
        ],

        "bin_centers": [
            -1.65, -1.35, -1.05, -0.75, -0.45, -0.15,
             0.15,  0.45,  0.75,  1.05,  1.35,  1.65,
        ],

        "dominance_convention": {
            "negative": "black-dominant",
            "positive": "white-dominant",
            "zero_line": "equal SNR_white and SNR_black",
        },

        "layers": {
            "L2/3": {
                "n_cells_reported": 69,
                "counts": [1, 6, 7, 11, 21, 18, 4, 0, 1, 0, 0, 0],
            },
            "L4A/B": {
                "n_cells_reported": 49,
                "counts": [0, 0, 0, 3, 9, 21, 13, 2, 1, 0, 0, 0],
            },
            "L4C": {
                "n_cells_reported": 44,
                "counts": [0, 0, 2, 4, 5, 16, 9, 3, 3, 2, 0, 0],
            },
            "L5/6": {
                "n_cells_reported": 13,
                "counts": [0, 2, 1, 1, 2, 2, 4, 1, 0, 0, 0, 0],
            },
        },
    },

"scatter": {
    "source": "scraped_article",
    "panel": "A-B",
    "x_label": "SNR_white",
    "y_label": "SNR_black",

    "axis_limits": {
        "x_display_range": [0, 30],
        "y_display_range": [0, 30],
        "overflow_display_value": 35.0,
    },

    "notes": [
        "Values are digitized from scatter points.",
        "The original axes display 0-30 plus an overflow bin for values >30.",
        "Digitized values above 30 are not exact SNR values; they indicate the top-coded >30 display bin.",
        "All digitized values above 30 were stored as 35.0 to preserve ordering without fake precision.",
    ],

    "marginal_histogram_bins": [0, 5, 10, 15, 20, 25, 30, 35],

    "layers": {
        "L2/3": {
            "n_points_digitized": 51,
            "n_topcoded_over_30": 6,

            "SNR_white": [
                19.392, 12.662, 4.677, 3.536, 1.711, 0.684,
                1.369, 3.308, 15.057, 2.053, 2.624, 0.570,
                1.711, 1.027, 4.221, 9.696, 10.266, 12.091,
                13.802, 3.080, 2.395, 0.684, 2.624, 3.194,
                10.380, 7.985, 3.422, 0.913, 1.483, 3.992,
                3.422, 1.027, 2.167, 4.791, 6.046, 6.616,
                5.133, 3.080, 1.255, 1.255, 3.080, 1.597,
                1.483, 1.255, 0.798, 2.624, 2.852, 3.878,
                4.677, 6.274, 3.650,
            ],

            "SNR_black": [
                35.0, 35.0, 35.0, 35.0, 35.0, 35.0,
                29.881, 29.075, 28.118, 25.201, 23.602, 23.040,
                22.465, 21.783, 22.113, 24.260, 21.292, 20.943,
                21.278, 19.152, 17.557, 16.651, 16.416, 15.501,
                16.044, 13.088, 12.991, 12.202, 11.972, 10.137,
                9.112, 7.867, 7.634, 7.853, 8.304, 6.705,
                5.228, 6.376, 6.497, 5.699, 5.121, 4.443,
                3.302, 2.163, 1.936, 2.728, 2.499, 3.179,
                2.492, 1.231, 7.743,
            ],
        },

        "L4C": {
            "n_points_digitized": 35,
            "n_topcoded_over_30": 1,

            "SNR_white": [
                5.418, 2.739, 11.676, 5.377, 9.459, 10.590,
                12.073, 26.934, 23.425, 23.077, 18.649, 15.255,
                15.137, 11.416, 9.501, 7.117, 9.022, 5.510,
                1.546, 0.639, 3.593, 3.823, 4.842, 5.744,
                6.874, 7.783, 0.992, 0.997, 1.910, 3.387,
                5.882, 2.146, 1.914, 0.894, 1.011,
            ],

            "SNR_black": [
                35.0, 24.308, 25.865, 15.207, 13.373, 13.710,
                10.750, 2.512, 2.411, 4.799, 8.225, 7.669,
                8.806, 4.843, 2.350, 4.178, 9.170, 9.865,
                10.335, 10.679, 8.055, 7.031, 7.027, 8.046,
                8.497, 7.698, 7.155, 5.905, 4.084, 2.601,
                1.455, 1.356, 2.834, 3.178, 2.269,
            ],
        },
    }
}

}