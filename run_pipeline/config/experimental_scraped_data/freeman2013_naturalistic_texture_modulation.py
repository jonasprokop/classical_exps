freeman2013_naturalistic_texture_modulation_scraped_data = {
    "texture_family_modulation": {
        "source": "scraped_article",
        "panel": "e",
        "x_label": "texture family",
        "y_label": "modulation index",

        "texture_family": list(range(1, 16)),

        "null_distribution": {
            "description": (
                "Gray band indicates 2.5th and 97.5th percentiles "
                "of the null distribution expected by chance."
            ),
            "percentiles": [2.5, 97.5],
        },

        "notes": [
            "Bars are mean modulation indices across neurons.",
            "Error bars are s.e.m. across neurons.",
            "Values digitized from the renormalized WPD project.",
        ],

        "V2": {
            "modulation_index": [
                0.2463, 0.2379, 0.2224, 0.2081, 0.2021,
                0.1746, 0.1436, 0.1078, 0.1018, 0.0684,
                0.0540, 0.0493, 0.0158, -0.0104, -0.0248,
            ],
            "errorbar_endpoint_low": [
                0.2128, 0.2057, 0.1985, 0.1818, 0.1806,
                0.1424, 0.1161, 0.0815, 0.0684, 0.0373,
                0.0254, 0.0230, -0.0128, -0.0403, -0.0534,
            ],
            "errorbar_endpoint_high": [
                0.2761, 0.2690, 0.2475, 0.2319, 0.2272,
                0.2104, 0.1746, 0.1340, 0.1340, 0.0958,
                0.0791, 0.0731, 0.0457, -0.0391, -0.0510,
            ],
            "summary": {
                "max_positive": {
                    "texture_family": 1,
                    "value": 0.2463,
                },
                "min_negative": {
                    "texture_family": 15,
                    "value": -0.0248,
                },
            },
        },

        "V1": {
            "modulation_index": [
                0.0087, 0.0863, -0.0116, 0.0457, 0.0325,
                0.0409, -0.0128, 0.0290, 0.0433, -0.0069,
                -0.0212, -0.0355, -0.0857, -0.0522, -0.0415,
            ],
            "errorbar_endpoint_low": [
                -0.0104, 0.0660, 0.0099, 0.0242, 0.0122,
                0.0194, 0.0063, 0.0099, 0.0194, 0.0206,
                0.0051, -0.0140, -0.0618, -0.0284, -0.0200,
            ],
            "errorbar_endpoint_high": [
                0.0278, 0.1078, -0.0307, 0.0684, 0.0528,
                0.0624, -0.0343, 0.0469, 0.0672, -0.0296,
                -0.0463, -0.0534, -0.1107, -0.0737, -0.0630,
            ],
            "summary": {
                "max_positive": {
                    "texture_family": 2,
                    "value": 0.0863,
                },
                "min_negative": {
                    "texture_family": 13,
                    "value": -0.0857,
                },
            },
        },
    },

    "v1_modulation_index_distribution": {
        "source": "scraped_article",
        "panel": "f",
        "area": "V1",
        "n_cells_reported": 102,

        "x_label": "modulation index",
        "y_label": "proportion of cells",

        "notes": [
            "Digitized as XY points from histogram bar tops.",
            "Bin centers are empirical digitized x-positions; exact original bin edges are not reconstructed.",
            "Proportions sum to approximately 1; small deviation is digitization noise.",
        ],

        "bin_centers_digitized": [
            -0.4194,
            -0.2386,
            -0.1819,
            -0.1231,
            -0.0599,
            -0.0011,
             0.0599,
             0.1209,
             0.1776,
             0.2386,
             0.3584,
        ],

        "proportion_of_cells": [
            0.0092,
            0.0184,
            0.0292,
            0.0384,
            0.1574,
            0.4215,
            0.2150,
            0.0691,
            0.0100,
            0.0192,
            0.0107,
        ],

        "summary": {
            "max_proportion": {
                "modulation_index": -0.0011,
                "proportion": 0.4215,
            },
            "most_negative_bin": {
                "modulation_index": -0.4194,
                "proportion": 0.0092,
            },
            "most_positive_bin": {
                "modulation_index": 0.3584,
                "proportion": 0.0107,
            },
        },
    },
}