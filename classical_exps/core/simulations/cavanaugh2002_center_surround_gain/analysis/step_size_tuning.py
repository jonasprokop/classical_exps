############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.core.tools.filtering_functions import *
from classical_exps.core.tools.utils import *
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import (
    _extract_size_tuning_markers_core,
)
from .shared import plot_scatter_hist, neuron_key, ensure_dir
from .data_size_tuning import load_size_tuning_results_bulk
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.core.tools.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy
from datetime import datetime
import json
import pandas as pd
import os
import openpyxl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D



plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "font.size": 10,
    "legend.frameon": False,
})

BLACK = "black"
DARK = "0.25"   # circular / main line
MID = "0.60"    # keep only if needed elsewhere
LIGHT = "0.88"
WHITE = "white"

CIRC_COLOR = BLACK
ANN_COLOR = "0.55"   # visibly different, still article-safe
MARKER_COLOR = BLACK
SUPPRESSION_COLOR = BLACK
GSF_COLOR = "0.45"
SURR_COLOR = BLACK
AMRF_COLOR = "0.70"

ARTICLE_COLOR = "black"
MODEL_COLOR = "#2F5D8C"



def size_tuning_results_1(
    *,
    loaded,
    h5_file,
    neuron_ids,
    filtered_neuron_ids,
    overwrite=False,
    run_mode="both",          # "raw" | "fit" | "both"
    plot_mode="both",         # "none" | "raw" | "fit" | "both"
    plot_dir="/project/results/nature_and_interactions/size_tuning/plots/",
    max_plot_neurons=None,
    plot_only_filtered=False,
    show_suppression=True,
    experimental_data_scraped=None,
    example_overlay_pairs=[("neuron_A", 0)],
    fit_err_thresh=None,
    supp_thresh=None,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)

    if "size_tuning_curves_active" not in loaded:
        raise KeyError("loaded missing 'size_tuning_curves_active'")

    st = loaded["size_tuning_curves_active"]
    radii = np.asarray(st["radii"], float)
    by_curve = st["by_id"]


    fits_by = loaded.get("size_tuning_fits", {}).get("by_id", {})

    group_path = "/size_tuning"
    subgroup_raw = group_path + "/results"
    subgroup_fit = group_path + "/results_fit"

    # decide which neurons get plotted
    plot_ids = filtered_neuron_ids if plot_only_filtered else neuron_ids
    if max_plot_neurons is not None:
        plot_ids = np.asarray(plot_ids[:int(max_plot_neurons)], dtype=int)
    plot_id_set = set(map(int, plot_ids))

    if overwrite:
        if run_mode in ("raw", "both"):
            clear_group(h5_file, subgroup_raw)
        if run_mode in ("fit", "both"):
            clear_group(h5_file, subgroup_fit)

    if run_mode in ("raw", "both"):
        group_init(h5_file=h5_file, group_path=subgroup_raw, group_args_str="")

    if run_mode in ("fit", "both"):
        group_init(h5_file=h5_file, group_path=subgroup_fit, group_args_str="")

    plot_mode_map = {
        "none": "none",
        "raw": "article",
        "fit": "compare",
        "both": "both",
    }
    plotter_mode = plot_mode_map[plot_mode]

    # ---------------------------------------
    # Article-faithful population splits
    # ---------------------------------------
    # Fig. 2-like scatter population:
    #   finite GSF + finite surround extent + SI > 0.10
    all_raw_scatter_GSF, all_raw_scatter_surr = [], []
    all_fit_scatter_GSF, all_fit_scatter_surr = [], []

    # Fig. 3-like SI histogram population:
    #   all neurons with either suppression OR saturation
    #   (i.e. excluding only "neither suppression nor saturation")
    all_raw_si_hist = []
    all_fit_si_hist = []

    # Diagnostics
    no_suppression = []
    no_saturation = []
    no_supp_asymptote = []
    excluded_no_supp_no_sat = []


    with h5py.File(h5_file, "a") as f:
        grp_raw = f.require_group(subgroup_raw) if run_mode in ("raw", "both") else None
        grp_fit = f.require_group(subgroup_fit) if run_mode in ("fit", "both") else None

        for nid in neuron_ids:
            nid = int(nid)

            if nid not in by_curve:
                continue

            circ = np.asarray(by_curve[nid]["circular"], float)
            ann = np.asarray(by_curve[nid].get("annular"), float) if "annular" in by_curve[nid] else None

            raw_analysis_markers = None
            raw_display_markers = None
            fit_analysis_markers = None
            fit_display_markers = None

            circular_fit = None
            annular_fit = None

            # ---------------------------------------
            # RAW metrics
            # ---------------------------------------
            if run_mode in ("raw", "both"):
                d_raw = _extract_size_tuning_markers_core(
                    radii=radii,
                    circular_curve=circ,
                    annular_curve=ann,
                    min_valid_radius=1e-9,
                    require_article_exclusion=True,
                )

                raw_analysis_markers, _ = build_analysis_markers(
                    d_raw,
                    has_annulus=(ann is not None),
                )

                raw_display_markers, raw_case_display = build_display_markers(
                    d_raw,
                    has_annulus=(ann is not None),
                    style="article_raw",
                )

                # article-style display diagnostics (caption logic)
                if raw_case_display == "B":
                    no_suppression.append(nid)
                elif raw_case_display == "C":
                    no_saturation.append(nid)
                elif raw_case_display == "D":
                    no_supp_asymptote.append(nid)

                # explicit paper exclusion tracking
                if (
                    (not bool(d_raw.get("suppression_present", False)))
                    and (not bool(d_raw.get("saturation_present", False)))
                ):
                    excluded_no_supp_no_sat.append(nid)

                k = neuron_key(nid)
                if k in grp_raw:
                    del grp_raw[k]

                grp_raw.create_dataset(
                    k,
                    data=np.asarray([
                        raw_analysis_markers["GSF"],
                        raw_analysis_markers["surround_extent"],
                        raw_analysis_markers["AMRF"],
                        raw_analysis_markers["SI"],
                        raw_analysis_markers["Ropt"],
                        raw_analysis_markers["Rsupp"],
                    ], float)
                )

                grp_raw[k].attrs["article_case"] = "full" if raw_case_display is None else raw_case_display
                grp_raw[k].attrs["suppression_present"] = bool(d_raw.get("suppression_present", False))
                grp_raw[k].attrs["saturation_present"] = bool(d_raw.get("saturation_present", False))
                grp_raw[k].attrs["suppression_asymptote_reached"] = bool(d_raw.get("suppression_asymptote_reached", False))
                grp_raw[k].attrs["valid"] = bool(d_raw.get("valid", False))

                # ---- Fig. 3 SI histogram pool ----
                if (
                    raw_analysis_markers is not None
                    and np.isfinite(raw_analysis_markers["SI"])
                    and (
                        bool(d_raw.get("suppression_present", False))
                        or bool(d_raw.get("saturation_present", False))
                    )
                ):
                    all_raw_si_hist.append(raw_analysis_markers["SI"])

                # ---- Fig. 2 scatter pool ----
                if (
                    raw_analysis_markers is not None
                    and np.isfinite(raw_analysis_markers["GSF"])
                    and np.isfinite(raw_analysis_markers["surround_extent"])
                    and np.isfinite(raw_analysis_markers["SI"])
                    and (raw_analysis_markers["SI"] > 0.10)
                ):
                    all_raw_scatter_GSF.append(raw_analysis_markers["GSF"])
                    all_raw_scatter_surr.append(raw_analysis_markers["surround_extent"])

            # ---------------------------------------
            # FIT metrics
            # ---------------------------------------
            if run_mode in ("fit", "both") and nid in fits_by:
                fit_blob = fits_by[nid]

                circular_fit = np.asarray(fit_blob.get("circular_fit", circ), float)
                if ann is not None:
                    annular_fit = np.asarray(fit_blob.get("annular_fit", ann), float)

                d_fit = _extract_size_tuning_markers_core(
                    radii=radii,
                    circular_curve=circular_fit,
                    annular_curve=annular_fit,
                    min_valid_radius=1e-9,
                    require_article_exclusion=False,
                )

                fit_analysis_markers, _ = build_analysis_markers(
                    d_fit,
                    has_annulus=(annular_fit is not None),
                )

                fit_display_markers, _ = build_display_markers(
                    d_fit,
                    has_annulus=(annular_fit is not None),
                    style="fit_compare",
                )

                k = neuron_key(nid)
                if k in grp_fit:
                    del grp_fit[k]

                grp_fit.create_dataset(
                    k,
                    data=np.asarray([
                        fit_analysis_markers["GSF"],
                        fit_analysis_markers["surround_extent"],
                        fit_analysis_markers["AMRF"],
                        fit_analysis_markers["SI"],
                        fit_analysis_markers["Ropt"],
                        fit_analysis_markers["Rsupp"],
                    ], float)
                )

                # ---- Fit SI histogram pool ----
                if (
                    fit_analysis_markers is not None
                    and np.isfinite(fit_analysis_markers["SI"])
                    and (
                        bool(d_fit.get("suppression_present", False))
                        or bool(d_fit.get("saturation_present", False))
                    )
                ):
                    all_fit_si_hist.append(fit_analysis_markers["SI"])

                # ---- Fit scatter pool ----
                if (
                    fit_analysis_markers is not None
                    and np.isfinite(fit_analysis_markers["GSF"])
                    and np.isfinite(fit_analysis_markers["surround_extent"])
                    and np.isfinite(fit_analysis_markers["SI"])
                    and (fit_analysis_markers["SI"] > 0.10)
                ):
                    all_fit_scatter_GSF.append(fit_analysis_markers["GSF"])
                    all_fit_scatter_surr.append(fit_analysis_markers["surround_extent"])

            # ---------------------------------------
            # Plotting
            # ---------------------------------------
            if (plot_mode != "none") and (nid in plot_id_set):
                plot_size_tuning_curve_with_markers(
                    neuron=str(nid),
                    radii=radii,
                    circular_curve_raw=circ,
                    annular_curve_raw=ann,
                    circular_curve_fit=circular_fit,
                    annular_curve_fit=annular_fit,
                    raw_markers=raw_display_markers,
                    fit_markers=fit_display_markers,
                    save_dir=plot_dir,
                    use_diameter=True,
                    plot_mode=plotter_mode,
                    show_suppression=show_suppression,
                )


    shared_model_support = None
    if len(all_raw_scatter_GSF) > 0 and len(all_raw_scatter_surr) > 0:
        shared_model_support = build_shared_model_support(
            2.0 * np.asarray(all_raw_scatter_GSF, dtype=float),
            2.0 * np.asarray(all_raw_scatter_surr, dtype=float),
            round_decimals=2,
        )

    
    # ---------------------------------------
    # Experimental scraped data branch
    # ---------------------------------------
    scraped_curve_payloads = {}
    if experimental_data_scraped is not None:
        scraped_plot_dir = os.path.join(plot_dir, "experimental_scraped")
        os.makedirs(scraped_plot_dir, exist_ok=True)

        subgroup_scraped = group_path + "/results_experimental_scraped"
        subgroup_scraped_pop = group_path + "/population_experimental_scraped"

        if overwrite:
            clear_group(h5_file, subgroup_scraped)
            clear_group(h5_file, subgroup_scraped_pop)

        group_init(h5_file=h5_file, group_path=subgroup_scraped, group_args_str="")
        group_init(h5_file=h5_file, group_path=subgroup_scraped_pop, group_args_str="")

        scraped_size = experimental_data_scraped.get("size_tunning", experimental_data_scraped)

        with h5py.File(h5_file, "a") as f:
            grp_scraped = f.require_group(subgroup_scraped)
            grp_scraped_pop = f.require_group(subgroup_scraped_pop)

            # ---------------------------------------
            # Store paper-scraped population summaries
            # ---------------------------------------
            diam_scatter = experimental_data_scraped.get("diametr_scatter", {})
            si_hist = experimental_data_scraped.get("SI_histogram", {})

            for k_src, k_dst in (
                ("gsf", "gsf"),
                ("surround_diametr", "surround_diametr"),
            ):
                arr = np.asarray(diam_scatter.get(k_src, []), dtype=float)
                if k_dst in grp_scraped_pop:
                    del grp_scraped_pop[k_dst]
                grp_scraped_pop.create_dataset(k_dst, data=arr)

            bins_raw = si_hist.get("bins", [])
            props = np.asarray(si_hist.get("SI_proportions", []), dtype=float)

            if "si_bin_edges" in grp_scraped_pop:
                del grp_scraped_pop["si_bin_edges"]
            if "si_proportions" in grp_scraped_pop:
                del grp_scraped_pop["si_proportions"]

            if len(bins_raw) > 0:
                edges = np.array([bins_raw[0][0]] + [b[1] for b in bins_raw], dtype=float)
                grp_scraped_pop.create_dataset("si_bin_edges", data=edges)

            grp_scraped_pop.create_dataset("si_proportions", data=props)
            grp_scraped_pop.attrs["source"] = "experimental_data_scraped"
            grp_scraped_pop.attrs["note"] = "Population summaries digitized from article"

            # ---------------------------------------
            # Store / plot scraped example neurons
            # ---------------------------------------
            n_scraped_examples = 0

            for neuron_scraped, tuning_data in scraped_size.items():
                classical = tuning_data.get("Classical_tunning", {})
                annular = tuning_data.get("Annular_tunning", {})

                circ_x = np.asarray(classical.get("X", []), float)
                circ_y = np.asarray(classical.get("Y", []), float)

                ann_x = np.asarray(annular.get("X", []), float)
                ann_y = np.asarray(annular.get("Y", []), float)

                # keep duplicates, only sort by X so downstream extraction behaves sensibly
                if circ_x.size == 0 or circ_y.size == 0:
                    raise ValueError(
                        f"experimental_data_scraped['size_tunning']['{neuron_scraped}']['Classical_tunning'] is missing X/Y data"
                    )

                m_c = np.isfinite(circ_x) & np.isfinite(circ_y)
                circ_x = circ_x[m_c]
                circ_y = circ_y[m_c]

                order_c = np.argsort(circ_x)
                circ_x = circ_x[order_c]
                circ_y = circ_y[order_c]

                # we are plotting in radii, and scraping diameters so we must devide 
                circ_x = circ_x * 0.5
                if ann_x.size > 0:
                    ann_x = ann_x * 0.5

                ann_curve = None
                use_annulus = False
                radii_for_scraped = circ_x

                if ann_x.size > 0 and ann_y.size > 0:
                    m_a = np.isfinite(ann_x) & np.isfinite(ann_y)
                    ann_x = ann_x[m_a]
                    ann_y = ann_y[m_a]

                    order_a = np.argsort(ann_x)
                    ann_x = ann_x[order_a]
                    ann_y = ann_y[order_a]

                    if ann_y.size != circ_y.size:
                        raise ValueError(
                            f"experimental_data_scraped['size_tunning']['{neuron_scraped}']: "
                            f"Classical_tunning and Annular_tunning must have same number of points "
                            f"when using the circular grid as the shared axis "
                            f"(got {circ_y.size} vs {ann_y.size})"
                        )

                    # Cavanaugh-style assumption:
                    # circular and annular conditions live on the same intended diameter grid.
                    # For scraped data, X mismatches are treated as digitization noise.
                    # We therefore keep the circular grid as the shared axis and only reuse annular Y.
                    ann_curve = ann_y
                    use_annulus = True

                d_scraped = _extract_size_tuning_markers_core(
                    radii=radii_for_scraped,
                    circular_curve=circ_y,
                    annular_curve=ann_curve if use_annulus else None,
                    min_valid_radius=1e-9,
                    require_article_exclusion=True,
                )

                scraped_analysis_markers, _ = build_analysis_markers(
                    d_scraped,
                    has_annulus=use_annulus,
                )

                scraped_display_markers, scraped_case_display = build_display_markers(
                    d_scraped,
                    has_annulus=use_annulus,
                    style="article_raw",
                )

                scraped_curve_payloads[neuron_scraped] = {
                    "label": f"Experimental {neuron_scraped}",
                    "radii": np.asarray(radii_for_scraped, dtype=float),
                    "circular": np.asarray(circ_y, dtype=float),
                    "annular": None if ann_curve is None else np.asarray(ann_curve, dtype=float),
                    "markers": scraped_display_markers,
                }

                if neuron_scraped in grp_scraped:
                    del grp_scraped[neuron_scraped]

                ds = grp_scraped.create_dataset(
                    neuron_scraped,
                    data=np.asarray([
                        scraped_analysis_markers["GSF"],
                        scraped_analysis_markers["surround_extent"],
                        scraped_analysis_markers["AMRF"],
                        scraped_analysis_markers["SI"],
                        scraped_analysis_markers["Ropt"],
                        scraped_analysis_markers["Rsupp"],
                    ], float)
                )

                ds.attrs["article_case"] = "full" if scraped_case_display is None else scraped_case_display
                ds.attrs["suppression_present"] = bool(d_scraped.get("suppression_present", False))
                ds.attrs["saturation_present"] = bool(d_scraped.get("saturation_present", False))
                ds.attrs["suppression_asymptote_reached"] = bool(d_scraped.get("suppression_asymptote_reached", False))
                ds.attrs["valid"] = bool(d_scraped.get("valid", False))
                ds.attrs["uses_shared_circular_grid"] = bool(use_annulus)

                # plot in article style only
                plot_size_tuning_curve_with_markers(
                    neuron=neuron_scraped,
                    radii=radii_for_scraped,
                    circular_curve_raw=circ_y,
                    annular_curve_raw=ann_curve if use_annulus else None,
                    circular_curve_fit=None,
                    annular_curve_fit=None,
                    raw_markers=scraped_display_markers,
                    fit_markers=None,
                    save_dir=scraped_plot_dir,
                    use_diameter=True,
                    plot_mode="article",
                    show_suppression=show_suppression,
                )

                n_scraped_examples += 1

        print("--------------------------------------")
        print("Experimental scraped size-tuning data")
        print(f"    > Stored/plotted {n_scraped_examples} scraped example neuron(s)")
        print(f"    > Population summaries stored in: {subgroup_scraped_pop}")
        print("--------------------------------------")
        print()

        st = loaded["size_tuning_curves_active"]
        radii_model = np.asarray(st["radii"], dtype=float)
        model_curves_by_id = st["by_id"]

        for scraped_id, model_nid in example_overlay_pairs:
            if scraped_id not in scraped_curve_payloads:
                print(f"Skipping overlay: scraped neuron {scraped_id!r} not available.")
                continue

            if model_nid not in model_curves_by_id:
                print(f"Skipping overlay: model neuron {model_nid} not available.")
                continue

            model_curve = model_curves_by_id[model_nid]

            model_payload = {
                "label": f"Model neuron {model_nid}",
                "radii": radii_model,
                "circular": np.asarray(model_curve["circular"], dtype=float),
                "annular": (
                    None
                    if "annular" not in model_curve
                    else np.asarray(model_curve["annular"], dtype=float)
                ),
                "markers": None,  # add later if wanted
            }

        st = loaded["size_tuning_curves_active"]
        radii_model = np.asarray(st["radii"], dtype=float)
        model_curves_by_id = st["by_id"]

        def _validate_overlay_payload(payload, *, name, require_suppression=True):
            if payload is None:
                raise ValueError(f"{name} overlay payload is None.")

            for key in ("radii", "circular"):
                if key not in payload:
                    raise KeyError(f"{name} overlay payload missing {key!r}.")

            radii_arr = np.asarray(payload["radii"], dtype=float)
            circ_arr = np.asarray(payload["circular"], dtype=float)

            if radii_arr.size < 2 or circ_arr.size < 2:
                raise ValueError(f"{name} overlay payload has too few points.")

            if payload.get("markers", None) is None:
                raise ValueError(f"{name} overlay payload has markers=None.")

            markers = payload["markers"]

            needed = ["GSF", "surround_extent"]
            if require_suppression:
                needed += ["Ropt", "Rsupp"]

            missing = []
            for k in needed:
                try:
                    v = float(markers.get(k, np.nan))
                except Exception:
                    v = np.nan

                if not np.isfinite(v):
                    missing.append(k)

            if missing:
                raise ValueError(
                    f"{name} overlay markers missing/non-finite values: {missing}. "
                    f"markers={markers}"
                )

        for scraped_id, model_nid in example_overlay_pairs:
            if scraped_id not in scraped_curve_payloads:
                print(f"Skipping overlay: scraped neuron {scraped_id!r} not available.")
                continue

            model_nid = int(model_nid)

            if model_nid not in model_curves_by_id:
                print(f"Skipping overlay: model neuron {model_nid} not available.")
                continue

            model_curve = model_curves_by_id[model_nid]

            model_circ = np.asarray(model_curve["circular"], dtype=float)

            model_ann = (
                None
                if "annular" not in model_curve
                else np.asarray(model_curve["annular"], dtype=float)
            )

            # Build model markers NOW, before the plot call.
            d_model_overlay = _extract_size_tuning_markers_core(
                radii=radii_model,
                circular_curve=model_circ,
                annular_curve=model_ann,
                min_valid_radius=1e-9,
                require_article_exclusion=False,
            )

            model_overlay_markers, _ = build_analysis_markers(
                d_model_overlay,
                has_annulus=(model_ann is not None),
            )

            model_payload = {
                "label": f"Model neuron {model_nid}",
                "radii": radii_model,
                "circular": model_circ,
                "annular": model_ann,
                "markers": model_overlay_markers,
            }

            # Use analysis markers for overlay, not display markers,
            # otherwise article-style cases can hide valid marker data.
            experimental_payload = dict(scraped_curve_payloads[scraped_id])

            _validate_overlay_payload(
                experimental_payload,
                name=f"Experiment {scraped_id}",
                require_suppression=True,
            )

            _validate_overlay_payload(
                model_payload,
                name=f"Model neuron {model_nid}",
                require_suppression=True,
            )

            print("--------------------------------------")
            print("Overlay payload check")
            print(f"    > Experiment: {scraped_id}")
            print(f"        markers: {experimental_payload['markers']}")
            print(f"    > Model: {model_nid}")
            print(f"        markers: {model_payload['markers']}")
            print("--------------------------------------")

            plot_size_tuning_curve_overlay_comparison(
                experimental_payload,
                model_payload,
                save_path=os.path.join(
                    plot_dir,
                    f"overlay_experimental_{scraped_id}_model_{model_nid}.svg",
                ),
                use_diameter=True,
                response_mode="minmax",
                show_markers=True,
                show_suppression=True,
            )
        _plot_scraped_population_summaries(
            experimental_data_scraped,
            save_root="/project/results/nature_and_interactions/size_tuning/",
            label="experimental_scraped",
            shared_support=shared_model_support,
        )
    
        # ---------------------------------------
        # Shared histogram axes (model vs scraped)
        # ---------------------------------------

        diam_scatter = experimental_data_scraped.get("diametr_scatter", {})
        gsf_d_scraped = np.asarray(diam_scatter.get("gsf", []), dtype=float)
        surr_d_scraped = np.asarray(diam_scatter.get("surround_diametr", []), dtype=float)

        gsf_d_model = 2.0 * np.asarray(all_raw_scatter_GSF, dtype=float)
        surr_d_model = 2.0 * np.asarray(all_raw_scatter_surr, dtype=float)

        # clean
        m_model = np.isfinite(gsf_d_model) & np.isfinite(surr_d_model) & (gsf_d_model > 0) & (surr_d_model > 0)
        gsf_d_model = gsf_d_model[m_model]
        surr_d_model = surr_d_model[m_model]

        m_scraped = np.isfinite(gsf_d_scraped) & np.isfinite(surr_d_scraped) & (gsf_d_scraped > 0) & (surr_d_scraped > 0)
        gsf_d_scraped = gsf_d_scraped[m_scraped]
        surr_d_scraped = surr_d_scraped[m_scraped]

        if gsf_d_model.size > 0 and gsf_d_scraped.size > 0:

            shared_bins, shared_ymax = build_shared_metric_hist_settings(
                model_gsf_d=gsf_d_model,
                model_surr_d=surr_d_model,
                scraped_gsf_d=gsf_d_scraped,
                scraped_surr_d=surr_d_scraped,
                include_ratio=True,
                n_log_bins=4,
            )

            # --- model plot ---
            plot_size_metric_histograms(
                gsf_d_model,
                surr_d_model,
                save_path="/project/results/nature_and_interactions/size_tuning/metrics_hist_model_shared.png",
                include_ratio=True,
                n_log_bins=4,
                show_titles=False,
                article_style=True,
                shared_bins=shared_bins,
                shared_ymax=shared_ymax,
            )

            # --- scraped plot ---
            plot_size_metric_histograms(
                gsf_d_scraped,
                surr_d_scraped,
                save_path="/project/results/nature_and_interactions/size_tuning/metrics_hist_scraped_shared.png",
                include_ratio=True,
                n_log_bins=4,
                show_titles=False,
                article_style=True,
                shared_bins=shared_bins,
                shared_ymax=shared_ymax,
            )

    # ---------------------------------------
    # Summary scatter(s)
    # ---------------------------------------
    def _scatter_summary(gsf_values, surround_values, si_hist_values, label, shared_support=None):
        gsf_values = np.asarray(gsf_values, float)
        surround_values = np.asarray(surround_values, float)
        si_hist_values = np.asarray(si_hist_values, float)

        # Fig. 2-like scatter subset
        valid_scatter = (
            np.isfinite(gsf_values)
            & np.isfinite(surround_values)
            & (gsf_values > 0)
            & (surround_values > 0)
        )
        gsf_values = gsf_values[valid_scatter]
        surround_values = surround_values[valid_scatter]

        # Fig. 3-like SI histogram subset
        si_hist_values = si_hist_values[np.isfinite(si_hist_values)]

        if gsf_values.size == 0 or surround_values.size == 0:
            print("--------------------------------------")
            print(f"GSF vs surround extent ({label})")
            print(f"    > Analysis made on {len(neuron_ids)} neurons")
            print("    > 0 neurons with finite GSF+surround extent")
            print("--------------------------------------\n")
            return

        gsf_diameters = 2.0 * gsf_values
        surround_diameters = 2.0 * surround_values
        diameter_ratio = surround_diameters / gsf_diameters

        plot_size_metric_histograms(
            gsf_diameters,
            surround_diameters,
            save_path=f"/project/results/nature_and_interactions/size_tuning/metrics_hist_{label}.png",
            title_prefix="",
            include_ratio=True,
            n_log_bins=4,
            show_titles=False,
            article_style=True,
        )

        plot_paired_binned_heatmaps(
            model_x=gsf_diameters,
            model_y=surround_diameters,
            scraped_x=gsf_diameters,
            scraped_y=surround_diameters,
            save_path=f"/project/results/nature_and_interactions/size_tuning/gsf_vs_surround_heatmap_{label}.png",
            x_label="GSF diameter (deg)",
            y_label="Surround diameter (deg)",
            title_left="",
            title_right="",
            figure_title="",
            n_bins_x=5,
            n_bins_y=5,
            edge_source="combined",
            annotate=True,
            normalize=False,
            use_power_norm=True,
            power_gamma=0.5,
            show_titles=False,
            article_style=True,
        )

        n_scatter = len(gsf_diameters)
        n_si = len(si_hist_values)

        gmean_gsf = float(np.exp(np.mean(np.log(gsf_diameters))))
        gmean_surround = float(np.exp(np.mean(np.log(surround_diameters))))
        gmean_ratio = float(np.exp(np.mean(np.log(diameter_ratio))))
        si_mean = float(np.mean(si_hist_values)) if n_si > 0 else np.nan

        print("--------------------------------------")
        print(f"GSF vs surround extent ({label})")
        print(f"    > Analysis made on {len(neuron_ids)} neurons")
        print(f"    > Scatter subset: {n_scatter} neurons with finite GSF+surround extent and SI > 0.10")
        print(f"    > SI histogram subset: {n_si} neurons with either suppression or saturation")
        print(
            f"    > gmean(GSF_diam)={gmean_gsf:.3f}  "
            f"gmean(surr_diam)={gmean_surround:.3f}  "
            f"gmean(ratio)={gmean_ratio:.3f}  "
            f"mean(SI)={si_mean:.3f}"
        )
        print("    > Plot :")

        plot_scatter_hist(
            x=gsf_diameters,
            y=surround_diameters,
            si_values=si_hist_values,
            x_gmean=gmean_gsf,
            y_gmean=gmean_surround,
            si_mean=si_mean,
            ratio_gmean=gmean_ratio,
            title="",
            x_label="GSF diameter (deg)",
            y_label="Surround diameter (deg)",
            scatter_save_path=f"/project/results/nature_and_interactions/size_tuning/gsf_vs_surround_{label}.svg",
            si_save_path=f"/project/results/nature_and_interactions/size_tuning/si_distribution_{label}.svg",
            ratio_save_path=f"/project/results/nature_and_interactions/size_tuning/ratio_distribution_{label}.svg",
        )
        print("--------------------------------------\n")

        if shared_support is not None:
            print(
                f"    > mean snap error: "
                f"GSF={mean_snap_error(gsf_diameters, shared_support['gsf_vals']):.3f}, "
                f"Surround={mean_snap_error(surround_diameters, shared_support['surr_vals']):.3f}"
            )

    if run_mode in ("raw", "both"):
        _scatter_summary(
            all_raw_scatter_GSF,
            all_raw_scatter_surr,
            all_raw_si_hist,
            "raw",
            shared_support=shared_model_support,
        )

    if run_mode in ("fit", "both"):
        _scatter_summary(
            all_fit_scatter_GSF,
            all_fit_scatter_surr,
            all_fit_si_hist,
            "fit",
            shared_support=shared_model_support,
        )

    if experimental_data_scraped is not None and shared_model_support is not None:
        if experimental_data_scraped is not None:
            _plot_model_vs_scraped_paired_heatmap(
                model_gsf_d=2.0 * np.asarray(all_raw_scatter_GSF, dtype=float),
                model_surr_d=2.0 * np.asarray(all_raw_scatter_surr, dtype=float),
                experimental_data_scraped=experimental_data_scraped,
                save_root="/project/results/nature_and_interactions/size_tuning/",
                label="",
                n_bins_x=5,
                n_bins_y=5,
            )

    if experimental_data_scraped is not None:
        si_hist = experimental_data_scraped.get("SI_histogram", {})
        bins_raw = si_hist.get("bins", [])
        article_props = np.asarray(si_hist.get("SI_proportions", []), dtype=float)

        if len(bins_raw) > 0 and article_props.size > 0 and len(all_raw_si_hist) > 0:
            article_edges = np.array(
                [bins_raw[0][0]] + [b[1] for b in bins_raw],
                dtype=float,
            )

        plot_si_histogram_comparison(
            article_bin_edges=article_edges,
            article_proportions=article_props,
            model_values=np.asarray(all_raw_si_hist, dtype=float),
            save_path="/project/results/nature_and_interactions/size_tuning/si_distribution_comparison.svg",
            article_label="Experiment",
            model_label="Model",
        )

        si_hist = experimental_data_scraped.get("SI_histogram", {})
        bins_raw = si_hist.get("bins", [])
        article_props = np.asarray(si_hist.get("SI_proportions", []), dtype=float)

        article_si_edges = None
        if len(bins_raw) > 0 and article_props.size > 0:
            article_si_edges = np.array(
                [bins_raw[0][0]] + [b[1] for b in bins_raw],
                dtype=float,
            )

        plot_shared_metric_histogram_comparison(
            model_gsf_d=gsf_d_model,
            model_surr_d=surr_d_model,
            scraped_gsf_d=gsf_d_scraped,
            scraped_surr_d=surr_d_scraped,
            shared_bins=shared_bins,
            shared_ymax=shared_ymax, 
            save_path="/project/results/nature_and_interactions/size_tuning/metrics_hist_shared_comparison.svg",
            model_si=np.asarray(all_raw_si_hist, dtype=float),
            scraped_si_bin_edges=article_si_edges,
            scraped_si_proportions=article_props,
            include_ratio=True,
            include_si=True,
            article_style=True,
            show_titles=False,
        )

        print("--------------------------------------")
        print("SI histogram comparison and metric histogram comparison")



    # ---------------------------------------
    # Diagnostics: neurons lacking parameters
    # ---------------------------------------
    def _print_case(label, arr):
        arr = np.asarray(arr, dtype=int)
        print(f"{label}: {len(arr)} neurons")
        if arr.size > 0:
            preview = ", ".join(map(str, arr[:10]))
            if arr.size > 10:
                preview += ", ..."
            print(f"    ids: {preview}")

    print("======================================")
    print("Size tuning parameter availability")
    print(f"Total neurons analysed: {len(neuron_ids)}")
    _print_case("No suppression (B)", no_suppression)
    _print_case("No response saturation (C)", no_saturation)
    _print_case("No suppression asymptote (D)", no_supp_asymptote)
    _print_case("Neither suppression nor saturation (paper-excluded)", excluded_no_supp_no_sat)
    print("======================================")
    print()

    # ---------------------------------------
    # Refresh loaded payload
    # ---------------------------------------
    loaded["size_results_raw"] = load_size_tuning_results_bulk(h5_file, neuron_ids, strict=False)

    loaded["size_results"] = loaded["size_results_raw"]

    if experimental_data_scraped is not None:
        loaded["size_results_experimental_scraped"] = {
            "group_path": "/size_tuning/results_experimental_scraped",
            "population_group_path": "/size_tuning/population_experimental_scraped",
        }

    return loaded


    
def size_tuning_results_2(
    neuron_ids,
    filtered_neuron_ids=None,   
    size_results=None,          
):
    
    ''' This function aims to visualize the distribution of the suppression index in the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean SI ...)
            - 3) It plots an histogram of the neuron SI values

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - see 'size_tuning_results_1'
    '''
    
    if filtered_neuron_ids is None:
        raise ValueError("size_tuning_results_2 requires filtered_neuron_ids (computed once in relay).")
    if size_results is None or "by_id" not in size_results:
        raise ValueError("size_tuning_results_2 requires size_results from bulk loader.")

    filtered_neuron_ids = np.asarray(filtered_neuron_ids, dtype=int)
    by_id = size_results["by_id"]

    # For each neuron, get SI (legacy index [-1], now via loader)
    all_SI = []
    for neuron_id in filtered_neuron_ids:
        d = by_id.get(int(neuron_id))
        if d is None:
            raise RuntimeError(
                f"Neuron {neuron_id} missing in size_results loader. "
                "Loader/filter mismatch — fix upstream."
            )
        all_SI.append(d["SI"])

    all_SI = np.array(all_SI)

    all_SI = np.asarray(all_SI, dtype=float)
    all_SI = all_SI[np.isfinite(all_SI)]

    n = len(neuron_ids)
    n_new = len(all_SI)

    if n_new == 0:
        print("--------------------------------------")
        print("Distribution of the Suppression Index :")
        print(f"    > Analysis made on {n} neurons")
        print("    > 0 neurons left after filtration")
        print("    > No finite SI values to plot")
        print("--------------------------------------")
        print()
        return

    mean_SI = round(float(np.mean(all_SI)), 2)
    max_SI = round(float(np.max(all_SI)), 1)
    bins = np.linspace(0, max(max_SI, 1), 6)

    # Show the results (unchanged)
    print("--------------------------------------")
    print("Distribution of the Suppression Index :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean value of the SI is {mean_SI}")
    print(f"    > Plot :")

    weights = np.ones(all_SI.shape, dtype=float) / len(all_SI)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.hist(
        all_SI,
        bins=bins,
        weights=weights,
        edgecolor=BLACK,
        facecolor=LIGHT,
        linewidth=1.2,
    )

    ax.set_xlabel("Suppression Index (SI)")
    ax.set_ylabel("Proportion of cells")
    ax.set_xticks(bins)

    if False:  # keep title off for article-style consistency
        ax.set_title(f"Distribution of the SI for the {len(all_SI)} neurons")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
    ax.grid(False)

    directory = "/project/results/nature_and_interactions/"
    os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "distribution_of_si.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("--------------------------------------")
    print()

def _draw_article_suppression(ax, x, markers, *, use_diameter=True, color=SUPPRESSION_COLOR):
    """
    Draw suppression exactly in the article sense:
    vertical reduction from Ropt to Rsupp, positioned near the large-stimulus region.
    """
    if markers is None:
        return

    try:
        Ropt = float(markers.get("Ropt", np.nan))
        Rsupp = float(markers.get("Rsupp", np.nan))
    except Exception:
        return

    if not (np.isfinite(Ropt) and np.isfinite(Rsupp)):
        return
    if Ropt <= Rsupp:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    xr = max(x_max - x_min, 1e-12)

    # Put the arrow in the large-stimulus region, preferably at surround extent
    x_arrow = None
    try:
        surr = float(markers.get("surround_extent", np.nan))
        if np.isfinite(surr) and surr >= 0:
            surr_x = 2.0 * surr if use_diameter else surr
            x_arrow = surr_x - 0.23 * xr   # small right offset from surround extent
    except Exception:
        pass

    # fallback: right side of the plot, but not at the edge
    if x_arrow is None or not np.isfinite(x_arrow):
        x_arrow = x_min + 0.82 * xr

    x_arrow = float(np.clip(x_arrow, x_min + 0.55 * xr, x_max - 0.23 * xr))

    cap = 0.02 * xr

    ax.annotate(
        "",
        xy=(x_arrow, Ropt),
        xytext=(x_arrow, Rsupp),
        arrowprops=dict(
            arrowstyle="<->",
            lw=1.4,
            color=color,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )

    ax.plot([x_arrow - cap, x_arrow + cap], [Ropt, Ropt], color=color, lw=1.1, zorder=6)
    ax.plot([x_arrow - cap, x_arrow + cap], [Rsupp, Rsupp], color=color, lw=1.1, zorder=6)

    ax.text(
        x_arrow + 0.02 * xr,
        0.5 * (Ropt + Rsupp),
        "Suppression",
        ha="left",
        va="center",
        fontsize=15,
        color=color,
        zorder=6,
    )

def _safe_float_or_nan(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def classify_article_case(d):
    """
    Figure-caption style display cases.
    """
    supp = bool(d.get("suppression_present", False))
    sat = bool(d.get("saturation_present", False))
    asym = bool(d.get("suppression_asymptote_reached", False))

    if not supp:
        return "B"
    elif not sat:
        return "C"
    elif not asym:
        return "D"
    else:
        return None


def _base_marker_dict(d, *, has_annulus):
    return {
        "GSF": _safe_float_or_nan(d.get("GSF")),
        "surround_extent": _safe_float_or_nan(d.get("surround_extent")),
        "AMRF": _safe_float_or_nan(d.get("AMRF")) if has_annulus else np.nan,
        "SI": _safe_float_or_nan(d.get("SI")),
        "Ropt": _safe_float_or_nan(d.get("Ropt")),
        "Rsupp": _safe_float_or_nan(d.get("Rsupp")),
    }


def _apply_marker_policy(markers, keep):
    """
    keep: dict like {"GSF": True, "surround_extent": False, ...}
    Any missing key defaults to True.
    """
    out = dict(markers)
    for k in ("GSF", "surround_extent", "AMRF", "SI", "Ropt", "Rsupp"):
        if not keep.get(k, True):
            out[k] = np.nan
    return out


ARTICLE_ANALYSIS_POLICY = {
    None: {"GSF": True,  "surround_extent": True,  "AMRF": True,  "SI": True,  "Ropt": True, "Rsupp": True},
    "B":  {"GSF": True,  "surround_extent": False, "AMRF": False, "SI": False, "Ropt": True, "Rsupp": False},
    "C":  {"GSF": False, "surround_extent": False, "AMRF": False, "SI": False, "Ropt": True, "Rsupp": False},
    "D":  {"GSF": True,  "surround_extent": True,  "AMRF": True,  "SI": True,  "Ropt": True, "Rsupp": True},
}

ARTICLE_DISPLAY_POLICY = {
    None: {"GSF": True,  "surround_extent": True,  "AMRF": True,  "SI": True,  "Ropt": True, "Rsupp": True},
    "B":  {"GSF": False, "surround_extent": False, "AMRF": False, "SI": False, "Ropt": False, "Rsupp": False},
    "C":  {"GSF": False, "surround_extent": False, "AMRF": False, "SI": False, "Ropt": False, "Rsupp": False},
    "D":  {"GSF": False, "surround_extent": False, "AMRF": False, "SI": False, "Ropt": False, "Rsupp": False},
}

FIT_DISPLAY_POLICY = {
    None: {"GSF": True, "surround_extent": True, "AMRF": True, "SI": True, "Ropt": True, "Rsupp": True},
    "B":  {"GSF": True, "surround_extent": True, "AMRF": True, "SI": True, "Ropt": True, "Rsupp": True},
    "C":  {"GSF": True, "surround_extent": True, "AMRF": True, "SI": True, "Ropt": True, "Rsupp": True},
    "D":  {"GSF": True, "surround_extent": True, "AMRF": True, "SI": True, "Ropt": True, "Rsupp": True},
}


def build_analysis_markers(d, *, has_annulus):
    """
    Markers for storage / population analysis.

    This uses the underlying boolean state, not the article-style
    display case, because article cases are illustrative and can
    collapse overlapping failure modes.
    """
    supp = bool(d.get("suppression_present", False))
    sat = bool(d.get("saturation_present", False))
    asym = bool(d.get("suppression_asymptote_reached", False))

    markers = _base_marker_dict(d, has_annulus=has_annulus)

    # Paper-excluded case:
    # neither suppression nor response saturation -> cannot estimate CRF extent
    if (not supp) and (not sat):
        markers["GSF"] = np.nan
        markers["surround_extent"] = np.nan
        markers["AMRF"] = np.nan
        markers["SI"] = np.nan
        markers["Rsupp"] = np.nan
        return markers, "C"

    # No suppression, but response saturates:
    # keep GSF, but surround-suppression quantities are not meaningful
    if not supp:
        markers["surround_extent"] = np.nan
        markers["SI"] = np.nan
        markers["Rsupp"] = np.nan
        markers["AMRF"] = np.nan
        return markers, "B"

    # Suppression present, but asymptote not reached:
    # paper keeps surround extent as largest stimulus presented
    if not asym:
        # keep GSF, surround_extent, SI, Rsupp
        # AMRF can stay if annular curve supports it
        return markers, "D"

    return markers, None


def build_display_markers(d, *, has_annulus, style="article_raw"):
    """
    Markers for plotting only.

    style:
        - 'article_raw' : raw article-style example plots
        - 'fit_compare' : fit diagnostic comparison plots
    """
    case = classify_article_case(d)
    markers = _base_marker_dict(d, has_annulus=has_annulus)

    if style == "article_raw":
        # In raw article-style plots, hide all markers for rejected/partial cases
        if case in ("B", "C", "D"):
            for k in ("GSF", "surround_extent", "AMRF", "SI", "Ropt", "Rsupp"):
                markers[k] = np.nan

    elif style == "fit_compare":
        # Diagnostic plots should show everything
        pass

    else:
        raise ValueError(f"Unknown display marker style: {style}")

    return markers, case

def _plot_scraped_population_summaries(
    experimental_data_scraped,
    *,
    save_root="/project/results/nature_and_interactions/size_tuning/",
    label="experimental_scraped",
    shared_support=None,
):
    os.makedirs(save_root, exist_ok=True)

    diam_scatter = experimental_data_scraped.get("diametr_scatter", {})
    si_hist = experimental_data_scraped.get("SI_histogram", {})

    gsf_d = np.asarray(diam_scatter.get("gsf", []), dtype=float)
    surr_d = np.asarray(diam_scatter.get("surround_diametr", []), dtype=float)

    m = np.isfinite(gsf_d) & np.isfinite(surr_d) & (gsf_d > 0) & (surr_d > 0)
    gsf_d = gsf_d[m]
    surr_d = surr_d[m]
    plot_size_metric_histograms(
        gsf_d,
        surr_d,
        save_path=os.path.join(save_root, f"metrics_hist_{label}.png"),
        title_prefix="",
        include_ratio=True,
        n_log_bins=4,
        show_titles=False,
        article_style=True,
    )

    bins_raw = si_hist.get("bins", [])
    props = np.asarray(si_hist.get("SI_proportions", []), dtype=float)

    si_mean = np.nan
    if len(bins_raw) > 0 and props.size > 0:
        bins_arr = np.asarray(bins_raw, dtype=float)
        if bins_arr.ndim != 2 or bins_arr.shape[1] != 2:
            raise ValueError(
                f"Scraped SI histogram bins must have shape (n_bins, 2), got {bins_arr.shape}"
            )
        if len(props) != len(bins_arr):
            raise ValueError(
                f"Scraped SI histogram mismatch: {len(props)} proportions for {len(bins_arr)} bins"
            )

        centers = 0.5 * (bins_arr[:, 0] + bins_arr[:, 1])
        if np.sum(props) > 0:
            si_mean = float(np.sum(centers * props) / np.sum(props))

    if gsf_d.size == 0 or surr_d.size == 0:
        print("--------------------------------------")
        print(f"GSF vs surround extent ({label})")
        print("    > No valid scraped diameter scatter data")
        print("--------------------------------------")
        print()
        return

    ratio = surr_d / gsf_d
    gmean_GSF = float(np.exp(np.mean(np.log(gsf_d))))
    gmean_surr = float(np.exp(np.mean(np.log(surr_d))))
    gmean_ratio = float(np.exp(np.mean(np.log(ratio))))

    plot_binned_2d_heatmap(
        gsf_d,
        surr_d,
        x_edges=np.logspace(np.log10(np.min(gsf_d)), np.log10(np.max(gsf_d)), 4),
        y_edges=np.logspace(np.log10(np.min(surr_d)), np.log10(np.max(surr_d)), 4),
        x_label="GSF diameter (deg)",
        y_label="Surround diameter (deg)",
        title="",
        save_path=os.path.join(save_root, f"gsf_vs_surround_heatmap_{label}.png"),
        annotate=True,
        normalize=False,
        use_power_norm=True,
        power_gamma=0.5,
        show_title=False,
        article_style=True,
    )

    print("--------------------------------------")
    print(f"GSF vs surround extent ({label})")
    print(f"    > Scraped scatter subset: {len(gsf_d)} points")
    print(
        f"    > gmean(GSF_diam)={gmean_GSF:.3f}  "
        f"gmean(surr_diam)={gmean_surr:.3f}  "
        f"gmean(ratio)={gmean_ratio:.3f}  "
        f"mean(SI)={si_mean:.3f}"
    )
    print("    > Plot :")

    plot_scatter_hist(
        x=gsf_d,
        y=surr_d,
        si_values=None,
        si_hist_bins=np.asarray(bins_raw, dtype=float) if len(bins_raw) > 0 else None,
        si_hist_weights=props if props.size > 0 else None,
        x_gmean=gmean_GSF,
        y_gmean=gmean_surr,
        si_mean=si_mean,
        ratio_gmean=gmean_ratio,
        title="",
        x_label="GSF diameter (deg)",
        y_label="surround diameter (deg)",
        scatter_save_path=os.path.join(save_root, f"gsf_vs_surround_{label}.png"),
        si_save_path=os.path.join(save_root, f"si_distribution_{label}.png"),
        ratio_save_path=os.path.join(save_root, f"ratio_distribution_{label}.png"),
    )

    print("--------------------------------------")
    print()


def _finalize_legend(ax, *, outside=True):
    handles, labels = ax.get_legend_handles_labels()

    # remove duplicates while preserving order
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    if not uniq_h:
        return

    if outside:
        ax.legend(
            uniq_h,
            uniq_l,
            frameon=False,
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            handlelength=2.2,
        )
    else:
        ax.legend(
            uniq_h,
            uniq_l,
            frameon=False,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            handlelength=2.2,
            columnspacing=1.2,
        )

def plot_size_tuning_curve_with_markers(
    neuron: str,
    radii,
    circular_curve_raw,
    annular_curve_raw=None,
    circular_curve_fit=None,
    annular_curve_fit=None,
    *,
    raw_markers=None,     # dict: {"GSF":..., "surround_extent":..., "AMRF":..., "SI":...}
    fit_markers=None,     # same keys
    save_dir="/project/results/nature_and_interactions/size_tuning/plots/",
    use_diameter=True,
    plot_mode="both",     # "both" | "article" | "compare" | "none"
    show_annular=True,
    show_suppression=True,
    show_titles=False,
    article_style=True,
):
    """
    Creates up to two plots:

    1) Article plot:
        - raw POINTS
        - RAW markers only

    2) Compare plot:
        - raw POINTS
        - fit CURVES (if available)
        - FIT markers only

    Returns:
        (path_article_or_None, path_compare_or_None)
    """

    ensure_dir(save_dir)

    if plot_mode not in ("both", "article", "compare", "none"):
        raise ValueError(f"Invalid plot_mode: {plot_mode}")

    if circular_curve_raw is None:
        return None, None

    radii = np.asarray(radii, dtype=float)
    circ_raw = np.asarray(circular_curve_raw, dtype=float)

    if radii.size < 2 or circ_raw.size < 2:
        return None, None

    n = min(len(radii), len(circ_raw))
    radii = radii[:n]
    circ_raw = circ_raw[:n]

    x = 2.0 * radii if use_diameter else radii
    xlab = "Diameter (deg)" if use_diameter else "Radius (deg)"

    ann_raw = None
    if annular_curve_raw is not None and show_annular:
        ann_raw = np.asarray(annular_curve_raw, dtype=float)[:n]

    circ_fit = None
    if circular_curve_fit is not None:
        circ_fit = np.asarray(circular_curve_fit, dtype=float)[:n]

    ann_fit = None
    if annular_curve_fit is not None and show_annular:
        ann_fit = np.asarray(annular_curve_fit, dtype=float)[:n]

    # -------------------------
    # poster sizing
    # -------------------------
    FIGSIZE = (10.5, 6.2)
    LABELSIZE = 22
    TICKSIZE = 15
    TITLESIZE = 22
    LEGENDSIZE = 20
    LINEWIDTH = 2.2
    FITLINEWIDTH = 2.4
    MARKERSIZE = 7.0
    TICKLEN = 6
    TICKWIDTH = 1.4
    MARKER_LW = 1.8

    def _mark(ax, val, *, label=None, ls="--", lw=MARKER_LW, color=MID):
        if val is None:
            return
        try:
            v = float(val)
        except Exception:
            return
        if not np.isfinite(v) or v < 0:
            return
        xv = 2.0 * v if use_diameter else v
        ax.axvline(xv, lw=lw, ls=ls, color=color, zorder=4, label=label)

    def _finalize_legend(ax, *, outside=True):
        handles, labels = ax.get_legend_handles_labels()

        # remove duplicates, preserve order
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)

        if not uniq_h:
            return

        if outside:
            ax.legend(
                uniq_h,
                uniq_l,
                frameon=False,
                fontsize=LEGENDSIZE,
                loc="center left",
                bbox_to_anchor=(1.005, 0.5),
                borderaxespad=0.0,
                handlelength=2.0,
                handletextpad=0.5,
                labelspacing=0.4,
                borderpad=0.2,
            )
        else:
            ax.legend(
                uniq_h,
                uniq_l,
                frameon=False,
                fontsize=LEGENDSIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=2,
                handlelength=2.0,
                handletextpad=0.5,
                columnspacing=1.0,
                labelspacing=0.4,
                borderpad=0.2,
            )

    want_article = plot_mode in ("both", "article")
    want_compare = plot_mode in ("both", "compare")

    has_fit = (circ_fit is not None) or (ann_fit is not None)
    path_article = None
    path_compare = None

    # -------------------------
    # 1) ARTICLE PLOT (RAW)
    # -------------------------
    if want_article:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.plot(
            x, circ_raw,
            marker="o", markersize=MARKERSIZE,
            mfc=CIRC_COLOR, mec=CIRC_COLOR,
            color=CIRC_COLOR, ls="-", lw=LINEWIDTH,
            label="Circular"
        )
        if ann_raw is not None:
            ax.plot(
                x, ann_raw,
                marker="o", markersize=MARKERSIZE,
                mfc=WHITE, mec=ANN_COLOR,
                color=ANN_COLOR, ls="-", lw=LINEWIDTH,
                label="Annular"
            )

        if raw_markers is not None:
            _mark(ax, raw_markers.get("GSF"), label="GSF", ls="--", lw=MARKER_LW, color=GSF_COLOR)
            _mark(ax, raw_markers.get("surround_extent"), label="Surround extent", ls=":", lw=MARKER_LW, color=SURR_COLOR)
            _mark(ax, raw_markers.get("AMRF"), label="AMRF", ls="-.", lw=MARKER_LW, color=AMRF_COLOR)

            if show_suppression:
                _draw_article_suppression(
                    ax,
                    x,
                    raw_markers,
                    use_diameter=use_diameter,
                    color=SUPPRESSION_COLOR,
                )

        ax.set_xlabel(xlab, fontsize=LABELSIZE)
        ax.set_ylabel("Response magnitude", fontsize=LABELSIZE)
        if show_titles:
            ax.set_title("Size tuning", fontsize=TITLESIZE)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=TICKLEN,
            width=TICKWIDTH,
            labelsize=TICKSIZE,
        )
        ax.grid(False)

        ax.set_xlim(left=0)

        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        _finalize_legend(ax, outside=True)

        path_article = os.path.join(save_dir, f"{neuron}_size_tuning_article.png")
        fig.subplots_adjust(right=0.78)
        fig.savefig(path_article, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # 2) COMPARE PLOT (FIT)
    # -------------------------
    if want_compare and has_fit:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.plot(
            x, circ_raw,
            marker="o", markersize=MARKERSIZE,
            mfc=CIRC_COLOR, mec=CIRC_COLOR,
            color=CIRC_COLOR, ls="None",
            label="Circular raw"
        )
        if ann_raw is not None:
            ax.plot(
                x, ann_raw,
                marker="o", markersize=MARKERSIZE,
                mfc=WHITE, mec=ANN_COLOR,
                color=ANN_COLOR, ls="None",
                label="Annular raw"
            )

        if circ_fit is not None:
            ax.plot(x, circ_fit, lw=FITLINEWIDTH, ls="-", color=CIRC_COLOR, label="Circular fit")
        if ann_fit is not None:
            ax.plot(x, ann_fit, lw=FITLINEWIDTH, ls="--", color=ANN_COLOR, label="Annular fit")

        if fit_markers is not None:
            _mark(ax, fit_markers.get("GSF"), label="GSF", ls="--", lw=MARKER_LW, color=GSF_COLOR)
            _mark(ax, fit_markers.get("surround_extent"), label="Surround extent", ls=":", lw=MARKER_LW, color=SURR_COLOR)
            _mark(ax, fit_markers.get("AMRF"), label="AMRF", ls="-.", lw=MARKER_LW, color=AMRF_COLOR)

        ax.set_xlabel(xlab, fontsize=LABELSIZE)
        ax.set_ylabel("Response magnitude", fontsize=LABELSIZE)
        if show_titles:
            ax.set_title("Size tuning", fontsize=TITLESIZE)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=TICKLEN,
            width=TICKWIDTH,
            labelsize=TICKSIZE,
        )
        ax.grid(False)

        _finalize_legend(ax, outside=True)

        path_compare = os.path.join(save_dir, f"{neuron}_size_tuning_compare.png")
        fig.subplots_adjust(right=0.78)
        fig.savefig(path_compare, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return path_article, path_compare


def plot_binned_2d_heatmap(
    x,
    y,
    *,
    x_edges,
    y_edges,
    x_label="GSF diameter (deg)",
    y_label="Surround diameter (deg)",
    title="",
    save_path=None,
    show=False,
    close=True,
    annotate=True,
    normalize=False,
    use_power_norm=True,
    power_gamma=0.5,
    show_title=False,
    article_style=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]

    if x.size == 0:
        raise ValueError("No valid x/y values for heatmap.")

    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    H = H.T

    if normalize and H.sum() > 0:
        H = H / H.sum()

    vmax = H.max() if H.max() > 0 else 1.0
    norm = PowerNorm(gamma=power_gamma, vmin=0, vmax=vmax) if use_power_norm else None

    x_centers = np.sqrt(np.asarray(x_edges[:-1]) * np.asarray(x_edges[1:]))
    y_centers = np.sqrt(np.asarray(y_edges[:-1]) * np.asarray(y_edges[1:]))

    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    im = ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        norm=norm,
        vmin=None if norm is not None else 0,
        vmax=None if norm is not None else vmax,
    )

    ax.set_xticks(np.arange(len(x_centers)))
    ax.set_yticks(np.arange(len(y_centers)))
    ax.set_xticklabels([f"{v:.2f}" for v in x_centers], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.2f}" for v in y_centers])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_title and title:
        ax.set_title(title)

    if article_style:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion of cells" if normalize else "Count")

    if annotate:
        thresh = 0.5 * vmax
        for iy in range(H.shape[0]):
            for ix in range(H.shape[1]):
                val = H[iy, ix]
                if val > 0:
                    txt = f"{100*val:.0f}%" if normalize else f"{int(val)}"
                    color = "white" if val >= thresh else "black"
                    ax.text(ix, iy, txt, ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    elif close:
        plt.close(fig)

def plot_size_metric_histograms(
    gsf_d,
    surr_d,
    *,
    save_path,
    title_prefix="",
    include_ratio=True,
    n_log_bins=4,
    show_titles=False,
    article_style=True,
    shared_bins=None,   # optional dict: {"gsf": ..., "surr": ..., "ratio": ...}
    shared_ymax=None,   # optional float
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    def _clean_positive(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0)]

    def _make_log_bins(vals, n_bins):
        vals = _clean_positive(vals)
        if vals.size == 0:
            raise ValueError("No positive finite values for log bins.")

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            return np.array([0.8 * vmin, 1.2 * vmax], dtype=float)

        return np.logspace(np.log10(vmin), np.log10(vmax), int(n_bins) + 1)

    def _style_log_x(ax, bins):
        ax.set_xscale("log")
        ax.set_xlim(float(bins[0]), float(bins[-1]))
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{b:.1f}" for b in bins], rotation=45, ha="right")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="minor", length=0)

    def _plot_hist(ax, vals, bins, *, xlabel, title=""):
        vals = _clean_positive(vals)
        weights = np.ones_like(vals) / len(vals)
        widths = np.diff(bins)

        hist, _ = np.histogram(vals, bins=bins, weights=weights)

        ax.bar(
            bins[:-1],
            hist,
            width=widths,
            align="edge",
            facecolor=LIGHT,
            edgecolor=BLACK,
            linewidth=1.2,
            alpha=1.0,
            zorder=2,
        )

        _style_log_x(ax, bins)
        ax.set_xlabel(xlabel)

        if show_titles:
            ax.set_title(f"{title_prefix} {title}".strip())

        if article_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=4, width=1)
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.2)

        return float(np.max(hist)) if hist.size else 0.0

    gsf_d = _clean_positive(gsf_d)
    surr_d = _clean_positive(surr_d)

    if gsf_d.size == 0 or surr_d.size == 0:
        raise ValueError("No valid GSF/surround values for histogram plotting.")

    ratio = np.array([], dtype=float)
    if include_ratio:
        ratio = _clean_positive(surr_d / gsf_d)

    panels = [
        ("gsf", "GSF diameter (deg)", "GSF", gsf_d),
        ("surr", "Surround diameter (deg)", "Surround", surr_d),
    ]
    if include_ratio and ratio.size > 0:
        panels.append(("ratio", "Surround / GSF", "Ratio", ratio))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.3))
    if len(panels) == 1:
        axes = [axes]

    y_max = 0.0

    for ax, (key, xlabel, short_title, vals) in zip(axes, panels):
        if shared_bins is not None and key in shared_bins:
            bins = np.asarray(shared_bins[key], dtype=float)
        else:
            bins = _make_log_bins(vals, n_log_bins)

        panel_ymax = _plot_hist(
            ax,
            vals,
            bins,
            xlabel=xlabel,
            title=short_title,
        )
        y_max = max(y_max, panel_ymax)

    final_ymax = float(shared_ymax) if shared_ymax is not None else max(1.08 * y_max, 0.05)

    for ax in axes:
        ax.set_ylim(0, final_ymax)

    axes[0].set_ylabel("Proportion of cells")
    axes[0].set_yticks(np.linspace(0, final_ymax, 4))

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(wspace=0.15)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def build_shared_model_support(
    gsf_d_model,
    surr_d_model,
    *,
    round_decimals=2,
):
    """
    Build shared discrete supports from the normal/model population.

    Returns
    -------
    support : dict with keys
        - gsf_vals
        - surr_vals
        - ratio_vals
        - gsf_edges
        - surr_edges
        - ratio_edges
    """
    gsf_d_model = np.asarray(gsf_d_model, dtype=float)
    surr_d_model = np.asarray(surr_d_model, dtype=float)

    m = np.isfinite(gsf_d_model) & np.isfinite(surr_d_model) & (gsf_d_model > 0) & (surr_d_model > 0)
    gsf_d_model = gsf_d_model[m]
    surr_d_model = surr_d_model[m]

    if gsf_d_model.size == 0 or surr_d_model.size == 0:
        raise ValueError("No valid model diameter scatter values for shared support.")

    gsf_vals = np.unique(np.round(gsf_d_model, round_decimals))
    surr_vals = np.unique(np.round(surr_d_model, round_decimals))
    ratio_vals = np.unique(np.round(surr_d_model / gsf_d_model, round_decimals))

    return {
        "gsf_vals": gsf_vals,
        "surr_vals": surr_vals,
        "ratio_vals": ratio_vals,
        "gsf_edges": discrete_value_edges(gsf_vals),
        "surr_edges": discrete_value_edges(surr_vals),
        "ratio_edges": discrete_value_edges(ratio_vals),
    }


def discrete_value_edges(vals):
    """
    Build histogram edges centered around discrete support values.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.unique(vals)

    if vals.size == 0:
        raise ValueError("No values for edge construction.")

    if vals.size == 1:
        step = max(abs(vals[0]) * 0.1, 0.05)
        return np.array([vals[0] - step, vals[0] + step], dtype=float)

    mids = 0.5 * (vals[:-1] + vals[1:])
    left = vals[0] - (mids[0] - vals[0])
    right = vals[-1] + (vals[-1] - mids[-1])

    return np.concatenate([[left], mids, [right]])


def snap_to_support(vals, support_vals):
    vals = np.asarray(vals, dtype=float)
    support_vals = np.asarray(support_vals, dtype=float)

    if support_vals.size == 0:
        raise ValueError("support_vals is empty.")

    out = np.full(vals.shape, np.nan, dtype=float)
    m = np.isfinite(vals)

    if np.any(m):
        d = np.abs(vals[m][:, None] - support_vals[None, :])
        idx = np.argmin(d, axis=1)
        out[m] = support_vals[idx]

    return out


def fmt_support_labels(vals, decimals=2):
    return [f"{float(v):.{decimals}f}" for v in vals]


def mean_snap_error(vals, support_vals):
    vals = np.asarray(vals, dtype=float)
    snapped = snap_to_support(vals, support_vals)
    m = np.isfinite(vals) & np.isfinite(snapped)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(vals[m] - snapped[m])))
def plot_paired_binned_heatmaps(
    model_x,
    model_y,
    scraped_x,
    scraped_y,
    *,
    save_path,
    x_label="GSF diameter (deg)",
    y_label="Surround diameter (deg)",
    title_left="",
    title_right="",
    figure_title="",
    n_bins_x=5,
    n_bins_y=5,
    edge_source="combined",   # kept for compatibility; now used only to choose support source
    annotate=True,
    normalize=True,          # False=count, True=proportion
    use_power_norm=True,
    power_gamma=0.5,
    show_titles=False,
    article_style=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    def _clean(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return x[valid], y[valid]

    model_x, model_y = _clean(model_x, model_y)
    scraped_x, scraped_y = _clean(scraped_x, scraped_y)

    if model_x.size == 0 or model_y.size == 0:
        raise ValueError("No valid model values.")
    if scraped_x.size == 0 or scraped_y.size == 0:
        raise ValueError("No valid scraped values.")

    if edge_source == "model":
        x_source = model_x
        y_source = model_y
    elif edge_source == "combined":
        x_source = np.concatenate([model_x, scraped_x])
        y_source = np.concatenate([model_y, scraped_y])
    else:
        raise ValueError(f"Unknown edge_source: {edge_source}")

    x_support = geometric_quantile_support(x_source, n_levels=n_bins_x)
    y_support = geometric_quantile_support(y_source, n_levels=n_bins_y)

    model_x_q, model_y_q = snap_xy_to_support(model_x, model_y, x_support, y_support)
    scraped_x_q, scraped_y_q = snap_xy_to_support(scraped_x, scraped_y, x_support, y_support)

    H_model = count_on_support_grid(
        model_x_q, model_y_q, x_support, y_support, normalize=normalize
    )
    H_scraped = count_on_support_grid(
        scraped_x_q, scraped_y_q, x_support, y_support, normalize=normalize
    )

    vmax = max(H_model.max(), H_scraped.max())
    if vmax <= 0:
        vmax = 1.0

    norm = PowerNorm(gamma=power_gamma, vmin=0, vmax=vmax) if use_power_norm else None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)

    ims = []
    for ax, H, ttl in zip(axes, [H_model, H_scraped], [title_left, title_right]):
        im = ax.imshow(
            H,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            norm=norm,
            vmin=None if norm is not None else 0,
            vmax=None if norm is not None else vmax,

        )
        ims.append(im)

        ax.set_xticks(np.arange(len(x_support)))
        ax.set_yticks(np.arange(len(y_support)))
        ax.set_xticklabels([f"{v:.2f}" for v in x_support], rotation=45, ha="right")
        ax.set_yticklabels([f"{v:.2f}" for v in y_support])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if show_titles and ttl:
            ax.set_title(ttl)

        if article_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=4, width=1)

        if annotate:
            thresh = 0.5 * vmax
            for iy in range(H.shape[0]):
                for ix in range(H.shape[1]):
                    val = H[iy, ix]
                    if val > 0:
                        txt = f"{100*val:.0f}%" if normalize else f"{int(val)}"
                        color = "white" if val >= thresh else "black"
                        ax.text(ix, iy, txt, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.95)
    cbar.set_label("Proportion of cells" if normalize else "Count")

    if show_titles and figure_title:
        fig.suptitle(figure_title)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_model_vs_scraped_paired_heatmap(
    model_gsf_d,
    model_surr_d,
    experimental_data_scraped,
    *,
    save_root,
    label="raw_vs_scraped",
    n_bins_x=3,
    n_bins_y=3,
):
    diam_scatter = experimental_data_scraped.get("diametr_scatter", {})
    gsf_d = np.asarray(diam_scatter.get("gsf", []), dtype=float)
    surr_d = np.asarray(diam_scatter.get("surround_diametr", []), dtype=float)

    m = np.isfinite(gsf_d) & np.isfinite(surr_d) & (gsf_d > 0) & (surr_d > 0)
    gsf_d = gsf_d[m]
    surr_d = surr_d[m]

    model_gsf_d = np.asarray(model_gsf_d, dtype=float)
    model_surr_d = np.asarray(model_surr_d, dtype=float)

    m_model = np.isfinite(model_gsf_d) & np.isfinite(model_surr_d) & (model_gsf_d > 0) & (model_surr_d > 0)
    model_gsf_d = model_gsf_d[m_model]
    model_surr_d = model_surr_d[m_model]

    if model_gsf_d.size == 0 or model_surr_d.size == 0 or gsf_d.size == 0 or surr_d.size == 0:
        print(f"[{label}] Skipping paired heatmap: missing valid data.")
        return

    plot_paired_binned_heatmaps(
        model_x=model_gsf_d,
        model_y=model_surr_d,
        scraped_x=gsf_d,
        scraped_y=surr_d,
        save_path=os.path.join(save_root, f"paired_binned_heatmap_{label}.png"),
        x_label="GSF diameter (deg)",
        y_label="Surround diameter (deg)",
        title_left="",
        title_right="",
        figure_title="",
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        edge_source="combined",
        annotate=True,
        normalize=True,
        use_power_norm=True,
        power_gamma=0.5,
        show_titles=False,
        article_style=True,
    )

    plot_model_vs_scraped_binned_1d(
        model_gsf_d=model_gsf_d,
        model_surr_d=model_surr_d,
        scraped_gsf_d=gsf_d,
        scraped_surr_d=surr_d,
        save_path=os.path.join(save_root, f"binned_1d_compare_{label}.png"),
        n_bins=5,
        article_style=True,
        show_titles=False,
    )

def geometric_quantile_support(vals, n_levels=5):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        raise ValueError("No positive finite values for support construction.")

    if vals.size == 1:
        return vals.copy()

    q = np.linspace(0.0, 1.0, int(n_levels))
    log_vals = np.log(vals)
    support = np.exp(np.quantile(log_vals, q))
    support = np.unique(np.asarray(support, dtype=float))

    # if quantiles collapse because of discreteness, pad from unique values
    if support.size < n_levels:
        unique_vals = np.unique(vals)
        if unique_vals.size <= n_levels:
            support = unique_vals
        else:
            q2 = np.linspace(0.0, 1.0, int(n_levels))
            idx = np.round(q2 * (len(unique_vals) - 1)).astype(int)
            support = unique_vals[idx]

    return np.asarray(support, dtype=float)


def snap_xy_to_support(x, y, x_support, y_support):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_support = np.asarray(x_support, dtype=float)
    y_support = np.asarray(y_support, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return np.array([]), np.array([])

    x_snapped = snap_to_support(x, x_support)
    y_snapped = snap_to_support(y, y_support)
    return x_snapped, y_snapped


def count_on_support_grid(x, y, x_support, y_support, normalize=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_support = np.asarray(x_support, dtype=float)
    y_support = np.asarray(y_support, dtype=float)

    H = np.zeros((len(y_support), len(x_support)), dtype=float)

    if len(x) == 0:
        return H

    x_idx = np.argmin(np.abs(x[:, None] - x_support[None, :]), axis=1)
    y_idx = np.argmin(np.abs(y[:, None] - y_support[None, :]), axis=1)

    for ix, iy in zip(x_idx, y_idx):
        H[iy, ix] += 1.0

    if normalize and H.sum() > 0:
        H /= H.sum()

    return H
def plot_model_vs_scraped_binned_1d(
    model_gsf_d,
    model_surr_d,
    scraped_gsf_d,
    scraped_surr_d,
    *,
    save_path,
    n_bins=5,
    article_style=True,
    show_titles=False,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def _clean_positive(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0)]

    def _make_shared_log_bins(a, b, n_bins):
        vals = np.concatenate([_clean_positive(a), _clean_positive(b)])
        if vals.size == 0:
            raise ValueError("No valid values for shared bin construction.")

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            return np.array([0.8 * vmin, 1.2 * vmax], dtype=float)

        return np.logspace(np.log10(vmin), np.log10(vmax), int(n_bins) + 1)

    def _hist_prop(vals, bins):
        vals = _clean_positive(vals)
        if vals.size == 0:
            return np.zeros(len(bins) - 1, dtype=float)

        weights = np.ones(vals.shape, dtype=float) / len(vals)
        hist, _ = np.histogram(vals, bins=bins, weights=weights)
        return hist.astype(float)

    def _fmt_edge_tick(v):
        if v >= 10:
            return f"{v:.0f}"
        elif v >= 1:
            return f"{v:.1f}"
        else:
            return f"{v:.2f}"

    def _padded_log_xlim(bins, pad_frac=0.03):
        bins = np.asarray(bins, dtype=float)
        lo = float(bins[0])
        hi = float(bins[-1])

        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        span = max(log_hi - log_lo, 1e-12)

        return (
            10 ** (log_lo - pad_frac * span),
            10 ** (log_hi + pad_frac * span),
        )

    def _split_bin_geometry_log_half(bins, inner_pad_frac=0.04):
        """
        Split each parent bin into two equal halves IN LOG SPACE.
        Left half  -> experiment
        Right half -> model

        inner_pad_frac adds a tiny pad inside each half so bars do not
        touch each other or the bin boundaries.
        """
        bins = np.asarray(bins, dtype=float)

        exp_left = []
        exp_width = []
        model_left = []
        model_width = []

        for left, right in zip(bins[:-1], bins[1:]):
            ll = np.log(left)
            rr = np.log(right)
            mid = 0.5 * (ll + rr)

            left_span = mid - ll
            right_span = rr - mid

            # left half = experiment
            e_ll = ll + inner_pad_frac * left_span
            e_rr = mid - inner_pad_frac * left_span

            # right half = model
            m_ll = mid + inner_pad_frac * right_span
            m_rr = rr - inner_pad_frac * right_span

            el = np.exp(e_ll)
            er = np.exp(e_rr)
            ml = np.exp(m_ll)
            mr = np.exp(m_rr)

            exp_left.append(el)
            exp_width.append(er - el)
            model_left.append(ml)
            model_width.append(mr - ml)

        return (
            np.asarray(exp_left, dtype=float),
            np.asarray(exp_width, dtype=float),
            np.asarray(model_left, dtype=float),
            np.asarray(model_width, dtype=float),
        )

    # -------------------------
    # poster sizing
    # -------------------------
    FIGSIZE = (15.0, 5.2)
    LABELSIZE = 20
    TICKSIZE = 14
    TITLESIZE = 20
    LEGENDSIZE = 20
    BAR_LW = 1.6
    GRID_LW = 0.8
    VLINE_LW = 0.8
    TICKLEN = 6
    TICKWIDTH = 1.4

    EXP_COLOR = ARTICLE_COLOR
    MOD_COLOR = MODEL_COLOR

    def _draw_panel(ax, model_vals, scraped_vals, bins, xlabel, title=""):
        model_hist = _hist_prop(model_vals, bins)
        scraped_hist = _hist_prop(scraped_vals, bins)

        exp_left, exp_width, model_left, model_width = _split_bin_geometry_log_half(
            bins,
            inner_pad_frac=0.04,
        )

        # Experiment first, black, left half
        ax.bar(
            exp_left,
            scraped_hist,
            width=exp_width,
            align="edge",
            facecolor=EXP_COLOR,
            edgecolor=EXP_COLOR,
            linewidth=BAR_LW,
            alpha=0.90,
            label="Experiment",
            zorder=4,
        )

        # Model second, blue, right half
        ax.bar(
            model_left,
            model_hist,
            width=model_width,
            align="edge",
            facecolor=MOD_COLOR,
            edgecolor=MOD_COLOR,
            linewidth=BAR_LW,
            alpha=0.85,
            label="Model",
            zorder=3,
        )

        ax.set_xscale("log")
        xlo, xhi = _padded_log_xlim(bins, pad_frac=0.03)
        ax.set_xlim(xlo, xhi)

        # keep ticks on original bin edges
        ax.set_xticks(bins)
        ax.set_xticklabels([_fmt_edge_tick(b) for b in bins], rotation=45, ha="right")
        ax.minorticks_off()

        ax.set_xlabel(xlabel, fontsize=LABELSIZE)

        if show_titles and title:
            ax.set_title(title, fontsize=TITLESIZE)

        ax.yaxis.grid(True, linestyle="--", linewidth=GRID_LW, alpha=0.35)
        ax.set_axisbelow(True)

        for b in bins[1:-1]:
            ax.axvline(b, color=BLACK, lw=VLINE_LW, alpha=0.08, zorder=1)

        if article_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                direction="out",
                length=TICKLEN,
                width=TICKWIDTH,
                labelsize=TICKSIZE,
            )

        return max(
            np.max(scraped_hist) if scraped_hist.size else 0.0,
            np.max(model_hist) if model_hist.size else 0.0,
        )

    # -------------------------
    # Clean inputs
    # -------------------------
    model_gsf_d = _clean_positive(model_gsf_d)
    model_surr_d = _clean_positive(model_surr_d)
    scraped_gsf_d = _clean_positive(scraped_gsf_d)
    scraped_surr_d = _clean_positive(scraped_surr_d)

    if (
        model_gsf_d.size == 0 or model_surr_d.size == 0
        or scraped_gsf_d.size == 0 or scraped_surr_d.size == 0
    ):
        raise ValueError("Missing valid model/scraped values for 1D comparison.")

    model_ratio = _clean_positive(model_surr_d / model_gsf_d)
    scraped_ratio = _clean_positive(scraped_surr_d / scraped_gsf_d)

    if model_ratio.size == 0 or scraped_ratio.size == 0:
        raise ValueError("Missing valid model/scraped ratio values for 1D comparison.")

    # -------------------------
    # Shared bins
    # -------------------------
    bins_gsf = _make_shared_log_bins(model_gsf_d, scraped_gsf_d, n_bins=n_bins)
    bins_surr = _make_shared_log_bins(model_surr_d, scraped_surr_d, n_bins=n_bins)
    bins_ratio = _make_shared_log_bins(model_ratio, scraped_ratio, n_bins=n_bins)

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=FIGSIZE,
        sharey=True,
        constrained_layout=True,
    )

    _draw_panel(
        axes[0],
        model_gsf_d,
        scraped_gsf_d,
        bins_gsf,
        xlabel="GSF diameter (deg)",
        title="GSF",
    )

    _draw_panel(
        axes[1],
        model_surr_d,
        scraped_surr_d,
        bins_surr,
        xlabel="Surround diameter (deg)",
        title="Surround",
    )

    _draw_panel(
        axes[2],
        model_ratio,
        scraped_ratio,
        bins_ratio,
        xlabel="Surround / GSF",
        title="Ratio",
    )

    # -------------------------
    # Fixed shared y-axis
    # -------------------------
    Y_MAX = 1.0
    Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
    Y_TICKLABELS = ["0", "0.25", "0.5", "0.75", "1"]

    for ax in axes:
        ax.set_ylim(0.0, Y_MAX)
        ax.set_yticks(Y_TICKS)

    axes[0].set_ylabel("Proportion of cells", fontsize=LABELSIZE)
    axes[0].set_yticklabels(Y_TICKLABELS)

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    axes[0].legend(
        frameon=False,
        fontsize=LEGENDSIZE,
        loc="upper right",
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.2,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _make_shared_log_bins(model_vals, scraped_vals, n_bins=5):
    vals = np.concatenate([model_vals, scraped_vals])
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size == 0:
        raise ValueError("No valid values for shared bin construction.")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if np.isclose(vmin, vmax):
        return np.array([0.8 * vmin, 1.2 * vmax], dtype=float)

    return np.logspace(np.log10(vmin), np.log10(vmax), int(n_bins) + 1)


def build_shared_metric_hist_settings(
    model_gsf_d,
    model_surr_d,
    scraped_gsf_d,
    scraped_surr_d,
    *,
    include_ratio=True,
    n_log_bins=4,
):
    import numpy as np

    def _clean_positive(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0)]

    def _make_log_bins(vals, n_bins):
        vals = _clean_positive(vals)
        if vals.size == 0:
            raise ValueError("No positive finite values for shared bins.")

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            return np.array([0.8 * vmin, 1.2 * vmax], dtype=float)

        return np.logspace(np.log10(vmin), np.log10(vmax), int(n_bins) + 1)

    def _max_prop_height(vals, bins):
        vals = _clean_positive(vals)
        if vals.size == 0:
            return 0.0
        weights = np.ones_like(vals) / len(vals)
        hist, _ = np.histogram(vals, bins=bins, weights=weights)
        return float(np.max(hist)) if hist.size else 0.0

    model_gsf_d = _clean_positive(model_gsf_d)
    model_surr_d = _clean_positive(model_surr_d)
    scraped_gsf_d = _clean_positive(scraped_gsf_d)
    scraped_surr_d = _clean_positive(scraped_surr_d)

    combined_gsf = np.concatenate([model_gsf_d, scraped_gsf_d])
    combined_surr = np.concatenate([model_surr_d, scraped_surr_d])

    shared_bins = {
        "gsf": _make_log_bins(combined_gsf, n_log_bins),
        "surr": _make_log_bins(combined_surr, n_log_bins),
    }

    ymax_candidates = [
        _max_prop_height(model_gsf_d, shared_bins["gsf"]),
        _max_prop_height(scraped_gsf_d, shared_bins["gsf"]),
        _max_prop_height(model_surr_d, shared_bins["surr"]),
        _max_prop_height(scraped_surr_d, shared_bins["surr"]),
    ]

    if include_ratio:
        model_ratio = _clean_positive(model_surr_d / model_gsf_d)
        scraped_ratio = _clean_positive(scraped_surr_d / scraped_gsf_d)
        combined_ratio = np.concatenate([model_ratio, scraped_ratio])

        shared_bins["ratio"] = _make_log_bins(combined_ratio, n_log_bins)
        ymax_candidates.extend([
            _max_prop_height(model_ratio, shared_bins["ratio"]),
            _max_prop_height(scraped_ratio, shared_bins["ratio"]),
        ])

    shared_ymax = max(1.08 * max(ymax_candidates), 0.05)

    return shared_bins, shared_ymax


def plot_size_tuning_curve_overlay_comparison(
    experimental_curve,
    model_curve,
    *,
    save_path,
    use_diameter=True,
    response_mode="minmax",  # "raw" | "peak_normalized" | "minmax"
    show_annular=True,
    show_markers=True,
    show_suppression=True,
    show_titles=False,
    article_style=True,
):
    """
    Overlay one experimental scraped size-tuning curve and one selected model curve.

    Visual grammar:
        source:
            Experiment = ARTICLE_COLOR
            Model      = MODEL_COLOR

        condition:
            Circular = solid line, filled marker
            Annular  = dashed line, open marker

        markers:
            GSF             = dotted vertical line
            Surround extent = dash-dot vertical line

    Normalisation:
        raw:
            no response scaling.

        peak_normalized:
            circular and annular responses divided by circular peak.
            Ropt/Rsupp divided by same circular peak.

        minmax:
            circular and annular responses jointly mapped to [0, 1]
            within each source independently.
            Ropt/Rsupp transformed using the same source-specific min/max.
            X-markers stay untouched.
    """

    if response_mode not in ("raw", "peak_normalized", "minmax"):
        raise ValueError(f"Invalid response_mode: {response_mode}")

    ensure_dir(os.path.dirname(save_path))

    # -------------------------
    # poster sizing
    # -------------------------
    FIGSIZE = (10.5, 6.2)
    LABELSIZE = 22
    TICKSIZE = 15
    TITLESIZE = 22
    LEGENDSIZE = 15
    LEGENDTITLESIZE = 15
    LINEWIDTH = 2.4
    MARKERSIZE = 7.0
    LEGEND_MARKERSIZE = 7.0
    TICKLEN = 6
    TICKWIDTH = 1.4
    MARKER_LW = 1.8

    EXP_COLOR = ARTICLE_COLOR
    MOD_COLOR = MODEL_COLOR
    CONDITION_COLOR = "0.25"
    MARKER_COLOR = "0.35"

    def _prepare_curve_payload(curve):
        if curve is None:
            return None

        radii = np.asarray(curve["radii"], dtype=float)
        circ = np.asarray(curve["circular"], dtype=float)

        if radii.size < 2 or circ.size < 2:
            return None

        n = min(radii.size, circ.size)
        radii = radii[:n]
        circ = circ[:n]

        ann = curve.get("annular", None)
        if ann is not None and show_annular:
            ann = np.asarray(ann, dtype=float)[:n]
        else:
            ann = None

        valid = np.isfinite(radii) & np.isfinite(circ)
        if ann is not None:
            valid = valid & np.isfinite(ann)

        radii = radii[valid]
        circ = circ[valid]
        if ann is not None:
            ann = ann[valid]

        if radii.size < 2 or circ.size < 2:
            return None

        return {
            "label": curve.get("label", ""),
            "radii": radii,
            "circular": circ,
            "annular": ann,
            "markers": curve.get("markers", None),
        }

    exp = _prepare_curve_payload(experimental_curve)
    mod = _prepare_curve_payload(model_curve)

    if exp is None or mod is None:
        return None

    def _copy_markers(markers):
        if markers is None:
            return None
        return dict(markers)

    def _normalize_response_marker_values(markers, *, mode, denom=None, lo=None, hi=None):
        """
        Normalize only response-valued marker entries.

        Keep untouched:
            GSF, surround_extent, AMRF, SI

        Normalize:
            Ropt, Rsupp
        """
        markers_n = _copy_markers(markers)
        if markers_n is None:
            return None

        for k in ("Ropt", "Rsupp"):
            try:
                v = float(markers_n.get(k, np.nan))
            except Exception:
                markers_n[k] = np.nan
                continue

            if not np.isfinite(v):
                markers_n[k] = np.nan
                continue

            if mode == "raw":
                markers_n[k] = v

            elif mode == "peak_normalized":
                if denom is None or not np.isfinite(denom) or denom <= 0:
                    markers_n[k] = np.nan
                else:
                    markers_n[k] = v / denom

            elif mode == "minmax":
                if (
                    lo is None
                    or hi is None
                    or not np.isfinite(lo)
                    or not np.isfinite(hi)
                    or hi <= lo
                ):
                    markers_n[k] = np.nan
                else:
                    markers_n[k] = (v - lo) / (hi - lo)

            else:
                raise ValueError(f"Invalid marker normalization mode: {mode}")

        return markers_n

    def _normalize_payload(circ, ann, markers):
        circ = np.asarray(circ, dtype=float)
        ann = None if ann is None else np.asarray(ann, dtype=float)

        if response_mode == "raw":
            markers_n = _normalize_response_marker_values(markers, mode="raw")
            return circ, ann, markers_n

        if response_mode == "peak_normalized":
            denom = float(np.nanmax(circ))

            if not np.isfinite(denom) or denom <= 0:
                markers_n = _normalize_response_marker_values(
                    markers,
                    mode="peak_normalized",
                    denom=np.nan,
                )
                return circ, ann, markers_n

            circ_n = circ / denom
            ann_n = None if ann is None else ann / denom

            markers_n = _normalize_response_marker_values(
                markers,
                mode="peak_normalized",
                denom=denom,
            )

            return circ_n, ann_n, markers_n

        if response_mode == "minmax":
            parts = [circ[np.isfinite(circ)]]
            if ann is not None:
                parts.append(ann[np.isfinite(ann)])

            vals = np.concatenate(parts) if len(parts) > 0 else np.array([], dtype=float)

            if vals.size == 0:
                markers_n = _normalize_response_marker_values(
                    markers,
                    mode="minmax",
                    lo=np.nan,
                    hi=np.nan,
                )
                return circ, ann, markers_n

            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            span = hi - lo

            if not np.isfinite(span) or span <= 0:
                circ_n = np.zeros_like(circ, dtype=float)
                ann_n = None if ann is None else np.zeros_like(ann, dtype=float)

                markers_n = _normalize_response_marker_values(
                    markers,
                    mode="minmax",
                    lo=np.nan,
                    hi=np.nan,
                )

                return circ_n, ann_n, markers_n

            circ_n = (circ - lo) / span
            ann_n = None if ann is None else (ann - lo) / span

            markers_n = _normalize_response_marker_values(
                markers,
                mode="minmax",
                lo=lo,
                hi=hi,
            )

            return circ_n, ann_n, markers_n

        raise ValueError(f"Invalid response_mode: {response_mode}")

    exp_circ, exp_ann, exp_markers = _normalize_payload(
        exp["circular"],
        exp["annular"],
        exp.get("markers", None),
    )

    mod_circ, mod_ann, mod_markers = _normalize_payload(
        mod["circular"],
        mod["annular"],
        mod.get("markers", None),
    )

    x_exp = 2.0 * exp["radii"] if use_diameter else exp["radii"]
    x_mod = 2.0 * mod["radii"] if use_diameter else mod["radii"]

    xlab = "Diameter (deg)" if use_diameter else "Radius (deg)"

    if response_mode == "raw":
        ylab = "Response magnitude"
    elif response_mode == "peak_normalized":
        ylab = "Peak-normalized response"
    elif response_mode == "minmax":
        ylab = "Normalized response"
    else:
        raise ValueError(f"Invalid response_mode: {response_mode}")

    def _mark(ax, val, *, ls="--", lw=MARKER_LW, color="0.25"):
        if val is None:
            return
        try:
            v = float(val)
        except Exception:
            return
        if not np.isfinite(v) or v < 0:
            return

        xv = 2.0 * v if use_diameter else v
        ax.axvline(
            xv,
            lw=lw,
            ls=ls,
            color=color,
            zorder=2,
            alpha=0.75,
        )

    def _add_overlay_legend(ax):
        """
        One compact semantic legend inside the axes, middle right.
        """
        handles = [
            # Source
            Line2D(
                [0], [0],
                color=EXP_COLOR,
                marker="o",
                markersize=LEGEND_MARKERSIZE,
                lw=LINEWIDTH,
                linestyle="-",
                markerfacecolor=EXP_COLOR,
                markeredgecolor=EXP_COLOR,
                label="Experiment",
            ),
            Line2D(
                [0], [0],
                color=MOD_COLOR,
                marker="s",
                markersize=LEGEND_MARKERSIZE,
                lw=LINEWIDTH,
                linestyle="-",
                markerfacecolor=MOD_COLOR,
                markeredgecolor=MOD_COLOR,
                label="Model",
            ),

            # Condition
            Line2D(
                [0], [0],
                color=CONDITION_COLOR,
                marker="o",
                markersize=LEGEND_MARKERSIZE,
                lw=LINEWIDTH,
                linestyle="-",
                markerfacecolor=CONDITION_COLOR,
                markeredgecolor=CONDITION_COLOR,
                label="Circular",
            ),
            Line2D(
                [0], [0],
                color=CONDITION_COLOR,
                marker="o",
                markersize=LEGEND_MARKERSIZE,
                lw=LINEWIDTH,
                linestyle="--",
                markerfacecolor=WHITE,
                markeredgecolor=CONDITION_COLOR,
                label="Annular",
            ),

            # Markers
            Line2D(
                [0], [0],
                color=MARKER_COLOR,
                lw=MARKER_LW,
                linestyle=":",
                label="GSF",
            ),
            Line2D(
                [0], [0],
                color=MARKER_COLOR,
                lw=MARKER_LW,
                linestyle="-.",
                label="Surround extent",
            ),
        ]

        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=12,
            loc="center right",
            bbox_to_anchor=(0.985, 0.50),
            borderaxespad=0.2,
            handlelength=1.8,
            handletextpad=0.45,
            labelspacing=0.25,
            borderpad=0.1,
        )
    
    def _draw_suppression_arrow(ax, x, markers, *, color, x_offset_frac=0.0, label_text=False):
        """
        Draw suppression arrow using already-normalized Ropt/Rsupp.

        x_offset_frac shifts arrows horizontally so experiment/model arrows
        do not sit exactly on top of each other.
        """
        if markers is None:
            return

        try:
            Ropt = float(markers.get("Ropt", np.nan))
            Rsupp = float(markers.get("Rsupp", np.nan))
        except Exception:
            return

        if not (np.isfinite(Ropt) and np.isfinite(Rsupp)):
            return
        if Ropt <= Rsupp:
            return

        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        xr = max(x_max - x_min, 1e-12)

        # Prefer around surround extent if available.
        x_arrow = None
        try:
            surr = float(markers.get("surround_extent", np.nan))
            if np.isfinite(surr) and surr >= 0:
                surr_x = 2.0 * surr if use_diameter else surr
                x_arrow = surr_x - 0.18 * xr
        except Exception:
            pass

        if x_arrow is None or not np.isfinite(x_arrow):
            x_arrow = x_min + 0.72 * xr

        x_arrow = x_arrow + x_offset_frac * xr
        x_arrow = float(np.clip(x_arrow, x_min + 0.50 * xr, x_max - 0.08 * xr))

        cap = 0.018 * xr

        ax.annotate(
            "",
            xy=(x_arrow, Ropt),
            xytext=(x_arrow, Rsupp),
            arrowprops=dict(
                arrowstyle="<->",
                lw=1.4,
                color=color,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=7,
        )

        ax.plot(
            [x_arrow - cap, x_arrow + cap],
            [Ropt, Ropt],
            color=color,
            lw=1.1,
            zorder=7,
        )
        ax.plot(
            [x_arrow - cap, x_arrow + cap],
            [Rsupp, Rsupp],
            color=color,
            lw=1.1,
            zorder=7,
        )

        if label_text:
            ax.text(
                x_arrow + 0.025 * xr,
                0.5 * (Ropt + Rsupp),
                "Suppression",
                ha="left",
                va="center",
                fontsize=12,
                color="0.15",
                zorder=8,
            )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # -------------------------
    # Experimental/article curves
    # -------------------------
    ax.plot(
        x_exp,
        exp_circ,
        marker="o",
        markersize=MARKERSIZE,
        mfc=EXP_COLOR,
        mec=EXP_COLOR,
        color=EXP_COLOR,
        ls="-",
        lw=LINEWIDTH,
        zorder=4,
    )

    if exp_ann is not None:
        ax.plot(
            x_exp,
            exp_ann,
            marker="o",
            markersize=MARKERSIZE,
            mfc=WHITE,
            mec=EXP_COLOR,
            color=EXP_COLOR,
            ls="--",
            lw=LINEWIDTH,
            zorder=4,
        )

    # -------------------------
    # Model curves
    # -------------------------
    ax.plot(
        x_mod,
        mod_circ,
        marker="s",
        markersize=MARKERSIZE,
        mfc=MOD_COLOR,
        mec=MOD_COLOR,
        color=MOD_COLOR,
        ls="-",
        lw=LINEWIDTH,
        zorder=3,
    )

    if mod_ann is not None:
        ax.plot(
            x_mod,
            mod_ann,
            marker="s",
            markersize=MARKERSIZE,
            mfc=WHITE,
            mec=MOD_COLOR,
            color=MOD_COLOR,
            ls="--",
            lw=LINEWIDTH,
            zorder=3,
        )

    # -------------------------
    # Markers
    # -------------------------
    if show_markers:
        if exp_markers is not None:
            _mark(
                ax,
                exp_markers.get("GSF"),
                ls=":",
                color=EXP_COLOR,
            )
            _mark(
                ax,
                exp_markers.get("surround_extent"),
                ls="-.",
                color=EXP_COLOR,
            )

        if show_suppression:
            _draw_suppression_arrow(
                ax,
                x_exp,
                exp_markers,
                color=EXP_COLOR,
                x_offset_frac=-0.018,
                label_text=True,
            )

        if mod_markers is not None:
            _mark(
                ax,
                mod_markers.get("GSF"),
                ls=":",
                color=MOD_COLOR,
            )
            _mark(
                ax,
                mod_markers.get("surround_extent"),
                ls="-.",
                color=MOD_COLOR,
            )

        if show_suppression:
            _draw_suppression_arrow(
                ax,
                x_mod,
                mod_markers,
                color=MOD_COLOR,
                x_offset_frac=0.018,
                label_text=False,
            )

    ax.set_xlabel(xlab, fontsize=LABELSIZE)
    ax.set_ylabel(ylab, fontsize=LABELSIZE)

    if show_titles:
        ax.set_title("Size tuning overlay", fontsize=TITLESIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=TICKLEN,
        width=TICKWIDTH,
        labelsize=TICKSIZE,
    )

    ax.grid(False)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    if response_mode == "minmax":
        ax.set_ylim(-0.03, 1.03)
    elif response_mode == "peak_normalized":
        ax.set_ylim(bottom=0)

    _add_overlay_legend(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_si_histogram_comparison(
    *,
    article_bin_edges,
    article_proportions,
    model_values,
    save_path,
    article_label="Experiment",
    model_label="Model",
    title=None,
    show_titles=False,
    article_style=True,
):
    """
    One unified side-by-side SI histogram comparison:
        - experimental/article data in black
        - model data in predefined blue

    Article data are pre-binned proportions.
    Model data are raw SI values binned onto the same article edges.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    edges = np.asarray(article_bin_edges, dtype=float)
    article_props = np.asarray(article_proportions, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    model_values = model_values[np.isfinite(model_values)]

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("article_bin_edges must be a 1D array with length >= 2.")

    n_bins = len(edges) - 1
    if article_props.ndim != 1 or article_props.size != n_bins:
        raise ValueError(
            f"article_proportions must have length {n_bins}, got {article_props.size}."
        )

    if model_values.size == 0:
        raise ValueError("No finite model SI values.")

    # defensive normalization of article proportions
    s = np.nansum(article_props)
    if np.isfinite(s) and s > 0:
        article_props = article_props / s

    # model -> same bins, proportions
    model_weights = np.ones(model_values.shape, dtype=float) / model_values.size
    model_props, _ = np.histogram(model_values, bins=edges, weights=model_weights)

    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bar_fraction = 0.45
    bar_widths = bar_fraction * widths

    # Inner edges touch at bin center:
    # Experiment is left of center, model is right of center.
    article_left = centers - bar_widths
    model_left = centers

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    ax.bar(
        article_left,
        article_props,
        width=bar_widths,
        align="edge",
        facecolor=ARTICLE_COLOR,
        edgecolor=ARTICLE_COLOR,
        linewidth=1.2,
        alpha=0.90,
        label=article_label,
        zorder=3,
    )

    ax.bar(
        model_left,
        model_props,
        width=bar_widths,
        align="edge",
        facecolor=MODEL_COLOR,
        edgecolor=MODEL_COLOR,
        linewidth=1.2,
        alpha=0.85,
        label=model_label,
        zorder=3,
    )

    ax.set_xlabel("Suppression Index (SI)")
    ax.set_ylabel("Proportion of cells")

    if show_titles and title:
        ax.set_title(title)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{x:.1f}" for x in edges])

    ymax = max(
        float(np.nanmax(article_props)) if article_props.size else 0.0,
        float(np.nanmax(model_props)) if model_props.size else 0.0,
    )
    ax.set_ylim(0, max(0.05, 1.10 * ymax))

    if article_style:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1)
        ax.grid(False)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return save_path

def plot_shared_metric_histogram_comparison(
    *,
    model_gsf_d,
    model_surr_d,
    scraped_gsf_d,
    scraped_surr_d,
    shared_bins,
    shared_ymax=None,  # kept only for call-site compatibility; ignored
    save_path,
    model_si=None,
    scraped_si_bin_edges=None,
    scraped_si_proportions=None,
    include_ratio=True,
    include_si=True,
    article_style=True,
    show_titles=False,
):
    """
    Shared metric histogram comparison.

    Panels:
        1) GSF diameter
        2) Surround diameter
        3) Surround / GSF
        4) Suppression Index (SI), if provided

    Visual convention:
        experiment/scraped = ARTICLE_COLOR, plotted first, left side of bin
        model              = MODEL_COLOR, plotted second, right side of bin

    Y-axis:
        fixed 0..1 by 0.2.

    Titles:
        drawn as inside-axis panel tags to avoid layout/title drift.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def _clean_positive(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0)]

    def _clean_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals)]

    def _hist_prop(vals, bins, *, positive=True):
        vals = _clean_positive(vals) if positive else _clean_finite(vals)
        if vals.size == 0:
            return np.zeros(len(bins) - 1, dtype=float)

        weights = np.ones(vals.shape, dtype=float) / len(vals)
        hist, _ = np.histogram(vals, bins=bins, weights=weights)
        return hist.astype(float)

    def _fmt_edge_tick(v):
        if v >= 10:
            return f"{v:.0f}"
        elif v >= 1:
            return f"{v:.1f}"
        else:
            return f"{v:.2f}"

    def _padded_log_xlim(bins, pad_frac=0.03):
        bins = np.asarray(bins, dtype=float)
        lo = float(bins[0])
        hi = float(bins[-1])

        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        span = max(log_hi - log_lo, 1e-12)

        return (
            10 ** (log_lo - pad_frac * span),
            10 ** (log_hi + pad_frac * span),
        )

    def _split_bin_geometry_log_center_touch(bins, bar_fraction=0.45):
        """
        Split each parent bin in LOG SPACE.

        Experiment:
            [center - bar_fraction * bin_width, center]

        Model:
            [center, center + bar_fraction * bin_width]
        """
        bins = np.asarray(bins, dtype=float)

        if np.any(~np.isfinite(bins)) or np.any(bins <= 0):
            raise ValueError("Log bins must be finite and strictly positive.")

        log_edges = np.log(bins)
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        log_widths = np.diff(log_edges)

        exp_log_left = log_centers - bar_fraction * log_widths
        exp_log_right = log_centers

        model_log_left = log_centers
        model_log_right = log_centers + bar_fraction * log_widths

        exp_left = np.exp(exp_log_left)
        exp_right = np.exp(exp_log_right)

        model_left = np.exp(model_log_left)
        model_right = np.exp(model_log_right)

        return (
            exp_left,
            exp_right - exp_left,
            model_left,
            model_right - model_left,
        )

    def _split_bin_geometry_linear_center_touch(edges, bar_fraction=0.45):
        """
        Same as log version, but in linear space.
        Used for SI.
        """
        edges = np.asarray(edges, dtype=float)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bar_widths = bar_fraction * widths
        exp_left = centers - bar_widths
        model_left = centers

        return exp_left, bar_widths, model_left, bar_widths

    def _normalise_props(props):
        props = np.asarray(props, dtype=float)
        s = np.nansum(props)
        if np.isfinite(s) and s > 0:
            props = props / s
        return props

    # -------------------------
    # poster sizing
    # -------------------------
    FIGSIZE_PER_PANEL = 4.1
    FIGHEIGHT = 5.2

    LABELSIZE = 20
    TICKSIZE = 14
    TITLESIZE = 18
    LEGENDSIZE = 18

    BAR_LW = 1.6
    GRID_LW = 0.8
    VLINE_LW = 0.8
    TICKLEN = 6
    TICKWIDTH = 1.4

    BAR_FRACTION = 0.45

    EXP_COLOR = ARTICLE_COLOR
    MOD_COLOR = MODEL_COLOR

    # -------------------------
    # Clean inputs
    # -------------------------
    model_gsf_d = _clean_positive(model_gsf_d)
    model_surr_d = _clean_positive(model_surr_d)
    scraped_gsf_d = _clean_positive(scraped_gsf_d)
    scraped_surr_d = _clean_positive(scraped_surr_d)

    if (
        model_gsf_d.size == 0
        or model_surr_d.size == 0
        or scraped_gsf_d.size == 0
        or scraped_surr_d.size == 0
    ):
        raise ValueError("Missing valid model/scraped metric values.")

    model_ratio = _clean_positive(model_surr_d / model_gsf_d)
    scraped_ratio = _clean_positive(scraped_surr_d / scraped_gsf_d)

    panels = [
        {
            "kind": "log",
            "key": "gsf",
            "model_vals": model_gsf_d,
            "exp_vals": scraped_gsf_d,
            "xlabel": "GSF diameter (deg)",
            "title": "GSF",
        },
        {
            "kind": "log",
            "key": "surr",
            "model_vals": model_surr_d,
            "exp_vals": scraped_surr_d,
            "xlabel": "Surround diameter (deg)",
            "title": "Surround",
        },
    ]

    if include_ratio:
        if model_ratio.size == 0 or scraped_ratio.size == 0:
            raise ValueError("Missing valid ratio values.")

        panels.append(
            {
                "kind": "log",
                "key": "ratio",
                "model_vals": model_ratio,
                "exp_vals": scraped_ratio,
                "xlabel": "Surround / GSF",
                "title": "Ratio",
            }
        )

    if include_si:
        has_si = (
            model_si is not None
            and scraped_si_bin_edges is not None
            and scraped_si_proportions is not None
        )

        if has_si:
            si_edges = np.asarray(scraped_si_bin_edges, dtype=float)
            si_props = np.asarray(scraped_si_proportions, dtype=float)

            if si_edges.ndim != 1 or si_edges.size < 2:
                raise ValueError("scraped_si_bin_edges must be a 1D array length >= 2.")

            if si_props.ndim != 1 or si_props.size != si_edges.size - 1:
                raise ValueError(
                    f"scraped_si_proportions must have length {si_edges.size - 1}, "
                    f"got {si_props.size}."
                )

            panels.append(
                {
                    "kind": "linear_prebinned_exp",
                    "key": "si",
                    "model_vals": _clean_finite(model_si),
                    "exp_props": _normalise_props(si_props),
                    "bins": si_edges,
                    "xlabel": "Suppression Index (SI)",
                    "title": "SI",
                }
            )

    # -------------------------
    # Figure layout
    # -------------------------
    width_ratios = [
        1.45 if panel["key"] == "si" else 1.0
        for panel in panels
    ]

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(FIGSIZE_PER_PANEL * sum(width_ratios), FIGHEIGHT),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    if len(panels) == 1:
        axes = [axes]

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.28,
        top=0.96,
        wspace=0.18,
    )

    def _fmt_edge_tick(v):
        return f"{float(v):.1f}"

    # -------------------------
    # Draw panels
    # -------------------------
    for ax, panel in zip(axes, panels):
        kind = panel["kind"]
        xlabel = panel["xlabel"]

        if kind == "log":
            key = panel["key"]
            if key not in shared_bins:
                raise KeyError(f"shared_bins missing key: {key}")

            bins = np.asarray(shared_bins[key], dtype=float)

            model_hist = _hist_prop(panel["model_vals"], bins, positive=True)
            exp_hist = _hist_prop(panel["exp_vals"], bins, positive=True)

            exp_left, exp_width, model_left, model_width = _split_bin_geometry_log_center_touch(
                bins,
                bar_fraction=BAR_FRACTION,
            )

            ax.set_xscale("log")
            xlo, xhi = _padded_log_xlim(bins, pad_frac=0.03)
            ax.set_xlim(xlo, xhi)

            ax.set_xticks(bins)
            ax.set_xticks(bins)
            ax.set_xticklabels(
                [_fmt_edge_tick(b) for b in bins],
                rotation=45,
                ha="right",
            )
            ax.minorticks_off()

            for b in bins[1:-1]:
                ax.axvline(
                    b,
                    color="black",
                    lw=VLINE_LW,
                    alpha=0.08,
                    zorder=1,
                )

        elif kind == "linear_prebinned_exp":
            bins = np.asarray(panel["bins"], dtype=float)

            exp_hist = np.asarray(panel["exp_props"], dtype=float)
            model_hist = _hist_prop(panel["model_vals"], bins, positive=False)

            exp_left, exp_width, model_left, model_width = _split_bin_geometry_linear_center_touch(
                bins,
                bar_fraction=BAR_FRACTION,
            )

            ax.set_xlim(float(bins[0]), float(bins[-1]))

            ax.set_xticks(bins)
            ax.set_xticklabels(
                [f"{float(b):.1f}" for b in bins],
                rotation=45,
                ha="right",
            )

            for b in bins[1:-1]:
                ax.axvline(
                    b,
                    color="black",
                    lw=VLINE_LW,
                    alpha=0.08,
                    zorder=1,
                )

        else:
            raise ValueError(f"Unknown panel kind: {kind}")

        # Experiment first, black, left side
        ax.bar(
            exp_left,
            exp_hist,
            width=exp_width,
            align="edge",
            facecolor=EXP_COLOR,
            edgecolor=EXP_COLOR,
            linewidth=BAR_LW,
            alpha=0.90,
            label="Experiment",
            zorder=4,
        )

        # Model second, blue, right side
        ax.bar(
            model_left,
            model_hist,
            width=model_width,
            align="edge",
            facecolor=MOD_COLOR,
            edgecolor=MOD_COLOR,
            linewidth=BAR_LW,
            alpha=0.85,
            label="Model",
            zorder=3,
        )

        # Panel tag inside axes. This does not drift.
        if show_titles:
            ax.text(
                0.5,
                0.955,
                panel["title"],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=TITLESIZE,
                fontweight="normal",
                color="0.1",
                zorder=30,
            )

        ax.set_xlabel(xlabel, fontsize=LABELSIZE)

        ax.yaxis.grid(True, linestyle="--", linewidth=GRID_LW, alpha=0.35)
        ax.set_axisbelow(True)

        if article_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                direction="out",
                length=TICKLEN,
                width=TICKWIDTH,
                labelsize=TICKSIZE,
            )

    # -------------------------
    # Fixed shared y-axis: 0..1 by 0.2
    # -------------------------
    Y_TICKS = np.arange(0.0, 1.0001, 0.2)

    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(Y_TICKS)

    axes[0].set_ylabel("Proportion of cells", fontsize=LABELSIZE)
    axes[0].set_yticklabels(
        [f"{v:.1f}" if v not in (0, 1) else f"{int(v)}" for v in Y_TICKS]
    )

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    # -------------------------
    # One legend
    # -------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles[:2],
        labels[:2],
        frameon=False,
        fontsize=LEGENDSIZE,
        loc="upper right",
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.2,
    )

    # -------------------------
    # Final spacing / save
    # -------------------------
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.22,
        top=0.97,
        wspace=0.18,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path