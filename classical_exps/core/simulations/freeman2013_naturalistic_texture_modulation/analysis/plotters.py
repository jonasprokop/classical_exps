import os
import numpy as np
import matplotlib.pyplot as plt 

from  classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.tools import (  
    _centers_to_edges,
    _finish_article_axes,
    _get_scraped_family_modulation,
    _get_scraped_v1_mi_distribution,
    _mi_to_texture_noise_ratio,
)

def plot_family_modulation_index_comparison(
    family_ids,
    model_mean_mi_family,
    *,
    save_path,
    scraped_mean_mi_family=None,
    scraped_yerr=None,
    scraped_label="Article V1",
):
    """
    Bar plot of mean modulation index by texture family.

    Assumes model_mean_mi_family is already aligned to family_ids.
    If scraped_mean_mi_family is supplied, it is assumed to be aligned too.

    Plot:
        - model bars
        - optional article/scraped bars
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    family_ids = np.asarray(family_ids).astype(str)
    model_mean_mi_family = np.asarray(model_mean_mi_family, dtype=float)

    if len(family_ids) != len(model_mean_mi_family):
        raise ValueError(
            "family_ids and model_mean_mi_family length mismatch: "
            f"{len(family_ids)} vs {len(model_mean_mi_family)}"
        )

    has_scraped = scraped_mean_mi_family is not None

    if has_scraped:
        scraped_mean_mi_family = np.asarray(scraped_mean_mi_family, dtype=float)

        if len(scraped_mean_mi_family) != len(model_mean_mi_family):
            raise ValueError(
                "scraped_mean_mi_family and model_mean_mi_family length mismatch: "
                f"{len(scraped_mean_mi_family)} vs {len(model_mean_mi_family)}"
            )

        if scraped_yerr is not None:
            scraped_yerr = np.asarray(scraped_yerr, dtype=float)

            if scraped_yerr.shape == (len(scraped_mean_mi_family),):
                pass

            elif scraped_yerr.shape == (2, len(scraped_mean_mi_family)):
                pass

            else:
                raise ValueError(
                    "scraped_yerr must have shape (n_family,) or (2, n_family), "
                    f"got {scraped_yerr.shape}"
                )

    x = np.arange(len(family_ids))

    fig, ax = plt.subplots(figsize=(8.2, 3.0))

    if not has_scraped:
        ax.bar(
            x,
            model_mean_mi_family,
            width=0.62,
            facecolor="0.35",
            edgecolor="black",
            linewidth=0.8,
            label="Model",
            zorder=2,
        )

        abs_max = float(np.nanmax(np.abs(model_mean_mi_family)))

    else:
        width = 0.36

        ax.bar(
            x - width / 2,
            model_mean_mi_family,
            width=width,
            facecolor="0.35",
            edgecolor="black",
            linewidth=0.8,
            label="Model",
            zorder=2,
        )

        ax.bar(
            x + width / 2,
            scraped_mean_mi_family,
            width=width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            label=scraped_label,
            zorder=2,
        )

        if scraped_yerr is not None:
            ax.errorbar(
                x + width / 2,
                scraped_mean_mi_family,
                yerr=scraped_yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                zorder=3,
            )

        abs_max = float(
            np.nanmax(
                np.abs(
                    np.concatenate([
                        model_mean_mi_family,
                        scraped_mean_mi_family,
                    ])
                )
            )
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    ylim = max(0.10, abs_max * 1.25)
    ax.set_ylim(-ylim, ylim)

    ax.set_xticks(x)
    ax.set_xticklabels(family_ids, fontsize=9)

    ax.set_xlabel("Texture family", fontsize=11)
    ax.set_ylabel("Modulation index", fontsize=11)

    if has_scraped:
        ax.legend(frameon=False, fontsize=9, ncol=2)

    _finish_article_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_neuron_mi_distribution_comparison(
    model_mean_mi_neuron,
    *,
    save_path,
    scraped_config=None,
):
    """
    Histogram of neuron-level mean MI.

    With scraped_config, bins the model into the article digitized bin geometry
    and draws model/article bars side-by-side.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_mean_mi_neuron = np.asarray(model_mean_mi_neuron, dtype=float)
    model_mean_mi_neuron = model_mean_mi_neuron[np.isfinite(model_mean_mi_neuron)]

    if len(model_mean_mi_neuron) == 0:
        raise ValueError("No finite model MI values for distribution plot.")

    scraped = _get_scraped_v1_mi_distribution(scraped_config)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    if scraped is None:
        weights = (
            np.ones_like(model_mean_mi_neuron, dtype=float)
            / len(model_mean_mi_neuron)
        )

        values, bins, _ = ax.hist(
            model_mean_mi_neuron,
            bins=31,
            edgecolor="black",
            linewidth=0.8,
            density=False,
            weights=weights,
            facecolor="0.55",
            label="Model",
        )

        x_min = min(-0.6, float(np.min(bins)))
        x_max = max(0.6, float(np.max(bins)))
        y_max = float(np.max(values)) if len(values) else 1.0

    else:
        centers, scraped_prop = scraped
        edges = _centers_to_edges(centers)

        model_counts, _ = np.histogram(
            model_mean_mi_neuron,
            bins=edges,
        )

        model_prop = model_counts.astype(float) / len(model_mean_mi_neuron)
        diffs = np.diff(centers)
        bin_step = float(np.min(diffs))

        # Make the within-pair gap equal to the between-pair gap.
        # For equally spaced centers:
        #     bin_step = 2 * bar_width + 2 * gap
        gap = 0.12 * bin_step
        bar_width = (bin_step - 2.0 * gap) / 2.0
        offset = (bar_width + gap) / 2.0

        ax.bar(
            centers - offset,
            model_prop,
            width=bar_width,
            facecolor="0.45",
            edgecolor="black",
            linewidth=0.8,
            # hatch="",
            label="Model",
            zorder=2,
        )

        ax.bar(
            centers + offset,
            scraped_prop,
            width=bar_width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            # hatch="///",
            label="Experimental",
            zorder=2,
        )

        x_min = min(-0.6, float(edges[0]))
        x_max = max(0.6, float(edges[-1]))
        y_max = float(np.max(np.concatenate([model_prop, scraped_prop])))

    model_mean = float(np.mean(model_mean_mi_neuron))

    # ax.axvline(
    #     0,
    #     color="black",
    #     linestyle="-",
    #     linewidth=1.0,
    #     label="zero",
    # )

    # ax.axvline(
    #     model_mean,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.3,
    #     label=f"Model mean = {model_mean:.3f}",
    # )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max(0.05, y_max * 1.18))

    ax.set_xticks([-0.5, 0.0, 0.5])

    ax.set_xlabel("Mean modulation index", fontsize=11)
    ax.set_ylabel("% of neurons", fontsize=11)

    ax.legend(frameon=False, fontsize=9)

    _finish_article_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_texture_noise_ratio_comparison(
    family_ids,
    model_mean_mi_family,
    *,
    save_path,
    scraped_mean_mi_family=None,
    scraped_yerr=None,
    scraped_label="Experimental",
):
    """
    Plot texture/noise response ratio by texture family.

    Assumes model_mean_mi_family is already aligned to family_ids.
    If scraped_mean_mi_family is supplied, it is assumed aligned too.

    Uses:
        MI = (T - N) / (T + N)
        T/N = (1 + MI) / (1 - MI)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    family_ids = np.asarray(family_ids).astype(str)
    model_mean_mi_family = np.asarray(model_mean_mi_family, dtype=float)

    if len(family_ids) != len(model_mean_mi_family):
        raise ValueError(
            "family_ids and model_mean_mi_family length mismatch: "
            f"{len(family_ids)} vs {len(model_mean_mi_family)}"
        )

    model_ratio = _mi_to_texture_noise_ratio(model_mean_mi_family)

    has_scraped = scraped_mean_mi_family is not None

    if has_scraped:
        scraped_mean_mi_family = np.asarray(scraped_mean_mi_family, dtype=float)

        if len(scraped_mean_mi_family) != len(model_mean_mi_family):
            raise ValueError(
                "scraped_mean_mi_family and model_mean_mi_family length mismatch: "
                f"{len(scraped_mean_mi_family)} vs {len(model_mean_mi_family)}"
            )

        scraped_ratio = _mi_to_texture_noise_ratio(scraped_mean_mi_family)

        ratio_yerr = None

        if scraped_yerr is not None:
            scraped_yerr = np.asarray(scraped_yerr, dtype=float)

            if scraped_yerr.shape == (len(scraped_mean_mi_family),):
                low_mi = scraped_mean_mi_family - scraped_yerr
                high_mi = scraped_mean_mi_family + scraped_yerr

            elif scraped_yerr.shape == (2, len(scraped_mean_mi_family)):
                low_mi = scraped_mean_mi_family - scraped_yerr[0]
                high_mi = scraped_mean_mi_family + scraped_yerr[1]

            else:
                raise ValueError(
                    "scraped_yerr must have shape (n_family,) or (2, n_family), "
                    f"got {scraped_yerr.shape}"
                )

            low_ratio = _mi_to_texture_noise_ratio(low_mi)
            high_ratio = _mi_to_texture_noise_ratio(high_mi)

            ratio_yerr = np.vstack([
                np.abs(scraped_ratio - low_ratio),
                np.abs(high_ratio - scraped_ratio),
            ])

    else:
        scraped_ratio = None
        ratio_yerr = None

    x = np.arange(len(family_ids))
    fig, ax = plt.subplots(figsize=(8.2, 3.0))

    if not has_scraped:
        ax.bar(
            x,
            model_ratio,
            width=0.62,
            facecolor="0.35",
            edgecolor="black",
            linewidth=0.8,
            label="Model",
            zorder=2,
        )

        y_max = float(np.nanmax(model_ratio))

    else:
        width = 0.36

        ax.bar(
            x - width / 2,
            model_ratio,
            width=width,
            facecolor="0.35",
            edgecolor="black",
            linewidth=0.8,
            label="Model",
            zorder=2,
        )

        ax.bar(
            x + width / 2,
            scraped_ratio,
            width=width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            label=scraped_label,
            zorder=2,
        )

        if ratio_yerr is not None:
            ax.errorbar(
                x + width / 2,
                scraped_ratio,
                yerr=ratio_yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                zorder=3,
            )

            y_max = float(
                np.nanmax(
                    np.concatenate([
                        model_ratio,
                        scraped_ratio + ratio_yerr[1],
                    ])
                )
            )
        else:
            y_max = float(
                np.nanmax(
                    np.concatenate([
                        model_ratio,
                        scraped_ratio,
                    ])
                )
            )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    ax.set_ylim(0, max(1.25, y_max * 1.12))

    ax.set_xticks(x)
    ax.set_xticklabels(family_ids, fontsize=9)

    ax.set_xlabel("Texture family", fontsize=11)
    ax.set_ylabel("Texture / noise response", fontsize=11)

    if has_scraped:
        ax.legend(frameon=False, fontsize=9, ncol=2)

    _finish_article_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



