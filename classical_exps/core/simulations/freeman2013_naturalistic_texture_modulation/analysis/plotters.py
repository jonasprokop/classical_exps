import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

from  classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.tools import (  
    _centers_to_edges,
    _finish_article_axes,
    _get_scraped_family_modulation,
    _get_scraped_v1_mi_distribution,
    _mi_to_texture_noise_ratio,
)

ARTICLE_COLOR = "black"
MODEL_COLOR = "#2F5D8C"

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
            facecolor=MODEL_COLOR,
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
        bar_width = 0.45 * bin_step

        ax.bar(
            centers - bar_width / 2,
            scraped_prop,
            width=bar_width,
            facecolor=ARTICLE_COLOR,
            edgecolor="black",
            linewidth=0.8,
            label="Experimental",
            zorder=2,
        )

        ax.bar(
            centers + bar_width / 2,
            model_prop,
            width=bar_width,
            facecolor=MODEL_COLOR,
            edgecolor="black",
            linewidth=0.8,
            label="Model",
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
    ax.set_ylabel("Proportion of neurons", fontsize=11)

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

def _p_to_text(p):
    if not np.isfinite(p):
        return "p = n/a"
    if p < 1e-4:
        return "p < 1e-4"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _weighted_mean_sem_from_digitized_hist(bin_centers, proportions, n_reported):
    """
    Approximate neuron-level mean/SEM from digitized histogram data.

    This is approximate because the original cell-level values are unavailable.
    """
    x = np.asarray(bin_centers, dtype=float)
    p = np.asarray(proportions, dtype=float)

    mask = np.isfinite(x) & np.isfinite(p)
    x = x[mask]
    p = p[mask]

    p = p / np.sum(p)

    mean = float(np.sum(p * x))

    # Approximate sample SD from binned distribution.
    var = float(np.sum(p * (x - mean) ** 2))

    n = int(n_reported)
    sem = float(np.sqrt(var) / np.sqrt(n))

    # Approximate one-sample test against zero.
    if sem > 0 and n > 1:
        tval = mean / sem
        p_zero = float(2.0 * stats.t.sf(abs(tval), df=n - 1))
    else:
        p_zero = np.nan

    return mean, sem, n, p_zero


def plot_texture_modulation_sign_summary(
    model_mean_mi_neuron,
    scraped_config,
    *,
    save_path="/project/results/texture_noise_response/texture_modulation_sign_summary.png",
    model_label="Model",
    article_label="Experimental V1",
):
    """
    Compare neuron-level mean texture modulation index against zero.

    Model:
        uses actual neuron-level MI values.

    Article V1:
        approximated from digitized panel-f histogram:
            bin centers + proportions + reported n_cells.

    Convention:
        MI > 0: texture response > noise response
        MI < 0: noise response > texture response
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_values = np.asarray(model_mean_mi_neuron, dtype=float)
    model_values = model_values[np.isfinite(model_values)]

    if len(model_values) == 0:
        raise ValueError("No finite model MI values.")

    model_mean = float(np.mean(model_values))
    model_n = int(len(model_values))
    model_sem = float(np.std(model_values, ddof=1) / np.sqrt(model_n))

    if model_n > 1 and model_sem > 0:
        _, model_p_zero = stats.ttest_1samp(model_values, popmean=0.0)
        model_p_zero = float(model_p_zero)
    else:
        model_p_zero = np.nan

    article_block = scraped_config["v1_modulation_index_distribution"]

    article_mean, article_sem, article_n, article_p_zero = (
        _weighted_mean_sem_from_digitized_hist(
            bin_centers=article_block["bin_centers_digitized"],
            proportions=article_block["proportion_of_cells"],
            n_reported=article_block["n_cells_reported"],
        )
    )

    # Approximate model vs article test.
    # Article distribution is binned/digitized, so this is not exact.
    diff = model_mean - article_mean
    se_diff = np.sqrt(model_sem ** 2 + article_sem ** 2)

    if se_diff > 0:
        z = diff / se_diff
        p_diff = float(2.0 * stats.norm.sf(abs(z)))
    else:
        p_diff = np.nan

    labels = [article_label, model_label]
    means = np.array([article_mean, model_mean], dtype=float)
    sems = np.array([article_sem, model_sem], dtype=float)
    ps_zero = [article_p_zero, model_p_zero]

    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    ax.bar(
        x,
        means,
        yerr=sems,
        width=0.62,
        color=[ARTICLE_COLOR, MODEL_COLOR],
        edgecolor="black",
        linewidth=0.9,
        capsize=4,
        error_kw=dict(elinewidth=1.1, capthick=1.1),
        zorder=2,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    max_abs = float(np.max(np.abs(means) + sems))
    ylim = max(0.06, max_abs * 2.0)
    ax.set_ylim(-ylim, ylim)

    for i, (mean, sem, p) in enumerate(zip(means, sems, ps_zero)):
        y = mean + sem if mean >= 0 else mean - sem
        offset = 0.07 * ylim if mean >= 0 else -0.07 * ylim
        va = "bottom" if mean >= 0 else "top"

        ax.text(
            x[i],
            y + offset,
            _stars(p),
            ha="center",
            va=va,
            fontsize=9,
        )


    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylabel("Mean modulation index", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Texture modulation sign summary:")
    print(
        f"    > {article_label}: mean={article_mean:.4f}, "
        f"SEM≈{article_sem:.4f}, n={article_n}, p_zero≈{article_p_zero:.4g}"
    )
    print(
        f"    > {model_label}: mean={model_mean:.4f}, "
        f"SEM={model_sem:.4f}, n={model_n}, p_zero={model_p_zero:.4g}"
    )
    print(f"    > model-vs-article approx. p={p_diff:.4g}")
    print(f"    > saved: {save_path}")


def _p_to_star_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _ttest_values_against_zero(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 2:
        return np.nan, np.nan, np.nan, n

    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n))

    if sem <= 0:
        return mean, sem, np.nan, n

    _, p = stats.ttest_1samp(values, popmean=0.0)

    return mean, sem, float(p), n


def _ttest_mean_sem_against_zero(mean, sem, n):
    mean = float(mean)
    sem = float(sem)
    n = int(n)

    if not np.isfinite(mean) or not np.isfinite(sem) or sem <= 0 or n <= 1:
        return np.nan

    t_val = mean / sem
    p = 2.0 * stats.t.sf(abs(t_val), df=n - 1)

    return float(p)


def plot_family_modulation_significance_comparison(
    *,
    family_ids,
    model_mi_neuron_family,
    scraped_config,
    save_path="/project/results/texture_noise_response/family_modulation_significance_comparison.svg",
    article_area="V1",
    article_n=None,
    model_label="Model",
    article_label="Experimental V1",
):
    """
    Per-family MI comparison with MI-vs-zero test.

    Model:
        exact one-sample t-test across model neurons for each family.

    Article:
        approximate t-test from scraped mean ± SEM and reported n.
        Raw article cell-level values are unavailable.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    family_ids = np.asarray(family_ids).astype(str)
    model_mi_neuron_family = np.asarray(model_mi_neuron_family, dtype=float)

    if model_mi_neuron_family.ndim != 2:
        raise ValueError(
            f"Expected model_mi_neuron_family [n_neuron, n_family], "
            f"got {model_mi_neuron_family.shape}"
        )

    if model_mi_neuron_family.shape[1] != len(family_ids):
        raise ValueError(
            f"Family count mismatch: model has {model_mi_neuron_family.shape[1]}, "
            f"family_ids has {len(family_ids)}"
        )

    article_block = scraped_config["texture_family_modulation"][article_area]

    article_mean = np.asarray(article_block["modulation_index"], dtype=float)
    article_low = np.asarray(article_block["errorbar_endpoint_low"], dtype=float)
    article_high = np.asarray(article_block["errorbar_endpoint_high"], dtype=float)

    if len(article_mean) != len(family_ids):
        raise ValueError(
            f"Article family count mismatch: article has {len(article_mean)}, "
            f"family_ids has {len(family_ids)}"
        )

    article_sem = 0.5 * (
        np.abs(article_mean - article_low)
        + np.abs(article_high - article_mean)
    )

    if article_n is None:
        article_n = scraped_config["v1_modulation_index_distribution"]["n_cells_reported"]

    model_mean = np.zeros(len(family_ids), dtype=float)
    model_sem = np.zeros(len(family_ids), dtype=float)
    model_p = np.zeros(len(family_ids), dtype=float)
    model_n = np.zeros(len(family_ids), dtype=int)

    for i in range(len(family_ids)):
        mean_i, sem_i, p_i, n_i = _ttest_values_against_zero(
            model_mi_neuron_family[:, i]
        )
        model_mean[i] = mean_i
        model_sem[i] = sem_i
        model_p[i] = p_i
        model_n[i] = n_i

    article_p = np.asarray([
        _ttest_mean_sem_against_zero(
            mean=article_mean[i],
            sem=article_sem[i],
            n=article_n,
        )
        for i in range(len(family_ids))
    ], dtype=float)

    csv_path = os.path.splitext(save_path)[0] + ".csv"

    with open(csv_path, "w") as f:
        f.write(
            "family,"
            "article_mean,article_sem,article_n,article_p_zero,article_sig,"
            "model_mean,model_sem,model_n,model_p_zero,model_sig\n"
        )

        for i, fam in enumerate(family_ids):
            f.write(
                f"{fam},"
                f"{article_mean[i]:.8f},"
                f"{article_sem[i]:.8f},"
                f"{article_n},"
                f"{article_p[i]:.8g},"
                f"{_p_to_star_marker(article_p[i])},"
                f"{model_mean[i]:.8f},"
                f"{model_sem[i]:.8f},"
                f"{model_n[i]},"
                f"{model_p[i]:.8g},"
                f"{_p_to_star_marker(model_p[i])}\n"
            )

    x = np.arange(len(family_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 3.0))

    # experiment first
    ax.bar(
        x - width / 2,
        article_mean,
        yerr=article_sem,
        width=width,
        facecolor=ARTICLE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=1.0),
        label=article_label,
        zorder=2,
    )

    # model second
    ax.bar(
        x + width / 2,
        model_mean,
        yerr=model_sem,
        width=width,
        facecolor=MODEL_COLOR,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=1.0),
        label=model_label,
        zorder=2,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=1,
    )

    all_y = np.concatenate([
        article_mean + article_sem,
        article_mean - article_sem,
        model_mean + model_sem,
        model_mean - model_sem,
    ])

    ylim_abs = max(0.12, float(np.nanmax(np.abs(all_y))) * 1.45)
    ax.set_ylim(-ylim_abs, ylim_abs)

    star_offset = 0.045 * ylim_abs

    for i in range(len(family_ids)):
        article_star = _p_to_star_marker(article_p[i])
        if article_star:
            y = (
                article_mean[i] + article_sem[i]
                if article_mean[i] >= 0
                else article_mean[i] - article_sem[i]
            )
            ax.text(
                x[i] - width / 2,
                y + star_offset if article_mean[i] >= 0 else y - star_offset,
                article_star,
                ha="center",
                va="bottom" if article_mean[i] >= 0 else "top",
                fontsize=6,
            )

        model_star = _p_to_star_marker(model_p[i])
        if model_star:
            y = (
                model_mean[i] + model_sem[i]
                if model_mean[i] >= 0
                else model_mean[i] - model_sem[i]
            )
            ax.text(
                x[i] + width / 2,
                y + star_offset if model_mean[i] >= 0 else y - star_offset,
                model_star,
                ha="center",
                va="bottom" if model_mean[i] >= 0 else "top",
                fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(family_ids, fontsize=9)

    ax.set_xlabel("Texture family", fontsize=11)
    ax.set_ylabel("Modulation index", fontsize=11)

    ax.legend(frameon=False, fontsize=9, ncol=2)

    _finish_article_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)