import numpy as np


def compute_freeman_modulation_index(tex_resp, noise_resp, eps=1e-8):
    """
    Compute Freeman-style texture/noise modulation index.

    Input shape:
        tex_resp   = [n_neuron, n_family, n_sample]
        noise_resp = [n_neuron, n_family, n_sample]

    Output:
        mi = [n_neuron, n_family]

    Important:
        Average responses across samples first.
        Then compute:
            MI = (texture_mean - noise_mean) / (texture_mean + noise_mean)

    Do NOT compute sample-wise MI and then average it.
    Mean of ratios != ratio of means.
    """
    tex_resp = np.asarray(tex_resp, dtype=float)
    noise_resp = np.asarray(noise_resp, dtype=float)

    if tex_resp.shape != noise_resp.shape:
        raise ValueError(
            f"Texture/noise response shapes differ: "
            f"tex={tex_resp.shape}, noise={noise_resp.shape}"
        )

    if tex_resp.ndim != 3:
        raise ValueError(
            f"Expected response arrays [n_neuron, n_family, n_sample], got {tex_resp.shape}"
        )

    tex_mean = np.mean(tex_resp, axis=2)
    noise_mean = np.mean(noise_resp, axis=2)

    mi = (tex_mean - noise_mean) / (tex_mean + noise_mean + eps)

    return mi

def _require_freeman_scraped_block(scraped_config, key):
    if scraped_config is None:
        return None

    if key not in scraped_config:
        available = ", ".join(scraped_config.keys())
        raise KeyError(
            f"scraped_texture_noise_config is missing key '{key}'. "
            f"Available keys: {available}"
        )

    return scraped_config[key]


def _get_scraped_family_modulation(scraped_config, area="V1"):
    block = _require_freeman_scraped_block(
        scraped_config,
        "texture_family_modulation",
    )

    if block is None:
        return None

    if area not in block:
        available = [k for k in block.keys() if k.upper() in {"V1", "V2"}]
        raise KeyError(
            f"No scraped family modulation for area '{area}'. "
            f"Available areas: {available}"
        )

    family_ids = np.asarray(block["texture_family"]).astype(str)
    mean = np.asarray(block[area]["modulation_index"], dtype=float)

    low = np.asarray(block[area].get("errorbar_endpoint_low", mean), dtype=float)
    high = np.asarray(block[area].get("errorbar_endpoint_high", mean), dtype=float)

    if len(family_ids) != len(mean):
        raise ValueError(
            "Scraped family count and modulation-index count differ: "
            f"{len(family_ids)} vs {len(mean)}"
        )

    yerr = np.vstack([
        np.abs(mean - low),
        np.abs(high - mean),
    ])

    return family_ids, mean, yerr


def _get_scraped_v1_mi_distribution(scraped_config):
    block = _require_freeman_scraped_block(
        scraped_config,
        "v1_modulation_index_distribution",
    )

    if block is None:
        return None

    centers = np.asarray(block["bin_centers_digitized"], dtype=float)
    proportions = np.asarray(block["proportion_of_cells"], dtype=float)

    if centers.shape != proportions.shape:
        raise ValueError(
            "Scraped MI distribution centers/proportions differ: "
            f"{centers.shape} vs {proportions.shape}"
        )

    order = np.argsort(centers)
    return centers[order], proportions[order]

def _centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("Need at least two bin centers to reconstruct edges.")

    mid = (centers[:-1] + centers[1:]) / 2.0
    first_width = centers[1] - centers[0]
    last_width = centers[-1] - centers[-2]

    return np.concatenate([
        [centers[0] - first_width / 2.0],
        mid,
        [centers[-1] + last_width / 2.0],
    ])

def _finish_article_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

def _mi_to_texture_noise_ratio(mi, eps=1e-8):
    mi = np.asarray(mi, dtype=float)
    return (1.0 + mi) / np.maximum(1.0 - mi, eps)








