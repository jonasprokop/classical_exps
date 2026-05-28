############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.core.tools.utils import *
## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.core.tools.utils import SingleCellModel
## Data storage
import h5py
from scipy.optimize import curve_fit
import time
import cv2
import matplotlib.pyplot as plt

import re
import hashlib
from pathlib import Path


IMG_NAME_RE = re.compile(
    r"^(?P<category>tex|noise)-(?P<res>\d+x\d+)-im(?P<family>\d+)-smp(?P<sample>\d+)\.png$"
)

def _spectral_centroid_2d(img, eps=1e-12):
    x = np.asarray(img, dtype=float)
    x = x - np.mean(x)

    fft = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(fft) ** 2

    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2
    cx = (w - 1) / 2

    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (np.max(r) + eps)

    return float(np.sum(r_norm * power) / (np.sum(power) + eps))


def _high_freq_fraction_2d(img, threshold=0.50, eps=1e-12):
    x = np.asarray(img, dtype=float)
    x = x - np.mean(x)

    fft = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(fft) ** 2

    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2
    cx = (w - 1) / 2

    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (np.max(r) + eps)

    high_mask = r_norm >= threshold

    return float(np.sum(power[high_mask]) / (np.sum(power) + eps))


def compute_selected_lowlevel_family_stats(tex_imgs, noise_imgs, dict_fam):
    """
    Compute five low-level image statistics from exact model-input tensors.

    Input:
        tex_imgs   [n_family, n_sample, y, x]
        noise_imgs [n_family, n_sample, y, x]

    Output:
        family_ids      [n_family]
        feature_names   [5]
        family_features [n_family, 5]
    """
    tex_np = tex_imgs.detach().cpu().numpy()
    noise_np = noise_imgs.detach().cpu().numpy()

    if tex_np.shape != noise_np.shape:
        raise ValueError(
            f"Texture/noise image shapes differ: tex={tex_np.shape}, noise={noise_np.shape}"
        )

    if tex_np.ndim != 4:
        raise ValueError(f"Expected [family, sample, y, x], got {tex_np.shape}")

    n_family, n_sample, _, _ = tex_np.shape

    family_ids = np.asarray(
        [fam for fam, fam_idx in sorted(dict_fam.items(), key=lambda kv: kv[1])],
        dtype=int,
    )

    feature_names = [
        "delta_rms_contrast",
        "delta_mean_luminance",
        "texture_noise_rms_difference",
        "delta_spectral_centroid",
        "delta_high_freq_fraction",
    ]

    family_features = np.zeros((n_family, len(feature_names)), dtype=float)

    for fam_i in range(n_family):
        vals_delta_std = []
        vals_delta_mean = []
        vals_rms_diff = []
        vals_delta_centroid = []
        vals_delta_high = []

        for smp_i in range(n_sample):
            tex = tex_np[fam_i, smp_i]
            noise = noise_np[fam_i, smp_i]

            vals_delta_std.append(float(np.std(tex) - np.std(noise)))
            vals_delta_mean.append(float(np.mean(tex) - np.mean(noise)))
            vals_rms_diff.append(float(np.sqrt(np.mean((tex - noise) ** 2))))

            vals_delta_centroid.append(
                _spectral_centroid_2d(tex) - _spectral_centroid_2d(noise)
            )

            vals_delta_high.append(
                _high_freq_fraction_2d(tex) - _high_freq_fraction_2d(noise)
            )

        family_features[fam_i, 0] = np.mean(vals_delta_std)
        family_features[fam_i, 1] = np.mean(vals_delta_mean)
        family_features[fam_i, 2] = np.mean(vals_rms_diff)
        family_features[fam_i, 3] = np.mean(vals_delta_centroid)
        family_features[fam_i, 4] = np.mean(vals_delta_high)

    return family_ids, feature_names, family_features




def _parse_texture_img_name(img_name):
    """
    Strictly parse Freeman-style texture/noise image names.

    Expected:
        tex-320x320-im13-smp2.png
        noise-320x320-im13-smp2.png
    """
    match = IMG_NAME_RE.match(img_name)

    if match is None:
        raise ValueError(
            f"Bad image filename: {img_name}. "
            "Expected e.g. tex-320x320-im13-smp2.png or noise-320x320-im13-smp2.png"
        )

    category = match.group("category")
    family = int(match.group("family"))
    sample = int(match.group("sample"))
    declared_res = match.group("res")

    return category, declared_res, family, sample


def _hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def audit_texture_noise_images(
    directory_imgs,
    num_samples=15,
    expected_num_families=15,
    require_complete=True,
):
    """
    Checks that the Freeman texture/noise image folder is structurally sane.

    Checks:
        - valid filenames
        - only tex/noise categories
        - deterministic sorted family/sample structure
        - every family has num_samples texture + num_samples noise images
        - every texture sample has matching noise sample
        - duplicate file hashes
        - image dimensions and value ranges
    """

    directory_imgs = Path(directory_imgs)

    if not directory_imgs.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory_imgs}")

    img_names = sorted(
        f.name for f in directory_imgs.iterdir()
        if f.is_file() and f.name.lower().endswith(".png")
    )

    if len(img_names) == 0:
        raise ValueError(f"No PNG images found in {directory_imgs}")

    records = []
    seen = set()

    for img_name in img_names:
        category, declared_res, family, sample = _parse_texture_img_name(img_name)

        if sample < 1:
            raise ValueError(f"Sample IDs should start from 1, got {img_name}")

        key = (category, family, sample)

        if key in seen:
            raise ValueError(f"Duplicate category/family/sample entry: {key}")

        seen.add(key)

        img_path = directory_imgs / img_name
        img = mpimg.imread(str(img_path))

        if img.ndim == 3:
            if img.shape[-1] == 4:
                # PNG with alpha. Fine, but alpha should not enter the model.
                img_for_stats = img[..., :3].mean(axis=-1)
            elif img.shape[-1] == 3:
                img_for_stats = img.mean(axis=-1)
            else:
                raise ValueError(f"Unexpected channel shape for {img_name}: {img.shape}")
        elif img.ndim == 2:
            img_for_stats = img
        else:
            raise ValueError(f"Unexpected image dimensions for {img_name}: {img.shape}")

        records.append({
            "file": img_name,
            "path": str(img_path),
            "category": category,
            "declared_res": declared_res,
            "family": family,
            "sample": sample,
            "shape": img.shape,
            "gray_shape": img_for_stats.shape,
            "min": float(np.min(img_for_stats)),
            "max": float(np.max(img_for_stats)),
            "mean": float(np.mean(img_for_stats)),
            "std": float(np.std(img_for_stats)),
            "hash": _hash_file(img_path),
        })

    families = sorted(set(r["family"] for r in records))

    if expected_num_families is not None and len(families) != expected_num_families:
        msg = (
            f"Expected {expected_num_families} texture families, "
            f"found {len(families)}: {families}"
        )
        if require_complete:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    # Check completeness
    missing = []

    for family in families:
        for sample in range(1, num_samples + 1):
            for category in ["tex", "noise"]:
                key = (category, family, sample)
                if key not in seen:
                    missing.append(key)

    if missing:
        msg = (
            f"Missing {len(missing)} expected images. "
            f"First missing entries: {missing[:20]}"
        )
        if require_complete:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    # Check extra samples
    extra = [
        (r["category"], r["family"], r["sample"])
        for r in records
        if r["sample"] > num_samples
    ]

    if extra:
        msg = f"Found samples beyond num_samples={num_samples}: {extra[:20]}"
        if require_complete:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    # Check shape consistency
    gray_shapes = sorted(set(r["gray_shape"] for r in records))

    if len(gray_shapes) != 1:
        raise ValueError(f"Inconsistent image sizes: {gray_shapes}")

    # Check blank images
    blankish = [r["file"] for r in records if r["std"] < 1e-8]

    if blankish:
        raise ValueError(f"Blank/nearly blank images found: {blankish[:20]}")

    # Check duplicate exact files
    hash_counts = {}
    for r in records:
        hash_counts.setdefault(r["hash"], []).append(r["file"])

    duplicate_groups = [
        files for files in hash_counts.values()
        if len(files) > 1
    ]

    if duplicate_groups:
        print("WARNING: Duplicate exact image files found:")
        for files in duplicate_groups[:10]:
            print("   ", files)

    fourier_rows, fourier_hard_problems, fourier_warnings = (
        audit_texture_noise_fourier_content(records)
    )

    if fourier_hard_problems:
        msg = (
            f"Fourier/content audit found {len(fourier_hard_problems)} hard issues. "
            "This may mean the noise images are not properly spectrally matched."
        )
        if require_complete:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    print("   > image audit passed")
    print(f"     families: {families}")
    print(f"     samples per family expected: {num_samples}")
    print(f"     image size: {gray_shapes[0]}")
    print(f"     total PNG files: {len(records)}")

    return records, families 


def _to_gray_float(img):
    """
    Convert image array to grayscale float32 for statistics / FFT checks.
    """
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    elif img.ndim != 2:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")

    return img.astype(np.float32)


def _load_gray_float(path):
    img = mpimg.imread(str(path))
    return _to_gray_float(img)


def _fft_power_phase(img, eps=1e-12):
    """
    Return log power spectrum and unit phase representation.

    Power spectrum should be similar for texture/noise.
    Phase should not be identical, otherwise the images are not proper phase-randomized controls.
    """
    x = img.astype(np.float32)
    x = x - np.mean(x)

    fft = np.fft.fft2(x)

    power = np.abs(fft) ** 2
    log_power = np.log10(power + eps)

    phase_unit = fft / (np.abs(fft) + eps)

    return log_power, phase_unit


def _corr_flat(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if a.size != b.size:
        raise ValueError(f"Cannot correlate arrays with different sizes: {a.size}, {b.size}")

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _radial_profile(arr):
    """
    Radially average a 2D array around its center.
    Useful for checking rotationally averaged spectrum similarity.
    """
    arr = np.asarray(arr)

    y, x = np.indices(arr.shape)
    cy = (arr.shape[0] - 1) / 2
    cx = (arr.shape[1] - 1) / 2

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    radial_sum = np.bincount(r.ravel(), weights=arr.ravel())
    radial_count = np.bincount(r.ravel())

    return radial_sum / np.maximum(radial_count, 1)


def _fftshift_power(power):
    return np.fft.fftshift(power)


def _phase_similarity(phase_a, phase_b):
    """
    Circular phase similarity.

    1.0 = same phase everywhere.
    Around 0-ish = unrelated phase.
    """
    sim = np.real(phase_a * np.conj(phase_b))
    return float(np.nanmean(sim))

def audit_texture_noise_fourier_content(
    records,
    *,
    min_power_corr_2d=0.85,
    min_power_corr_radial=0.98,
    max_phase_similarity=0.25,
    max_pixel_corr=0.75,
):
    """
    Checks whether each tex/noise pair behaves like Freeman-style spectrally matched noise.

    Hard checks:
        - high radial Fourier power correlation
        - low phase similarity
        - low pixel correlation

    Soft diagnostic:
        - 2D Fourier power correlation.
          Exact 2D power matching can be lower after synthesis/export/cropping,
          so this should warn, not kill the run.
    """

    by_key = {}

    for r in records:
        key = (r["family"], r["sample"])
        by_key.setdefault(key, {})[r["category"]] = r

    fourier_rows = []
    hard_problems = []
    warnings = []

    for key in sorted(by_key.keys()):
        family, sample = key
        pair = by_key[key]

        if "tex" not in pair or "noise" not in pair:
            hard_problems.append(
                f"Missing tex/noise pair for family={family}, sample={sample}"
            )
            continue

        tex = _load_gray_float(pair["tex"]["path"])
        noise = _load_gray_float(pair["noise"]["path"])

        if tex.shape != noise.shape:
            hard_problems.append(
                f"Shape mismatch for family={family}, sample={sample}: "
                f"tex={tex.shape}, noise={noise.shape}"
            )
            continue

        tex_log_power, tex_phase = _fft_power_phase(tex)
        noise_log_power, noise_phase = _fft_power_phase(noise)

        power_corr_2d = _corr_flat(tex_log_power, noise_log_power)

        tex_radial = _radial_profile(_fftshift_power(tex_log_power))
        noise_radial = _radial_profile(_fftshift_power(noise_log_power))
        power_corr_radial = _corr_flat(tex_radial, noise_radial)

        phase_sim = _phase_similarity(tex_phase, noise_phase)
        pixel_corr = _corr_flat(tex, noise)

        row = {
            "family": family,
            "sample": sample,
            "power_corr_2d": power_corr_2d,
            "power_corr_radial": power_corr_radial,
            "phase_similarity": phase_sim,
            "pixel_corr": pixel_corr,
        }
        fourier_rows.append(row)

        # Soft diagnostic only.
        if power_corr_2d < min_power_corr_2d:
            warnings.append(
                f"Low 2D power-spectrum correlation for family={family}, sample={sample}: "
                f"{power_corr_2d:.3f} < {min_power_corr_2d}"
            )

        # Hard checks.
        if power_corr_radial < min_power_corr_radial:
            hard_problems.append(
                f"Low radial power-spectrum correlation for family={family}, sample={sample}: "
                f"{power_corr_radial:.3f} < {min_power_corr_radial}"
            )

        if abs(phase_sim) > max_phase_similarity:
            hard_problems.append(
                f"Suspiciously high phase similarity for family={family}, sample={sample}: "
                f"{phase_sim:.3f} > {max_phase_similarity}"
            )

        if abs(pixel_corr) > max_pixel_corr:
            hard_problems.append(
                f"Suspiciously high pixel correlation for family={family}, sample={sample}: "
                f"{pixel_corr:.3f} > {max_pixel_corr}"
            )

    if fourier_rows:
        power_2d = np.array([r["power_corr_2d"] for r in fourier_rows], dtype=float)
        power_radial = np.array([r["power_corr_radial"] for r in fourier_rows], dtype=float)
        phase = np.array([r["phase_similarity"] for r in fourier_rows], dtype=float)
        pix = np.array([r["pixel_corr"] for r in fourier_rows], dtype=float)

        print("   > Fourier content audit:")
        print(
            f"     2D power corr:     "
            f"mean={np.nanmean(power_2d):.3f}, min={np.nanmin(power_2d):.3f}"
        )
        print(
            f"     radial power corr: "
            f"mean={np.nanmean(power_radial):.3f}, min={np.nanmin(power_radial):.3f}"
        )
        print(
            f"     phase similarity:  "
            f"mean={np.nanmean(phase):.3f}, max_abs={np.nanmax(np.abs(phase)):.3f}"
        )
        print(
            f"     pixel corr:        "
            f"mean={np.nanmean(pix):.3f}, max_abs={np.nanmax(np.abs(pix)):.3f}"
        )

    if warnings:
        print("WARNING: Fourier/content audit soft warnings:")
        for p in warnings[:20]:
            print("   ", p)
        if len(warnings) > 20:
            print(f"   ... and {len(warnings) - 20} more")

    if hard_problems:
        print("WARNING: Fourier/content audit hard problems:")
        for p in hard_problems[:20]:
            print("   ", p)
        if len(hard_problems) > 20:
            print(f"   ... and {len(hard_problems) - 20} more")

    return fourier_rows, hard_problems, warnings


def save_one_stimulus_pair_per_family(
    tex_imgs,
    noise_imgs,
    dict_fam,
    *,
    save_dir="/project/results/texture_noise_response/stimulus_samples/",
    sample_strategy="first",   # "first", "middle", or "random"
    seed=42,
    scale_mode="global",       # "global" or "per_image"
    save_npy=True,
    save_pair_png=True,
    save_montage=True,
):
    """
    Save one texture/noise stimulus pair per family before model evaluation.

    Inputs:
        tex_imgs   [n_family, n_sample, y, x]
        noise_imgs [n_family, n_sample, y, x]

    What gets saved:
        - one exact .npy texture image per family
        - one exact .npy noise image per family
        - one visual pair PNG per family
        - one montage PNG with all families
        - one CSV manifest with family/sample/file paths

    scale_mode:
        "global"    -> one grayscale range from all selected images.
                       Best for comparing actual incoming contrast across families.
        "per_image" -> each image auto-scaled separately.
                       Prettier, but can lie about relative contrast.
    """
    os.makedirs(save_dir, exist_ok=True)

    tex_np = tex_imgs.detach().cpu().numpy()
    noise_np = noise_imgs.detach().cpu().numpy()

    if tex_np.shape != noise_np.shape:
        raise ValueError(
            f"Texture/noise stimulus shapes differ: tex={tex_np.shape}, noise={noise_np.shape}"
        )

    if tex_np.ndim != 4:
        raise ValueError(f"Expected [family, sample, y, x], got {tex_np.shape}")

    n_family, n_sample, _, _ = tex_np.shape

    # dict_fam maps original family id -> tensor index.
    # Sort by tensor index to recover the actual tensor order.
    family_order = [
        family for family, fam_idx in sorted(dict_fam.items(), key=lambda kv: kv[1])
    ]

    if len(family_order) != n_family:
        raise ValueError(
            f"dict_fam length does not match tensor family axis: "
            f"{len(family_order)} vs {n_family}"
        )

    if sample_strategy == "first":
        sample_ids = np.zeros(n_family, dtype=int)

    elif sample_strategy == "middle":
        sample_ids = np.full(n_family, n_sample // 2, dtype=int)

    elif sample_strategy == "random":
        rng = np.random.default_rng(seed)
        sample_ids = rng.integers(0, n_sample, size=n_family)

    else:
        raise ValueError(
            f"Unknown sample_strategy={sample_strategy}. "
            "Use 'first', 'middle', or 'random'."
        )

    selected_tex = np.stack(
        [tex_np[fam_idx, sample_ids[fam_idx]] for fam_idx in range(n_family)],
        axis=0,
    )
    selected_noise = np.stack(
        [noise_np[fam_idx, sample_ids[fam_idx]] for fam_idx in range(n_family)],
        axis=0,
    )

    shown = np.concatenate([selected_tex, selected_noise], axis=0)
    global_vmin = float(np.min(shown))
    global_vmax = float(np.max(shown))

    if np.isclose(global_vmin, global_vmax):
        global_vmax = global_vmin + 1e-6

    def _get_limits(img):
        if scale_mode == "global":
            return global_vmin, global_vmax

        if scale_mode == "per_image":
            vmin = float(np.min(img))
            vmax = float(np.max(img))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            return vmin, vmax

        raise ValueError("scale_mode must be 'global' or 'per_image'")

    manifest_rows = []

    # ------------------------------------------------------------------
    # Save one pair per family
    # ------------------------------------------------------------------
    for fam_idx, family in enumerate(family_order):
        sample_id = int(sample_ids[fam_idx])

        tex_img = selected_tex[fam_idx]
        noise_img = selected_noise[fam_idx]

        base = f"family_{int(family):02d}_sample_{sample_id + 1:02d}"

        tex_npy_path = os.path.join(save_dir, f"{base}_texture.npy")
        noise_npy_path = os.path.join(save_dir, f"{base}_noise.npy")
        pair_png_path = os.path.join(save_dir, f"{base}_pair.png")

        if save_npy:
            np.save(tex_npy_path, tex_img)
            np.save(noise_npy_path, noise_img)
        else:
            tex_npy_path = ""
            noise_npy_path = ""

        if save_pair_png:
            tex_vmin, tex_vmax = _get_limits(tex_img)
            noise_vmin, noise_vmax = _get_limits(noise_img)

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(4.8, 2.4),
                constrained_layout=True,
            )

            im = axes[0].imshow(
                tex_img,
                cmap="gray",
                vmin=tex_vmin,
                vmax=tex_vmax,
                interpolation="nearest",
            )

            axes[1].imshow(
                noise_img,
                cmap="gray",
                vmin=noise_vmin,
                vmax=noise_vmax,
                interpolation="nearest",
            )

            axes[0].set_title("texture", fontsize=10)
            axes[1].set_title("noise", fontsize=10)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            if scale_mode == "global":
                cbar = fig.colorbar(
                    im,
                    ax=axes.ravel().tolist(),
                    shrink=0.75,
                    pad=0.03,
                )
                cbar.set_label("model input value", fontsize=8)
                cbar.ax.tick_params(labelsize=7)

            fig.suptitle(
                f"family {family}, sample {sample_id + 1}",
                fontsize=10,
            )

            plt.savefig(pair_png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            pair_png_path = ""

        manifest_rows.append({
            "family": int(family),
            "sample": sample_id + 1,
            "texture_npy": tex_npy_path,
            "noise_npy": noise_npy_path,
            "pair_png": pair_png_path,
            "texture_min": float(np.min(tex_img)),
            "texture_max": float(np.max(tex_img)),
            "noise_min": float(np.min(noise_img)),
            "noise_max": float(np.max(noise_img)),
        })

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------
    manifest_path = os.path.join(save_dir, "one_pair_per_family_manifest.csv")

    with open(manifest_path, "w") as f:
        f.write(
            "family,sample,texture_npy,noise_npy,pair_png,"
            "texture_min,texture_max,noise_min,noise_max\n"
        )

        for row in manifest_rows:
            f.write(
                f"{row['family']},"
                f"{row['sample']},"
                f"{row['texture_npy']},"
                f"{row['noise_npy']},"
                f"{row['pair_png']},"
                f"{row['texture_min']:.8f},"
                f"{row['texture_max']:.8f},"
                f"{row['noise_min']:.8f},"
                f"{row['noise_max']:.8f}\n"
            )

    # ------------------------------------------------------------------
    # Save montage with all families
    # ------------------------------------------------------------------
    if save_montage:
        fig, axes = plt.subplots(
            n_family,
            2,
            figsize=(5.4, max(2.0, 1.55 * n_family)),
            constrained_layout=True,
        )

        if n_family == 1:
            axes = np.asarray([axes])

        last_im = None

        for fam_idx, family in enumerate(family_order):
            sample_id = int(sample_ids[fam_idx])

            tex_img = selected_tex[fam_idx]
            noise_img = selected_noise[fam_idx]

            tex_vmin, tex_vmax = _get_limits(tex_img)
            noise_vmin, noise_vmax = _get_limits(noise_img)

            ax_tex = axes[fam_idx, 0]
            ax_noise = axes[fam_idx, 1]

            last_im = ax_tex.imshow(
                tex_img,
                cmap="gray",
                vmin=tex_vmin,
                vmax=tex_vmax,
                interpolation="nearest",
            )

            ax_noise.imshow(
                noise_img,
                cmap="gray",
                vmin=noise_vmin,
                vmax=noise_vmax,
                interpolation="nearest",
            )

            ax_tex.set_ylabel(
                f"fam {family}\nsmp {sample_id + 1}",
                fontsize=8,
                rotation=0,
                ha="right",
                va="center",
            )

            if fam_idx == 0:
                ax_tex.set_title("texture", fontsize=11)
                ax_noise.set_title("noise", fontsize=11)

            for ax in (ax_tex, ax_noise):
                ax.set_xticks([])
                ax.set_yticks([])

        if scale_mode == "global":
            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                shrink=0.72,
                pad=0.02,
            )
            cbar.set_label(
                f"model input value\nmin={global_vmin:.3f}, max={global_vmax:.3f}",
                fontsize=9,
            )
            cbar.ax.tick_params(labelsize=8)

        else:
            fig.text(
                0.98,
                0.02,
                "Each image auto-scaled separately",
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.3",
            )

        fig.suptitle(
            f"One stimulus pair per family passed to model "
            f"({sample_strategy}, {scale_mode} scale)",
            fontsize=12,
        )

        montage_path = os.path.join(
            save_dir,
            f"montage_one_pair_per_family_{sample_strategy}_{scale_mode}.png",
        )

        plt.savefig(montage_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"   > one stimulus pair per family saved to: {save_dir}")
    print(f"   > manifest: {manifest_path}")
    print(f"   > shown tensor range: min={global_vmin:.4f}, max={global_vmax:.4f}")


def _safe_kurtosis(x, eps=1e-8):
    """
    Excess-ish kurtosis proxy without scipy dependency.
    Higher = heavier-tailed pixel distribution.
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = np.mean(x)
    sd = np.std(x)

    if sd < eps:
        return np.nan

    z = (x - mu) / sd
    return float(np.mean(z ** 4))


def _spectral_summary(img, eps=1e-12):
    """
    Simple Fourier summaries for one 2D image.

    Returns:
        total_power
        low_freq_fraction
        high_freq_fraction
        spectral_centroid
    """
    x = np.asarray(img, dtype=float)
    x = x - np.mean(x)

    fft = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(fft) ** 2

    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2
    cx = (w - 1) / 2

    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (np.max(r) + eps)

    total_power = float(np.sum(power) + eps)

    low_mask = r_norm <= 0.25
    high_mask = r_norm >= 0.50

    low_freq_fraction = float(np.sum(power[low_mask]) / total_power)
    high_freq_fraction = float(np.sum(power[high_mask]) / total_power)

    spectral_centroid = float(np.sum(r_norm * power) / total_power)

    return total_power, low_freq_fraction, high_freq_fraction, spectral_centroid


def compute_texture_noise_image_statistics(tex_imgs, noise_imgs, dict_fam):
    """
    Compute low-level image statistics from the exact tensors passed to the model.

    Inputs:
        tex_imgs   [n_family, n_sample, y, x]
        noise_imgs [n_family, n_sample, y, x]

    Returns:
        family_ids      [n_family]
        feature_names   list[str]
        family_features [n_family, n_feature]
        sample_features [n_family, n_sample, n_feature]
    """
    tex_np = tex_imgs.detach().cpu().numpy()
    noise_np = noise_imgs.detach().cpu().numpy()

    if tex_np.shape != noise_np.shape:
        raise ValueError(
            f"Texture/noise image tensor mismatch: tex={tex_np.shape}, noise={noise_np.shape}"
        )

    if tex_np.ndim != 4:
        raise ValueError(f"Expected [family, sample, y, x], got {tex_np.shape}")

    n_family, n_sample, _, _ = tex_np.shape

    family_ids = np.asarray(
        [fam for fam, idx in sorted(dict_fam.items(), key=lambda kv: kv[1])],
        dtype=int,
    )

    if len(family_ids) != n_family:
        raise ValueError(
            f"Family-id count does not match tensor axis: {len(family_ids)} vs {n_family}"
        )

    feature_names = [
        "tex_mean",
        "noise_mean",
        "tex_std",
        "noise_std",
        "delta_mean_tex_minus_noise",
        "delta_std_tex_minus_noise",
        "rms_tex_minus_noise",
        "abs_rms_tex_minus_noise",
        "tex_kurtosis",
        "noise_kurtosis",
        "delta_kurtosis_tex_minus_noise",
        "tex_total_power",
        "noise_total_power",
        "delta_total_power_tex_minus_noise",
        "tex_low_freq_fraction",
        "noise_low_freq_fraction",
        "delta_low_freq_fraction_tex_minus_noise",
        "tex_high_freq_fraction",
        "noise_high_freq_fraction",
        "delta_high_freq_fraction_tex_minus_noise",
        "tex_spectral_centroid",
        "noise_spectral_centroid",
        "delta_spectral_centroid_tex_minus_noise",
    ]

    sample_features = np.zeros(
        (n_family, n_sample, len(feature_names)),
        dtype=float,
    )

    for fam_i in range(n_family):
        for smp_i in range(n_sample):
            tex = tex_np[fam_i, smp_i]
            noise = noise_np[fam_i, smp_i]

            tex_mean = float(np.mean(tex))
            noise_mean = float(np.mean(noise))

            tex_std = float(np.std(tex))
            noise_std = float(np.std(noise))

            diff = tex - noise
            rms_diff = float(np.sqrt(np.mean(diff ** 2)))

            tex_k = _safe_kurtosis(tex)
            noise_k = _safe_kurtosis(noise)

            (
                tex_total_power,
                tex_low_freq_fraction,
                tex_high_freq_fraction,
                tex_spectral_centroid,
            ) = _spectral_summary(tex)

            (
                noise_total_power,
                noise_low_freq_fraction,
                noise_high_freq_fraction,
                noise_spectral_centroid,
            ) = _spectral_summary(noise)

            vals = [
                tex_mean,
                noise_mean,
                tex_std,
                noise_std,
                tex_mean - noise_mean,
                tex_std - noise_std,
                rms_diff,
                abs(rms_diff),
                tex_k,
                noise_k,
                tex_k - noise_k,
                tex_total_power,
                noise_total_power,
                tex_total_power - noise_total_power,
                tex_low_freq_fraction,
                noise_low_freq_fraction,
                tex_low_freq_fraction - noise_low_freq_fraction,
                tex_high_freq_fraction,
                noise_high_freq_fraction,
                tex_high_freq_fraction - noise_high_freq_fraction,
                tex_spectral_centroid,
                noise_spectral_centroid,
                tex_spectral_centroid - noise_spectral_centroid,
            ]

            sample_features[fam_i, smp_i, :] = vals

    family_features = np.nanmean(sample_features, axis=1)

    return family_ids, feature_names, family_features, sample_features
