from __future__ import annotations

import math

import torch

from classical_exps.core.models.run_saved_v1_model import load_saved_V1_model


MODEL_DIR = "/project/classical_exps/core/models/model_files/model_v1_musa"


# =============================================================================
# Stimulus helpers
# =============================================================================

def rescale(x, old_min, old_max, new_min, new_max):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def make_fullfield_grating(
    *,
    orientation: float,
    spatial_frequency: float = 2.0,
    phase: float = 0.0,
    contrast: float = 1.0,
    img_res=(93, 93),
    size: float = 2.67,
    pixel_min: float = -1.7876,
    pixel_max: float = 2.1919,
) -> torch.Tensor:
    """
    Full-field sine grating.

    orientation: radians, in [0, pi)
    spatial_frequency: cycles / visual degree
    phase: radians

    Returns:
        Tensor shape (1, H, W), matching the old grayscale pipeline input.
    """
    h, w = img_res

    y = torch.linspace(-size / 2, size / 2, h)
    x = torch.linspace(-size / 2, size / 2, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    coord = xx * math.cos(orientation) + yy * math.sin(orientation)

    stim = torch.sin(2 * math.pi * spatial_frequency * coord + phase)
    stim = stim * contrast

    stim = rescale(stim, -1, 1, pixel_min, pixel_max)

    return stim.reshape(1, h, w).float()


# =============================================================================
# Model sanity test
# =============================================================================

def run_orientation_sanity_test(
    *,
    model_dir: str = MODEL_DIR,
    n_orientations: int = 12,
    spatial_frequency: float = 2.0,
    phase: float = 0.0,
    contrast: float = 1.0,
    img_res=(93, 93),
    size: float = 2.67,
    pixel_min: float = -1.7876,
    pixel_max: float = 2.1919,
    top_k: int = 15,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_saved_V1_model(
        model_dir,
        device=device,
        in_shape=(3, 93, 93),
        repeat_grayscale_to_rgb=True,
    )

    gray_val = (pixel_min + pixel_max) / 2
    gray = torch.ones(1, *img_res) * gray_val

    y_gray = model.predict_all(gray)  # (1, n_neurons)

    orientations = torch.linspace(0, torch.pi, steps=n_orientations + 1)[:-1]

    responses = []

    for ori in orientations:
        stim = make_fullfield_grating(
            orientation=float(ori),
            spatial_frequency=spatial_frequency,
            phase=phase,
            contrast=contrast,
            img_res=img_res,
            size=size,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
        )

        y = model.predict_all(stim)
        responses.append((y - y_gray).detach().cpu().squeeze(0))

    # shape: (n_orientations, n_neurons)
    responses = torch.stack(responses, dim=0)

    n_neurons = responses.shape[1]

    preferred_idx = responses.argmax(dim=0)
    preferred_ori_deg = orientations[preferred_idx].cpu() * 180.0 / math.pi

    peak_delta = responses.max(dim=0).values
    min_delta = responses.min(dim=0).values
    tuning_range = peak_delta - min_delta

    mean_delta_per_ori = responses.mean(dim=1)

    print("\n" + "=" * 88)
    print("Saved V1 model orientation sanity test")
    print("=" * 88)

    print(f"device: {device}")
    print(f"n_neurons: {n_neurons}")
    print(f"n_orientations: {n_orientations}")
    print(f"spatial_frequency: {spatial_frequency}")
    print(f"phase: {phase}")
    print(f"contrast: {contrast}")

    print("\nOutput sanity")
    print(f"gray response shape: {tuple(y_gray.shape)}")
    print(
        "gray min/mean/max: "
        f"{float(y_gray.min()):.4f} / "
        f"{float(y_gray.mean()):.4f} / "
        f"{float(y_gray.max()):.4f}"
    )

    print("\nPopulation mean delta by orientation")
    for ori, mean_resp in zip(orientations, mean_delta_per_ori):
        ori_deg = float(ori * 180.0 / math.pi)
        print(f"  {ori_deg:6.1f} deg: {float(mean_resp): .4f}")

    bins = torch.linspace(0, 180, steps=n_orientations + 1)
    hist = torch.histc(preferred_ori_deg.float(), bins=n_orientations, min=0, max=180)

    print("\nPreferred orientation histogram")
    for left, right, count in zip(bins[:-1], bins[1:], hist):
        print(f"  {float(left):6.1f}–{float(right):6.1f} deg: {int(count)}")

    print("\nResponse modulation")
    print(f"peak delta mean/median/max: "
          f"{float(peak_delta.mean()):.4f} / "
          f"{float(peak_delta.median()):.4f} / "
          f"{float(peak_delta.max()):.4f}")
    print(f"tuning range mean/median/max: "
          f"{float(tuning_range.mean()):.4f} / "
          f"{float(tuning_range.median()):.4f} / "
          f"{float(tuning_range.max()):.4f}")

    top = torch.topk(tuning_range, k=min(top_k, n_neurons))

    print(f"\nTop {len(top.indices)} neurons by orientation modulation")
    for rank, neuron_id in enumerate(top.indices.tolist(), start=1):
        best_ori = float(preferred_ori_deg[neuron_id])
        peak = float(peak_delta[neuron_id])
        trange = float(tuning_range[neuron_id])

        print(
            f"{rank:02d}. neuron {neuron_id:03d} | "
            f"pref={best_ori:6.1f} deg | "
            f"peak_delta={peak: .4f} | "
            f"range={trange: .4f}"
        )

    print("\nExample tuning curves for top neurons")
    for neuron_id in top.indices[:5].tolist():
        print(f"\nneuron {neuron_id:03d}:")
        for ori, resp in zip(orientations, responses[:, neuron_id]):
            ori_deg = float(ori * 180.0 / math.pi)
            print(f"  {ori_deg:6.1f} deg: {float(resp): .4f}")


def main() -> None:
    run_orientation_sanity_test(
        n_orientations=12,
        spatial_frequency=2.0,
        phase=0.0,
        contrast=1.0,
        top_k=15,
    )


if __name__ == "__main__":
    main()