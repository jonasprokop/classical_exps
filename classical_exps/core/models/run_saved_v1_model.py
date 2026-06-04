from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn

# Keep these imports local to your project layout.
# You said you will handle exact import paths.
from .model_files.model_v1_musa.model_class import (
    ModelConfig,
    build_model,
)


# =============================================================================
# Checkpoint / config loading
# =============================================================================

def load_saved_v1_checkpoint_config(
    model_dir: str,
    *,
    config_name: str = "config.json",
    checkpoint_name: str = "model.pt",
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load saved V1 model config and checkpoint.

    Expected model_dir:
        model_dir/
            config.json
            model.pt

    Expected checkpoint:
        {
            "model": state_dict,
            "best_val_corr": float,
            "epoch": int,
            "loss_state": ...
        }
    """
    model_dir_path = Path(model_dir)

    config_path = model_dir_path / config_name
    checkpoint_path = model_dir_path / checkpoint_name

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config dict, got {type(config)}")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    if "model_cfg" not in config:
        raise KeyError(f"config.json missing key 'model_cfg'. Found: {sorted(config)}")

    if "n_neurons" not in config:
        raise KeyError(f"config.json missing key 'n_neurons'. Found: {sorted(config)}")

    if "model" not in checkpoint:
        raise KeyError(f"checkpoint missing key 'model'. Found: {sorted(checkpoint)}")

    return config, checkpoint


# =============================================================================
# Exact reconstruction
# =============================================================================

def rebuild_v1_model(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    in_shape: Tuple[int, int, int] = (3, 93, 93),
    device: str | torch.device | None = None,
    strict: bool = True,
    mean_activity: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Rebuild the saved V1 model from config + checkpoint.

    This reconstructs:
        CoreReadoutModel(
            core=TransferLearningCore(...),
            readout=PointPooled2d(...)
        )

    Then loads:
        checkpoint["model"]

    Notes:
        - in_shape should be (3, 93, 93) for the inspected ConvNeXt checkpoint.
        - n_neurons is read from config["n_neurons"].
        - model_cfg is mapped into ModelConfig.
    """
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cfg = ModelConfig(**config["model_cfg"])
    outdims = int(config["n_neurons"])

    model = build_model(
        in_shape=in_shape,
        outdims=outdims,
        cfg=cfg,
        device=device_obj,
        mean_activity=mean_activity,
    )

    state_dict = checkpoint["model"]
    incompatible = model.load_state_dict(state_dict, strict=strict)

    # If strict=False, surface diagnostics instead of hiding them.
    if not strict:
        if incompatible.missing_keys:
            print("[rebuild_v1_model] Missing keys:")
            for key in incompatible.missing_keys:
                print(f"  - {key}")

        if incompatible.unexpected_keys:
            print("[rebuild_v1_model] Unexpected keys:")
            for key in incompatible.unexpected_keys:
                print(f"  - {key}")

    model.to(device_obj)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model


# =============================================================================
# Inference wrapper
# =============================================================================

class InferenceV1Model(nn.Module):
    """
    Inference-only wrapper for saved V1 models.

    Provides one unified interface:
        model(x)                         -> all neurons
        model(x, neuron_ids=ids)          -> selected neurons
        model.predict_all(x)              -> all neurons
        model.predict_neurons(x, ids)     -> selected neurons

    This wrapper intentionally disables training.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device | None = None,
        repeat_grayscale_to_rgb: bool = True,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.repeat_grayscale_to_rgb = repeat_grayscale_to_rgb

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError(
                "InferenceV1Model is inference-only. "
                "Training is intentionally disabled."
            )

        self.model.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        neuron_ids: Optional[Any] = None,
    ) -> torch.Tensor:
        if neuron_ids is None:
            return self.predict_all(x)

        return self.predict_neurons(x, neuron_ids)

    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict all neurons.

        Returns:
            Tensor of shape (B, n_neurons).
        """
        x = self._prepare_input(x)

        self.model.eval()
        with torch.inference_mode():
            return self.model(x)

    def predict_neurons(
        self,
        x: torch.Tensor,
        neuron_ids: Any,
    ) -> torch.Tensor:
        """
        Predict selected neurons.

        Current implementation computes full readout and slices afterwards.
        This is correct and safe. If needed, later optimize by indexing inside
        PointPooled2d directly.

        Returns:
            Tensor of shape (B, len(neuron_ids)).
        """
        y = self.predict_all(x)
        return y[:, neuron_ids]

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input shape/device for ConvNeXt.

        Accepted:
            (B, H, W)
            (B, 1, H, W)
            (B, 3, H, W)

        Output:
            (B, 3, H, W), float32, on self.device.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        x = x.float()

        if x.ndim == 3:
            x = x[:, None, :, :]

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (B,H,W), (B,1,H,W), or (B,3,H,W). "
                f"Got {tuple(x.shape)}"
            )

        if x.shape[1] == 1 and self.repeat_grayscale_to_rgb:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[1] != 3:
            raise ValueError(
                "Saved ConvNeXt V1 model expects 3 input channels. "
                f"Got shape {tuple(x.shape)}"
            )

        return x.to(self.device)


# =============================================================================
# Public loader
# =============================================================================

def load_saved_V1_model(
    model_dir: str,
    *,
    in_shape: Tuple[int, int, int] = (3, 93, 93),
    device: str | torch.device | None = None,
    strict: bool = True,
    repeat_grayscale_to_rgb: bool = True,
) -> InferenceV1Model:
    """
    Load saved V1 model as inference-only pipeline-compatible object.

    This is the function to export/import into simulations.

    Example:
        all_neurons_model = load_saved_V1_model(
            "/project/model_files/model_v1_musa",
            device=device,
        )
    """
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    config, checkpoint = load_saved_v1_checkpoint_config(
        model_dir,
        map_location=device_obj,
    )

    raw_model = rebuild_v1_model(
        config,
        checkpoint,
        in_shape=in_shape,
        device=device_obj,
        strict=strict,
    )

    return InferenceV1Model(
        raw_model,
        device=device_obj,
        repeat_grayscale_to_rgb=repeat_grayscale_to_rgb,
    )


# Optional alias with boring lowercase spelling.
load_saved_v1_model = load_saved_V1_model