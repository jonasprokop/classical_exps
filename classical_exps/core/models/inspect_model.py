from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL_DIR = Path("/project/model_files/model_v1_musa")
CONFIG_PATH = MODEL_DIR / "config.json"
PT_PATH = MODEL_DIR / "model.pt"

def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def inspect_json(path: Path) -> None:
    print_header(f"CONFIG: {path}")

    if not path.exists():
        print("Missing config.json")
        return

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"type(config): {type(config)}")

    if isinstance(config, dict):
        print("\nTop-level keys:")
        for key in sorted(config):
            value = config[key]
            print(f"  - {key}: {type(value).__name__}")

        print("\nFull config:")
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(config)


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"  {name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )


def inspect_state_dict(state_dict: dict) -> None:
    tensor_items = {
        key: value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }

    print(f"state_dict keys: {len(state_dict)}")
    print(f"tensor keys: {len(tensor_items)}")

    print("\nFirst 40 tensor keys:")
    for i, (key, value) in enumerate(tensor_items.items()):
        if i >= 40:
            break
        summarize_tensor(key, value)

    non_tensor = [
        key for key, value in state_dict.items()
        if not isinstance(value, torch.Tensor)
    ]

    if non_tensor:
        print("\nNon-tensor keys:")
        for key in non_tensor[:40]:
            print(f"  - {key}: {type(state_dict[key]).__name__}")


def inspect_pt(path: Path) -> None:
    print_header(f"PT FILE: {path}")

    if not path.exists():
        print("Missing model.pt")
        return

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        print("torch.load failed.")
        print(type(exc).__name__, exc)
        return

    print(f"type(obj): {type(obj)}")

    if isinstance(obj, torch.nn.Module):
        print("\nLoaded full torch.nn.Module.")
        print(obj)
        return

    if isinstance(obj, dict):
        print("\nTop-level dict keys:")
        for key in obj.keys():
            print(f"  - {key}: {type(obj[key]).__name__}")

        # Common checkpoint formats
        for candidate in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if candidate in obj:
                print_header(f"Nested candidate: {candidate}")
                nested = obj[candidate]
                print(f"type({candidate}): {type(nested)}")

                if isinstance(nested, torch.nn.Module):
                    print("Nested full torch.nn.Module:")
                    print(nested)
                elif isinstance(nested, dict):
                    inspect_state_dict(nested)
                else:
                    print(nested)

                return

        # Maybe the top-level dict itself is a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            print("\nTop-level object looks like a raw state_dict.")
            inspect_state_dict(obj)
            return

        print("\nTop-level dict does not look like a plain state_dict.")
        print("Printing tensor-like entries if any:")
        inspect_state_dict(obj)
        return

    print("\nUnknown object type. repr:")
    print(repr(obj))


def main() -> None:
    inspect_json(CONFIG_PATH)
    inspect_pt(PT_PATH)


if __name__ == "__main__":
    main()