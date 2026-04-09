from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

import torch

_PRINTED_TAGS: Set[str] = set()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_device(
    device: Optional[str] = None,
    *,
    require_cuda: Optional[bool] = None,
) -> torch.device:
    """
    Resolve the torch device to use.

    Precedence:
      1) `device` arg (e.g. "cuda", "cuda:0", "cpu")
      2) env var `HYPERGRAD_DEVICE`
      3) auto: "cuda" if available else "cpu"

    If `require_cuda` is True (or env var `HYPERGRAD_REQUIRE_CUDA=1`), raises if CUDA is not usable.
    """
    if device is None:
        device = os.environ.get("HYPERGRAD_DEVICE")

    if require_cuda is None:
        require_cuda = _env_flag("HYPERGRAD_REQUIRE_CUDA", default=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={device!r}, but torch.cuda.is_available() is False.")

    if require_cuda and resolved.type != "cuda":
        raise RuntimeError(
            f"CUDA required (set via require_cuda or HYPERGRAD_REQUIRE_CUDA=1), but resolved device is {resolved}."
        )

    if resolved.type == "cuda":
        try:
            if resolved.index is not None:
                torch.cuda.set_device(resolved.index)
        except Exception:
            # Best-effort: if set_device fails for any reason, let PyTorch raise later when CUDA is used.
            pass

    return resolved


def device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": str(device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        info.update(
            {
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_current_device": int(cur),
                "cuda_device_name": torch.cuda.get_device_name(cur),
            }
        )
        props = torch.cuda.get_device_properties(cur)
        info.update(
            {
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )

    return info


def print_device_info(device: torch.device, *, tag: Optional[str] = None) -> None:
    prefix = f"[{tag}] " if tag else ""
    info = device_info(device)

    print(
        prefix
        + "device={device} cuda_available={cuda_available} torch={torch_version} torch_cuda={torch_cuda_version}".format(
            **info
        )
    )
    if info.get("cuda_visible_devices") is not None:
        print(prefix + f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}")

    if info.get("cuda_available"):
        print(
            prefix
            + "cuda_device_count={cuda_device_count} cuda_current_device={cuda_current_device} cuda_name={cuda_device_name}".format(
                **info
            )
        )
        print(prefix + "cuda_capability={cuda_capability} cuda_total_memory_gb={cuda_total_memory_gb}".format(**info))


def maybe_print_device_info(device: torch.device, *, tag: str) -> None:
    if tag in _PRINTED_TAGS:
        return
    print_device_info(device, tag=tag)
    _PRINTED_TAGS.add(tag)

