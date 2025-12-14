"""Device detection utilities for M1 Mac / CUDA / CPU."""

import torch


def get_device() -> torch.device:
    """Get the best available device for training/inference.

    Returns:
        torch.device: MPS for M1 Mac, CUDA for NVIDIA GPU, or CPU as fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Get a human-readable device name."""
    if device.type == "mps":
        return "MPS (Metal Performance Shaders)"
    elif device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"
