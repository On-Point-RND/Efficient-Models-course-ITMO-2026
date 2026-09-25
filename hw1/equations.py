"""Analytical Homework 1 equations. All functions broadcast NumPy inputs.

The memory equation counts model parameters and simultaneously live tensors.
It deliberately excludes implementation-dependent cuDNN workspaces; therefore
it is a lower-bound model for torch.cuda.max_memory_allocated(), not a promise
of the exact allocation observed for every cuDNN algorithm.
"""

from __future__ import annotations

import numpy as np


PARAMETERS = 1_040_324  # Six bias-free convolutions and two biased Linear layers.


def flops(image_size, batch):
    """Forward arithmetic operations; 1 multiply-accumulate = 2 FLOPs."""
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float))
    return b * (17_714 * s**2 + 313_344)


def memory(image_size, batch):
    """Ideal live-tensor peak, bytes, with input retained by the caller."""
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float))
    return 4 * PARAMETERS + 52 * b * s**2


def bytes_moved(image_size, batch):
    """Nominal tensor traffic, bytes, including separate in-place ReLU kernels.

    Each layer reads its input and writes its output once, each in-place ReLU
    reads and writes its output, and model weights are read once. Cache reuse,
    cuDNN transformations, and temporary workspaces are intentionally omitted.
    """
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float))
    return 4 * (PARAMETERS + b * (91 * s**2 + 2_148))


def latency(image_size, batch, theta):
    """Roofline latency in seconds; theta was fitted on training points only."""
    f = flops(image_size, batch)
    d = bytes_moved(image_size, batch)
    return theta["launch_ms"] / 1_000 + np.maximum(
        f / (theta["compute_tflops"] * 1e12),
        d / (theta["bandwidth_tbps"] * 1e12),
    )


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules from effective power times predicted latency."""
    return theta_energy["gpu_power_w"] * latency(image_size, batch, theta_energy)
