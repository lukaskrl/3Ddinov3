from typing import Optional, Sequence, Tuple

import torch


def ct_window_and_normalize(
    volume: torch.Tensor,
    window: Tuple[float, float] = (-1000.0, 400.0),
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Basic CT preprocessing for 3D volumes.

    Args:
        volume: Tensor of shape (C, D, H, W).
        window: (min, max) HU window to clip to.
        mean/std: Optional per‑channel mean and std (after windowing and scaling).
                  If None, we simply scale to [0, 1] without extra normalization.
    """
    assert volume.ndim == 4, f"Expected (C, D, H, W) tensor, got {tuple(volume.shape)}"

    v_min, v_max = window
    volume = torch.clamp(volume, v_min, v_max)

    # Scale to [0, 1]
    volume = (volume - v_min) / (v_max - v_min + 1e-6)

    if mean is not None and std is not None:
        # Broadcast per‑channel mean/std over (D, H, W)
        c = volume.shape[0]
        assert len(mean) == c and len(std) == c, "mean/std length must match number of channels"
        mean_t = torch.as_tensor(mean, dtype=volume.dtype, device=volume.device).view(c, 1, 1, 1)
        std_t = torch.as_tensor(std, dtype=volume.dtype, device=volume.device).view(c, 1, 1, 1)
        volume = (volume - mean_t) / (std_t + 1e-6)

    return volume


def make_ct_3d_base_transform(
    window: Tuple[float, float] = (-1000.0, 400.0),
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Factory returning a callable that applies CT windowing + normalization
    to tensors of shape (C, D, H, W).
    """

    def _transform(volume: torch.Tensor) -> torch.Tensor:
        return ct_window_and_normalize(volume, window=window, mean=mean, std=std)

    return _transform

