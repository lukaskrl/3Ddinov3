from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


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


class RandResizedCrop3Dd:
    """
    Dictionary-based 3D analogue of torchvision's RandomResizedCrop, operating
    on tensors of shape (C, D, H, W).

    This transform:
      1. Samples a target volume within `scale` fraction of the input volume.
      2. Samples two aspect ratios (depth/height and height/width) log-uniformly
         within `ratio`.
      3. Derives crop sizes (d, h, w) from the sampled volume + ratios; if valid,
         crops the volume.
      4. If no valid sample is found, falls back to a centered cubic crop of the
         minimal dimension.
      5. Resizes the crop to `roi_size` using 3D interpolation.

    Arguments mirror the hand-written _random_3d_crop used in DataAugmentationDINO3D,
    but in a MONAI-style dictionary transform.
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        roi_size: Tuple[int, int, int],
        scale: Tuple[float, float],
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.roi_size = tuple(int(x) for x in roi_size)
        self.scale = scale
        self.ratio = ratio

    def _sample_crop(self, volume: torch.Tensor) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        assert volume.ndim == 4, f"Expected (C, D, H, W), got {tuple(volume.shape)}"
        _, D, H, W = volume.shape
        log_ratio_low, log_ratio_high = np.log(self.ratio[0]), np.log(self.ratio[1])
        volume_total = D * H * W

        def _attempt():
            # Target volume sampled uniformly in scale range.
            target_vol = float(np.random.uniform(self.scale[0], self.scale[1]) * volume_total)
            # Sample two aspect ratios in log space to avoid skew.
            log_r_dh = float(np.random.uniform(log_ratio_low, log_ratio_high))
            log_r_hw = float(np.random.uniform(log_ratio_low, log_ratio_high))
            r_dh = np.exp(log_r_dh)  # depth / height
            r_hw = np.exp(log_r_hw)  # height / width

            # Solve for (d, h, w) given target_vol and ratios:
            #   target_vol = d * h * w
            #   d = r_dh * h ; h = r_hw * w  => target_vol = r_dh * r_hw * h^3
            h = max(1, int(round((target_vol / (r_dh * r_hw)) ** (1.0 / 3.0))))
            d = max(1, int(round(r_dh * h)))
            w = max(1, int(round(h / r_hw)))
            return d, h, w

        # Try multiple times to find a valid crop; fall back if none succeed.
        for _ in range(10):
            cd, ch, cw = _attempt()
            if cd <= D and ch <= H and cw <= W:
                break
        else:
            # Fallback: central cubic crop with minimal dimension to keep behavior deterministic
            min_d = min(D, H, W)
            cd = ch = cw = min_d

        # Sample random crop position.
        sd = 0 if cd >= D else np.random.randint(0, D - cd + 1)
        sh = 0 if ch >= H else np.random.randint(0, H - ch + 1)
        sw = 0 if cw >= W else np.random.randint(0, W - cw + 1)

        return (cd, ch, cw), (sd, sh, sw)

    def _crop_and_resize(self, volume: torch.Tensor) -> torch.Tensor:
        (cd, ch, cw), (sd, sh, sw) = self._sample_crop(volume)
        cropped = volume[:, sd : sd + cd, sh : sh + ch, sw : sw + cw]
        if cropped.shape[1:] != self.roi_size:
            cropped = F.interpolate(
                cropped.unsqueeze(0),  # (1, C, D, H, W)
                size=self.roi_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)  # (C, D, H, W)
        assert (
            cropped.shape[1:] == self.roi_size
        ), f"RandResizedCrop3Dd: expected {self.roi_size}, got {cropped.shape[1:]}"
        return cropped

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self._crop_and_resize(img)
        return d

