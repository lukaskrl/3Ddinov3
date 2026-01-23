from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .transforms_3d import RandResizedCrop3Dd, make_ct_3d_base_transform
import monai


class DataAugmentationDINO3D(object):
    """
    3D analogue of DataAugmentationDINO for volumetric CT.

    Operates on tensors of shape (C, D, H, W) and returns a dict with the same
    keys as the 2D version ("global_crops", "global_crops_teacher",
    "local_crops", "offsets", optionally "gram_teacher_crops") so that the
    rest of the training pipeline can stay mostly unchanged.
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_size_3d: Tuple[int, int, int],
        local_crops_size_3d: Tuple[int, int, int],
        gram_teacher_crops_size_3d: Optional[Tuple[int, int, int]] = None,
        gram_teacher_no_distortions: bool = False,
        local_crops_subset_of_global_crops: bool = False,
        patch_size_3d: Tuple[int, int, int] = (2, 16, 16),
        horizontal_flips: bool = True,
        ct_window: Tuple[float, float] = (-1000.0, 400.0),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        ratio_3d: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size_3d = global_crops_size_3d
        self.local_crops_size_3d = local_crops_size_3d
        self.gram_teacher_crops_size_3d = gram_teacher_crops_size_3d
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size_3d = patch_size_3d
        self.horizontal_flips = horizontal_flips
        # Log-uniform aspect-ratio range, mirroring torchvision RandomResizedCrop default
        self.ratio_3d = ratio_3d

        self.normalize = make_ct_3d_base_transform(window=ct_window, mean=mean, std=std)

    def _random_3d_crop(
        self,
        volume: torch.Tensor,
        out_size: Tuple[int, int, int],
        scale: Tuple[float, float],
        ratio: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Random 3D resized crop on (C, D, H, W) mirroring torchvision's RandomResizedCrop:
        1) Sample a target volume within `scale` fraction of the input volume.
        2) Sample two aspect ratios (depth/height and height/width) log-uniformly within `ratio`.
        3) Derive crop sizes from the sampled volume + aspect ratios; if valid, crop.
        4) If no valid sample is found, fall back to a central crop of the minimal dimension.
        5) Resize the crop to `out_size` using trilinear interpolation.

        Args:
            volume: (C, D, H, W) tensor
            out_size: Target size (D, H, W) after resizing
            scale: (min_scale, max_scale) for random volume fraction sampling
            ratio: (min_ratio, max_ratio) log-uniform range for aspect ratios; defaults to self.ratio_3d

        Returns:
            cropped: (C, D, H, W) tensor of exactly out_size
            offset: (sd, sh, sw) original crop offset before resizing
        """
        assert volume.ndim == 4
        _, D, H, W = volume.shape
        td, th, tw = out_size
        ratio = ratio if ratio is not None else self.ratio_3d
        log_ratio_low, log_ratio_high = np.log(ratio[0]), np.log(ratio[1])
        volume_total = D * H * W

        def _attempt():
            # target volume sampled uniformly in scale range (matches torchvision RandomResizedCrop)
            target_vol = float(np.random.uniform(scale[0], scale[1]) * volume_total)
            # sample two aspect ratios in log space to avoid skew
            log_r_dh = float(np.random.uniform(log_ratio_low, log_ratio_high))
            log_r_hw = float(np.random.uniform(log_ratio_low, log_ratio_high))
            r_dh = np.exp(log_r_dh)  # depth / height
            r_hw = np.exp(log_r_hw)  # height / width

            # Solve for (d, h, w) given target_vol and ratios
            # target_vol = d * h * w
            # d = r_dh * h ; h = r_hw * w  => target_vol = r_dh * r_hw * h^3
            h = max(1, int(round((target_vol / (r_dh * r_hw)) ** (1.0 / 3.0))))
            d = max(1, int(round(r_dh * h)))
            w = max(1, int(round(h / r_hw)))
            return d, h, w

        # Try multiple times to find a valid crop; fall back if none succeed
        for _ in range(10):
            cd, ch, cw = _attempt()
            if cd <= D and ch <= H and cw <= W:
                break
        else:
            # Fallback: central crop with minimal dimension to keep behavior deterministic when volume is small
            min_d = min(D, H, W)
            cd = ch = cw = min_d

        # Sample random crop position
        if cd >= D:
            sd = 0
        else:
            sd = np.random.randint(0, D - cd + 1)
        
        if ch >= H:
            sh = 0
        else:
            sh = np.random.randint(0, H - ch + 1)
        
        if cw >= W:
            sw = 0
        else:
            sw = np.random.randint(0, W - cw + 1)
        
        # Extract variable-size crop
        cropped = volume[:, sd : sd + cd, sh : sh + ch, sw : sw + cw]
        # Resize to exact target size using 3D interpolation
        if cropped.shape[1:] != out_size:
            # Use trilinear interpolation for 3D resize
            cropped = torch.nn.functional.interpolate(
                cropped.unsqueeze(0),  # Add batch dim: (1, C, D, H, W)
                size=out_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dim: (C, D, H, W)
        # Ensure output is exactly the requested size
        assert cropped.shape[1:] == out_size, f"Expected {out_size}, got {cropped.shape[1:]}"
        
        return cropped, (sd, sh, sw)

    def _maybe_flip(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.horizontal_flips:
            return volume
        # Flip along width as "horizontal" by convention; we could also randomly flip other axes.
        if np.random.rand() < 0.5:
            volume = torch.flip(volume, dims=(-1,))
        return volume

    def __call__(self, volume: torch.Tensor) -> Dict[str, Any]:
        # Assume input is already a tensor (C, D, H, W)
        output: Dict[str, Any] = {}
        output["weak_flag"] = True

        # Global crops
        g1, g1_offset = self._random_3d_crop(
            volume,
            out_size=self.global_crops_size_3d,
            scale=self.global_crops_scale,
        )
        g1 = self._maybe_flip(g1)

        g2, g2_offset = self._random_3d_crop(
            volume,
            out_size=self.global_crops_size_3d,
            scale=self.global_crops_scale,
        )
        g2 = self._maybe_flip(g2)

        g1_norm = self.normalize(g1)
        g2_norm = self.normalize(g2)

        output["global_crops"] = [g1_norm, g2_norm]

        # Teacher crops: for now just share same crops/normalization
        output["global_crops_teacher"] = [g1_norm, g2_norm]

        # Gram teacher crops (optional, using same crops by default)
        if self.gram_teacher_crops_size_3d is not None:
            gt1, _ = self._random_3d_crop(
                volume,
                out_size=self.gram_teacher_crops_size_3d,
                scale=self.global_crops_scale,
            )
            gt2, _ = self._random_3d_crop(
                volume,
                out_size=self.gram_teacher_crops_size_3d,
                scale=self.global_crops_scale,
            )
            gt1 = self.normalize(gt1)
            gt2 = self.normalize(gt2)
            output["gram_teacher_crops"] = [gt1, gt2]

        # Local crops
        local_crops: List[torch.Tensor] = []
        local_offsets: List[Tuple[int, int, int]] = []

        if self.local_crops_subset_of_global_crops:
            # Make local crops inside g1/g2 volumes, respecting patch alignment if desired.
            base_volumes = [g1] * (self.local_crops_number // 2) + [g2] * (
                self.local_crops_number - self.local_crops_number // 2
            )
            gs_d, gs_h, gs_w = self.global_crops_size_3d
            ls_d, ls_h, ls_w = self.local_crops_size_3d
            pd, ph, pw = self.patch_size_3d

            for base in base_volumes:
                # sample offsets in patch space then map back to voxel coordinates
                max_d = max((gs_d - ls_d) // pd, 0)
                max_h = max((gs_h - ls_h) // ph, 0)
                max_w = max((gs_w - ls_w) // pw, 0)
                rd = np.random.randint(0, max_d + 1) * pd if max_d > 0 else 0
                rh = np.random.randint(0, max_h + 1) * ph if max_h > 0 else 0
                rw = np.random.randint(0, max_w + 1) * pw if max_w > 0 else 0

                crop = base[:, rd : rd + ls_d, rh : rh + ls_h, rw : rw + ls_w]
                crop = self.normalize(crop)
                local_crops.append(crop)
                local_offsets.append((rd, rh, rw))
        else:
            for _ in range(self.local_crops_number):
                lc, offset = self._random_3d_crop(
                    volume,
                    out_size=self.local_crops_size_3d,
                    scale=self.local_crops_scale,
                )
                lc = self._maybe_flip(lc)
                lc = self.normalize(lc)
                local_crops.append(lc)
                local_offsets.append(offset)

        output["local_crops"] = local_crops
        output["offsets"] = local_offsets if self.local_crops_subset_of_global_crops else ()

        return output


class DataAugmentationDINO3DMonai(object):
    """
    MONAI-based 3D analogue of DataAugmentationDINO for volumetric CT.

    The goal is to mirror the public API and output structure of
    `DataAugmentationDINO3D` while delegating spatial transforms to MONAI.

    - Input:  tensor of shape (C, D, H, W)
    - Output: dict with keys:
        * "weak_flag"
        * "global_crops"          : list[Tensor]
        * "global_crops_teacher"  : list[Tensor]
        * "local_crops"           : list[Tensor]
        * "gram_teacher_crops"    : optional list[Tensor]
        * "offsets"               : () for now (no patch-aligned offsets)
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_size_3d: Tuple[int, int, int],
        local_crops_size_3d: Tuple[int, int, int],
        gram_teacher_crops_size_3d: Optional[Tuple[int, int, int]] = None,
        gram_teacher_no_distortions: bool = False,
        local_crops_subset_of_global_crops: bool = False,
        patch_size_3d: Tuple[int, int, int] = (16, 16, 16),
        horizontal_flips: bool = True,
        ct_window: Tuple[float, float] = (-1000.0, 2000.0),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        ratio_3d: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        # MONAI / CT specific knobs (all expected to come from cfg.crops.*)
        foreground_threshold: Optional[float] = None,
        foreground_crop_prob: float = 0.0,
        min_foreground_ratio: float = 0.0,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size_3d = global_crops_size_3d
        self.local_crops_size_3d = local_crops_size_3d
        self.gram_teacher_crops_size_3d = gram_teacher_crops_size_3d
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size_3d = patch_size_3d
        self.horizontal_flips = horizontal_flips
        self.ratio_3d = ratio_3d

        # Store CT / foreground parameters for future, more advanced MONAI transforms
        self.ct_window = ct_window
        self.mean = mean
        self.std = std
        self.foreground_threshold = foreground_threshold
        self.foreground_crop_prob = foreground_crop_prob
        self.min_foreground_ratio = min_foreground_ratio

        # Build MONAI spatial transforms. We operate on dictionary data
        # {"img": tensor} to stay compatible with other MONAI components.
        t = monai.transforms

        # Intensity + normalization using MONAI:
        # 1) Clip to CT window and scale to [0, 1] with ScaleIntensityRanged
        # 2) Optionally apply per-channel mean/std with NormalizeIntensityd
        ct_min, ct_max = ct_window
        intensity_transforms: List[monai.transforms.Transform] = [
            t.ScaleIntensityRanged(
                keys=["img"],
                a_min=ct_min,
                a_max=ct_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
        if mean is not None and std is not None:
            intensity_transforms.append(
                t.NormalizeIntensityd(
                    keys=["img"],
                    subtrahend=mean,
                    divisor=std,
                    channel_wise=False,
                )
            )
        self._intensity = t.Compose(intensity_transforms)

        # Global crops pipeline: random resized crop using scale + optional flip.
        global_spatial = [
            RandResizedCrop3Dd(
                keys=["img"],
                roi_size=self.global_crops_size_3d,
                scale=self.global_crops_scale,
                ratio=self.ratio_3d,
            ),
        ]
        if self.horizontal_flips:
            # Treat the last spatial axis as "horizontal" by convention.
            global_spatial.append(
                t.RandFlipd(
                    keys=["img"],
                    prob=0.5,
                    spatial_axis=-1,
                )
            )
        self.global_transform = t.Compose(global_spatial + [self._intensity])

        # Local crops pipeline: analogous but with local crop size + local scale.
        local_spatial = [
            RandResizedCrop3Dd(
                keys=["img"],
                roi_size=self.local_crops_size_3d,
                scale=self.local_crops_scale,
                ratio=self.ratio_3d,
            ),
        ]
        if self.horizontal_flips:
            local_spatial.append(
                t.RandFlipd(
                    keys=["img"],
                    prob=0.5,
                    spatial_axis=-1,
                )
            )
        self.local_transform = t.Compose(local_spatial + [self._intensity])

        # Gram-teacher pipeline: potentially lighter / distortion-free if requested.
        if self.gram_teacher_crops_size_3d is not None:
            gram_spatial = [
                RandResizedCrop3Dd(
                    keys=["img"],
                    roi_size=self.gram_teacher_crops_size_3d,
                    scale=self.global_crops_scale,
                    ratio=self.ratio_3d,
                ),
            ]
            if not self.gram_teacher_no_distortions and self.horizontal_flips:
                gram_spatial.append(
                    t.RandFlipd(
                        keys=["img"],
                        prob=0.5,
                        spatial_axis=-1,
                    )
                )
            self.gram_transform = t.Compose(gram_spatial + [self._intensity])
        else:
            self.gram_transform = None

    def __call__(self, volume: torch.Tensor) -> Dict[str, Any]:
        """
        Apply MONAI-based 3D DINO augmentations.

        Args:
            volume: Tensor of shape (C, D, H, W).
        """
        assert volume.ndim == 4, f"Expected (C, D, H, W), got {tuple(volume.shape)}"

        output: Dict[str, Any] = {}
        output["weak_flag"] = True

        # Global crops (2 views).
        data = {"img": volume}
        g1_dict = self.global_transform(data)
        g2_dict = self.global_transform(data)
        g1 = g1_dict["img"]
        g2 = g2_dict["img"]

        output["global_crops"] = [g1, g2]
        # For now, teacher sees the same crops/normalization; can be made lighter later.
        output["global_crops_teacher"] = [g1, g2]

        # Gram-teacher crops, if enabled.
        if self.gram_transform is not None:
            gt1 = self.gram_transform(data)["img"]
            gt2 = self.gram_transform(data)["img"]
            output["gram_teacher_crops"] = [gt1, gt2]

        # Local crops.
        local_crops: List[torch.Tensor] = []
        # NOTE: MONAI transforms don't expose crop coordinates directly;
        # for now, we don't return offsets (same as the non-subset path
        # in the original 3D augmentation).
        for _ in range(self.local_crops_number):
            lc_dict = self.local_transform(data)
            local_crops.append(lc_dict["img"])

        output["local_crops"] = local_crops
        output["offsets"] = ()  # no patch-aligned offsets from MONAI crops (yet)

        return output
