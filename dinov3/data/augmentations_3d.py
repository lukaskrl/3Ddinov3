from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .transforms_3d import make_ct_3d_base_transform


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

        self.normalize = make_ct_3d_base_transform(window=ct_window, mean=mean, std=std)

    def _random_3d_crop(
        self,
        volume: torch.Tensor,
        out_size: Tuple[int, int, int],
        scale: Tuple[float, float],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Random 3D crop on (C, D, H, W).
        We ignore true area scaling for simplicity and instead just pick a random
        start index such that the crop fits.
        """
        assert volume.ndim == 4
        _, D, H, W = volume.shape
        td, th, tw = out_size

        if td > D or th > H or tw > W:
            # If requested crop is larger than volume, just center‑crop / padless crop.
            sd = max((D - td) // 2, 0)
            sh = max((H - th) // 2, 0)
            sw = max((W - tw) // 2, 0)
        else:
            sd = np.random.randint(0, D - td + 1)
            sh = np.random.randint(0, H - th + 1)
            sw = np.random.randint(0, W - tw + 1)

        cropped = volume[:, sd : sd + td, sh : sh + th, sw : sw + tw]
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

