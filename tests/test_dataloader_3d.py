import os
import sys

import numpy as np
import torch

# Allow running tests from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.augmentations_3d import DataAugmentationDINO3D
from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator3D, MaskingGenerator


def _print_stats(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, min={x.min().item():.4f}, max={x.max().item():.4f}, "
        f"mean={x.mean().item():.4f}, finite={torch.isfinite(x).all().item()}"
    )


def test_3d_dataloader_single_batch(tmp_path):
    """End-to-end test of 3D data path: dataset -> aug -> collate.

    Uses a single synthetic CT volume saved as .npy and checks that all
    intermediate tensors are finite and have expected shapes.
    """

    # 1) Create a dummy 3D volume on disk: (D, H, W)
    D, H, W = 64, 96, 96
    vol = np.random.randn(D, H, W).astype("float32") * 100.0  # some HU-like range
    npy_path = tmp_path / "vol_0.npy"
    np.save(npy_path, vol)

    # 2) Dataset
    ds = CTVolumeDataset(root=str(tmp_path))
    assert len(ds) == 1

    vol_tensor, target = ds[0]
    assert vol_tensor.shape == (1, D, H, W)
    assert target == 0
    assert torch.isfinite(vol_tensor).all()
    _print_stats("raw_volume", vol_tensor)

    # 3) 3D augmentation
    aug = DataAugmentationDINO3D(
        global_crops_scale=(0.5, 1.0),
        local_crops_scale=(0.1, 0.5),
        local_crops_number=2,
        global_crops_size_3d=(32, 64, 64),
        local_crops_size_3d=(16, 32, 32),
        gram_teacher_crops_size_3d=None,
        gram_teacher_no_distortions=False,
        local_crops_subset_of_global_crops=False,
        patch_size_3d=(2, 16, 16),
        horizontal_flips=True,
        ct_window=(-1000.0, 400.0),
        mean=None,
        std=None,
    )

    sample = (aug(vol_tensor), target)

    # 4) Collate function + masking (single batch of size 1)
    D_g, H_g, W_g = 32, 64, 64
    D_p, H_p, W_p = 2, 16, 16
    n_tokens = (D_g // D_p) * (H_g // H_p) * (W_g // W_p)
    mask_gen = MaskingGenerator3D(input_size=(D_g // D_p, H_g // H_p, W_g // W_p), max_num_patches=int(0.5 * n_tokens))

    batch = collate_data_and_cast(
        samples_list=[sample],
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_gen,
        random_circular_shift=False,
        local_batch_size=None,
    )

    # 5) Check shapes and finiteness
    for key in ["collated_global_crops", "collated_local_crops"]:
        assert key in batch
        tensor = batch[key]
        assert torch.isfinite(tensor).all()
        _print_stats(key, tensor)

    masks = batch["collated_masks"]
    assert masks.dtype == torch.bool
    _print_stats("collated_masks", masks.float())

    masks_weight = batch["masks_weight"]
    assert torch.isfinite(masks_weight).all()
    _print_stats("masks_weight", masks_weight)

    # No NaNs anywhere in the batch dictionary
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), f"NaNs detected in {key}"


def test_2d_dataloader_single_batch():
    """Analogous end-to-end test for 2D path: image -> 2D aug -> collate."""

    # 1) Dummy RGB image (C, H, W)
    C, H, W = 3, 224, 224
    img = torch.randn(C, H, W)
    _print_stats("raw_image", img)

    # 2) 2D augmentation (same hyper-params style as DINOv2)
    aug = DataAugmentationDINO(
        global_crops_scale=(0.5, 1.0),
        local_crops_scale=(0.1, 0.5),
        local_crops_number=2,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
    )

    sample = (aug(img), 0)

    # 3) Collate + 2D masking (single sample)
    img_size = 224
    patch_size = 16
    n_tokens = (img_size // patch_size) ** 2
    mask_gen = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=int(0.5 * n_tokens),
    )

    batch = collate_data_and_cast(
        samples_list=[sample],
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_gen,
        random_circular_shift=False,
        local_batch_size=None,
    )

    for key in ["collated_global_crops", "collated_local_crops"]:
        assert key in batch
        tensor = batch[key]
        assert torch.isfinite(tensor).all()
        _print_stats(f"2d_{key}", tensor)

    masks = batch["collated_masks"]
    _print_stats("2d_collated_masks", masks.float())
    masks_weight = batch["masks_weight"]
    _print_stats("2d_masks_weight", masks_weight)

    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), f"NaNs detected in 2D {key}"
