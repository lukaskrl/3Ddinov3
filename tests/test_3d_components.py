import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.augmentations_3d import DataAugmentationDINO3D
from dinov3.data.masking import MaskingGenerator3D
from dinov3.models.vision_transformer import DinoVisionTransformer3D


def test_ct_volume_dataset_basic(tmp_path):
    # Create a fake 3D volume on disk
    vol = np.random.randn(32, 64, 64).astype("float32")
    npy_path = tmp_path / "vol_0.npy"
    np.save(npy_path, vol)

    ds = CTVolumeDataset(root=str(tmp_path))
    assert len(ds) == 1

    tensor, target = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 32, 64, 64)
    assert torch.isfinite(tensor).all()
    assert target == 0


def test_3d_augmentation_output_shapes_and_finiteness():
    # Simple synthetic volume (C, D, H, W)
    vol = torch.randn(1, 64, 96, 96)

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

    out = aug(vol)
    assert "global_crops" in out and "local_crops" in out
    assert len(out["global_crops"]) == 2
    assert len(out["local_crops"]) == 2

    for g in out["global_crops"]:
        assert g.shape == (1, 32, 64, 64)
        assert torch.isfinite(g).all()

    for l in out["local_crops"]:
        assert l.shape == (1, 16, 32, 32)
        assert torch.isfinite(l).all()


def test_masking_generator_3d_shapes_and_counts():
    D_p, H_p, W_p = 4, 8, 8
    N = D_p * H_p * W_p
    gen = MaskingGenerator3D(input_size=(D_p, H_p, W_p), max_num_patches=int(0.5 * N))

    num_mask = int(0.3 * N)
    mask = gen(num_masking_patches=num_mask)

    assert mask.shape == (D_p, H_p, W_p)
    assert mask.dtype == np.bool_

    # Exact count is enforced via complete_mask_randomly
    assert mask.sum() == num_mask


def test_dinovisiontransformer3d_forward_no_nans():
    # Choose parameters consistent with 3D RoPE constraint (per-head dim % 6 == 0)
    B, C, D, H, W = 2, 1, 64, 192, 192
    x = torch.zeros(B, C, D, H, W)

    model = DinoVisionTransformer3D(
        img_size=H,
        patch_size=16,
        patch_size_d=2,
        in_chans=C,
        embed_dim=960,
        depth=2,       # keep small for unit test speed
        num_heads=16,
    )
    model.init_weights()

    with torch.no_grad():
        out = model.forward_features(x, masks=None)

    assert isinstance(out, dict)
    # Check that key tensors exist and are finite
    for key in ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"]:
        assert key in out
        tensor = out[key]
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()
