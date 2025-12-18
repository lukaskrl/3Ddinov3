import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.data.augmentations_3d import DataAugmentationDINO3D
from dinov3.data.masking import MaskingGenerator, MaskingGenerator3D
from dinov3.models.vision_transformer import DinoVisionTransformer, DinoVisionTransformer3D


def test_2d_augmentation_basic():
    img = torch.randn(3, 224, 224)
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
    out = aug(img)
    assert "global_crops" in out and "local_crops" in out
    assert len(out["global_crops"]) == 2
    assert len(out["local_crops"]) == 2


def test_masking_generators_2d_vs_3d_token_count():
    H_p, W_p = 8, 8
    D_p = 4
    N_2d = H_p * W_p
    N_3d = D_p * H_p * W_p

    gen2d = MaskingGenerator(input_size=(H_p, W_p), max_num_patches=int(0.5 * N_2d))
    gen3d = MaskingGenerator3D(input_size=(D_p, H_p, W_p), max_num_patches=int(0.5 * N_3d))

    num_mask_2d = int(0.25 * N_2d)
    num_mask_3d = int(0.25 * N_3d)

    mask2d = gen2d(num_masking_patches=num_mask_2d)
    mask3d = gen3d(num_masking_patches=num_mask_3d)

    assert mask2d.shape == (H_p, W_p)
    assert mask3d.shape == (D_p, H_p, W_p)
    assert mask2d.sum() == num_mask_2d
    assert mask3d.sum() == num_mask_3d


def test_2d_vs_3d_backbone_no_nan():
    # 2D backbone
    x2d = torch.zeros(2, 3, 224, 224)
    model2d = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=2,
        num_heads=12,
    )
    model2d.init_weights()
    with torch.no_grad():
        out2d = model2d.forward_features(x2d, masks=None)
    for key in ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"]:
        assert key in out2d
        assert torch.isfinite(out2d[key]).all()

    # 3D backbone
    x3d = torch.zeros(2, 1, 64, 192, 192)
    model3d = DinoVisionTransformer3D(
        img_size=192,
        patch_size=16,
        patch_size_d=2,
        in_chans=1,
        embed_dim=960,
        depth=2,
        num_heads=16,
    )
    model3d.init_weights()
    with torch.no_grad():
        out3d = model3d.forward_features(x3d, masks=None)
    for key in ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"]:
        assert key in out3d
        assert torch.isfinite(out3d[key]).all()
