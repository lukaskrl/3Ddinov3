import os
import sys

import torch

# Allow running tests from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.models.vision_transformer import DinoVisionTransformer, DinoVisionTransformer3D


def _print_stats_tensor(name: str, x: torch.Tensor) -> None:
    if x.numel() == 0:
        print(f"{name}: shape={tuple(x.shape)}, empty tensor")
    else:
        print(
            f"{name}: shape={tuple(x.shape)}, "
            f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
            f"mean={x.mean().item():.4f}, finite={torch.isfinite(x).all().item()}"
        )


def test_dino3d_forward_features_and_intermediate_layers():
    """Test the 3D ViT backbone end-to-end on a dummy volume."""

    B, C, D, H, W = 2, 1, 64, 192, 192
    x = torch.zeros(B, C, D, H, W)

    model = DinoVisionTransformer3D(
        img_size=H,
        patch_size=16,
        patch_size_d=2,
        in_chans=C,
        embed_dim=960,
        depth=4,
        num_heads=16,
    )
    model.init_weights()

    # forward_features
    with torch.no_grad():
        out = model.forward_features(x, masks=None)

    assert isinstance(out, dict)
    for key in ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"]:
        assert key in out
        tensor = out[key]
        assert torch.isfinite(tensor).all(), f"NaNs in 3D {key}"
        _print_stats_tensor(f"3d_{key}", tensor)

    # get_intermediate_layers last block
    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)

    assert isinstance(feats, tuple) and len(feats) == 1
    vol_feat, cls_token = feats[0]
    assert vol_feat.ndim == 5  # (B, C, D', H', W')
    assert cls_token.shape[0] == B
    assert torch.isfinite(vol_feat).all()
    assert torch.isfinite(cls_token).all()

    _print_stats_tensor("3d_vol_feat", vol_feat)
    _print_stats_tensor("3d_cls_token", cls_token)


def test_dino2d_forward_features_and_intermediate_layers():
    """Analogous test for 2D DinoVisionTransformer."""

    B, C, H, W = 2, 3, 224, 224
    x = torch.zeros(B, C, H, W)

    model = DinoVisionTransformer(
        img_size=H,
        patch_size=16,
        in_chans=C,
        embed_dim=768,
        depth=4,
        num_heads=12,
    )
    model.init_weights()

    # forward_features
    with torch.no_grad():
        out = model.forward_features(x, masks=None)

    assert isinstance(out, dict)
    for key in ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"]:
        assert key in out
        tensor = out[key]
        assert torch.isfinite(tensor).all(), f"NaNs in 2D {key}"
        _print_stats_tensor(f"2d_{key}", tensor)

    # get_intermediate_layers last block
    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)

    assert isinstance(feats, tuple) and len(feats) == 1
    feat_map, cls_token = feats[0]
    # 2D reshape=True returns (B, C, H', W')
    assert feat_map.ndim == 4
    assert cls_token.shape[0] == B
    assert torch.isfinite(feat_map).all()
    assert torch.isfinite(cls_token).all()

    _print_stats_tensor("2d_feat_map", feat_map)
    _print_stats_tensor("2d_cls_token", cls_token)
