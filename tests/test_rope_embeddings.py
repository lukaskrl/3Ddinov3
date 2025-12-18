import os
import sys

import torch

# Allow running tests from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.layers.rope_position_encoding import RopePositionEmbedding, RopePositionEmbedding3D


def test_rope_2d_basic():
    embed_dim = 768
    num_heads = 12
    H, W = 14, 14

    rope = RopePositionEmbedding(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=100.0,
        min_period=None,
        max_period=None,
    )

    sin, cos = rope(H=H, W=W)
    assert sin.shape == cos.shape
    assert sin.shape[0] == H * W  # HW positions
    assert sin.shape[1] == embed_dim // num_heads  # per-head dim

    print("2D RoPE sin/cos shape:", sin.shape)
    assert torch.isfinite(sin).all()
    assert torch.isfinite(cos).all()


def test_rope_3d_basic():
    # Use settings consistent with 3D constraint (per-head dim % 6 == 0)
    embed_dim = 960
    num_heads = 16
    D, H, W = 4, 8, 8

    rope3d = RopePositionEmbedding3D(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=100.0,
        min_period=None,
        max_period=None,
    )

    sin, cos = rope3d(D=D, H=H, W=W)
    assert sin.shape == cos.shape
    assert sin.shape[0] == D * H * W
    assert sin.shape[1] == embed_dim // num_heads

    print("3D RoPE sin/cos shape:", sin.shape)
    assert torch.isfinite(sin).all()
    assert torch.isfinite(cos).all()
