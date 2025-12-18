# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def make_3tuple(x):
    """
    Utility mirroring make_2tuple, but for (D, H, W).
    Accepts either an int or a length‑3 tuple.
    """
    if isinstance(x, tuple):
        assert len(x) == 3
        return x

    assert isinstance(x, int)
    return (x, x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class PatchEmbed3D(nn.Module):
    """
    3D volumetric patch embedding: (B, C, D, H, W) -> (B, N, D_embed) or (B, D_p, H_p, W_p, D_embed)

    This is a direct 3D analogue of `PatchEmbed` using Conv3d. When `flatten_embedding`
    is False we keep a 3D grid of patches which is useful for passing structured shapes
    to the transformer (e.g. for positional encoding).
    """

    def __init__(
        self,
        img_size_3d: Union[int, Tuple[int, int, int]] = (64, 224, 224),
        patch_size_3d: Union[int, Tuple[int, int, int]] = (2, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        volume_DHW = make_3tuple(img_size_3d)
        patch_DHW = make_3tuple(patch_size_3d)
        patch_grid_size = (
            volume_DHW[0] // patch_DHW[0] if volume_DHW[0] > 0 else 0,
            volume_DHW[1] // patch_DHW[1] if volume_DHW[1] > 0 else 0,
            volume_DHW[2] // patch_DHW[2] if volume_DHW[2] > 0 else 0,
        )

        self.img_size_3d = volume_DHW
        self.patch_size_3d = patch_DHW
        self.patches_resolution = patch_grid_size
        self.num_patches = (
            patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]
            if all(patch_grid_size)
            else 0
        )

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_DHW,
            stride=patch_DHW,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, D, H, W)
        assert x.ndim == 5, f"PatchEmbed3D expects (B, C, D, H, W) but got {tuple(x.shape)}"
        x = self.proj(x)  # (B, embed_dim, D_p, H_p, W_p)
        D, H, W = x.size(2), x.size(3), x.size(4)
        if self.flatten_embedding:
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
            x = self.norm(x)
        else:
            # Keep structured 3D grid, with channels last to mirror 2D PatchEmbed
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D_p, H_p, W_p, embed_dim)
            x = self.norm(x)
        return x

    def flops(self) -> float:
        Dp, Hp, Wp = self.patches_resolution
        flops = (
            Dp
            * Hp
            * Wp
            * self.embed_dim
            * self.in_chans
            * (self.patch_size_3d[0] * self.patch_size_3d[1] * self.patch_size_3d[2])
        )
        if self.norm is not None:
            flops += Dp * Hp * Wp * self.embed_dim
        return flops

    def reset_parameters(self):
        # Simple uniform init analogous to 2D case but for 3D kernels.
        k = 1 / (self.in_chans * (self.patch_size_3d[0] * self.patch_size_3d[1] * self.patch_size_3d[2]))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
