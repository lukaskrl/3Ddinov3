# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        norm_last_layer=False,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        if norm_last_layer:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        else:
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(self, 'norm_last_layer') and self.norm_last_layer:
                # With weight_norm, initialize weight_v (the actual parameter), not weight (computed property)
                # weight_g (initialized to 1) controls the magnitude separately.
                if hasattr(m, 'weight_v'):
                    # This is a weight_norm'd layer - must initialize weight_v directly
                    trunc_normal_(m.weight_v, std=0.02)
                    if hasattr(m, 'weight_g'):
                        # Ensure weight_g stays at 1 as intended
                        m.weight_g.data.fill_(1)
                else:
                    # Fallback for non-weight_norm'd layers (shouldn't happen in practice)
                    trunc_normal_(m.weight, std=0.02)
            elif hasattr(self, 'last_layer') and m is self.last_layer:
                # Without weight_norm, use larger std for the last layer
                # so Sinkhorn-Knopp has enough logit variance to preserve structure.
                trunc_normal_(m.weight, std=0.3)
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
