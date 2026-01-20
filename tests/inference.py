# %% Imports
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.transforms_3d import make_ct_3d_base_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval


# %% Paths and basic config
# Path to a single distributed checkpoint directory (integer‑named subdir)
path_to_checkpoint = "/home/lukas/projects/3Ddinov3/work_dir/artillery_gram_anchoring/ckpt/51599"

# Directory with CT volumes stored as .npy files
path_to_data = "/home/lukas/data/ARTILLERY/images_binary"

# Training config used for this checkpoint (3D CT DINOv3 config)
path_to_config = "/home/lukas/projects/3Ddinov3/dinov3/configs/ssl_ct3d_config.yaml"


# %% Model loading utilities
def load_backbone_from_checkpoint(
    config_file: str,
    ckpt_dir: str,
) -> tuple[torch.nn.Module, torch.dtype]:
    """
    Build the 3D DINOv3 backbone and load weights from a distributed checkpoint.
    Returns the model in eval mode on CUDA and the autocast dtype.
    """
    # Ensure the DINOv3 distributed utilities are initialized, even for single‑GPU.
    if not distributed.is_enabled():
        distributed.enable()

    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=ckpt_dir,
        shard_unsharded_model=False,
        output_dir="",
        opts=[],
    )
    config = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(config, setup_args.pretrained_weights)
    model.cuda()
    model.eval()

    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype, config


def build_ct_dataset(data_root: str, config) -> CTVolumeDataset:
    """
    Build a simple CT dataset that mirrors the preprocessing used in training:
    windowing + normalization, but without random cropping/augmentations.
    """
    # Pull CT window / normalization from the config if available.
    ct_window = getattr(config.crops, "ct_window", (-1000.0, 400.0))
    ct_mean = getattr(config.crops, "ct_mean", None)
    ct_std = getattr(config.crops, "ct_std", None)

    # YAML may specify null -> becomes None in config
    mean = ct_mean if ct_mean is not None else None
    std = ct_std if ct_std is not None else None

    base_transform = make_ct_3d_base_transform(window=tuple(ct_window), mean=mean, std=std)

    dataset = CTVolumeDataset(
        root=data_root,
        transform=base_transform,
        split=None,
    )
    return dataset


def pick_first_volume_path(data_root: str) -> str:
    """
    Helper to pick the first .npy file in a directory, for convenience.
    """
    root_path = Path(data_root)
    npy_files = sorted(p for p in root_path.glob("*.npy") if p.is_file())
    if not npy_files:
        raise RuntimeError(f"No .npy volumes found under {data_root}")
    return str(npy_files[0])



# %% Script entry point (optional when used in Jupyter)
#%%
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3D DINOv3 inference script.")

    # --- Load model ---
    print(f"Loading model from config={path_to_config} and checkpoint={path_to_checkpoint}")
    model, autocast_dtype, config = load_backbone_from_checkpoint(
        config_file=path_to_config,
        ckpt_dir=path_to_checkpoint,
    )

    # --- Load data ---
    volume_path = pick_first_volume_path(path_to_data)
    print(f"Using volume: {volume_path}")

    dataset = build_ct_dataset(path_to_data, config)
    # Find index of the chosen volume in the dataset
    idx = dataset._paths.index(volume_path)
    volume, _ = dataset[idx]  # (C, D, H, W)

    # Add batch dim and move to GPU: (1, C, D, H, W)
    volume = volume.unsqueeze(0).cuda(non_blocking=True)

    # --- Forward pass to get feature maps ---
    model_requires_kwargs = {"is_training": False} if "is_training" in model.forward.__code__.co_varnames else {}

    with torch.no_grad(), torch.cuda.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        # For DINOv3 backbones, it's more informative to use forward_features
        feats = model.forward_features(volume)

        # Also fetch last-layer volumetric feature map via get_intermediate_layers (reshaped)
        last_vol_feats_tuple = model.get_intermediate_layers(
            volume,
            n=1,
            reshape=True,
            return_class_token=True,
            return_extra_tokens=False,
            norm=True,
        )

    # feats is a dict with keys like:
    #   "x_norm_clstoken": (B, C)
    #   "x_storage_tokens": (B, S, C)
    #   "x_norm_patchtokens": (B, N, C)
    cls_token = feats["x_norm_clstoken"]  # (1, C)
    patch_tokens = feats["x_norm_patchtokens"]  # (1, N, C)

    # last_vol_feats_tuple is a tuple with one element because n=1
    (vol_feats, vol_cls_token) = last_vol_feats_tuple[0]
    # vol_feats: (B, C, D', H', W')

    print("\n=== Inference statistics ===")
    print(f"Input volume shape (B, C, D, H, W): {tuple(volume.shape)}")
    print(f"CLS token shape: {tuple(cls_token.shape)}")
    print(f"Patch tokens shape (B, N, C): {tuple(patch_tokens.shape)}")
    print(f"Last-layer volumetric feature map shape (B, C, D', H', W'): {tuple(vol_feats.shape)}")

    # CLS token stats
    cls_mean = cls_token.mean().item()
    cls_std = cls_token.std().item()
    cls_min = cls_token.min().item()
    cls_max = cls_token.max().item()
    print("\nCLS token stats:")
    print(f"  mean={cls_mean:.6f}, std={cls_std:.6f}, min={cls_min:.6f}, max={cls_max:.6f}")

    # Patch token stats (flatten over tokens and channels)
    patch_mean = patch_tokens.mean().item()
    patch_std = patch_tokens.std().item()
    patch_min = patch_tokens.min().item()
    patch_max = patch_tokens.max().item()
    print("\nPatch token stats (all tokens, all channels):")
    print(f"  mean={patch_mean:.6f}, std={patch_std:.6f}, min={patch_min:.6f}, max={patch_max:.6f}")

    # Volumetric feature map stats
    vol_mean = vol_feats.mean().item()
    vol_std = vol_feats.std().item()
    vol_min = vol_feats.min().item()
    vol_max = vol_feats.max().item()
    print("\nLast-layer volumetric feature map stats:")
    print(f"  mean={vol_mean:.6f}, std={vol_std:.6f}, min={vol_min:.6f}, max={vol_max:.6f}")