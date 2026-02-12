# %% Imports
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.transforms_3d import make_ct_3d_base_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval


#Paths and basic config
# Path to a single distributed checkpoint directory (integer‑named subdir)
# path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/mri_no_accumulation_efficiency/ckpt/108799"
path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/mri_hrft_resumed/ckpt/7199"

# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Directory with CT volumes stored as .npy files
path_to_data = "/home/lukas/data/brain-t1-dataset"

# Training config used for this checkpoint (3D CT DINOv3 config)
# path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_mri3d_config.yaml"
path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_mri3d_stage3_hrft.yaml"


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
    volume_files = sorted(
        p
        for p in root_path.iterdir()
        if p.is_file() and (p.name.endswith(".npy") or p.name.endswith(".nii.gz"))
    )
    if not volume_files:
        raise RuntimeError(f"No .npy or .nii.gz volumes found under {data_root}")
    return str(volume_files[0])


def _strip_nii_gz_suffix(path_str: str) -> str:
    name = Path(path_str).name
    return name[:-7] if name.endswith(".nii.gz") else Path(path_str).stem


def _show_activation_visuals(activation_volume: np.ndarray) -> None:
    """
    Save center slices and MIP views for a 3D activation volume.
    activation_volume: (D, H, W) numpy array
    """
    d, h, w = activation_volume.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2


    axial = activation_volume[d_mid, :, :]
    coronal = activation_volume[:, h_mid, :]
    sagittal = activation_volume[:, :, w_mid]

    mip_d = activation_volume.max(axis=0)
    mip_h = activation_volume.max(axis=1)
    mip_w = activation_volume.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial, cmap="viridis")
    axes[0].set_title("Center axial (D)")
    axes[1].imshow(coronal, cmap="viridis")
    axes[1].set_title("Center coronal (H)")
    axes[2].imshow(sagittal, cmap="viridis")
    axes[2].set_title("Center sagittal (W)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mip_d, cmap="viridis")
    axes[0].set_title("MIP over D")
    axes[1].imshow(mip_h, cmap="viridis")
    axes[1].set_title("MIP over H")
    axes[2].imshow(mip_w, cmap="viridis")
    axes[2].set_title("MIP over W")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(6, 4))
    plt.hist(activation_volume.ravel(), bins=100, color="#4C78A8", alpha=0.9)
    plt.title("Activation magnitude histogram")
    plt.xlabel("Activation magnitude")
    plt.ylabel("Count")
    fig.tight_layout()
    plt.show()


def _show_input_visuals(volume: np.ndarray, title_prefix: str = "Input") -> None:
    """
    Visualize center slices and MIP views for the input 3D volume.
    volume: (D, H, W) numpy array
    """
    d, h, w = volume.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    axial = volume[d_mid, :, :]
    coronal = volume[:, h_mid, :]
    sagittal = volume[:, :, w_mid]

    mip_d = volume.max(axis=0)
    mip_h = volume.max(axis=1)
    mip_w = volume.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial, cmap="gray")
    axes[0].set_title(f"{title_prefix} center axial (D)")
    axes[1].imshow(coronal, cmap="gray")
    axes[1].set_title(f"{title_prefix} center coronal (H)")
    axes[2].imshow(sagittal, cmap="gray")
    axes[2].set_title(f"{title_prefix} center sagittal (W)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mip_d, cmap="gray")
    axes[0].set_title(f"{title_prefix} MIP over D")
    axes[1].imshow(mip_h, cmap="gray")
    axes[1].set_title(f"{title_prefix} MIP over H")
    axes[2].imshow(mip_w, cmap="gray")
    axes[2].set_title(f"{title_prefix} MIP over W")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def _show_topk_channel_mips(features: np.ndarray, channel_indices: list[int]) -> None:
    """
    Save MIP visualizations for selected feature channels.
    features: (C, D, H, W) numpy array
    """
    n = len(channel_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, ch in zip(axes, channel_indices):
        mip = features[ch].max(axis=0)
        ax.imshow(mip, cmap="viridis")
        ax.set_title(f"Channel {ch} MIP")
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def _show_overlay_visuals(
    input_volume: np.ndarray,
    similarity_volume: np.ndarray,
    title_prefix: str = "Cosine similarity overlay",
    alpha: float = 0.45,
) -> None:
    """
    Overlay similarity map on input volume center slices.
    input_volume: (D, H, W) numpy array
    similarity_volume: (D, H, W) numpy array in [0, 1]
    """
    d, h, w = input_volume.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    axial_img = input_volume[d_mid, :, :]
    coronal_img = input_volume[:, h_mid, :]
    sagittal_img = input_volume[:, :, w_mid]

    axial_sim = similarity_volume[d_mid, :, :]
    coronal_sim = similarity_volume[:, h_mid, :]
    sagittal_sim = similarity_volume[:, :, w_mid]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial_img, cmap="gray")
    axes[0].imshow(axial_sim, cmap="inferno", alpha=alpha, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"{title_prefix} axial")

    axes[1].imshow(coronal_img, cmap="gray")
    axes[1].imshow(coronal_sim, cmap="inferno", alpha=alpha, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{title_prefix} coronal")

    axes[2].imshow(sagittal_img, cmap="gray")
    axes[2].imshow(sagittal_sim, cmap="inferno", alpha=alpha, vmin=0.0, vmax=1.0)
    axes[2].set_title(f"{title_prefix} sagittal")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()



# %% Script entry point (optional when used in Jupyter)
#%%
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3D DINOv3 inference script.")

    # After setting CUDA_VISIBLE_DEVICES, the selected GPU is cuda:0
    torch.cuda.set_device(0)

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

    # Visualize input volume (center slices + MIPs) before moving to GPU
    input_volume_np = volume[0].detach().cpu().numpy()
    _show_input_visuals(input_volume_np, title_prefix="Input")

    # Add batch dim and move to GPU: (1, C, D, H, W)
    volume = volume.unsqueeze(0).cuda(non_blocking=True)

    # --- Forward pass to get feature maps ---
    model_requires_kwargs = {"is_training": False} if "is_training" in model.forward.__code__.co_varnames else {}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
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

    # --- 3D feature map statistics & visualizations ---
    with torch.no_grad():
        feat_cpu = vol_feats.detach().float().cpu()[0]  # (C, D, H, W)
        activation = torch.linalg.vector_norm(feat_cpu, dim=0)  # (D, H, W)

        # Cosine similarity to center feature vector
        c, d_f, h_f, w_f = feat_cpu.shape
        d_mid_f, h_mid_f, w_mid_f = d_f // 2, h_f // 2, w_f // 2
        center_feat = feat_cpu[:, d_mid_f, h_mid_f, w_mid_f]  # (C,)
        feat_norm = F.normalize(feat_cpu, dim=0)
        center_norm = F.normalize(center_feat, dim=0)
        sim_map = (feat_norm * center_norm[:, None, None, None]).sum(dim=0)
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)

        # Upsample similarity map to input resolution for overlay
        sim_map_up = F.interpolate(
            sim_map[None, None, ...],
            size=input_volume_np.shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        sim_map_np = sim_map_up.cpu().numpy()

        activation_np = activation.numpy()
        a_mean = float(activation.mean())
        a_std = float(activation.std())
        a_min = float(activation.min())
        a_max = float(activation.max())
        p1, p5, p50, p95, p99 = np.percentile(activation_np, [1, 5, 50, 95, 99])
        high_activity_ratio = float((activation_np > p95).mean())

        channel_mean = feat_cpu.abs().mean(dim=(1, 2, 3))
        topk = min(5, channel_mean.numel())
        topk_vals, topk_idx = torch.topk(channel_mean, k=topk)

    print("\nFeature map activation stats (L2 over channels):")
    print(
        "  mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}, p1={:.6f}, p5={:.6f}, p50={:.6f}, p95={:.6f}, p99={:.6f}".format(
            a_mean, a_std, a_min, a_max, p1, p5, p50, p95, p99
        )
    )
    print(f"  high-activity ratio (>p95): {high_activity_ratio:.4f}")

    print("\nTop channels by mean |activation|:")
    for rank, (idx, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
        print(f"  {rank:02d}. channel={idx} mean|act|={val:.6f}")

    _show_activation_visuals(activation_np)
    _show_topk_channel_mips(feat_cpu.numpy(), topk_idx.tolist()[:3])
    _show_overlay_visuals(input_volume_np, sim_map_np, title_prefix="Cosine similarity")