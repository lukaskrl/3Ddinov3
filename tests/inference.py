# %% Imports
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import nibabel as nib
from scipy import ndimage

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config

from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.transforms_3d import make_ct_3d_base_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval


#Paths and basic config
# Path to a single distributed checkpoint directory (integer‑named subdir)
path_to_checkpoint = "/home/lukas/projects/3Ddinov3/work_dir/failure_1gpu/checkpoint_51399.pth"
# path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/mri_full_training_centering.pth"


# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Directory with CT volumes stored as .npy files
path_to_data = "/home/lukas/data/brain-t1-dataset"


# Training config used for this checkpoint (3D CT DINOv3 config)
# path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_mri3d_config.yaml"
path_to_config = "/home/lukas/projects/3Ddinov3/work_dir/failure_1gpu/config.yaml"
# path_to_config = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/config.yaml"

# Optional resizing for more fine-grained features
# sample_path = "/home/lukas/data/brain-t1-dataset/a9e965db61d02e2772f4819290c2362f.nii.gz"
sample_path = None
resize_shape = (1000, 1000, 1000)  # If set, must be a tuple of (D, H, W) for the desired input size. Overrides resize_scale if both are set.
resize_scale = None
# sample_path = "/home/lukas/data/OASIS/nifti_converted/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.nii.gz"  # If set, must be a .npy or .nii.gz file path to a single volume for testing instead of dataset
head_mask_threshold = None
use_head_mask_for_pca = False  # If True: fit PCA on masked feature voxels and visualize PCA maps only within mask.
pca_component_offset = 10  # Which PCA component to start from (0=components 0-2, 3=components 3-5, etc.)

def load_backbone_from_checkpoint(
    config_file: str,
    ckpt_dir: str,
):
    """
    Build the 3D DINOv3 backbone and load weights from a distributed checkpoint.
    Returns the model in eval mode on CUDA, the autocast dtype, and the config.
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
    ct_window = getattr(config.crops, "ct_window", (0, 1000.0))
    ct_mean = getattr(config.crops, "ct_mean", None)
    ct_std = getattr(config.crops, "ct_std", None)

    # YAML may specify null -> becomes None in config
    mean = ct_mean if ct_mean is not None else None
    std = ct_std if ct_std is not None else None

    base_transform = make_ct_3d_base_transform(window=ct_window, mean=mean, std=std)

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


def load_single_volume_from_file(
    file_path: str,
    transform,
) -> torch.Tensor:
    """
    Load a single volume from a .npy or .nii.gz file and apply the transform.
    
    Args:
        file_path: Path to the .npy or .nii.gz file
        transform: Transform function to apply (e.g., ct_window_and_normalize)
    
    Returns:
        volume: Tensor of shape (C, D, H, W)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Volume file not found: {file_path}")
    
    # Load the volume
    if path.suffix == '.npy':
        arr = np.load(file_path)
    elif file_path.endswith('.nii.gz') or path.suffix == '.gz':
        nii = nib.load(file_path)
        arr = nii.get_fdata()
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Expected .npy or .nii.gz")
    
    arr = arr.squeeze()
    # Convert to tensor and add channel dimension if needed
    if arr.ndim == 3:
        # Single channel: (D, H, W) -> (1, D, H, W)
        volume = torch.from_numpy(arr).float().unsqueeze(0)
    elif arr.ndim == 4:
        # Already has channel dimension: (C, D, H, W)
        volume = torch.from_numpy(arr).float()
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")
    
    # Apply transform
    volume = transform(volume)
    print(f"Loaded volume from {file_path} with shape {volume.shape} after transform")
    return volume


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

def _show_mask_overlay(volume: np.ndarray, mask: np.ndarray, title_prefix: str = "Head mask") -> None:
    d, h, w = volume.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    views = [
        (volume[d_mid, :, :], mask[d_mid, :, :], f"{title_prefix} axial (D={d_mid})"),
        (volume[:, h_mid, :], mask[:, h_mid, :], f"{title_prefix} coronal (H={h_mid})"),
        (volume[:, :, w_mid], mask[:, :, w_mid], f"{title_prefix} sagittal (W={w_mid})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (img, msk, title) in zip(axes, views):
        ax.imshow(img, cmap="gray")
        ax.imshow(np.ma.masked_where(~msk, msk), cmap="autumn", alpha=0.35, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def make_head_mask_and_crop(
    volume: np.ndarray,
    threshold: float,
    closing_iterations: int = 2,
    closing_kernel_size: int = 5,
    visualize: bool = True,
):
    mask = volume > threshold
    structure = np.ones((closing_kernel_size, closing_kernel_size, closing_kernel_size), dtype=bool)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=closing_iterations)
    mask = ndimage.binary_fill_holes(mask)

    labeled, num_labels = ndimage.label(mask)
    if num_labels == 0:
        raise RuntimeError("Head mask is empty. Try lowering head_mask_threshold.")

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax())
    head_mask = labeled == largest_component

    d_idx, h_idx, w_idx = np.where(head_mask)
    d0, d1 = int(d_idx.min()), int(d_idx.max()) + 1
    h0, h1 = int(h_idx.min()), int(h_idx.max()) + 1
    w0, w1 = int(w_idx.min()), int(w_idx.max()) + 1
    crop_slices = (slice(d0, d1), slice(h0, h1), slice(w0, w1))

    if visualize:
        _show_mask_overlay(volume, head_mask, title_prefix="Head mask overlay")

    return volume[crop_slices], head_mask, crop_slices



# %% Script entry point (optional when used in Jupyter)
#%%
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this 3D DINOv3 inference script.")



# After setting CUDA_VISIBLE_DEVICES, the selected GPU is cuda:0

# --- Load model ---
print(f"Loading model from config={path_to_config} and checkpoint={path_to_checkpoint}")
model, autocast_dtype, config = load_backbone_from_checkpoint(
    config_file=path_to_config,
    ckpt_dir=path_to_checkpoint,
)
#%%
# --- Load data ---
if sample_path is not None and Path(sample_path).is_file():
    # Load a specific .nii.gz or .npy file directly
    print(f"Loading specific volume from: {sample_path}")
    
    # Get CT window and normalization parameters from config
    ct_window = getattr(config.crops, "ct_window", (0, 1000.0))
    ct_mean = getattr(config.crops, "ct_mean", None)
    ct_std = getattr(config.crops, "ct_std", None)
    mean = ct_mean if ct_mean is not None else None
    std = ct_std if ct_std is not None else None
    
    # Create transform
    base_transform = make_ct_3d_base_transform(window=ct_window, mean=mean, std=std)
    
    # Load and transform the volume
    volume = load_single_volume_from_file(sample_path, base_transform)
    
else:
    # Original behavior: load from dataset
    volume_path = pick_first_volume_path(path_to_data)
    print(f"Using volume: {volume_path}")

    dataset = build_ct_dataset(path_to_data, config)
    # Find index of the chosen volume in the dataset
    idx = dataset._paths.index(volume_path)

    volume, _ = dataset[idx]  # (C, D, H, W)


# Visualize input volume (center slices + MIPs) before moving to GPU
input_volume_np = volume[0].detach().cpu().numpy()
_show_input_visuals(input_volume_np, title_prefix="Input")


if head_mask_threshold is not None:
    cropped_np, head_mask_np, crop_slices = make_head_mask_and_crop(
        input_volume_np,
        threshold=head_mask_threshold,
        closing_iterations=2,
        closing_kernel_size=5,
        visualize=True,
    )
    d_slice, h_slice, w_slice = crop_slices
    volume = volume[:, d_slice, h_slice, w_slice]
    input_volume_np = cropped_np
    cropped_head_mask_np = head_mask_np[d_slice, h_slice, w_slice]
    _show_input_visuals(input_volume_np, title_prefix="Cropped head")



if resize_shape is not None and resize_scale is not None:
    raise ValueError("Set only one of resize_shape or resize_scale.")

if resize_shape is not None:
    volume = F.interpolate(
        volume.unsqueeze(0),
        size=resize_shape,
        mode="trilinear",
        align_corners=False,
    )[0]
    if head_mask_threshold is not None:
        cropped_head_mask_np = F.interpolate(
            torch.from_numpy(cropped_head_mask_np.astype(np.float32))[None, None, ...],
            size=resize_shape,
            mode="nearest",
        )[0, 0].numpy()
elif resize_scale is not None:
    volume = F.interpolate(
        volume.unsqueeze(0),
        scale_factor=resize_scale,
        mode="trilinear",
        align_corners=False,
    )[0]
    
# Add batch dim and move to GPU: (1, C, D, H, W)
volume = volume.unsqueeze(0).cuda(non_blocking=True)
#%%
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


# make another set of 3 images with only the sagital slices with one slice up and down from the cetner



#%%
# --- 3D feature map statistics & visualizations ---
with torch.no_grad():
    feat_cpu = vol_feats.detach().float().cpu()[0]  # (C, D, H, W)
    c, d_f, h_f, w_f = feat_cpu.shape

    # Bring cropped input mask to feature-map resolution
    if head_mask_threshold is not None:
        feature_mask = F.interpolate(
            torch.from_numpy(cropped_head_mask_np.astype(np.float32))[None, None, ...],
            size=(d_f, h_f, w_f),
            mode="nearest",
        )[0, 0] > 0.5
        feature_mask_np = feature_mask.numpy()

    feats_flat = feat_cpu.permute(1, 2, 3, 0).reshape(-1, c)  # (N, C)
    feats_np_all = feats_flat.numpy().astype(np.float32)

    if use_head_mask_for_pca:
        mask_flat_np = feature_mask_np.reshape(-1)

        feats_np_fit = feats_np_all[mask_flat_np]
        print(f"Using masked PCA fit on {feats_np_fit.shape[0]} / {feats_np_all.shape[0]} feature voxels")
    else:
        feats_np_fit = feats_np_all
        if use_head_mask_for_pca:
            print("Mask is empty at feature resolution. Falling back to full-volume PCA fit.")
        else:
            print("Using full-volume PCA fit.")

    # PCA over channel dimension using scikit-learn
    # Compute enough components to accommodate the offset
    n_components_needed = min(pca_component_offset + 3, feats_np_fit.shape[1])
    pca = PCA(n_components=n_components_needed)
    pca.fit(feats_np_fit)
    pca_proj = pca.transform(feats_np_all)  # (N, n_components)

    # First component as activation map
    activation = pca_proj[:, 0].reshape(d_f, h_f, w_f)
    if use_head_mask_for_pca:
        activation = np.where(feature_mask_np, activation, 0.0)

    # RGB visualization: 3 PCA components starting from pca_component_offset normalized to [0, 1]
    rgb_channels = []
    for i in range(3):
        component_idx = pca_component_offset + i
        if component_idx < pca_proj.shape[1]:
            ch = pca_proj[:, component_idx].reshape(d_f, h_f, w_f)
            if use_head_mask_for_pca and mask_flat_np.any():
                ch_masked = ch[feature_mask_np]
                ch_min, ch_max = ch_masked.min(), ch_masked.max()
            else:
                ch_min, ch_max = ch.min(), ch.max()
            ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
            if use_head_mask_for_pca:
                ch_norm = np.where(feature_mask_np, ch_norm, 0.0)
            rgb_channels.append(ch_norm)
        else:
            # If we don't have this component, fill with zeros
            rgb_channels.append(np.zeros((d_f, h_f, w_f)))
    pca_rgb = np.stack(rgb_channels, axis=-1)  # (D, H, W, 3)

    # Cosine similarity to center feature vector
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

    activation_np = activation  # already numpy from PCA
    if use_head_mask_for_pca and mask_flat_np.any():
        activation_values = activation_np[feature_mask_np]
    else:
        activation_values = activation_np.reshape(-1)

    a_mean = float(activation_values.mean())
    a_std = float(activation_values.std())
    a_min = float(activation_values.min())
    a_max = float(activation_values.max())
    p1, p5, p50, p95, p99 = np.percentile(activation_values, [1, 5, 50, 95, 99])
    high_activity_ratio = float((activation_values > p95).mean())

    channel_mean = feat_cpu.abs().mean(dim=(1, 2, 3))
    topk = min(5, channel_mean.numel())
    topk_vals, topk_idx = torch.topk(channel_mean, k=topk)

print("\nFeature map activation stats (PCA-1 projection):")
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

print(f"\nPCA explained variance ratio (components {pca_component_offset}-{pca_component_offset+2}):")
for i in range(3):
    component_idx = pca_component_offset + i
    if component_idx < len(pca.explained_variance_ratio_):
        var = pca.explained_variance_ratio_[component_idx]
        print(f"  PC{component_idx+1}: {var:.4f}")
    else:
        print(f"  PC{component_idx+1}: N/A (not computed)")

# Visualize PCA-RGB at center slices
print(f"\nPCA-RGB visualization (components {pca_component_offset}-{pca_component_offset+2} as RGB)...")
d_mid, h_mid, w_mid = d_f // 2, h_f // 2, w_f // 2
axial_rgb = pca_rgb[d_mid, :, :]
coronal_rgb = pca_rgb[:, h_mid, :, :]
sagittal_rgb = pca_rgb[:, :, w_mid, :]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(axial_rgb, interpolation="nearest")
axes[0].set_title(f"PCA-RGB center axial (PC{pca_component_offset+1}-{pca_component_offset+3})")
axes[1].imshow(coronal_rgb, interpolation="nearest")
axes[1].set_title(f"PCA-RGB center coronal (PC{pca_component_offset+1}-{pca_component_offset+3})")
axes[2].imshow(sagittal_rgb, interpolation="nearest")
axes[2].set_title(f"PCA-RGB center sagittal (PC{pca_component_offset+1}-{pca_component_offset+3})")
for ax in axes:

    ax.axis("off")
fig.tight_layout()
plt.show()

plt.imshow(sagittal_rgb, interpolation="nearest")
plt.axis("off")
plt.show()


#%%
# --- Cosine similarity visualization based on point coordinates ---
print("\n=== Cosine similarity to point feature ===")
point_coordinates = (40, 25, 30)  # (D, H, W) in feature space


# Clamp coordinates to valid range
point_d = min(max(point_coordinates[0], 0), d_f - 1)
point_h = min(max(point_coordinates[1], 0), h_f - 1)
point_w = min(max(point_coordinates[2], 0), w_f - 1)
print(f"Using point coordinates: ({point_d}, {point_h}, {point_w})")

# Extract feature at the point and compute cosine similarity to all features
point_feat = feat_cpu[:, point_d, point_h, point_w]  # (C,)
feat_norm = F.normalize(feat_cpu, dim=0)  # (C, D, H, W)
point_norm = F.normalize(point_feat, dim=0)  # (C,)

# Cosine similarity: dot product of normalized features
similarity = (feat_norm * point_norm[:, None, None, None]).sum(dim=0)  # (D, H, W)
similarity_np = similarity.cpu().numpy()

# Normalize to [0, 1] for visualization
sim_min, sim_max = similarity_np.min(), similarity_np.max()
similarity_norm = (similarity_np - sim_min) / (sim_max - sim_min + 1e-8)

print(f"Cosine similarity range: [{sim_min:.6f}, {sim_max:.6f}]")

# Visualize at center slices
print("\nVisualizing cosine similarity at center slices...")
axial_sim = similarity_norm[point_d, :, :]
coronal_sim = similarity_norm[:, point_h, :]
sagittal_sim = similarity_norm[:, :, point_w]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(axial_sim, cmap="coolwarm")
axes[0].plot(point_w, point_h, "r.", markersize=15)
axes[0].set_title(f"Cosine similarity axial (D={point_d})")

axes[1].imshow(coronal_sim, cmap="coolwarm")
axes[1].plot(point_w, point_d, "r.", markersize=15)
axes[1].set_title(f"Cosine similarity coronal (H={point_h})")

axes[2].imshow(sagittal_sim, cmap="coolwarm")
axes[2].plot(point_h, point_d, "r.", markersize=15)
axes[2].set_title(f"Cosine similarity sagittal (W={point_w})")

for ax in axes:
    ax.axis("off")
fig.colorbar(axes[0].imshow(axial_sim, cmap="coolwarm"), ax=axes[0], label="Similarity")
fig.tight_layout()
plt.show()

# Upsample similarity map to input resolution for overlay
similarity_up = F.interpolate(
    torch.from_numpy(similarity_norm).unsqueeze(0).unsqueeze(0).float(),
    size=input_volume_np.shape,
    mode="trilinear",
    align_corners=False,
)[0, 0].numpy()

plt.imshow(axial_sim, cmap="coolwarm")
plt.axis("off")
plt.show()


