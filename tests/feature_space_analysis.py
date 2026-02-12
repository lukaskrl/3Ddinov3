#%%
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
path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/mri_hrft_resumed/ckpt/7599"

# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Directory with CT volumes stored as .npy files
path_to_data = "/home/lukas/data/brain-t1-dataset"

# Training config used for this checkpoint (3D CT DINOv3 config)
# path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_mri3d_config.yaml"
path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_mri3d_stage3_hrft.yaml"


# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def load_backbone_from_checkpoint(config_file: str, ckpt_dir: str):
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
    ct_window = getattr(config.crops, "ct_window", (-1000.0, 400.0))
    ct_mean = getattr(config.crops, "ct_mean", None)
    ct_std = getattr(config.crops, "ct_std", None)

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
    root_path = Path(data_root)
    volume_files = sorted(
        p
        for p in root_path.iterdir()
        if p.is_file() and (p.name.endswith(".npy") or p.name.endswith(".nii.gz"))
    )
    if not volume_files:
        raise RuntimeError(f"No .npy or .nii.gz volumes found under {data_root}")
    return str(volume_files[0])


def analyze_feature_space(features: torch.Tensor):
    """
    features: (C, D, H, W) on CPU (float)
    """
    c, d, h, w = features.shape
    n = d * h * w
    feats = features.reshape(c, n).T  # (N, C)

    # Norm statistics
    norms = torch.linalg.vector_norm(feats, dim=1)
    norms_np = norms.numpy()

    # Cosine similarity to mean feature
    mean_feat = feats.mean(dim=0)
    feats_norm = F.normalize(feats, dim=1)
    mean_norm = F.normalize(mean_feat, dim=0)
    cos_to_mean = (feats_norm @ mean_norm).clamp(-1.0, 1.0).numpy()

    # Feature variance per channel
    channel_var = feats.var(dim=0)
    channel_var_np = channel_var.numpy()

    print("\n=== Feature space analysis ===")
    print(f"Tokens: {n}, Channels: {c}")
    print(
        "Norms: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            norms_np.mean(), norms_np.std(), norms_np.min(), norms_np.max()
        )
    )
    print(
        "Cosine to mean: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            cos_to_mean.mean(), cos_to_mean.std(), cos_to_mean.min(), cos_to_mean.max()
        )
    )
    print(
        "Channel variance: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            channel_var_np.mean(), channel_var_np.std(), channel_var_np.min(), channel_var_np.max()
        )
    )

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(norms_np, bins=80, color="#4C78A8", alpha=0.9)
    axes[0].set_title("Token L2 norms")
    axes[1].hist(cos_to_mean, bins=80, color="#F58518", alpha=0.9)
    axes[1].set_title("Cosine to mean feature")
    axes[2].hist(channel_var_np, bins=80, color="#54A24B", alpha=0.9)
    axes[2].set_title("Channel variance")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis script.")

    torch.cuda.set_device(0)

    print(f"Loading model from config={path_to_config} and checkpoint={path_to_checkpoint}")
    model, autocast_dtype, config = load_backbone_from_checkpoint(
        config_file=path_to_config,
        ckpt_dir=path_to_checkpoint,
    )

    dataset = build_ct_dataset(path_to_data, config)
    volume_path = pick_first_volume_path(path_to_data)
    print(f"Using volume: {volume_path}")
    idx = dataset._paths.index(volume_path)

    volume, _ = dataset[idx]  # (C, D, H, W)
    volume = volume.unsqueeze(0).cuda(non_blocking=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
        last_vol_feats_tuple = model.get_intermediate_layers(
            volume,
            n=1,
            reshape=True,
            return_class_token=True,
            return_extra_tokens=False,
            norm=True,
        )

    (vol_feats, _vol_cls_token) = last_vol_feats_tuple[0]
    feat_cpu = vol_feats.detach().float().cpu()[0]  # (C, D, H, W)

    analyze_feature_space(feat_cpu)
