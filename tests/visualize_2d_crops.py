"""
Visualize global and local 2D crops from DataAugmentationDINO on a natural image.

Run from repo root:
  PYTHONPATH=. python tests/visualize_2d_crops.py
"""
#%%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Allow imports from the repo when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.augmentations import DataAugmentationDINO


# --- Paths ---
path_to_config = "/home/lukas/3Ddinov3/dinov3/configs/ssl_default_config.yaml"
image_path = "/home/lukas/3Ddinov3/2026-02-12_11-51.png"  # <-- set to your image


def load_config(config_file: str):
    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights="",CTVolume:root=/home/lukas/data/brain-t1-dataset
        shard_unsharded_model=False,
        output_dir="",
        opts=[],
    )
    return setup_config(setup_args, strict_cfg=False)


def _denormalize(img: torch.Tensor, mean, std) -> torch.Tensor:
    """
    img: (C, H, W) normalized tensor
    mean/std: list or tuple length C
    """
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device)[:, None, None]
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device)[:, None, None]
    out = img * std_t + mean_t
    return out.clamp(0.0, 1.0)


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [0,1] to (H, W, C) numpy."""
    return img.permute(1, 2, 0).detach().cpu().numpy()


def _overlay_patch_grid(
    ax: plt.Axes,
    height: int,
    width: int,
    patch_size: int,
    *,
    color: str = "white",
    alpha: float = 0.35,
    linewidth: float = 0.5,
) -> None:
    if patch_size is None or patch_size <= 0:
        return
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, width + 1, patch_size), minor=True)
    ax.set_yticks(np.arange(0, height + 1, patch_size), minor=True)
    ax.grid(which="minor", color=color, alpha=alpha, linewidth=linewidth)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_frame_on(False)


def _visualize_crops(
    original: Image.Image,
    crops: list[torch.Tensor],
    labels: list[str],
    mean,
    std,
    patch_size: int,
) -> None:
    cols = 4
    total = 1 + len(crops)
    rows = int(np.ceil(total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    # Plot original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("original")
    axes[0, 0].axis("off")

    # Plot crops
    idx = 1
    for crop, label in zip(crops, labels):
        r, c = divmod(idx, cols)
        crop_denorm = _denormalize(crop, mean, std)
        crop_np = _to_numpy(crop_denorm)
        axes[r, c].imshow(crop_np)
        _overlay_patch_grid(axes[r, c], crop_np.shape[0], crop_np.shape[1], patch_size)
        axes[r, c].set_title(label)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        axes[r, c].set_frame_on(False)
        idx += 1

    # Hide any leftover axes
    for k in range(idx, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = load_config(path_to_config)

    aug = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        teacher_no_color_jitter=getattr(cfg.crops, "teacher_no_color_jitter", False),
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        patch_size=cfg.student.patch_size,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
    )

    img = Image.open(image_path).convert("RGB")
    sample = aug(img)

    global_crops = sample["global_crops"]
    local_crops = sample["local_crops"]

    crops = []
    labels = []
    for i, g in enumerate(global_crops):
        crops.append(g)
        labels.append(f"global[{i}]")
    for i, l in enumerate(local_crops):
        crops.append(l)
        labels.append(f"local[{i}]")

    _visualize_crops(img, crops, labels, cfg.crops.rgb_mean, cfg.crops.rgb_std, cfg.student.patch_size)
