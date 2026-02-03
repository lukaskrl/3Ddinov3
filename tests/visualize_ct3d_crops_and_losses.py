"""
Utility script to:
  1) Load the CT 3D DINOv3 config.
  2) Build the exact 3D augmentation / cropping pipeline used in training.
  3) Apply it to a single CT volume and visualize the global & local crops.
  4) Build a one-sample batch and compute the DINO global/local crop losses
     (and iBOT loss) using SSLMetaArch.forward_backward.

Run from the repo root:

  PYTHONPATH=. python tests/visualize_ct3d_crops_and_losses.py
"""
#%%
import os
import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Allow imports from the repo when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator3D
from dinov3.train.ssl_meta_arch import SSLMetaArch


def _parse_ct_root_from_cfg(cfg) -> str:
    """
    Parse the CTVolume root path from cfg.train.dataset_path, which looks like:
      'CTVolume:root=/path/to/data'
    """
    ds_str = cfg.train.dataset_path
    tokens = ds_str.split(":")
    if tokens[0] != "CTVolume":
        raise ValueError(f"Expected CTVolume dataset, got {tokens[0]}")
    root = None
    for token in tokens[1:]:
        key, value = token.split("=")
        if key == "root":
            root = value
    if root is None:
        raise ValueError(f"Could not find 'root=' in cfg.train.dataset_path={ds_str}")
    return root


def _build_single_sample_batch(cfg, aug, volume: torch.Tensor, disable_flips=False):
    """
    Apply 3D DINO augmentation to a single volume and collate it into
    a training-style batch (with masks and mask_indices_list) for one sample.
    
    Args:
        disable_flips: If True, temporarily disable horizontal_flips for visualization.
    """
    # Temporarily disable flips if requested (for consistent visualization)
    original_flip_setting = aug.horizontal_flips
    if disable_flips:
        aug.horizontal_flips = False
    
    # 1) Apply augmentation
    sample_dict = aug(volume)  # keys: global_crops, local_crops, ...
    
    # Restore original flip setting
    aug.horizontal_flips = original_flip_setting
    
    sample = (sample_dict, 0)  # dummy target

    # 2) Compute patch grid and masking generator from config
    D_g, H_g, W_g = tuple(cfg.crops.global_crops_size_3d)
    D_p, H_p, W_p = tuple(cfg.crops.patch_size_3d)

    n_tokens = (D_g // D_p) * (H_g // H_p) * (W_g // W_p)
    grid_size = (D_g // D_p, H_g // H_p, W_g // W_p)

    mask_gen = MaskingGenerator3D(
        input_size=grid_size,
        max_num_patches=int(0.5 * n_tokens),
    )

    batch = collate_data_and_cast(
        samples_list=[sample],
        mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_gen,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=None,
    )

    # Training loop normally adds this field
    batch["global_batch_size"] = 1
    return batch, sample_dict


def _visualize_crops(sample_dict, original_volume=None, show_original=False, foreground_threshold=None):
    """
    Visualize a few slices from the global and local crops.
    Assumes C=1 (single-channel CT).
    
    Args:
        sample_dict: Output from DataAugmentationDINO3D
        original_volume: Optional original volume (C, D, H, W) to show for comparison
        show_original: If True and original_volume provided, show original first
    """
    global_crops = sample_dict["global_crops"]
    local_crops = sample_dict["local_crops"]

    n_globals = len(global_crops)
    n_locals = len(local_crops)

    # Take at most 2 globals and 4 locals for plotting
    globals_to_show = global_crops[: min(2, n_globals)]
    locals_to_show = local_crops[: min(4, n_locals)]

    crops_to_plot = []
    crop_labels = []
    
    if show_original and original_volume is not None:
        crops_to_plot.append(original_volume)
        crop_labels.append("original")
    
    crops_to_plot.extend(globals_to_show)
    crop_labels.extend([f"global[{i}]" for i in range(len(globals_to_show))])
    
    crops_to_plot.extend(locals_to_show)
    crop_labels.extend([f"local[{i}]" for i in range(len(locals_to_show))])

    n_rows = len(crops_to_plot)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    def _plot_volume(ax_row, vol, title_prefix):
        # vol: (C, D, H, W)
        vol_np = vol.detach().cpu().numpy() if isinstance(vol, torch.Tensor) else vol
        if vol_np.ndim == 4:
            assert vol_np.shape[0] == 1, "Expected single-channel CT volume"
            vol_np = vol_np[0]  # Remove channel dim: (D, H, W)
        
        D, H, W = vol_np.shape

        mid_d = D // 2
        mid_h = H // 2
        mid_w = W // 2

        # Axial (z-mid), coronal (y-mid), sagittal (x-mid) views
        slices = [
            vol_np[mid_d, :, :],  # axial
            vol_np[:, mid_h, :],  # coronal
            vol_np[:, :, mid_w],  # sagittal
        ]
        titles = [f"{title_prefix} axial", f"{title_prefix} coronal", f"{title_prefix} sagittal"]

        for ax, img, t in zip(ax_row, slices, titles):
            ax.imshow(img, cmap="gray", origin="lower")  # origin='lower' for medical convention
            ax.set_title(t, fontsize=10)
            ax.axis("off")

    for row, (vol, label) in enumerate(zip(crops_to_plot, crop_labels)):
        _plot_volume(axes[row], vol, label)

    if foreground_threshold is None:
        fg_title = "foreground_threshold: N/A"
    else:
        fg_title = f"foreground_threshold: {foreground_threshold}"
    plt.suptitle(
        "3D CT Crops Visualization\n"
        f"({fg_title}; Note: horizontal_flips may cause crops to appear flipped)",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()
    plt.show()


def _summarize_crops(sample_dict):
    def _summary(vol, name):
        m = float(vol.mean().item())
        s = float(vol.std().item())
        nz = float((vol.abs() > 1e-6).float().mean().item())
        print(f"{name}: mean={m:.4f} std={s:.4f} nonzero_frac={nz:.4f}")

    def _foreground_summary(vol, name, threshold=-500.0):
        # Convert back to approximate HU values for foreground analysis
        # vol is normalized to [0,1] from [-1000, 400] window by default
        hu_approx = vol * 2400.0 - 1000.0  # rough approximation
        fg_mask = hu_approx > threshold
        fg_ratio = float(fg_mask.float().mean().item())
        print(f"{name}: foreground_ratio={fg_ratio:.4f} (HU > {threshold})")

    print("\n=== Crop stats (mean/std/nonzero_frac) ===")
    for i, g in enumerate(sample_dict["global_crops"]):
        _summary(g, f"global[{i}]")
        _foreground_summary(g, f"global[{i}]")
    for i, l in enumerate(sample_dict["local_crops"]):
        _summary(l, f"local[{i}]")
        _foreground_summary(l, f"local[{i}]")
    if "gram_teacher_crops" in sample_dict:
        for i, gt in enumerate(sample_dict["gram_teacher_crops"]):
            _summary(gt, f"gram_teacher[{i}]")
            _foreground_summary(gt, f"gram_teacher[{i}]")


def _pca_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Very small PCA helper that returns a 2D projection.
    x: [N, D]
    """
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on [N, D] is fine here (N is tiny: n_crops * B)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    components = vh[:2].T  # [D, 2]
    return x @ components  # [N, 2]


def _entropy_from_probs(p: torch.Tensor) -> float:
    p = p.float().clamp_min(1e-8)
    return float((-(p * p.log()).sum(dim=-1)).mean().item())


def _plot_cls_diagnostics(cfg, model: SSLMetaArch, batch: dict):
    """
    Visualize the CLS tokens / logits that go into the DINO loss:
      - student_logits: student_{global,local}["cls_after_head"]  [n_crops, B, K]
      - teacher_probs: teacher_global["cls_centered"]            [n_crops, B, K]
    """
    device = next(model.parameters()).device
    n_global_crops = 2
    n_local_crops = cfg.crops.local_crops_number
    B = batch["collated_local_crops"].shape[0] // n_local_crops

    global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
    local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
    masks = batch["collated_masks"].to(device, non_blocking=True)
    mask_indices_list = batch["mask_indices_list"].to(device, non_blocking=True)
    masks_weight = batch["masks_weight"].to(device, non_blocking=True)
    n_masked_patches_tensor = batch["n_masked_patches"].to(device, non_blocking=True)

    teacher_crops = batch.get("collated_global_crops_teacher", None)
    if teacher_crops is None:
        teacher_crops = global_crops
    else:
        teacher_crops = teacher_crops.to(device, non_blocking=True)

    teacher_temp = cfg.teacher.teacher_temp

    with torch.no_grad():
        teacher_global = model.get_teacher_output(
            teacher_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=batch["upperbound"],
        )
        student_global, student_local = model.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=batch["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

    # --- Extract tensors ---
    t_cls_pre = teacher_global["cls_pre_head"].reshape(-1, teacher_global["cls_pre_head"].shape[-1])  # [2B, D]
    sg_cls_pre = student_global["cls_pre_head"].reshape(-1, student_global["cls_pre_head"].shape[-1])  # [2B, D]
    sl_cls_pre = student_local["cls_pre_head"].reshape(-1, student_local["cls_pre_head"].shape[-1])  # [n_local*B, D]

    t_logits = teacher_global["cls_after_head"]  # [2, B, K]
    sg_logits = student_global["cls_after_head"]  # [2, B, K]
    sl_logits = student_local["cls_after_head"]  # [n_local, B, K]
    t_probs = teacher_global["cls_centered"]  # [2, B, K]

    # --- Numeric sanity ---
    print("\n=== CLS/Logits diagnostics (what DINO loss actually uses) ===")
    print(f"K (prototypes): {t_logits.shape[-1]}")
    def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().mean().item())

    def _mean_abs_diff_centered_over_k(a: torch.Tensor, b: torch.Tensor) -> float:
        # Remove per-sample mean over K (tests “different only by constant shift”)
        a = a.float() - a.float().mean(dim=-1, keepdim=True)
        b = b.float() - b.float().mean(dim=-1, keepdim=True)
        return float((a - b).abs().mean().item())

    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float().flatten(1)
        b = b.float().flatten(1)
        return float(F.cosine_similarity(a, b, dim=1).mean().item())

    def _topk_str(v: torch.Tensor, k: int = 5) -> str:
        # v: [K]
        vals, idx = torch.topk(v.float(), k)
        pairs = ", ".join([f"{int(i)}:{float(x):.4f}" for i, x in zip(idx, vals)])
        return pairs

    # Patch embeddings (pre-head) to detect where collapse happens
    t_patch_pre = teacher_global["patch_pre_head"]  # [2, B, P, D]
    sg_patch_pre = student_global["patch_pre_head"]  # [2, B, P, D]

    print(f"teacher_probs entropy: {_entropy_from_probs(t_probs.flatten(0, 1)):.4f}")
    print(f"student_global softmax entropy: {_entropy_from_probs(torch.softmax(sg_logits.flatten(0, 1), dim=-1)):.4f}")
    print(f"student_local softmax entropy: {_entropy_from_probs(torch.softmax(sl_logits.flatten(0, 1), dim=-1)):.4f}")

    # Crop differences
    if t_probs.shape[0] >= 2:
        diff_probs = _mean_abs_diff(t_probs[0], t_probs[1])
        print(f"teacher_probs crop0-vs-crop1 mean|diff|: {diff_probs:.6f}")
        # Also compare teacher logits (raw + mean-centered over K)
        diff_t_logits = _mean_abs_diff(t_logits[0], t_logits[1])
        diff_t_logits_centered = _mean_abs_diff_centered_over_k(t_logits[0], t_logits[1])
        print(f"teacher_logits crop0-vs-crop1 mean|diff|: {diff_t_logits:.6f}")
        print(f"teacher_logits crop0-vs-crop1 mean|diff| (mean-centered over K): {diff_t_logits_centered:.6f}")
        
        # Compare teacher CLS pre-head (D-dim)
        t_pre = teacher_global["cls_pre_head"]  # [2, B, D]
        diff_t_pre = _mean_abs_diff(t_pre[0], t_pre[1])
        print(f"teacher_cls_pre_head crop0-vs-crop1 mean|diff|: {diff_t_pre:.6f}")

        # Same diagnostics for student global logits
        diff_sg_logits = _mean_abs_diff(sg_logits[0], sg_logits[1])
        diff_sg_logits_centered = _mean_abs_diff_centered_over_k(sg_logits[0], sg_logits[1])
        print(f"student_global_logits crop0-vs-crop1 mean|diff|: {diff_sg_logits:.6f}")
        print(f"student_global_logits crop0-vs-crop1 mean|diff| (mean-centered over K): {diff_sg_logits_centered:.6f}")

        # Patch embeddings diff & cosine
        diff_t_patch = _mean_abs_diff(t_patch_pre[0], t_patch_pre[1])
        diff_sg_patch = _mean_abs_diff(sg_patch_pre[0], sg_patch_pre[1])
        cos_t_patch = _cosine(t_patch_pre[0], t_patch_pre[1])
        cos_sg_patch = _cosine(sg_patch_pre[0], sg_patch_pre[1])
        print(f"teacher_patch_pre crop0-vs-crop1 mean|diff|: {diff_t_patch:.6f}, cos={cos_t_patch:.6f}")
        print(f"student_patch_pre crop0-vs-crop1 mean|diff|: {diff_sg_patch:.6f}, cos={cos_sg_patch:.6f}")

        # Print top-k prototype activations for each teacher crop (batch-mean if B>1)
        t_logits_mean0 = t_logits[0].mean(dim=0)  # [K]
        t_logits_mean1 = t_logits[1].mean(dim=0)  # [K]
        t_probs_mean0 = t_probs[0].mean(dim=0)  # [K]
        t_probs_mean1 = t_probs[1].mean(dim=0)  # [K]
        print(f"teacher_logits top5 crop0: {_topk_str(t_logits_mean0)}")
        print(f"teacher_logits top5 crop1: {_topk_str(t_logits_mean1)}")
        print(f"teacher_probs  top5 crop0: {_topk_str(t_probs_mean0)}")
        print(f"teacher_probs  top5 crop1: {_topk_str(t_probs_mean1)}")
    if teacher_crops.shape[0] >= 2:
        diff_tc = (teacher_crops[0] - teacher_crops[1]).abs().mean().item()
        print(f"teacher_crops input crop0-vs-crop1 mean|diff|: {diff_tc:.6f}")
    diff_t_vs_s = (teacher_crops - global_crops).abs().mean().item()
    print(f"teacher_crops vs student global_crops mean|diff|: {diff_t_vs_s:.6f}")

    # --- PCA plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    proj_pre = _pca_2d(torch.cat([t_cls_pre, sg_cls_pre, sl_cls_pre], dim=0).detach().cpu())
    n_t = t_cls_pre.shape[0]
    n_sg = sg_cls_pre.shape[0]
    t_xy = proj_pre[:n_t]
    sg_xy = proj_pre[n_t : n_t + n_sg]
    sl_xy = proj_pre[n_t + n_sg :]
    axes[0].scatter(t_xy[:, 0], t_xy[:, 1], c="tab:orange", label="teacher cls_pre", s=60, marker="o")
    axes[0].scatter(sg_xy[:, 0], sg_xy[:, 1], c="tab:blue", label="student global cls_pre", s=60, marker="^")
    axes[0].scatter(sl_xy[:, 0], sl_xy[:, 1], c="tab:green", label="student local cls_pre", s=40, marker=".")
    axes[0].set_title("PCA(2D) of CLS pre-head (D-dim)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PCA on logits (K-dim)
    logits_all = torch.cat(
        [
            t_logits.reshape(-1, t_logits.shape[-1]),
            sg_logits.reshape(-1, sg_logits.shape[-1]),
            sl_logits.reshape(-1, sl_logits.shape[-1]),
        ],
        dim=0,
    ).detach().cpu()
    proj_logits = _pca_2d(logits_all)
    n_tl = t_logits.numel() // t_logits.shape[-1]
    n_sgl = sg_logits.numel() // sg_logits.shape[-1]
    tl_xy = proj_logits[:n_tl]
    sgl_xy = proj_logits[n_tl : n_tl + n_sgl]
    sll_xy = proj_logits[n_tl + n_sgl :]
    axes[1].scatter(tl_xy[:, 0], tl_xy[:, 1], c="tab:orange", label="teacher cls_after_head", s=60, marker="o")
    axes[1].scatter(sgl_xy[:, 0], sgl_xy[:, 1], c="tab:blue", label="student global cls_after_head", s=60, marker="^")
    axes[1].scatter(sll_xy[:, 0], sll_xy[:, 1], c="tab:green", label="student local cls_after_head", s=40, marker=".")
    axes[1].set_title("PCA(2D) of CLS after-head logits (K-dim)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Per-crop probability curves (teacher + student globals) ---
    # (Useful to spot collapse to uniform or extremely peaky distributions)
    t_probs_cpu = t_probs.detach().cpu().squeeze(1)  # [2, K] if B=1 else [2, B, K] -> squeeze only if B=1
    sg_probs_cpu = torch.softmax(sg_logits.detach().cpu(), dim=-1).squeeze(1)
    # Handle B>1: plot mean over batch
    if t_probs_cpu.ndim == 3:
        t_probs_cpu = t_probs_cpu.mean(dim=1)
    if sg_probs_cpu.ndim == 3:
        sg_probs_cpu = sg_probs_cpu.mean(dim=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i in range(t_probs_cpu.shape[0]):
        axes[0].plot(t_probs_cpu[i].numpy(), label=f"teacher_probs crop{i}")
    axes[0].set_title("Teacher probs per global crop (mean over batch)")
    axes[0].set_ylabel("prob")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i in range(sg_probs_cpu.shape[0]):
        axes[1].plot(sg_probs_cpu[i].numpy(), label=f"student_global_softmax crop{i}")
    axes[1].set_title("Student global softmax probs per crop (mean over batch)")
    axes[1].set_xlabel("prototype index")
    axes[1].set_ylabel("prob")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Distributed init (simple single-process enable) ---
    if not distributed.is_enabled():
        distributed.enable()

    # --- Load config ---
    config_file = "/home/lukas/projects/3Ddinov3/dinov3/configs/ssl_ct3d_config.yaml"
    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights="",
        shard_unsharded_model=False,
        output_dir="",
        opts=[],
    ) 
    cfg = setup_config(setup_args, strict_cfg=False)

    # For this visualization script we do not need FSDP/meta-device init; build
    # the student/teacher on real devices instead of "meta".
    cfg.debug_no_meta = True

    # --- Dataset / volume ---
    data_root = _parse_ct_root_from_cfg(cfg)
    dataset = CTVolumeDataset(root=data_root)
    if len(dataset) == 0:
        raise RuntimeError(f"No volumes found under {data_root}")

    volume, _ = dataset[0]  # (C, D, H, W)

    # --- Build model + augmentation using the standard (non-meta) path ---
    model = SSLMetaArch(cfg)
    model.init_weights()
    if torch.cuda.is_available():
        model.cuda()
    aug = model.build_data_augmentation_dino(cfg)
    print("config and model loaded.", cfg)

    # Print foreground cropping configuration
    print("=== Foreground Cropping Configuration ===")
    print(f"foreground_threshold: {getattr(aug, 'foreground_threshold', 'N/A')}")
    print(f"foreground_crop_prob: {getattr(aug, 'foreground_crop_prob', 'N/A')}")
    print(f"min_foreground_ratio: {getattr(aug, 'min_foreground_ratio', 'N/A')}")
    print()

    # --- Build one-sample batch and visualize crops ---
    print("\n=== Visualizing crops WITH augmentation (including random flips) ===")
    batch, sample_dict = _build_single_sample_batch(cfg, aug, volume, disable_flips=False)
    _visualize_crops(
        sample_dict,
        original_volume=volume,
        show_original=True,
        foreground_threshold=getattr(aug, "foreground_threshold", None),
    )


    # Print simple stats on crops
    _summarize_crops(sample_dict)

    # --- Compute losses (DINO global/local + iBOT) for this one batch ---
#%%
    teacher_temp = cfg.teacher.teacher_temp

    loss, metrics = model.forward_backward(
        batch,
        teacher_temp=teacher_temp,
        iteration=0,
    )

    print("\n=== Single-batch loss breakdown ===")
    print(f"total_loss: {loss.item():.4f}")
    
    # Print detailed loss info
    n_global = len(sample_dict["global_crops"])
    n_local = len(sample_dict["local_crops"])
    print(f"\nCrop counts: {n_global} global crops, {n_local} local crops")
    print(f"Global loss compares: {n_global} student globals vs {n_global} teacher globals")
    print(f"Local loss compares: {n_local} student locals vs {n_global} teacher globals")
    
    for key in ["dino_global_crops_loss", "dino_local_crops_loss", "ibot_loss", "dino_local_loss_weight"]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: <not present in metrics>")

    # --- Visualize CLS tokens / logits that DINO loss uses ---
    _plot_cls_diagnostics(cfg, model, batch)



# %%
