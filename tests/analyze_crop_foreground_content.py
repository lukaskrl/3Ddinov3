"""
Analyze foreground content in crops to understand the background matching problem.

This script:
1. Loads a CT volume
2. Generates crops using current augmentation settings
3. Computes foreground ratios for each crop
4. Visualizes crops and their foreground content
5. Helps identify if crops are too background-heavy
"""
#%%
import os
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

# Allow imports from the repo when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.augmentations_3d import DataAugmentationDINO3D


def compute_foreground_ratio(volume, threshold=500):
    """
    Compute the ratio of foreground voxels in a volume.
    
    Args:
        volume: (C, D, H, W) tensor with HU values
        threshold: HU threshold for foreground (above this is foreground)
    
    Returns:
        ratio: float between 0 and 1
    """
    if volume.ndim == 4:
        hu_values = volume[0]  # (D, H, W)
    else:
        hu_values = volume
    
    foreground_mask = hu_values > threshold
    foreground_ratio = foreground_mask.float().mean().item()
    return foreground_ratio


def analyze_crops(cfg, volume_path=None, n_samples=10):
    """
    Analyze foreground content in generated crops.
    """
    # Build augmentation
    aug = DataAugmentationDINO3D(
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size_3d=tuple(cfg.crops.global_crops_size_3d),
        local_crops_size_3d=tuple(cfg.crops.local_crops_size_3d),
        gram_teacher_crops_size_3d=(
            tuple(cfg.crops.gram_teacher_crops_size_3d)
            if getattr(cfg.crops, "gram_teacher_crops_size_3d", None) is not None
            else None
        ),
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        patch_size_3d=tuple(cfg.crops.patch_size_3d),
        horizontal_flips=cfg.crops.horizontal_flips,
        ct_window=getattr(cfg.crops, "ct_window", (-1000.0, 400.0)),
        mean=getattr(cfg.crops, "ct_mean", None),
        std=getattr(cfg.crops, "ct_std", None),
        foreground_threshold=getattr(cfg.crops, "foreground_threshold", 500),
        foreground_crop_prob=getattr(cfg.crops, "foreground_crop_prob", 0.0),
        min_foreground_ratio=getattr(cfg.crops, "min_foreground_ratio", 0.0),
    )
    
    # Load dataset
    if volume_path:
        # Load single volume
        dataset = CTVolumeDataset(root=volume_path, transform=None)
        volumes = [dataset[0][0]]  # Get first volume
    else:
        # Parse dataset path from config
        ds_str = cfg.train.dataset_path
        if ":" in ds_str:
            ds_type, params = ds_str.split(":", 1)
            if ds_type == "CTVolume":
                params_dict = {}
                for param in params.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params_dict[key] = value
                root = params_dict.get("root")
                if root:
                    dataset = CTVolumeDataset(root=root, transform=None)
                    # Sample a few volumes
                    n_volumes = min(n_samples, len(dataset))
                    volumes = [dataset[i][0] for i in range(n_volumes)]
                else:
                    raise ValueError("Could not parse dataset root from config")
            else:
                raise ValueError(f"Unsupported dataset type: {ds_type}")
        else:
            raise ValueError("Could not parse dataset path")
    
    # Analyze crops
    all_stats = {
        'global_crops': [],
        'local_crops': [],
    }
    
    for vol_idx, volume in enumerate(volumes):
        print(f"\nAnalyzing volume {vol_idx + 1}/{len(volumes)}")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume foreground ratio: {compute_foreground_ratio(volume, aug.foreground_threshold):.3f}")
        
        # Generate crops
        sample_dict = aug(volume)
        
        # Analyze global crops
        for i, global_crop in enumerate(sample_dict['global_crops']):
            # Denormalize to get HU values (approximate - depends on normalization)
            # For analysis, we'll use the original crop before normalization
            # But we need to get it from the augmentation process
            # For now, analyze the normalized crop (less accurate but still informative)
            fg_ratio = compute_foreground_ratio(global_crop, aug.foreground_threshold)
            all_stats['global_crops'].append({
                'volume_idx': vol_idx,
                'crop_idx': i,
                'foreground_ratio': fg_ratio,
                'shape': global_crop.shape,
            })
            print(f"  Global crop {i}: foreground_ratio={fg_ratio:.3f}")
        
        # Analyze local crops
        for i, local_crop in enumerate(sample_dict['local_crops']):
            fg_ratio = compute_foreground_ratio(local_crop, aug.foreground_threshold)
            all_stats['local_crops'].append({
                'volume_idx': vol_idx,
                'crop_idx': i,
                'foreground_ratio': fg_ratio,
                'shape': local_crop.shape,
            })
            print(f"  Local crop {i}: foreground_ratio={fg_ratio:.3f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    global_ratios = [s['foreground_ratio'] for s in all_stats['global_crops']]
    local_ratios = [s['foreground_ratio'] for s in all_stats['local_crops']]
    
    print(f"\nGlobal Crops:")
    print(f"  Count: {len(global_ratios)}")
    print(f"  Mean foreground ratio: {np.mean(global_ratios):.3f}")
    print(f"  Std foreground ratio: {np.std(global_ratios):.3f}")
    print(f"  Min foreground ratio: {np.min(global_ratios):.3f}")
    print(f"  Max foreground ratio: {np.max(global_ratios):.3f}")
    print(f"  Crops with <30% foreground: {sum(r < 0.3 for r in global_ratios)} ({100*sum(r < 0.3 for r in global_ratios)/len(global_ratios):.1f}%)")
    print(f"  Crops with <50% foreground: {sum(r < 0.5 for r in global_ratios)} ({100*sum(r < 0.5 for r in global_ratios)/len(global_ratios):.1f}%)")
    
    print(f"\nLocal Crops:")
    print(f"  Count: {len(local_ratios)}")
    print(f"  Mean foreground ratio: {np.mean(local_ratios):.3f}")
    print(f"  Std foreground ratio: {np.std(local_ratios):.3f}")
    print(f"  Min foreground ratio: {np.min(local_ratios):.3f}")
    print(f"  Max foreground ratio: {np.max(local_ratios):.3f}")
    print(f"  Crops with <30% foreground: {sum(r < 0.3 for r in local_ratios)} ({100*sum(r < 0.3 for r in local_ratios)/len(local_ratios):.1f}%)")
    print(f"  Crops with <50% foreground: {sum(r < 0.5 for r in local_ratios)} ({100*sum(r < 0.5 for r in local_ratios)/len(local_ratios):.1f}%)")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if np.mean(global_ratios) < 0.3:
        print("⚠️  WARNING: Global crops are mostly background (<30% foreground)")
        print("   The model will learn to match background regions!")
    elif np.mean(global_ratios) < 0.5:
        print("⚠️  CAUTION: Global crops have moderate foreground (30-50%)")
        print("   Consider increasing foreground requirements")
    else:
        print("✓ Global crops have good foreground content (>50%)")
    
    if np.mean(local_ratios) < 0.3:
        print("⚠️  WARNING: Local crops are mostly background (<30% foreground)")
        print("   When matched to global crops, model learns 'background→background'")
    elif np.mean(local_ratios) < 0.5:
        print("⚠️  CAUTION: Local crops have moderate foreground (30-50%)")
        print("   Consider increasing foreground requirements")
    else:
        print("✓ Local crops have good foreground content (>50%)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if np.mean(global_ratios) < 0.5 or np.mean(local_ratios) < 0.5:
        print("1. Enable foreground-biased cropping:")
        print("   foreground_crop_prob: 1.0")
        print("   min_foreground_ratio: 0.5")
        print()
    
    if not cfg.crops.localcrops_subset_of_globalcrops:
        print("2. Make local crops subsets of global crops:")
        print("   localcrops_subset_of_globalcrops: true")
        print("   This ensures local crops share context with global crops")
        print()
    
    if aug.foreground_threshold > 0:
        print("3. Consider lowering foreground threshold:")
        print(f"   Current: {aug.foreground_threshold} HU")
        print("   Suggested: -500 HU (soft tissue and above) or 0 HU (all tissue)")
        print("   This will include more anatomical structures")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of global crop foreground ratios
    ax = axes[0, 0]
    ax.hist(global_ratios, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.3, color='orange', linestyle='--', label='30% threshold')
    ax.axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
    ax.set_xlabel('Foreground Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Global Crops: Foreground Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram of local crop foreground ratios
    ax = axes[0, 1]
    ax.hist(local_ratios, bins=20, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(x=0.3, color='orange', linestyle='--', label='30% threshold')
    ax.axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
    ax.set_xlabel('Foreground Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Local Crops: Foreground Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comparison
    ax = axes[1, 0]
    ax.boxplot([global_ratios, local_ratios], labels=['Global', 'Local'])
    ax.set_ylabel('Foreground Ratio')
    ax.set_title('Foreground Ratio: Global vs Local Crops')
    ax.grid(True, alpha=0.3)
    
    # Scatter: global vs local (if same volume)
    ax = axes[1, 1]
    # Group by volume
    for vol_idx in range(len(volumes)):
        vol_global = [s['foreground_ratio'] for s in all_stats['global_crops'] if s['volume_idx'] == vol_idx]
        vol_local = [s['foreground_ratio'] for s in all_stats['local_crops'] if s['volume_idx'] == vol_idx]
        if vol_global and vol_local:
            # Plot mean for this volume
            ax.scatter(np.mean(vol_global), np.mean(vol_local), 
                      label=f'Volume {vol_idx}', s=100, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Global Crop Foreground Ratio (mean)')
    ax.set_ylabel('Local Crop Foreground Ratio (mean)')
    ax.set_title('Foreground Ratio: Global vs Local (per volume)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(cfg.train.output_dir) / 'crop_foreground_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Analyze foreground content in CT crops')
    parser.add_argument('--config', type=str, default='dinov3/configs/ssl_ct3d_config.yaml',
                       help='Path to config file')
    parser.add_argument('--volume-path', type=str, default=None,
                       help='Path to single CT volume file (optional, overrides dataset)')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of volumes to sample from dataset')
    args = parser.parse_args()
    
    # Setup config
    setup_args = DinoV3SetupArgs(config_file=args.config)
    cfg = setup_config(setup_args)
    
    # Initialize distributed (required for some components)
    distributed.init_distributed_mode()
    
    # Analyze crops
    stats = analyze_crops(cfg, volume_path=args.volume_path, n_samples=args.n_samples)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
