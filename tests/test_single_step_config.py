"""
Test a simple forward pass through the model using the ssl_ct3d_debug.yaml config.

This test verifies that:
1. Config loading works correctly
2. Model can be instantiated
3. A forward pass completes without errors
4. Outputs are finite
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow running tests from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinov3.configs import DinoV3SetupArgs, get_cfg_from_args, setup_job, exit_job
from dinov3.models import build_model_from_cfg
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.data.masking import MaskingGenerator3D


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_model_forward_pass(tmp_path):
    """Test a simple forward pass through the model using ssl_ct3d_debug.yaml config."""
    
    # 1. Load config from ssl_ct3d_debug.yaml
    config_path = Path(__file__).parent.parent / "dinov3" / "configs" / "ssl_ct3d_debug.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    args = DinoV3SetupArgs(
        config_file=str(config_path),
        output_dir=str(output_dir),
        opts=[],
    )
    
    cfg = get_cfg_from_args(args, strict=False)
    
    # 2. Build student and teacher backbones directly (no distributed setup needed)
    # Note: build_model_from_cfg creates models on "meta" device
    student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
    
    # 3. Move models from meta device to CUDA and initialize weights
    # Use to_empty() to properly move meta tensors to CUDA
    print("=== Moving models to CUDA ===")
    student_backbone.to_empty(device="cuda")
    teacher_backbone.to_empty(device="cuda")
    
    print("=== Initializing weights ===")
    student_backbone.init_weights()
    teacher_backbone.init_weights()
    
    # Check for NaNs in model weights after initialization
    print("\n=== Checking model weights after initialization ===")
    student_nan_params = sum(torch.isnan(p).any().item() for p in student_backbone.parameters())
    teacher_nan_params = sum(torch.isnan(p).any().item() for p in teacher_backbone.parameters())
    print(f"Student backbone params with NaNs: {student_nan_params}")
    print(f"Teacher backbone params with NaNs: {teacher_nan_params}")
    
    if student_nan_params > 0:
        print("WARNING: Student backbone has NaN parameters!")
        for name, param in student_backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"  {name}: shape={param.shape}, NaN count={torch.isnan(param).sum().item()}")
    
    if teacher_nan_params > 0:
        print("WARNING: Teacher backbone has NaN parameters!")
        for name, param in teacher_backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"  {name}: shape={param.shape}, NaN count={torch.isnan(param).sum().item()}")
    
    student_backbone.eval()
    teacher_backbone.eval()
    
    # 4. Create dummy 3D input data matching the config
    # From config: global_crops_size_3d: [64, 192, 192], batch_size_per_gpu: 2
    batch_size = cfg.train.batch_size_per_gpu
    n_global_crops = 2
    n_local_crops = cfg.crops.local_crops_number
    
    # Global crops: [n_global_crops, B, C, D, H, W]
    D, H, W = cfg.crops.global_crops_size_3d
    C = cfg.student.in_chans  # 1 for CT volumes
    global_crops = torch.randn(n_global_crops, batch_size, C, D, H, W, device="cuda")
    
    # Local crops: [n_local_crops, B, C, D, H, W]
    D_local, H_local, W_local = cfg.crops.local_crops_size_3d
    local_crops = torch.randn(n_local_crops, batch_size, C, D_local, H_local, W_local, device="cuda")
    
    # 5. Forward pass through student backbone
    with torch.no_grad():
        # Flatten crops for backbone input
        global_crops_flat = global_crops.flatten(0, 1)  # [n_global_crops * B, C, D, H, W]
        local_crops_flat = local_crops.flatten(0, 1)    # [n_local_crops * B, C, D, H, W]
        
        print("\n=== Input Data Check ===")
        print(f"Global crops shape: {global_crops_flat.shape}")
        print(f"Global crops - min: {global_crops_flat.min().item():.4f}, max: {global_crops_flat.max().item():.4f}, mean: {global_crops_flat.mean().item():.4f}")
        print(f"Global crops has NaNs: {torch.isnan(global_crops_flat).any().item()}")
        print(f"Global crops has Infs: {torch.isinf(global_crops_flat).any().item()}")
        print(f"Local crops shape: {local_crops_flat.shape}")
        print(f"Local crops - min: {local_crops_flat.min().item():.4f}, max: {local_crops_flat.max().item():.4f}, mean: {local_crops_flat.mean().item():.4f}")
        print(f"Local crops has NaNs: {torch.isnan(local_crops_flat).any().item()}")
        print(f"Local crops has Infs: {torch.isinf(local_crops_flat).any().item()}")
        
        # Forward through student backbone (can handle multiple inputs)
        # Use forward_features to get dict outputs, or is_training=True
        print("\n=== Student Backbone Forward Pass ===")
        outputs = student_backbone.forward_features(
            [global_crops_flat, local_crops_flat],
            masks=[None, None],  # No masking for simple forward pass
        )
        # forward_features returns a list of dicts when given a list of inputs
        global_out, local_out = outputs[0], outputs[1]
        
        print("\n=== Student Global Outputs ===")
        for key, value in global_out.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    print(f"  {key}: shape={value.shape}, EMPTY TENSOR")
                else:
                    print(f"  {key}: shape={value.shape}, min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
                    print(f"    Has NaNs: {torch.isnan(value).any().item()}, Has Infs: {torch.isinf(value).any().item()}")
                    if torch.isnan(value).any():
                        print(f"    WARNING: NaN detected in {key}!")
                        nan_count = torch.isnan(value).sum().item()
                        print(f"    NaN count: {nan_count} / {value.numel()}")
        
        print("\n=== Student Local Outputs ===")
        for key, value in local_out.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    print(f"  {key}: shape={value.shape}, EMPTY TENSOR")
                else:
                    print(f"  {key}: shape={value.shape}, min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
                    print(f"    Has NaNs: {torch.isnan(value).any().item()}, Has Infs: {torch.isinf(value).any().item()}")
                    if torch.isnan(value).any():
                        print(f"    WARNING: NaN detected in {key}!")
                        nan_count = torch.isnan(value).sum().item()
                        print(f"    NaN count: {nan_count} / {value.numel()}")
    
    # 6. Check outputs are present and finite
    assert "x_norm_clstoken" in global_out
    assert "x_storage_tokens" in global_out
    assert "x_norm_patchtokens" in global_out
    
    assert "x_norm_clstoken" in local_out
    assert "x_storage_tokens" in local_out
    assert "x_norm_patchtokens" in local_out
    
    # Check global outputs
    global_cls = global_out["x_norm_clstoken"]
    global_patch = global_out["x_norm_patchtokens"]
    
    assert global_cls.shape[0] == n_global_crops * batch_size
    assert torch.isfinite(global_cls).all(), "Global CLS tokens contain non-finite values"
    assert torch.isfinite(global_patch).all(), "Global patch tokens contain non-finite values"
    
    # Check local outputs
    local_cls = local_out["x_norm_clstoken"]
    local_patch = local_out["x_norm_patchtokens"]
    
    assert local_cls.shape[0] == n_local_crops * batch_size
    assert torch.isfinite(local_cls).all(), "Local CLS tokens contain non-finite values"
    assert torch.isfinite(local_patch).all(), "Local patch tokens contain non-finite values"
    
    # 7. Test teacher backbone forward pass
    with torch.no_grad():
        print("\n=== Teacher Backbone Forward Pass ===")
        # For single input, forward_features returns a dict (not a list)
        # When given a list, it returns a list of dicts
        teacher_out = teacher_backbone.forward_features(
            global_crops_flat,
            masks=None,
        )
        # If it's a list (shouldn't happen for single input), take first element
        if isinstance(teacher_out, list):
            teacher_out = teacher_out[0]
        
        print("\n=== Teacher Outputs ===")
        for key, value in teacher_out.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    print(f"  {key}: shape={value.shape}, EMPTY TENSOR")
                else:
                    print(f"  {key}: shape={value.shape}, min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
                    print(f"    Has NaNs: {torch.isnan(value).any().item()}, Has Infs: {torch.isinf(value).any().item()}")
                    if torch.isnan(value).any():
                        print(f"    WARNING: NaN detected in {key}!")
                        nan_count = torch.isnan(value).sum().item()
                        print(f"    NaN count: {nan_count} / {value.numel()}")
    
    assert "x_norm_clstoken" in teacher_out
    assert "x_storage_tokens" in teacher_out
    assert "x_norm_patchtokens" in teacher_out
    
    teacher_cls = teacher_out["x_norm_clstoken"]
    teacher_patch = teacher_out["x_norm_patchtokens"]
    
    assert teacher_cls.shape[0] == n_global_crops * batch_size
    assert torch.isfinite(teacher_cls).all(), "Teacher CLS tokens contain non-finite values"
    assert torch.isfinite(teacher_patch).all(), "Teacher patch tokens contain non-finite values"
    
    print(f"✓ Forward pass completed successfully!")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Global CLS shape: {global_cls.shape}")
    print(f"  Global patch shape: {global_patch.shape}")
    print(f"  Local CLS shape: {local_cls.shape}")
    print(f"  Local patch shape: {local_patch.shape}")
    print(f"  Teacher CLS shape: {teacher_cls.shape}")
    print(f"  Teacher patch shape: {teacher_patch.shape}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_loss_calculation(tmp_path):
    """Test loss calculation using SSLMetaArch with ssl_ct3d_debug.yaml config."""
    
    # Set up single-process distributed environment (required for loss calculation)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")  # Different port from other test
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    # 1. Load config from ssl_ct3d_debug.yaml
    config_path = Path(__file__).parent.parent / "dinov3" / "configs" / "ssl_ct3d_debug.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    args = DinoV3SetupArgs(
        config_file=str(config_path),
        output_dir=str(output_dir),
        opts=[],
    )
    
    cfg = get_cfg_from_args(args, strict=False)
    
    # 2. Set up distributed (required for loss calculation)
    setup_job(
        output_dir=str(output_dir),
        distributed_enabled=True,
        logging_enabled=False,
        seed=0,
        restrict_print_to_main_process=False,
    )
    
    try:
        # 3. Create SSLMetaArch model
        # Note: SSLMetaArch creates models on "meta" device
        print("=== Creating SSLMetaArch ===")
        model = SSLMetaArch(cfg)
    
        # 4. Prepare for distributed training (sets up FSDP)
        model.prepare_for_distributed_training()
        
        # 5. Move model from meta device to CUDA and initialize weights
        # Fill with NaNs first (as done in training), then move to CUDA
        model._apply(
            lambda t: torch.full_like(
                t,
                fill_value=float("nan") if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
                device="cuda",
            ),
            recurse=True,
        )
        
        print("=== Initializing weights ===")
        model.init_weights()
        model.train()
        
        # Check for NaNs in model weights after initialization
        print("\n=== Checking model weights after initialization ===")
        total_nan_params = sum(torch.isnan(p).any().item() for p in model.parameters())
        print(f"Total params with NaNs: {total_nan_params}")
        if total_nan_params > 0:
            print("WARNING: Model has NaN parameters!")
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"  {name}: shape={param.shape}, NaN count={torch.isnan(param).sum().item()}")
        
        # 6. Create dummy 3D input data matching the config
        batch_size = cfg.train.batch_size_per_gpu
        n_global_crops = 2
        n_local_crops = cfg.crops.local_crops_number
        
        D, H, W = cfg.crops.global_crops_size_3d
        D_local, H_local, W_local = cfg.crops.local_crops_size_3d
        C = cfg.student.in_chans
        
        # Create crops in the format expected by collate_data_and_cast
        # Format: [n_crops, B, C, D, H, W] -> flatten to [n_crops * B, C, D, H, W]
        global_crops = torch.randn(n_global_crops * batch_size, C, D, H, W, device="cuda")
        local_crops = torch.randn(n_local_crops * batch_size, C, D_local, H_local, W_local, device="cuda")
        
        # 7. Create masks for iBOT
        # Calculate number of tokens
        patch_size_d, patch_size_h, patch_size_w = cfg.crops.patch_size_3d
        D_p = D // patch_size_d
        H_p = H // patch_size_h
        W_p = W // patch_size_w
        n_tokens = D_p * H_p * W_p
        
        mask_generator = MaskingGenerator3D(
            input_size=(D_p, H_p, W_p),
            max_num_patches=int(0.5 * n_tokens),
        )
        
        # Create masks for each sample
        mask_ratio_min, mask_ratio_max = cfg.ibot.mask_ratio_min_max
        mask_probability = cfg.ibot.mask_sample_probability
        
        n_samples_masked = int(n_global_crops * batch_size * mask_probability)
        masks_list = []
        upperbound = 0
        
        for i in range(n_global_crops * batch_size):
            if i < n_samples_masked:
                # Sample mask ratio
                mask_ratio = torch.rand(1).item() * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
                num_masking_patches = int(n_tokens * mask_ratio)
                mask = mask_generator(num_masking_patches=num_masking_patches)
                # Convert numpy array to torch tensor
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                upperbound += num_masking_patches
            else:
                mask = torch.zeros((D_p, H_p, W_p), dtype=torch.bool)
            masks_list.append(mask.flatten())
        
        collated_masks = torch.stack(masks_list)  # [n_global_crops * B, n_tokens]
        mask_indices_list = collated_masks.flatten().nonzero().flatten()
        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
        n_masked_patches = torch.tensor([mask_indices_list.shape[0]], dtype=torch.long, device="cuda")
        
        # 8. Prepare data dictionary
        data = {
            "collated_global_crops": global_crops,
            "collated_local_crops": local_crops,
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "n_masked_patches": n_masked_patches,
            "upperbound": upperbound,
            "global_batch_size": batch_size,
        }
        
        print("\n=== Data Batch Info ===")
        print(f"Global crops shape: {global_crops.shape}")
        print(f"Local crops shape: {local_crops.shape}")
        print(f"Masks shape: {collated_masks.shape}")
        print(f"Mask indices list length: {mask_indices_list.shape[0]}")
        print(f"Number of masked patches: {n_masked_patches.item()}")
        print(f"Upperbound: {upperbound}")
        
        # 9. Get teacher temperature
        teacher_temp = cfg.teacher.teacher_temp
        
        # 10. Forward-backward pass
        print("\n=== Running forward_backward ===")
        total_loss, metrics_dict = model.forward_backward(
            data,
            teacher_temp=teacher_temp,
            iteration=0,
        )
        
        # 11. Print all loss values and check for NaNs
        print("\n=== Loss Values ===")
        print(f"Total loss: {total_loss.item():.6f}")
        print(f"Total loss has NaN: {torch.isnan(total_loss).item()}")
        print(f"Total loss has Inf: {torch.isinf(total_loss).item()}")
        
        if torch.isnan(total_loss):
            print("ERROR: Total loss is NaN!")
        
        print("\n=== Individual Loss Components ===")
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    print(f"  {key}: EMPTY TENSOR")
                else:
                    val_item = value.item() if value.numel() == 1 else value
                    print(f"  {key}: {val_item}")
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        print(f"    Has NaNs: {torch.isnan(value).any().item()}, Has Infs: {torch.isinf(value).any().item()}")
                        if torch.isnan(value).any():
                            print(f"    WARNING: NaN detected in {key}!")
                            nan_count = torch.isnan(value).sum().item()
                            print(f"    NaN count: {nan_count} / {value.numel()}")
            else:
                print(f"  {key}: {value}")
        
        # 12. Verify all losses are finite
        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss}"
        assert total_loss.item() > 0, f"Total loss should be positive: {total_loss.item()}"
        
        # Check individual loss components
        required_losses = ["dino_local_crops_loss", "dino_global_crops_loss", "ibot_loss", "koleo_loss"]
        for loss_name in required_losses:
            assert loss_name in metrics_dict, f"Missing loss: {loss_name}"
            loss_value = metrics_dict[loss_name]
            if isinstance(loss_value, torch.Tensor):
                assert torch.isfinite(loss_value).all(), f"{loss_name} is not finite: {loss_value}"
        
        print("\n✓ Loss calculation completed successfully!")
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  DINO local loss: {metrics_dict['dino_local_crops_loss'].item():.6f}")
        print(f"  DINO global loss: {metrics_dict['dino_global_crops_loss'].item():.6f}")
        print(f"  iBOT loss: {metrics_dict['ibot_loss'].item():.6f}")
        print(f"  KoLeo loss: {metrics_dict['koleo_loss'].item():.6f}")
    
    finally:
        # Clean up distributed
        exit_job(distributed_enabled=True, logging_enabled=False)









