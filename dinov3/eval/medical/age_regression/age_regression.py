"""
Age Regression Prototype using DINOv3 3D Backbone
================================================

This script evaluates a trained DINOv3 backbone on age regression task using OASIS-1 dataset.

Features:
- Supports both linear regression (fast) and MLP regression (better performance)
- Extracts features using DINOv3 backbone
- Handles 3D volume cropping similar to training
- Includes multiple regression metrics (MAE, MSE, R², Pearson correlation)
- Simple train/val/test split with stratification by age groups
"""
#%%
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# Add parent dir to path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from monai.transforms import (
    Compose,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    Rand3DElastic,
    RandFlip,
    RandRotate90,
    RandScaleIntensity,
    RandShiftIntensity,
    ScaleIntensityRange,
    Resize,
)
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval
import random
from monai.transforms import (
    Compose, EnsureType, LoadImage, NormalizeIntensity, Rand3DElastic, RandFlip, RandRotate90, RandScaleIntensity, RandShiftIntensity, ScaleIntensityRange, Resize
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Configuration & Data Structures
# ============================================================================

@dataclass
class RegressionConfig:
    """Configuration for age regression evaluation."""
    # Paths
    checkpoint_path: str = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/checkpoint_final.pth"
    config_path: str = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/config.yaml"
    # oasis_csv_path: str = "/home/lukas/3Ddinov3/dinov3/eval/medical/age_regression/oasis_data.csv"
    oasis_csv_path: str = "/home/lukas/data/base_t1w_SegmentationPreproc_v2/filename-age.csv"
    volumes_root_dir: str = "/home/lukas/data/base_t1w_SegmentationPreproc_v2"  # OASIS nifti converted directory
    output_dir: str = "/home/lukas/3Ddinov3/dinov3/eval/medical/age_regression/age_regression_results"
    \
    # Data handling
    resize_shape_3d: Tuple[int, int, int] = (200, 200, 200)  # Resize volumes to this shape for backbone
    resize_scale: Optional[float] = None  # Set to scale volume (e.g., 0.5 for half size)
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    age_stratification_bins: int = 5  # For stratified split
    
    # Overfitting mode (for debugging)
    overfit_mode: bool = False  # If True, train on small subset and skip validation
    overfit_n_samples: int = 100  # Number of samples to use for overfitting
    
    # Training
    batch_size: int = 16
    num_workers: Optional[int] = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.0  # No regularization in overfit mode
    epochs: int = 10
    freeze_backbone: bool = True
    mlp_hidden_dims: Optional[List[int]] = None  # If None, defaults to [512, 256]
    
    # Regression head
    regression_head_type: str = "mlp"  # "linear", "mlp", or "both"
    mlp_dropout: float = 0.1
    
    # Feature extraction
    use_multilayer_features: bool = False  # If True, extract from last N layers
    n_layers_for_features: int = 4  # Number of last layers to extract features from
    
    # Device
    device: str = "cuda:0"
    mixed_precision: bool = True

    # Timing
    enable_timing: bool = True
    timing_log_interval: int = 50

    # GPU transforms
    train_transforms_on_gpu: bool = True
    eval_transforms_on_gpu: bool = False
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [1024, 512, 256]
        if self.num_workers is None:
            self.num_workers = min(8, os.cpu_count() or 4)


# ============================================================================

# =====================
# Data Loading Section
# =====================

def build_transforms(cfg: RegressionConfig, is_train: bool):
    base_transforms = [
        LoadImage(image_only=True),
        EnsureType(),
        Resize(cfg.resize_shape_3d, mode="bilinear", dtype=np.float32),
        ScaleIntensityRange(a_min=0.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
    if is_train:
        base_transforms += [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate90(prob=0.5, max_k=3),
            Rand3DElastic(
                prob=0.2,
                sigma_range=(3.0, 5.0),
                magnitude_range=(50.0, 150.0),
                translate_range=(0, 0, 0),
                rotate_range=(0.05, 0.05, 0.05),
                scale_range=(0.05, 0.05, 0.05),
            ),
            RandScaleIntensity(factors=0.1, prob=0.5),
            RandShiftIntensity(offsets=0.1, prob=0.5),
        ]
    return Compose(base_transforms)


def _find_cases(csv_path: str, volumes_root: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Try to find the subject id column
    for col in ["subject_id", "ID", "id", "filename"]:
        if col in df.columns:
            subject_col = col
            break
    else:
        raise ValueError("CSV must contain one of: subject_id, ID, id")
    cases = {"train": [], "val": [], "test": []}
    for i, row in df.iterrows():
        subject_id = str(row[subject_col])
        age = float(row["age"])
        # Assume the CSV has a column 'image' or 'path' with the full path or relative path
        image_path = row.get("image") or row.get("path")
        if not image_path:
            # Try to reconstruct path as before (legacy)
            subject_dir = Path(volumes_root) / subject_id / "PROCESSED" / "MPRAGE" / "T88_111"
            t88_gfc_files = list(subject_dir.glob(f"{subject_id}*_t88_gfc.nii.gz"))
            if not t88_gfc_files:
                t88_gfc_files = list(subject_dir.glob("*_t88_gfc.nii.gz"))
            if t88_gfc_files:
                image_path = str(t88_gfc_files[0])
            else:
                continue
        # Determine split by folder name in path
        lower_path = str(image_path).lower()
        if "/train/" in lower_path or "\\train\\" in lower_path:
            split = "train"
        elif "/val/" in lower_path or "/valid/" in lower_path or "\\val\\" in lower_path or "\\valid\\" in lower_path:
            split = "val"
        elif "/test/" in lower_path or "\\test\\" in lower_path:
            split = "test"
        else:
            # If not found, skip (or assign to train by default)
            continue
        cases[split].append({"image": str(image_path), "age": age})
    return cases

def split_cases(cases, cfg: RegressionConfig):
    # cases is now a dict with keys 'train', 'val', 'test'
    train_cases = cases.get("train", [])
    val_cases = cases.get("val", [])
    test_cases = cases.get("test", [])
    return train_cases, val_cases, test_cases

class AgeRegressionDataset(Dataset):
    def __init__(self, cases, transform):
        self.cases = cases
        self.transform = transform
    def __len__(self):
        return len(self.cases)
    def __getitem__(self, idx):
        item = self.cases[idx]
        image = self.transform(item["image"])
        age = float(item["age"])
        print(f"Loaded case {idx}: image={item['image']}, age={age}, image shape after transform: {image.shape}")
        return image, age



class OASISAgeRegressionDataset(Dataset):
    """
    OASIS-1 Age Regression Dataset
    
    Expects:
    - CSV file with columns: subject_id, age (and possibly paths)
    - Volume files in volumes_root_dir
    """
    
    def __init__(
        self,
        csv_path: str,
        volumes_root: str,
        config_yaml_path: str,
        resize_shape_3d: Tuple[int, int, int] = (192, 192, 192),
        transform_type: str = "eval",  # "train" or "eval"
        indices: Optional[List[int]] = None,
        enable_timing: bool = False,
        timing_log_interval: int = 50,
        apply_transforms: bool = True,
    ):
        """
        Args:
            csv_path: Path to OASIS CSV with subject_id and age columns
            volumes_root: Root directory containing volume files
            config_yaml_path: Path to training config for transforms
            resize_shape_3d: Target shape for resizing volumes (D, H, W)
            transform_type: "train" for augmentation or "eval" for no augmentation
            indices: Optional subset of indices to use
        """
        self.volumes_root = Path(volumes_root)
        self.csv_path = csv_path
        self.transform_type = transform_type
        self.enable_timing = enable_timing
        self.timing_log_interval = timing_log_interval
        self.apply_transforms = apply_transforms
        self._timing_count = 0
        self._timing_load = 0.0
        self._timing_resize = 0.0
        self._timing_transform = 0.0
        self._timing_total = 0.0
        self._nifti_loader = LoadImage(image_only=True)
        self._subject_to_volume_path: Dict[str, Optional[str]] = {}
        
        # Load metadata
        self.metadata = pd.read_csv(csv_path)
        if indices is not None:
            self.metadata = self.metadata.iloc[indices].reset_index(drop=True)

        if "subject_id" in self.metadata.columns:
            subject_col = "subject_id"
        elif "ID" in self.metadata.columns:
            subject_col = "ID"
        elif "id" in self.metadata.columns:
            subject_col = "id"
        else:
            raise ValueError("CSV must contain one of: subject_id, ID, id")

        self.ages: List[float] = self.metadata["age"].astype(float).tolist()
        self.subject_ids: List[str] = self.metadata[subject_col].astype(str).tolist()
        
        # Load config for transforms
        if not distributed.is_enabled():
            distributed.enable()
        
        setup_args = DinoV3SetupArgs(
            config_file=config_yaml_path,
            pretrained_weights="",
            shard_unsharded_model=False,
            output_dir="",
            opts=[],
        )
        self.config = setup_config(setup_args, strict_cfg=False)
        self.resize_shape_3d = resize_shape_3d

        self.volume_paths: List[Optional[str]] = [self._find_volume_file(subject_id) for subject_id in self.subject_ids]
        missing_files = sum(path is None for path in self.volume_paths)
        if missing_files > 0:
            logger.warning(
                f"Could not resolve volume file for {missing_files}/{len(self.volume_paths)} samples during dataset init"
            )

    def _load_volume(self, volume_path: str) -> torch.Tensor:
        """Load volume from disk. Supports .npy and .nii.gz formats."""
        volume_path = Path(volume_path)
        
        if volume_path.suffix == ".npy":
            volume = np.load(volume_path)
        elif volume_path.suffix == ".gz" or volume_path.suffixes[-2:] == [".nii", ".gz"]:
            volume = self._nifti_loader(str(volume_path))
        else:
            raise ValueError(f"Unsupported volume format: {volume_path}")

        if torch.is_tensor(volume):
            volume = volume.float()
        else:
            volume = torch.from_numpy(volume).float()  # Convert to torch.Tensor

        # Ensure channel-first format

        volume = volume.squeeze().unsqueeze(0)  # (1, D, H, W)
        # if volume.ndim == 3:
        #     volume = volume.unsqueeze(0)  # (1, D, H, W)
        return volume  # (1, D, H, W)

    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            (volume, age) where volume is (1, D, H, W) and age is float
        """
        age = self.ages[idx]
        volume_path = self.volume_paths[idx]
        subject_id = self.subject_ids[idx]
        
        if volume_path is None:
            raise FileNotFoundError(f"Volume file not found for subject {subject_id}")
        
        # Load and process volume
        if self.enable_timing:
            t_start = time.perf_counter()

        t0 = time.perf_counter() if self.enable_timing else None
        volume = self._load_volume(volume_path)  # (1, D, H, W)
        if self.enable_timing:
            t1 = time.perf_counter()
            self._timing_load += t1 - t0
        print(f"Loaded volume for subject {subject_id}, shape: {volume.shape}, age: {age}")
        t0 = time.perf_counter() if self.enable_timing else None
        # volume = self._resize_volume(volume, self.resize_shape_3d)  # Resize to (1, D', H', W')
        if self.enable_timing:
            t1 = time.perf_counter()
            self._timing_resize += t1 - t0

        if self.transform is not None:
            t0 = time.perf_counter() if self.enable_timing else None
            volume = self.transform(volume)  # Apply normalization/augmentation
            print(f"Applied transforms for subject {subject_id}, resulting shape: {volume.shape}")

        return volume, age
    
    def _find_volume_file(self, subject_id: str) -> Optional[str]:
        """Find T88_gfc volume file for a given subject."""
        if subject_id in self._subject_to_volume_path:
            return self._subject_to_volume_path[subject_id]

        # OASIS structure: {subject_id}/PROCESSED/MPRAGE/T88_111/*_t88_gfc.nii.gz
        # Look for the T88_gfc file (not the masked version)
        subject_dir = self.volumes_root / subject_id / "PROCESSED" / "MPRAGE" / "T88_111"
        
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            self._subject_to_volume_path[subject_id] = None
            return None
        
        # Find T88 gfc file (non-masked version)
        t88_gfc_files = list(subject_dir.glob(f"{subject_id}*_t88_gfc.nii.gz"))
        
        if t88_gfc_files:
            resolved = str(t88_gfc_files[0])
            self._subject_to_volume_path[subject_id] = resolved
            return resolved
        
        # Fallback: try any t88_gfc file in the directory
        t88_gfc_files = list(subject_dir.glob("*_t88_gfc.nii.gz"))
        if t88_gfc_files:
            resolved = str(t88_gfc_files[0])
            self._subject_to_volume_path[subject_id] = resolved
            return resolved
        
        logger.warning(f"No T88_gfc file found for subject {subject_id}")
        self._subject_to_volume_path[subject_id] = None
        return None

def create_dataloaders(cfg: RegressionConfig):
    cases = _find_cases(cfg.oasis_csv_path, cfg.volumes_root_dir)
    if not cases:
        raise RuntimeError(f"No cases found in {cfg.volumes_root_dir} from {cfg.oasis_csv_path}")
    train_cases, val_cases, test_cases = split_cases(cases, cfg)
    logger.info(
        "OASIS split: total=%d train=%d val=%d test=%d",
        len(cases), len(train_cases), len(val_cases), len(test_cases)
    )
    train_ds = OASISAgeRegressionDataset(train_cases, config_yaml_path=cfg.config_path, resize_shape_3d=cfg.resize_shape_3d, volumes_root=)
    val_ds = OASISAgeRegressionDataset(val_cases,  config_yaml_path=cfg.config_path, resize_shape_3d=cfg.resize_shape_3d)
    test_ds = OASISAgeRegressionDataset(test_cases, config_yaml_path=cfg.config_path, resize_shape_3d=cfg.resize_shape_3d)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
# ============================================================================
# Regression Heads
# ============================================================================

class LinearRegressionHead(nn.Module):
    """Simple linear regression head with age constraint [0-100]."""
    
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features) feature tensor
        
        Returns:
            (B, 1) age predictions
        """
        # No constraint during training - let network learn the full range
        # Clipping to [0, 100] can be done during evaluation if needed
        return self.linear(x)


class MLPRegressionHead(nn.Module):
    """MLP regression head with age constraint [0-100]."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features) feature tensor
        
        Returns:
            (B, 1) age predictions
        """
        # No constraint during training - let network learn the full range
        # Clipping to [0, 100] can be done during evaluation if needed
        return self.model(x)


# ============================================================================
# Feature Extractor
# ============================================================================

class FeatureExtractor(nn.Module):
    """Wrapper to extract features from DINOv3 backbone."""
    
    def __init__(self, backbone: nn.Module, device: str, use_multilayer: bool = False, n_layers: int = 4):
        super().__init__()
        self.backbone = backbone
        # Keep backbone in eval mode but allow gradients to flow for on-the-fly augmentation
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.device = device
        self.use_multilayer = use_multilayer
        self.n_layers = n_layers
        
        if use_multilayer:
            logger.info(f"Using multi-layer features from last {n_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features from backbone.
        
        Args:
            x: (B, C, D, H, W) volumes
        
        Returns:
            (B, feat_dim) CLS token features (or concatenated multi-layer features)
        """
        x = x.to(self.device)
        
        # Use no_grad only for feature extraction, allowing gradients through data transforms
        with torch.no_grad():
            if not self.use_multilayer:
                # Single layer (final layer CLS token only)
                features = self.backbone.forward_features(x)
                cls_token = features["x_norm_clstoken"]  # (B, C)
                return cls_token
            else:
                # Multi-layer: extract CLS tokens from last N layers
                return self._extract_multilayer_features(x)
    
    def _extract_multilayer_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS tokens from the last N layers and concatenate them.
        
        Args:
            x: (B, C, D, H, W) volumes
        
        Returns:
            (B, feat_dim * n_layers) concatenated CLS tokens from last N layers
        """
        # Prepare tokens
        x, spatial_dims = self.backbone.prepare_tokens_with_masks(x, masks=None)
        
        # Store features from last N layers
        layer_features = []
        n_blocks = len(self.backbone.blocks)
        start_layer = max(0, n_blocks - self.n_layers)
        
        # Forward through blocks
        for idx, blk in enumerate(self.backbone.blocks):
            if self.backbone.rope_embed is not None:
                # Handle both 2D and 3D cases
                if len(spatial_dims) == 3:  # 3D case: (D, H, W)
                    rope_sincos = self.backbone.rope_embed(D=spatial_dims[0], H=spatial_dims[1], W=spatial_dims[2])
                else:  # 2D case: (H, W)
                    rope_sincos = self.backbone.rope_embed(H=spatial_dims[0], W=spatial_dims[1])
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            
            # Save CLS token from last N layers
            if idx >= start_layer:
                # Apply normalization to CLS token
                if self.backbone.untie_cls_and_patch_norms:
                    x_norm_cls = self.backbone.cls_norm(x[:, 0:1])  # (B, 1, C)
                else:
                    x_norm_cls = self.backbone.norm(x[:, 0:1])  # (B, 1, C)
                
                cls_token = x_norm_cls[:, 0]  # (B, C)
                layer_features.append(cls_token)
        
        # Concatenate features from all layers
        concatenated = torch.cat(layer_features, dim=1)  # (B, C * n_layers)
        return concatenated


# ============================================================================
# End-to-End Model for On-The-Fly Training
# ============================================================================

class DINOAgeRegressionModel(nn.Module):
    """End-to-end model combining feature extraction and regression head."""
    
    def __init__(self, feature_extractor: nn.Module, regression_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regression_head = regression_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) volumes
        Returns:
            (B, 1) age predictions
        """
        features = self.feature_extractor(x)
        predictions = self.regression_head(features)
        return predictions


# ============================================================================
# Training & Evaluation
# ============================================================================

class RegressionTrainer:
    """Trainer for age regression."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set device first
        self.device = torch.device(config.device)
        
        # Load backbone
        logger.info("Loading DINOv3 backbone...")
        self.backbone, self.autocast_dtype, self.dino_config = self._load_backbone()
        
        # Feature extractor
        if config.use_multilayer_features:
            logger.info(f"Using multi-layer features: extracting from last {config.n_layers_for_features} layers")
        else:
            logger.info("Using single-layer features: final CLS token only")
            
        self.feature_extractor = FeatureExtractor(
            self.backbone, 
            config.device,
            use_multilayer=config.use_multilayer_features,
            n_layers=config.n_layers_for_features
        )

        
        # Get feature dimension
        dummy_volume = torch.randn(1, 1, 192, 192, 192).to(self.device)
        with torch.no_grad():
            dummy_feats = self.feature_extractor(dummy_volume)
        self.feat_dim = dummy_feats.shape[1]
        logger.info(f"Feature dimension: {self.feat_dim}")
    
    def _load_backbone(self) -> Tuple[nn.Module, torch.dtype, dict]:
        """Load DINOv3 backbone from checkpoint."""
        if not distributed.is_enabled():
            distributed.enable()
        
        setup_args = DinoV3SetupArgs(
            config_file=self.config.config_path,
            pretrained_weights=self.config.checkpoint_path,
            shard_unsharded_model=False,
            output_dir="",
            opts=[],
        )
        config = setup_config(setup_args, strict_cfg=False)
        model = build_model_for_eval(config, self.config.checkpoint_path)
        model = model.to(self.device)
        model.eval()
        
        autocast_dtype = get_autocast_dtype(config)
        return model, autocast_dtype, config
    
    def extract_all_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for entire dataset."""
        all_features = []
        all_ages = []
        
        logger.info("Extracting features...")
        for i, (volumes, ages) in enumerate(dataloader):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(dataloader)} batches")
            
            with torch.no_grad():
                feats = self.feature_extractor(volumes)
            
            all_features.append(feats.numpy())
            all_ages.extend(ages)
        
        features = np.concatenate(all_features, axis=0)
        ages = np.array(all_ages)
        
        logger.info(f"Extracted features shape: {features.shape}, ages shape: {ages.shape}")
        return features, ages
    
    @staticmethod
    def verify_frozen_parameters(model: nn.Module, feature_extractor: nn.Module, head: nn.Module):
        """Verify which parameters are trainable vs frozen."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        backbone_params = sum(p.numel() for p in feature_extractor.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in feature_extractor.backbone.parameters() if p.requires_grad)
        
        head_params = sum(p.numel() for p in head.parameters())
        head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        
        logger.info("\n" + "="*60)
        logger.info("PARAMETER VERIFICATION")
        logger.info("="*60)
        logger.info(f"Total parameters:          {total_params:,}")
        logger.info(f"Trainable parameters:      {trainable_params:,}")
        logger.info(f"Frozen parameters:         {frozen_params:,}")
        logger.info(f"")
        logger.info(f"Backbone parameters:       {backbone_params:,}")
        logger.info(f"Backbone trainable:        {backbone_trainable:,} {'✓ FROZEN' if backbone_trainable == 0 else '✗ NOT FROZEN!'}")
        logger.info(f"")
        logger.info(f"Regression head parameters: {head_params:,}")
        logger.info(f"Regression head trainable:  {head_trainable:,} {'✓ TRAINABLE' if head_trainable == head_params else '✗ NOT ALL TRAINABLE!'}")
        logger.info("="*60 + "\n")
    
    def train_linear_head(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """Train and evaluate linear regression head with on-the-fly feature extraction."""
        logger.info("\n" + "="*60)
        logger.info("Training Linear Regression Head (On-The-Fly)")
        logger.info("="*60)
        
        # Create regression head
        head = LinearRegressionHead(self.feat_dim).to(self.device)
        
        # Create end-to-end model
        model = DINOAgeRegressionModel(self.feature_extractor, head)
        model.to(self.device)
        
        # Only optimize the regression head
        optimizer = torch.optim.Adam(head.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Verify which parameters are trainable
        self.verify_frozen_parameters(model, self.feature_extractor, head)
        
        # Training loop
        best_val_loss = float("inf")
        best_head_state = None
        log_interval = 100 if self.config.overfit_mode else 10
        
        # Track losses for plotting
        train_losses = []
        val_losses = []
        
        first_batch_check = True  # For gradient verification
        
        for epoch in range(self.config.epochs):
            # Train
            head.train()
            epoch_loss = 0.0
            n_batches = 0
            n_samples = 0
            epoch_start_time = time.perf_counter()

            if self.config.enable_timing:
                data_time = 0.0
                transfer_time = 0.0
                forward_time = 0.0
                backward_time = 0.0
                step_time = 0.0
                batch_time = 0.0
                end_time = time.perf_counter()
            
            for volumes, ages in train_loader:
                if self.config.enable_timing:
                    batch_start = time.perf_counter()
                    data_time += batch_start - end_time

                volumes = volumes.to(self.device, non_blocking=True)
                ages = ages.float().to(self.device, non_blocking=True).view(-1, 1)

                if self.config.enable_timing:
                    t_after_transfer = time.perf_counter()
                    transfer_time += t_after_transfer - batch_start
                optimizer.zero_grad()
                pred = model(volumes)

                if self.config.enable_timing:
                    t_after_forward = time.perf_counter()
                    forward_time += t_after_forward - t_after_transfer

                loss = criterion(pred, ages)
                loss.backward()
                if self.config.enable_timing:
                    t_after_backward = time.perf_counter()
                    backward_time += t_after_backward - t_after_forward
                
                optimizer.step()

                if self.config.enable_timing:
                    t_after_step = time.perf_counter()
                    step_time += t_after_step - t_after_backward
                    batch_time += t_after_step - batch_start
                    end_time = t_after_step
                
                epoch_loss += loss.item()
                n_batches += 1
                n_samples += ages.shape[0]
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            if self.config.enable_timing and ((epoch + 1) % log_interval == 0 or epoch == 0):
                avg_data = data_time / n_batches
                avg_transfer = transfer_time / n_batches
                avg_forward = forward_time / n_batches
                avg_backward = backward_time / n_batches
                avg_step = step_time / n_batches
                avg_batch = batch_time / n_batches
                epoch_total_time = time.perf_counter() - epoch_start_time
                samples_per_sec = n_samples / max(epoch_total_time, 1e-8)
                data_wait_pct = (data_time / max(batch_time, 1e-8)) * 100.0
                logger.info(
                    "Timing (s/batch): "
                    f"data {avg_data:.4f}, transfer {avg_transfer:.4f}, "
                    f"forward {avg_forward:.4f}, backward {avg_backward:.4f}, step {avg_step:.4f}, total {avg_batch:.4f}"
                )
                logger.info(
                    f"Perf: {samples_per_sec:.2f} samples/s | Data wait: {data_wait_pct:.1f}%"
                )
            
            # Validation (skip in overfit mode for speed)
            if val_loader is not None and not self.config.overfit_mode:
                head.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for volumes, ages in val_loader:
                        volumes = volumes.to(self.device, non_blocking=True)
                        ages = ages.float().to(self.device, non_blocking=True).view(-1, 1)
                        pred = model(volumes)
                        loss = criterion(pred, ages)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                
                if (epoch + 1) % log_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % log_interval == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f}")
        
        # Use best head (skip in overfit mode, just use final)
        if best_head_state is not None and not self.config.overfit_mode:
            head.load_state_dict(best_head_state)
        
        return {"head": head, "model": model, "train_losses": train_losses, "val_losses": val_losses}
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        set_name: str = "Test"
    ) -> Dict[str, float]:
        """Evaluate regression model with on-the-fly feature extraction."""
        model.eval()
        
        all_predictions = []
        all_ages = []
        
        with torch.no_grad():
            for volumes, ages in dataloader:
                volumes = volumes.to(self.device, non_blocking=True)
                pred = model(volumes).cpu().numpy().flatten()
                all_predictions.extend(pred)
                all_ages.extend(ages.numpy())
        
        predictions = np.array(all_predictions)
        ages = np.array(all_ages)
        
        # Compute metrics
        mae = np.mean(np.abs(predictions - ages))
        mse = np.mean((predictions - ages) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((predictions - ages) ** 2) / np.sum((ages - ages.mean()) ** 2))
        pearson_r, pearson_p = stats.pearsonr(predictions, ages)
        
        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
        }
        
        logger.info(f"\n{set_name} Set Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        model_type: str,
        save_path: Optional[str] = None
    ):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
        
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title(f'{model_type} - Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curve saved to {save_path}")
        
        plt.close()
    
    def run_experiment(self):
        """Run the full regression experiment."""
        logger.info(f"\nInitializing Age Regression Experiment")
        if self.config.overfit_mode:
            logger.info("=" * 60)
            logger.info("⚠️  OVERFITTING MODE ENABLED")
            logger.info(f"Training on {self.config.overfit_n_samples} samples")
            logger.info(f"No validation, zero regularization")
            logger.info("=" * 60)
        logger.info(f"Config: {self.config}")

        # Use new dataloader logic (like segmentation pipeline)
        train_loader, val_loader, test_loader = create_dataloaders(self.config)

        results = {}

        if self.config.regression_head_type in ["linear", "both"]:
            logger.info("\n" + "="*80)
            logger.info("LINEAR REGRESSION HEAD (With On-The-Fly Augmentation)")
            logger.info("="*80)
            trained_linear = self.train_linear_head(
                train_loader,
                val_loader
            )
            metrics = self.evaluate(
                trained_linear["model"],
                test_loader,
                "Test"
            )
            results["linear"] = metrics
            plot_path = os.path.join(self.config.output_dir, "linear_training_curve.png")
            self.plot_training_curves(
                trained_linear["train_losses"],
                trained_linear["val_losses"],
                "Linear Regression",
                plot_path
            )

        if self.config.regression_head_type in ["mlp", "both"]:
            logger.info("\n" + "="*80)
            logger.info("MLP REGRESSION HEAD (With On-The-Fly Augmentation)")
            logger.info("="*80)
            trained_mlp = self.train_mlp_head(
                train_loader,
                val_loader
            )
            metrics = self.evaluate(
                trained_mlp["model"],
                test_loader,
                "Test"
            )
            results["mlp"] = metrics
            plot_path = os.path.join(self.config.output_dir, "mlp_training_curve.png")
            self.plot_training_curves(
                trained_mlp["train_losses"],
                trained_mlp["val_losses"],
                "MLP Regression",
                plot_path
            )

        # Save results
        results_df = pd.DataFrame(results).T
        results_path = os.path.join(self.config.output_dir, "results.csv")
        results_df.to_csv(results_path)
        logger.info(f"\nResults saved to {results_path}")
        logger.info(f"\n{results_df}")
        return results
    
    def train_mlp_head(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """Train and evaluate MLP regression head with on-the-fly feature extraction."""
        logger.info("\n" + "="*60)
        logger.info("Training MLP Regression Head (On-The-Fly)")
        logger.info("="*60)
        
        # Create regression head
        head = MLPRegressionHead(
            self.feat_dim,
            self.config.mlp_hidden_dims,
            self.config.mlp_dropout
        ).to(self.device)
        
        # Create end-to-end model
        model = DINOAgeRegressionModel(self.feature_extractor, head)
        model.to(self.device)
        
        # Only optimize the regression head
        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()
        
        # Verify which parameters are trainable
        self.verify_frozen_parameters(model, self.feature_extractor, head)
        
        # Training loop
        best_val_loss = float("inf")
        best_head_state = None
        patience = 1000 if not self.config.overfit_mode else 999999  # No early stopping in overfit mode
        patience_counter = 0
        log_interval = 100 if self.config.overfit_mode else 10
        
        # Track losses for plotting
        train_losses = []
        val_losses = []
        
        first_batch_check = True  # For gradient verification
        
        for epoch in range(self.config.epochs):
            # Train
            head.train()
            epoch_loss = 0.0
            n_batches = 0
            n_samples = 0
            epoch_start_time = time.perf_counter()

            if self.config.enable_timing:
                data_time = 0.0
                transfer_time = 0.0
                forward_time = 0.0
                backward_time = 0.0
                step_time = 0.0
                batch_time = 0.0
                end_time = time.perf_counter()
            print(f"Epoch {epoch+1}/{self.config.epochs} - Starting training loop...")
            for volumes, ages in train_loader:
                if self.config.enable_timing:
                    batch_start = time.perf_counter()
                    data_time += batch_start - end_time

                volumes = volumes.to(self.device, non_blocking=True)

                ages = ages.float().to(self.device, non_blocking=True).view(-1, 1)

                if self.config.enable_timing:
                    t_after_transfer = time.perf_counter()
                    transfer_time += t_after_transfer - batch_start
                
                optimizer.zero_grad()
                pred = model(volumes)

                if self.config.enable_timing:
                    t_after_forward = time.perf_counter()
                    forward_time += t_after_forward - t_after_transfer

                loss = criterion(pred, ages)
                loss.backward()
                
                # Verify gradients on first batch
                if first_batch_check:
                    backbone_has_grad = any(p.grad is not None for p in self.feature_extractor.backbone.parameters())
                    head_has_grad = any(p.grad is not None for p in head.parameters())
                    logger.info(f"Gradient Check (first batch):")
                    logger.info(f"  Backbone has gradients: {backbone_has_grad} {'✗ UNEXPECTED!' if backbone_has_grad else '✓'}")
                    logger.info(f"  Head has gradients: {head_has_grad} {'✓ EXPECTED' if head_has_grad else '✗ NO GRADIENTS!'}")
                    first_batch_check = False
                
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)

                if self.config.enable_timing:
                    t_after_backward = time.perf_counter()
                    backward_time += t_after_backward - t_after_forward

                optimizer.step()

                if self.config.enable_timing:
                    t_after_step = time.perf_counter()
                    step_time += t_after_step - t_after_backward
                    batch_time += t_after_step - batch_start
                    end_time = t_after_step
                
                epoch_loss += loss.item()
                n_batches += 1
                n_samples += ages.shape[0]
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            if self.config.enable_timing and ((epoch + 1) % log_interval == 0 or epoch == 0):
                avg_data = data_time / n_batches
                avg_transfer = transfer_time / n_batches
                avg_forward = forward_time / n_batches
                avg_backward = backward_time / n_batches
                avg_step = step_time / n_batches
                avg_batch = batch_time / n_batches
                epoch_total_time = time.perf_counter() - epoch_start_time
                samples_per_sec = n_samples / max(epoch_total_time, 1e-8)
                data_wait_pct = (data_time / max(batch_time, 1e-8)) * 100.0
                logger.info(
                    "Timing (s/batch): "
                    f"data {avg_data:.4f}, transfer {avg_transfer:.4f}, "
                    f"forward {avg_forward:.4f}, backward {avg_backward:.4f}, step {avg_step:.4f}, total {avg_batch:.4f}"
                )
                logger.info(
                    f"Perf: {samples_per_sec:.2f} samples/s | Data wait: {data_wait_pct:.1f}%"
                )
            
            # Validation
            if val_loader is not None and not self.config.overfit_mode:
                head.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for volumes, ages in val_loader:
                        volumes = volumes.to(self.device, non_blocking=True)

                        ages = ages.float().to(self.device, non_blocking=True).view(-1, 1)
                        pred = model(volumes)
                        loss = criterion(pred, ages)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % log_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % log_interval == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f}")
        
        # Use best head (skip in overfit mode, just use final)
        if best_head_state is not None and not self.config.overfit_mode:
            head.load_state_dict(best_head_state)
        
        return {"head": head, "model": model, "train_losses": train_losses, "val_losses": val_losses}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = RegressionConfig()
    trainer = RegressionTrainer(config)
    results = trainer.run_experiment()
