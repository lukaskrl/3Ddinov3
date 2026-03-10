from __future__ import annotations
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
	Compose,
	CropForegroundd,
	EnsureChannelFirstd,
	EnsureTyped,
	Lambdad,
	LoadImaged,
	Orientationd,
	RandFlipd,
	RandRotate90d,
	RandScaleIntensityd,
	RandShiftIntensityd,
	RandSpatialCropd,
	ScaleIntensityRangePercentilesd,
	SpatialPadd,
	Spacingd,
)

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SegmentationConfig:
	checkpoint_path: str = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/checkpoint_final.pth"
	config_path: str = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/config.yaml"
	bratsdataset_path: str = "/home/lukas/data/brats"
	output_dir: str = "/home/lukas/3Ddinov3/dinov3/eval/medical/segmentation/results"

	modality_suffix: str = "-t1n.nii"
	label_suffix: str = "-seg.nii"
	binary_segmentation: bool = True

	roi_size: Tuple[int, int, int] = (96, 96, 96)
	pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)

	train_ratio: float = 0.7
	val_ratio: float = 0.15
	test_ratio: float = 0.15

	batch_size: int = 2
	num_workers: int = 4
	epochs: int = 30
	lr: float = 1e-4
	backbone_lr_scale: float = 0.1
	weight_decay: float = 1e-5

	freeze_backbone: bool = True
	head_hidden_dim: int = 256

	mixed_precision: bool = True
	device: str = "cuda:1"
	seed: int = 42
	val_every: int = 1


class DinoSegmentationModel(nn.Module):
	def __init__(self, backbone: nn.Module, freeze_backbone: bool, head_hidden_dim: int):
		super().__init__()
		self.backbone = backbone
		self.freeze_backbone = freeze_backbone

		if hasattr(backbone, "embed_dim"):
			embed_dim = int(backbone.embed_dim)
		else:
			raise ValueError("Backbone does not expose embed_dim; cannot build segmentation head.")

		self.head = nn.Sequential(
			nn.Conv3d(embed_dim, head_hidden_dim, kernel_size=3, padding=1),
			nn.InstanceNorm3d(head_hidden_dim),
			nn.GELU(),
			nn.Conv3d(head_hidden_dim, 1, kernel_size=1),
		)

		self._set_backbone_trainable(not freeze_backbone)

	def _set_backbone_trainable(self, trainable: bool) -> None:
		for parameter in self.backbone.parameters():
			parameter.requires_grad = trainable

	def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
		if self.freeze_backbone:
			with torch.no_grad():
				features = self.backbone.forward_features(x)
		else:
			features = self.backbone.forward_features(x)

		patch_tokens = features["x_norm_patchtokens"]
		batch_size, n_patches, channels = patch_tokens.shape

		patch_size = int(getattr(self.backbone, "patch_size", 16))
		patch_size_d = int(getattr(self.backbone, "patch_size_d", patch_size))

		depth = x.shape[2] // patch_size_d
		height = x.shape[3] // patch_size
		width = x.shape[4] // patch_size

		if depth * height * width != n_patches:
			raise RuntimeError(
				"Patch-grid mismatch: got "
				f"{n_patches} tokens but expected {depth}*{height}*{width}. "
				"Ensure ROI size is divisible by patch sizes."
			)

		feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, channels, depth, height, width).contiguous()
		return feature_map

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feature_map = self._backbone_features(x)
		logits = self.head(feature_map)
		logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
		return logits


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _find_cases(dataset_root: Path, modality_suffix: str, label_suffix: str) -> List[Dict[str, str]]:
	label_files = sorted(dataset_root.rglob(f"*{label_suffix}"))
	cases: List[Dict[str, str]] = []
	for label_path in label_files:
		case_stem = str(label_path).replace(label_suffix, "")
		image_path = Path(case_stem + modality_suffix)
		if image_path.is_file():
			cases.append({"image": str(image_path), "label": str(label_path)})
	return cases


def split_cases(cases: Sequence[Dict[str, str]], cfg: SegmentationConfig):
	if len(cases) < 3:
		raise ValueError("Need at least 3 cases to create train/val/test splits.")

	total = len(cases)
	n_train = max(1, int(total * cfg.train_ratio))
	n_val = max(1, int(total * cfg.val_ratio))
	n_test = total - n_train - n_val
	if n_test <= 0:
		n_test = 1
		if n_train > n_val:
			n_train -= 1
		else:
			n_val -= 1

	shuffled = list(cases)
	random.shuffle(shuffled)

	train_cases = shuffled[:n_train]
	val_cases = shuffled[n_train : n_train + n_val]
	test_cases = shuffled[n_train + n_val :]
	return train_cases, val_cases, test_cases


def build_transforms(cfg: SegmentationConfig, is_train: bool):
	transforms: List = [
		LoadImaged(keys=["image", "label"]),
		EnsureChannelFirstd(keys=["image", "label"]),
		Orientationd(keys=["image", "label"], axcodes="RAS"),
		Spacingd(keys=["image", "label"], pixdim=cfg.pixdim, mode=("bilinear", "nearest")),
		ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
		CropForegroundd(keys=["image", "label"], source_key="image"),
		SpatialPadd(keys=["image", "label"], spatial_size=cfg.roi_size),
	]

	if is_train:
		transforms.extend(
			[
				RandSpatialCropd(keys=["image", "label"], roi_size=cfg.roi_size, random_size=False),
				RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
				RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
				RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
				RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
				RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
				RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
			]
		)
	else:
		transforms.append(RandSpatialCropd(keys=["image", "label"], roi_size=cfg.roi_size, random_center=False, random_size=False))

	if cfg.binary_segmentation:
		transforms.append(Lambdad(keys="label", func=lambda x: (x > 0).astype(np.float32)))

	transforms.append(EnsureTyped(keys=["image", "label"]))
	return Compose(transforms)


def create_dataloaders(cfg: SegmentationConfig):
	root = Path(cfg.bratsdataset_path)
	if not root.exists():
		raise FileNotFoundError(f"Dataset path not found: {cfg.bratsdataset_path}")

	cases = _find_cases(root, cfg.modality_suffix, cfg.label_suffix)
	if not cases:
		raise RuntimeError(
			f"No cases found under {cfg.bratsdataset_path} with modality suffix {cfg.modality_suffix} and label suffix {cfg.label_suffix}."
		)

	train_cases, val_cases, test_cases = split_cases(cases, cfg)
	logger.info(
		"BraTS split: total=%d train=%d val=%d test=%d",
		len(cases),
		len(train_cases),
		len(val_cases),
		len(test_cases),
	)

	train_ds = Dataset(data=train_cases, transform=build_transforms(cfg, is_train=True))
	val_ds = Dataset(data=val_cases, transform=build_transforms(cfg, is_train=False))
	test_ds = Dataset(data=test_cases, transform=build_transforms(cfg, is_train=False))

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


def load_dinov3_backbone(cfg: SegmentationConfig, device: torch.device):
	if not distributed.is_enabled():
		distributed.enable()

	setup_args = DinoV3SetupArgs(
		config_file=cfg.config_path,
		pretrained_weights=cfg.checkpoint_path,
		shard_unsharded_model=False,
		output_dir="",
		opts=[],
	)
	dino_cfg = setup_config(setup_args, strict_cfg=False)
	backbone = build_model_for_eval(dino_cfg, cfg.checkpoint_path)
	backbone = backbone.to(device)
	backbone.eval()
	autocast_dtype = get_autocast_dtype(dino_cfg)
	return backbone, autocast_dtype


def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
	autocast_enabled: bool,
	autocast_dtype: torch.dtype,
) -> Dict[str, float]:
	model.eval()
	dice_metric = DiceMetric(include_background=True, reduction="mean")
	hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")

	total_loss = 0.0
	n_batches = 0
	with torch.no_grad():
		for batch in loader:
			images = batch["image"].to(device)
			labels = batch["label"].to(device).float()

			with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype):
				logits = model(images)
				loss = criterion(logits, labels)

			total_loss += float(loss.item())
			n_batches += 1

			probs = torch.sigmoid(logits)
			preds = (probs > 0.5).float()
			dice_metric(y_pred=preds, y=labels)
			hd95_metric(y_pred=preds, y=labels)

	dice = float(torch.nan_to_num(dice_metric.aggregate(), nan=0.0).item())
	hd95 = float(torch.nan_to_num(hd95_metric.aggregate(), nan=0.0, posinf=0.0).item())
	dice_metric.reset()
	hd95_metric.reset()

	return {
		"loss": total_loss / max(n_batches, 1),
		"dice": dice,
		"hd95": hd95,
	}


def build_optimizer(model: DinoSegmentationModel, cfg: SegmentationConfig) -> torch.optim.Optimizer:
	if cfg.freeze_backbone:
		return torch.optim.AdamW(model.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

	param_groups = [
		{"params": model.backbone.parameters(), "lr": cfg.lr * cfg.backbone_lr_scale},
		{"params": model.head.parameters(), "lr": cfg.lr},
	]
	return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train(cfg: SegmentationConfig):
	os.makedirs(cfg.output_dir, exist_ok=True)
	set_seed(cfg.seed)

	# Initialize TensorBoard SummaryWriter
	writer = SummaryWriter(log_dir=cfg.output_dir)

	device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	train_loader, val_loader, test_loader = create_dataloaders(cfg)
	backbone, autocast_dtype = load_dinov3_backbone(cfg, device)
	model = DinoSegmentationModel(
		backbone=backbone,
		freeze_backbone=cfg.freeze_backbone,
		head_hidden_dim=cfg.head_hidden_dim,
	).to(device)

	optimizer = build_optimizer(model, cfg)
	scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")
	criterion = DiceCELoss(sigmoid=True)

	best_val_dice = -1.0
	best_model_path = Path(cfg.output_dir) / "best_model.pt"


	for epoch in range(1, cfg.epochs + 1):
		model.train()
		running_loss = 0.0

		for batch in train_loader:
			images = batch["image"].to(device)
			labels = batch["label"].to(device).float()

			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(
				device_type=device.type,
				enabled=cfg.mixed_precision and device.type == "cuda",
				dtype=autocast_dtype,
			):
				logits = model(images)
				loss = criterion(logits, labels)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			running_loss += float(loss.item())

		train_loss = running_loss / max(len(train_loader), 1)

		if epoch % cfg.val_every == 0:
			val_metrics = evaluate(
				model=model,
				loader=val_loader,
				device=device,
				criterion=criterion,
				autocast_enabled=cfg.mixed_precision and device.type == "cuda",
				autocast_dtype=autocast_dtype,
			)

			logger.info(
				"Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | val_hd95=%.4f",
				epoch,
				cfg.epochs,
				train_loss,
				val_metrics["loss"],
				val_metrics["dice"],
				val_metrics["hd95"],
			)

			# TensorBoard logging
			writer.add_scalar("Loss/train", train_loss, epoch)
			writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
			writer.add_scalar("Dice/val", val_metrics["dice"], epoch)
			writer.add_scalar("HD95/val", val_metrics["hd95"], epoch)

			if val_metrics["dice"] > best_val_dice:
				best_val_dice = val_metrics["dice"]
				torch.save(
					{
						"epoch": epoch,
						"model_state": model.state_dict(),
						"optimizer_state": optimizer.state_dict(),
						"best_val_dice": best_val_dice,
						"config": asdict(cfg),
					},
					best_model_path,
				)
				logger.info("Saved new best model to %s", best_model_path)
		else:
			logger.info("Epoch %d/%d | train_loss=%.4f", epoch, cfg.epochs, train_loss)
			# Still log train loss for every epoch
			writer.add_scalar("Loss/train", train_loss, epoch)

	if best_model_path.exists():
		checkpoint = torch.load(best_model_path, map_location=device)
		model.load_state_dict(checkpoint["model_state"], strict=True)

	test_metrics = evaluate(
		model=model,
		loader=test_loader,
		device=device,
		criterion=criterion,
		autocast_enabled=cfg.mixed_precision and device.type == "cuda",
		autocast_dtype=autocast_dtype,
	)

	logger.info("Test metrics | loss=%.4f | dice=%.4f | hd95=%.4f", test_metrics["loss"], test_metrics["dice"], test_metrics["hd95"])

	# TensorBoard logging for test metrics
	writer.add_scalar("Loss/test", test_metrics["loss"], cfg.epochs)
	writer.add_scalar("Dice/test", test_metrics["dice"], cfg.epochs)
	writer.add_scalar("HD95/test", test_metrics["hd95"], cfg.epochs)

	results = {
		"best_val_dice": best_val_dice,
		"test": test_metrics,
		"freeze_backbone": cfg.freeze_backbone,
	}
	results_path = Path(cfg.output_dir) / "results.json"
	with results_path.open("w", encoding="utf-8") as handle:
		json.dump(results, handle, indent=2)
	logger.info("Saved results to %s", results_path)

	writer.close()


def parse_args() -> SegmentationConfig:
	parser = argparse.ArgumentParser(description="DINOv3 medical segmentation training with simple 3D head")
	parser.add_argument("--checkpoint-path", type=str, default=SegmentationConfig.checkpoint_path)
	parser.add_argument("--config-path", type=str, default=SegmentationConfig.config_path)
	parser.add_argument("--bratsdataset-path", type=str, default=SegmentationConfig.bratsdataset_path)
	parser.add_argument("--output-dir", type=str, default=SegmentationConfig.output_dir)
	parser.add_argument("--epochs", type=int, default=SegmentationConfig.epochs)
	parser.add_argument("--batch-size", type=int, default=SegmentationConfig.batch_size)
	parser.add_argument("--num-workers", type=int, default=SegmentationConfig.num_workers)
	parser.add_argument("--lr", type=float, default=SegmentationConfig.lr)
	parser.add_argument("--weight-decay", type=float, default=SegmentationConfig.weight_decay)
	parser.add_argument("--backbone-lr-scale", type=float, default=SegmentationConfig.backbone_lr_scale)
	parser.add_argument("--seed", type=int, default=SegmentationConfig.seed)
	parser.add_argument("--device", type=str, default=SegmentationConfig.device)
	parser.add_argument("--modality-suffix", type=str, default=SegmentationConfig.modality_suffix)
	parser.add_argument("--label-suffix", type=str, default=SegmentationConfig.label_suffix)
	parser.add_argument("--head-hidden-dim", type=int, default=SegmentationConfig.head_hidden_dim)
	parser.add_argument("--val-every", type=int, default=SegmentationConfig.val_every)
	parser.add_argument("--roi-size", type=int, nargs=3, default=list(SegmentationConfig.roi_size))
	parser.add_argument("--pixdim", type=float, nargs=3, default=list(SegmentationConfig.pixdim))
	parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DINOv3 backbone and train only segmentation head")
	parser.add_argument("--finetune-backbone", action="store_true", help="Finetune backbone jointly with segmentation head")
	parser.add_argument("--no-mixed-precision", action="store_true")

	args = parser.parse_args()

	if args.freeze_backbone and args.finetune_backbone:
		raise ValueError("Use either --freeze-backbone or --finetune-backbone, not both.")

	freeze_backbone = True
	if args.finetune_backbone:
		freeze_backbone = False
	elif args.freeze_backbone:
		freeze_backbone = True

	return SegmentationConfig(
		checkpoint_path=args.checkpoint_path,
		config_path=args.config_path,
		bratsdataset_path=args.bratsdataset_path,
		output_dir=args.output_dir,
		modality_suffix=args.modality_suffix,
		label_suffix=args.label_suffix,
		roi_size=tuple(args.roi_size),
		pixdim=tuple(args.pixdim),
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		epochs=args.epochs,
		lr=args.lr,
		backbone_lr_scale=args.backbone_lr_scale,
		weight_decay=args.weight_decay,
		freeze_backbone=freeze_backbone,
		head_hidden_dim=args.head_hidden_dim,
		mixed_precision=not args.no_mixed_precision,
		device=args.device,
		seed=args.seed,
		val_every=args.val_every,
	)


if __name__ == "__main__":
	config = parse_args()
	train(config)