"""
Quick statistical check of features extracted from DINOv3 backbone.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from age_regression import RegressionConfig, OASISAgeRegressionDataset, RegressionTrainer
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_features():
    """Extract features and run statistical analysis."""
    logger.info("="*60)
    logger.info("Feature Statistical Analysis")
    logger.info("="*60)
    
    config = RegressionConfig()
    trainer = RegressionTrainer(config)
    
    # Load dataset (use first 20 samples for quick analysis)
    dataset = OASISAgeRegressionDataset(
        csv_path=config.oasis_csv_path,
        volumes_root=config.volumes_root_dir,
        config_yaml_path=config.config_path,
        resize_shape_3d=config.resize_shape_3d,
    )
    
    n_samples = min(20, len(dataset))
    from torch.utils.data import Subset
    subset = Subset(dataset, range(n_samples))
    loader = DataLoader(subset, batch_size=4, num_workers=0)
    
    logger.info(f"\nExtracting features from {n_samples} samples...")
    features, ages = trainer.extract_all_features(loader)
    
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE STATISTICS")
    logger.info('='*60)
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Ages shape: {ages.shape}")
    logger.info(f"Ages range: [{ages.min():.1f}, {ages.max():.1f}]")
    
    # Overall statistics
    logger.info(f"\n{'='*60}")
    logger.info("Overall Feature Statistics:")
    logger.info('='*60)
    logger.info(f"  Mean: {features.mean():.6f}")
    logger.info(f"  Std:  {features.std():.6f}")
    logger.info(f"  Min:  {features.min():.6f}")
    logger.info(f"  Max:  {features.max():.6f}")
    
    # Per-feature statistics
    feature_means = features.mean(axis=0)  # (1080,)
    feature_stds = features.std(axis=0)    # (1080,)
    feature_mins = features.min(axis=0)    # (1080,)
    feature_maxs = features.max(axis=0)    # (1080,)
    
    logger.info(f"\n{'='*60}")
    logger.info("Per-Feature Statistics:")
    logger.info('='*60)
    logger.info(f"  Mean across features: {feature_means.mean():.6f} ± {feature_means.std():.6f}")
    logger.info(f"  Std across features:  {feature_stds.mean():.6f} ± {feature_stds.std():.6f}")
    
    # Check for dead features (low variance)
    dead_features = np.sum(feature_stds < 1e-6)
    low_var_features = np.sum(feature_stds < 0.01)
    logger.info(f"\n  Dead features (std < 1e-6): {dead_features} / {features.shape[1]}")
    logger.info(f"  Low variance features (std < 0.01): {low_var_features} / {features.shape[1]}")
    
    # Feature correlation with age
    logger.info(f"\n{'='*60}")
    logger.info("Feature-Age Correlation:")
    logger.info('='*60)
    
    correlations = []
    for i in range(features.shape[1]):
        corr, _ = pearsonr(features[:, i], ages)
        correlations.append(corr)
    
    correlations = np.array(correlations)
    abs_correlations = np.abs(correlations)
    
    logger.info(f"  Mean |correlation|: {abs_correlations.mean():.4f}")
    logger.info(f"  Max |correlation|:  {abs_correlations.max():.4f}")
    logger.info(f"  Features with |corr| > 0.3: {np.sum(abs_correlations > 0.3)}")
    logger.info(f"  Features with |corr| > 0.5: {np.sum(abs_correlations > 0.5)}")
    
    # Top correlated features
    top_k = 10
    top_indices = np.argsort(abs_correlations)[-top_k:][::-1]
    logger.info(f"\n  Top {top_k} age-correlated features:")
    for idx in top_indices:
        logger.info(f"    Feature {idx}: r={correlations[idx]:.4f}")
    
    # Sample-wise feature statistics
    logger.info(f"\n{'='*60}")
    logger.info("Per-Sample Statistics:")
    logger.info('='*60)
    sample_norms = np.linalg.norm(features, axis=1)
    logger.info(f"  L2 norm per sample: {sample_norms.mean():.2f} ± {sample_norms.std():.2f}")
    logger.info(f"  L2 norm range: [{sample_norms.min():.2f}, {sample_norms.max():.2f}]")
    
    # Check if features are normalized
    sample_means = features.mean(axis=1)
    sample_stds = features.std(axis=1)
    logger.info(f"  Mean per sample: {sample_means.mean():.6f} ± {sample_means.std():.6f}")
    logger.info(f"  Std per sample:  {sample_stds.mean():.6f} ± {sample_stds.std():.6f}")
    
    # Feature value distribution
    logger.info(f"\n{'='*60}")
    logger.info("Feature Distribution:")
    logger.info('='*60)
    logger.info(f"  % of features > 0: {(features > 0).mean() * 100:.1f}%")
    logger.info(f"  % of features < 0: {(features < 0).mean() * 100:.1f}%")
    logger.info(f"  % of features == 0: {(features == 0).mean() * 100:.1f}%")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = [np.percentile(features, p) for p in percentiles]
    logger.info(f"\n  Feature percentiles:")
    for p, v in zip(percentiles, values):
        logger.info(f"    {p}%: {v:.6f}")
    
    # Check feature diversity (are all samples similar?)
    logger.info(f"\n{'='*60}")
    logger.info("Sample Similarity:")
    logger.info('='*60)
    
    # Compute pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(features)
    
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(similarities, k=1)
    pairwise_sims = similarities[triu_indices]
    
    logger.info(f"  Mean pairwise cosine similarity: {pairwise_sims.mean():.4f}")
    logger.info(f"  Std pairwise cosine similarity:  {pairwise_sims.std():.4f}")
    logger.info(f"  Min pairwise cosine similarity:  {pairwise_sims.min():.4f}")
    logger.info(f"  Max pairwise cosine similarity:  {pairwise_sims.max():.4f}")
    
    if pairwise_sims.mean() > 0.95:
        logger.warning("\n⚠️  WARNING: Very high similarity between samples!")
        logger.warning("   Features may not be distinctive enough.")
    
    # Print a few raw feature values
    logger.info(f"\n{'='*60}")
    logger.info("Sample Feature Values (first 3 samples, first 10 features):")
    logger.info('='*60)
    for i in range(min(3, n_samples)):
        logger.info(f"  Sample {i} (age={ages[i]:.0f}): {features[i, :10]}")

if __name__ == "__main__":
    analyze_features()
