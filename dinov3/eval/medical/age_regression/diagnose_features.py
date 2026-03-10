"""
Diagnose why we can't overfit beyond 4-5 samples.
"""
#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from age_regression import RegressionConfig, OASISAgeRegressionDataset, RegressionTrainer
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_feature_space():
    """Analyze the feature space to understand overfitting limitations."""
    logger.info("="*80)
    logger.info("DIAGNOSING FEATURE SPACE LIMITATIONS")
    logger.info("="*80)
    
    config = RegressionConfig()
    config.use_multilayer_features = False  # Use single layer for simplicity
    trainer = RegressionTrainer(config)
    
    # Load dataset
    dataset = OASISAgeRegressionDataset(
        csv_path=config.oasis_csv_path,
        volumes_root=config.volumes_root_dir,
        config_yaml_path=config.config_path,
        resize_shape_3d=config.resize_shape_3d,
    )
    
    # Extract features for first 10 samples
    n_samples = 10
    from torch.utils.data import Subset
    subset = Subset(dataset, range(n_samples))
    loader = DataLoader(subset, batch_size=2, num_workers=0)
    
    logger.info(f"\nExtracting features from {n_samples} samples...")
    features, ages = trainer.extract_all_features(loader)
    
    logger.info(f"\nFeatures shape: {features.shape}")
    logger.info(f"Ages: {ages}")
    
    # Normalize features
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # 1. Check feature rank (effective dimensionality)
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE RANK ANALYSIS")
    logger.info('='*80)
    
    U, S, Vt = np.linalg.svd(features_norm, full_matrices=False)
    
    logger.info(f"  Singular values (top 20): {S[:20]}")
    logger.info(f"  Singular value ratios: {S[:10] / S[0]}")
    
    # Effective rank (number of singular values that capture 99% of variance)
    cumsum_var = np.cumsum(S**2) / np.sum(S**2)
    eff_rank_99 = np.searchsorted(cumsum_var, 0.99) + 1
    eff_rank_95 = np.searchsorted(cumsum_var, 0.95) + 1
    
    logger.info(f"\n  Effective rank (95% variance): {eff_rank_95}")
    logger.info(f"  Effective rank (99% variance): {eff_rank_99}")
    logger.info(f"  Feature matrix rank: {np.linalg.matrix_rank(features_norm)}")
    
    if eff_rank_99 < n_samples:
        logger.warning(f"\n  ⚠️  PROBLEM: Effective rank ({eff_rank_99}) < num samples ({n_samples})")
        logger.warning("      Features lie in a low-dimensional subspace!")
    
    # 2. Check pairwise feature distances
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE SPACE DISTANCES")
    logger.info('='*80)
    
    from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
    
    dist_matrix = euclidean_distances(features_norm)
    cos_sim_matrix = cosine_similarity(features_norm)
    
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_samples, k=1)
    pairwise_dists = dist_matrix[triu_idx]
    pairwise_sims = cos_sim_matrix[triu_idx]
    
    logger.info(f"  Pairwise Euclidean distances:")
    logger.info(f"    Mean: {pairwise_dists.mean():.4f}")
    logger.info(f"    Std:  {pairwise_dists.std():.4f}")
    logger.info(f"    Min:  {pairwise_dists.min():.4f}")
    logger.info(f"    Max:  {pairwise_dists.max():.4f}")
    
    logger.info(f"\n  Pairwise cosine similarities:")
    logger.info(f"    Mean: {pairwise_sims.mean():.4f}")
    logger.info(f"    Std:  {pairwise_sims.std():.4f}")
    logger.info(f"    Min:  {pairwise_sims.min():.4f}")
    logger.info(f"    Max:  {pairwise_sims.max():.4f}")
    
    if pairwise_sims.mean() > 0.9:
        logger.warning(f"\n  ⚠️  PROBLEM: Very high cosine similarity ({pairwise_sims.mean():.3f})")
        logger.warning("      Samples are nearly identical in feature space!")
    
    # 3. Check feature-age relationship
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE-AGE RELATIONSHIP")
    logger.info('='*80)
    
    # Project features onto age direction
    age_normalized = (ages - ages.mean()) / ages.std()
    
    # Find the direction in feature space most correlated with age
    correlations = np.array([pearsonr(features_norm[:, i], ages)[0] for i in range(features_norm.shape[1])])
    best_feature_idx = np.argmax(np.abs(correlations))
    best_corr = correlations[best_feature_idx]
    
    logger.info(f"  Best single feature correlation with age: {best_corr:.4f}")
    logger.info(f"  Feature index: {best_feature_idx}")
    
    # 4. Try to fit with increasing sample sizes
    logger.info(f"\n{'='*80}")
    logger.info("OVERFITTING CAPACITY TEST")
    logger.info('='*80)
    
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    
    for n in range(2, min(11, len(features_norm) + 1)):
        X_train = features_norm[:n]
        y_train = ages[:n]
        
        # Try to fit with zero regularization (lstsq for numerical stability)
        from numpy.linalg import lstsq
        X_with_bias = np.column_stack([X_train, np.ones(n)])
        coef, residuals, rank, s = lstsq(X_with_bias, y_train, rcond=None)
        
        pred = X_with_bias @ coef
        mae = mean_absolute_error(y_train, pred)
        
        logger.info(f"  {n} samples: MAE={mae:.2f}, matrix rank={rank}, condition={s[0]/s[-1]:.1e}")
        
        if mae > 5.0:
            logger.warning(f"    ⚠️  Cannot fit {n} samples perfectly (MAE > 5)!")
    
    # 5. Check if samples are linearly separable
    logger.info(f"\n{'='*80}")
    logger.info("LINEAR SEPARABILITY TEST")
    logger.info('='*80)
    
    # For each pair of samples with different ages, check if they're separable
    age_pairs_similar = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            age_diff = abs(ages[i] - ages[j])
            feature_dist = np.linalg.norm(features_norm[i] - features_norm[j])
            cos_sim = np.dot(features_norm[i], features_norm[j]) / (
                np.linalg.norm(features_norm[i]) * np.linalg.norm(features_norm[j])
            )
            
            if age_diff > 10 and cos_sim > 0.95:
                age_pairs_similar.append((i, j, ages[i], ages[j], age_diff, feature_dist, cos_sim))
    
    if age_pairs_similar:
        logger.warning(f"\n  ⚠️  Found {len(age_pairs_similar)} sample pairs with:")
        logger.warning("      - Large age difference (>10 years)")
        logger.warning("      - Very similar features (cosine sim > 0.95)")
        logger.warning("\n  These pairs are NOT linearly separable:")
        for i, j, age_i, age_j, age_diff, dist, sim in age_pairs_similar[:5]:
            logger.warning(f"    Samples {i},{j}: ages={age_i:.0f},{age_j:.0f} (Δ={age_diff:.0f}), "
                         f"dist={dist:.4f}, cosim={sim:.4f}")
    
    # 6. Visualize feature projection onto best 2D subspace
    logger.info(f"\n{'='*80}")
    logger.info("PCA PROJECTION")
    logger.info('='*80)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_norm)
    
    logger.info(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")
    logger.info(f"  Variance explained by PC2: {pca.explained_variance_ratio_[1]:.4f}")
    logger.info(f"  Total variance (2 PCs): {pca.explained_variance_ratio_[:2].sum():.4f}")
    
    logger.info(f"\n  Sample positions in 2D PCA space:")
    for i in range(n_samples):
        logger.info(f"    Sample {i} (age={ages[i]:.0f}): PC1={features_2d[i,0]:7.3f}, PC2={features_2d[i,1]:7.3f}")
    
    return features_norm, ages

if __name__ == "__main__":
    features, ages = analyze_feature_space()
