#%%
import os
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    hdbscan = None

import dinov3.distributed as distributed
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.data.datasets.ct_volume import CTVolumeDataset
from dinov3.data.transforms_3d import make_ct_3d_base_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval

#Paths and basic config
# Path to a single distributed checkpoint directory (integer‑named subdir)
path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/vit_large_patch8/checkpoint_40399.pth"
# path_to_checkpoint = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/checkpoint_final.pth"

# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Directory with CT volumes stored as .npy files
path_to_data = "/home/lukas/data/brain-t1-dataset"

# Training config used for this checkpoint (3D CT DINOv3 config)
# path_to_config = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/config.yaml"
path_to_config = "/home/lukas/3Ddinov3/work_dir/vit_large_patch8/config.yaml"

# Select which GPU to use (0-based index). Must be set before any CUDA ops.
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# K-means analysis settings
run_kmeans = True
kmeans_k = 8
kmeans_sample_size = 20000
kmeans_pca_points = 5000
kmeans_random_state = 0

# Other unlabeled analysis settings
run_dimred = True
run_gmm = True
gmm_k = 8
run_spectral = True
spectral_k = 8
run_dbscan = True
dbscan_eps = 0.5
dbscan_min_samples = 10
run_hdbscan = True
hdbscan_min_cluster_size = 30
run_anisotropy = True
run_knn_consistency = True
knn_k = 10
knn_sample_size = 5000
run_spatial_coherence = True
run_aug_similarity = True
aug_noise_std = 0.02
aug_dropout_p = 0.1
dimred_sample_size = 5000
tsne_points = 2000
umap_points = 5000
num_layers = 4
num_volumes = 1
report_path = "/home/lukas/3Ddinov3/work_dir/mri_full_training_centering/feature_space_report_final.txt"


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


def pick_volume_paths(data_root: str, num_volumes: int) -> list[str]:
    root_path = Path(data_root)
    volume_files = sorted(
        p
        for p in root_path.iterdir()
        if p.is_file() and (p.name.endswith(".npy") or p.name.endswith(".nii.gz"))
    )
    if not volume_files:
        raise RuntimeError(f"No .npy or .nii.gz volumes found under {data_root}")
    return [str(p) for p in volume_files[: max(1, num_volumes)]]


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


def _sample_and_normalize(
    features: torch.Tensor,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    c, d, h, w = features.shape
    n = d * h * w
    feats = features.reshape(c, n).T  # (N, C)
    rng = np.random.default_rng(random_state)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        feats = feats[idx]
    feats_np = feats.numpy().astype(np.float32)
    feats_np = feats_np / (np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12)
    return feats_np


def _cluster_metrics(feats_np: np.ndarray, labels: np.ndarray, name: str):
    unique = np.unique(labels)
    if unique.size < 2:
        print(f"{name} metrics: skipped (need at least 2 clusters)")
        return
    if -1 in unique:
        mask = labels != -1
        feats_np = feats_np[mask]
        labels = labels[mask]
        unique = np.unique(labels)
        if unique.size < 2:
            print(f"{name} metrics: skipped (need at least 2 clusters)")
            return

    sil = silhouette_score(feats_np, labels, metric="euclidean")
    ch = calinski_harabasz_score(feats_np, labels)
    db = davies_bouldin_score(feats_np, labels)
    print(f"{name} metrics: silhouette={sil:.6f}, CH={ch:.6f}, DB={db:.6f}")


def analyze_dimensionality_reduction(
    features: torch.Tensor,
    sample_size: int,
    tsne_n: int,
    umap_n: int,
    random_state: int,
):
    feats_np = _sample_and_normalize(features, sample_size, random_state)
    print("\n=== Dimensionality reduction ===")
    pca = PCA(n_components=min(64, feats_np.shape[1]), random_state=random_state)
    pca.fit(feats_np)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"PCA: top-10 variance={cum[min(9, len(cum)-1)]:.4f}, top-20 variance={cum[min(19, len(cum)-1)]:.4f}")
    print(f"PCA: components for 90% variance={int(np.searchsorted(cum, 0.9) + 1)}")

    if tsne_n > 0:
        n = min(tsne_n, feats_np.shape[0])
        tsne = TSNE(n_components=2, init="pca", random_state=random_state)
        emb = tsne.fit_transform(feats_np[:n])
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.6)
        plt.title("t-SNE (no labels)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    if umap_n > 0 and umap is not None:
        n = min(umap_n, feats_np.shape[0])
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        emb = reducer.fit_transform(feats_np[:n])
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.6)
        plt.title("UMAP (no labels)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    elif umap_n > 0 and umap is None:
        print("UMAP: skipped (package not installed)")


def analyze_anisotropy(features: torch.Tensor, random_state: int):
    feats_np = _sample_and_normalize(features, 20000, random_state)
    print("\n=== Feature anisotropy ===")
    feats_np = feats_np - feats_np.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(feats_np, full_matrices=False)
    eig = s**2
    p = eig / (eig.sum() + 1e-12)
    eff_rank = np.exp(-(p * np.log(p + 1e-12)).sum())
    part_ratio = 1.0 / (p**2).sum()
    print(f"Effective rank: {eff_rank:.3f}")
    print(f"Participation ratio: {part_ratio:.3f}")


def analyze_knn_consistency(
    features: torch.Tensor,
    k: int,
    sample_size: int,
    random_state: int,
    noise_std: float,
    dropout_p: float,
):
    feats_np = _sample_and_normalize(features, sample_size, random_state)
    rng = np.random.default_rng(random_state)
    noise_a = rng.normal(scale=noise_std, size=feats_np.shape).astype(np.float32)
    noise_b = rng.normal(scale=noise_std, size=feats_np.shape).astype(np.float32)
    mask_a = rng.random(feats_np.shape) > dropout_p
    mask_b = rng.random(feats_np.shape) > dropout_p
    view_a = feats_np * mask_a + noise_a
    view_b = feats_np * mask_b + noise_b

    view_a = view_a / (np.linalg.norm(view_a, axis=1, keepdims=True) + 1e-12)
    view_b = view_b / (np.linalg.norm(view_b, axis=1, keepdims=True) + 1e-12)

    nn = NearestNeighbors(n_neighbors=min(k + 1, view_a.shape[0]))
    nn.fit(view_a)
    neigh_a = nn.kneighbors(return_distance=False)[:, 1:]
    nn.fit(view_b)
    neigh_b = nn.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for a, b in zip(neigh_a, neigh_b):
        overlaps.append(len(set(a).intersection(set(b))) / max(1, len(a)))
    print("\n=== kNN consistency (feature augmentations) ===")
    print(f"kNN overlap@{k}: mean={np.mean(overlaps):.4f}, std={np.std(overlaps):.4f}")


def analyze_spatial_coherence(features: torch.Tensor):
    print("\n=== Spatial coherence ===")
    feats = features.float()
    feats = feats / (torch.linalg.vector_norm(feats, dim=0, keepdims=True) + 1e-12)
    sims = []
    sims.append((feats[:, 1:, :, :] * feats[:, :-1, :, :]).sum(dim=0))
    sims.append((feats[:, :, 1:, :] * feats[:, :, :-1, :]).sum(dim=0))
    sims.append((feats[:, :, :, 1:] * feats[:, :, :, :-1]).sum(dim=0))
    sims_all = torch.cat([s.reshape(-1) for s in sims])
    sims_np = sims_all.cpu().numpy()
    print(
        "Neighbor cosine: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            sims_np.mean(), sims_np.std(), sims_np.min(), sims_np.max()
        )
    )


def analyze_aug_similarity(features: torch.Tensor, random_state: int, noise_std: float):
    feats_np = _sample_and_normalize(features, 10000, random_state)
    rng = np.random.default_rng(random_state)
    noise_a = rng.normal(scale=noise_std, size=feats_np.shape).astype(np.float32)
    noise_b = rng.normal(scale=noise_std, size=feats_np.shape).astype(np.float32)
    view_a = feats_np + noise_a
    view_b = feats_np + noise_b
    view_a = view_a / (np.linalg.norm(view_a, axis=1, keepdims=True) + 1e-12)
    view_b = view_b / (np.linalg.norm(view_b, axis=1, keepdims=True) + 1e-12)
    pos = (view_a * view_b).sum(axis=1)
    neg = (view_a * np.roll(view_b, shift=1, axis=0)).sum(axis=1)
    print("\n=== Augmentation similarity proxy ===")
    print(f"Positive cosine: mean={pos.mean():.6f}, std={pos.std():.6f}")
    print(f"Negative cosine: mean={neg.mean():.6f}, std={neg.std():.6f}")


def analyze_feature_clustering(
    features: torch.Tensor,
    k: int = 8,
    sample_size: int = 20000,
    pca_points: int = 5000,
    random_state: int = 0,
):
    """
    K-means clustering on token features with optional PCA visualization.
    features: (C, D, H, W) on CPU (float)
    """
    c, d, h, w = features.shape
    n = d * h * w
    feats = features.reshape(c, n).T  # (N, C)

    rng = np.random.default_rng(random_state)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        feats = feats[idx]

    feats_np = feats.numpy().astype(np.float32)
    feats_np = feats_np / (np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12)

    print("\n=== K-means clustering ===")
    print(f"Tokens used: {feats_np.shape[0]}, Channels: {feats_np.shape[1]}, K: {k}")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(feats_np)

    inertia = kmeans.inertia_
    sizes = np.bincount(labels, minlength=k)
    print(f"Inertia: {inertia:.6f}")
    print("Cluster sizes:", sizes.tolist())

    if k > 1 and feats_np.shape[0] >= k:
        sil_sample = feats_np
        if feats_np.shape[0] > 5000:
            idx = rng.choice(feats_np.shape[0], size=5000, replace=False)
            sil_sample = feats_np[idx]
            sil_labels = labels[idx]
        else:
            sil_labels = labels
        sil = silhouette_score(sil_sample, sil_labels, metric="euclidean")
        print(f"Silhouette score (sampled): {sil:.6f}")

    if pca_points > 0:
        plot_n = min(pca_points, feats_np.shape[0])
        plot_idx = rng.choice(feats_np.shape[0], size=plot_n, replace=False)
        pca = PCA(n_components=2, random_state=random_state)
        emb2d = pca.fit_transform(feats_np[plot_idx])
        plot_labels = labels[plot_idx]

        plt.figure(figsize=(6, 5))
        plt.scatter(emb2d[:, 0], emb2d[:, 1], s=4, c=plot_labels, cmap="tab10", alpha=0.7)
        plt.title("K-means clusters (PCA projection)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


def analyze_clustering_suite(
    features: torch.Tensor,
    random_state: int,
):
    feats_np = _sample_and_normalize(features, 20000, random_state)

    if run_kmeans:
        kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(feats_np)
        print("\n=== K-means clustering ===")
        print(f"Tokens used: {feats_np.shape[0]}, Channels: {feats_np.shape[1]}, K: {kmeans_k}")
        print(f"Inertia: {kmeans.inertia_:.6f}")
        print("Cluster sizes:", np.bincount(labels, minlength=kmeans_k).tolist())
        _cluster_metrics(feats_np, labels, "K-means")

    if run_gmm:
        gmm = GaussianMixture(n_components=gmm_k, covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(feats_np)
        print("\n=== GMM clustering ===")
        print(f"Tokens used: {feats_np.shape[0]}, Channels: {feats_np.shape[1]}, K: {gmm_k}")
        print(f"BIC: {gmm.bic(feats_np):.6f}, AIC: {gmm.aic(feats_np):.6f}")
        _cluster_metrics(feats_np, labels, "GMM")

    if run_spectral:
        spectral = SpectralClustering(
            n_clusters=spectral_k,
            assign_labels="kmeans",
            random_state=random_state,
            affinity="nearest_neighbors",
        )
        labels = spectral.fit_predict(feats_np)
        print("\n=== Spectral clustering ===")
        print(f"Tokens used: {feats_np.shape[0]}, Channels: {feats_np.shape[1]}, K: {spectral_k}")
        _cluster_metrics(feats_np, labels, "Spectral")

    if run_dbscan:
        dbs = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbs.fit_predict(feats_np)
        print("\n=== DBSCAN clustering ===")
        num_noise = int((labels == -1).sum())
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Clusters: {num_clusters}, Noise points: {num_noise}")
        _cluster_metrics(feats_np, labels, "DBSCAN")

    if run_hdbscan:
        if hdbscan is None:
            print("\n=== HDBSCAN clustering ===")
            print("HDBSCAN: skipped (package not installed)")
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
            labels = clusterer.fit_predict(feats_np)
            print("\n=== HDBSCAN clustering ===")
            num_noise = int((labels == -1).sum())
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Clusters: {num_clusters}, Noise points: {num_noise}")
            _cluster_metrics(feats_np, labels, "HDBSCAN")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis script.")

    torch.cuda.set_device(0)

    report_file = open(report_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(sys.stdout, report_file)
    try:
        print("Feature analysis report")
        print(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Config: {path_to_config}")
        print(f"Checkpoint: {path_to_checkpoint}")
        print(f"Data root: {path_to_data}")
        print(f"Volumes: {num_volumes}, Layers: {num_layers}")

        print(f"Loading model from config={path_to_config} and checkpoint={path_to_checkpoint}")
        model, autocast_dtype, config = load_backbone_from_checkpoint(
            config_file=path_to_config,
            ckpt_dir=path_to_checkpoint,
        )

        dataset = build_ct_dataset(path_to_data, config)
        volume_paths = pick_volume_paths(path_to_data, num_volumes)
        print(f"Using {len(volume_paths)} volume(s)")

        for vol_idx, volume_path in enumerate(volume_paths, start=1):
            print(f"\n################ Volume {vol_idx}/{len(volume_paths)} ################")
            print(f"Using volume: {volume_path}")
            idx = dataset._paths.index(volume_path)

            volume, _ = dataset[idx]  # (C, D, H, W)
            volume = volume.unsqueeze(0).cuda(non_blocking=True)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
                layers_feats_tuple = model.get_intermediate_layers(
                    volume,
                    n=num_layers,
                    reshape=True,
                    return_class_token=True,
                    return_extra_tokens=False,
                    norm=True,
                )

            for layer_idx, (vol_feats, _vol_cls_token) in enumerate(layers_feats_tuple, start=1):
                feat_cpu = vol_feats.detach().float().cpu()[0]  # (C, D, H, W)
                print(f"\n================ Layer {layer_idx}/{num_layers} ================")

                analyze_feature_space(feat_cpu)

                if run_dimred:
                    analyze_dimensionality_reduction(
                        feat_cpu,
                        sample_size=dimred_sample_size,
                        tsne_n=tsne_points,
                        umap_n=umap_points,
                        random_state=kmeans_random_state,
                    )

                if run_anisotropy:
                    analyze_anisotropy(feat_cpu, random_state=kmeans_random_state)

                if run_spatial_coherence:
                    analyze_spatial_coherence(feat_cpu)

                if run_aug_similarity:
                    analyze_aug_similarity(
                        feat_cpu,
                        random_state=kmeans_random_state,
                        noise_std=aug_noise_std,
                    )

                if run_knn_consistency:
                    analyze_knn_consistency(
                        feat_cpu,
                        k=knn_k,
                        sample_size=knn_sample_size,
                        random_state=kmeans_random_state,
                        noise_std=aug_noise_std,
                        dropout_p=aug_dropout_p,
                    )

                analyze_clustering_suite(feat_cpu, random_state=kmeans_random_state)

                if run_kmeans:
                    analyze_feature_clustering(
                        feat_cpu,
                        k=kmeans_k,
                        sample_size=kmeans_sample_size,
                        pca_points=kmeans_pca_points,
                        random_state=kmeans_random_state,
                    )
    finally:
        sys.stdout = original_stdout
        report_file.close()
        print(f"Report written to: {report_path}")
