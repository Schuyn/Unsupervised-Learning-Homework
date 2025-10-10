#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Chuyang Su <cs4570@columbia.edu>
Date: 2025-10-10
FilePath: /Unsupervised-Learning-Homework/Homework 1/Code/Problem_1_b.py
Description:
    GR5244 HW1 Part 1(b): Manifold learning on sklearn digits (n=1797, p=64).
        Methods: Kernel PCA, Spectral Embedding, Classical MDS, Metric MDS, t-SNE, UMAP, Autoencoder.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import SpectralEmbedding, MDS, TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
import umap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    
def set_seed(seed: int):
    import random, os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_embedding(X2d, y, title, outpath):
    """Simple 2D scatter with digits colormap; no axes ticks; safe colormap API."""
    cmap = plt.colormaps.get_cmap('tab10')
    plt.figure(figsize=(6, 5), dpi=120)
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=y, s=12, cmap=cmap, alpha=0.9, edgecolors='none')
    plt.title(title)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    cbar = plt.colorbar(sc, ticks=range(10))
    cbar.set_label("Digit label")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def kmeans_scores(X2d, y, seed, k=10):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X2d)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    sil = silhouette_score(X2d, labels)
    return ari, nmi, sil


def classical_mds(X, n_components=2):
    """
    Classical MDS (Torgerson/Gower): eigendecomposition of double-centered
    squared-distance matrix. Returns low-d coords (can be rotated/reflected).
    """
    D = pairwise_distances(X, metric='euclidean')
    D2 = D ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(B)
    # Take top components
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    # Keep only positive eigenvalues
    pos = evals > 0
    evals = evals[pos]; evecs = evecs[:, pos]
    evals_k = evals[:n_components]
    evecs_k = evecs[:, :n_components]
    X_emb = evecs_k * np.sqrt(np.maximum(evals_k, 0))
    return X_emb


def auto_gamma_rbf(X):
    """
    Heuristic gamma for RBF kernel (Kernel PCA): gamma = 1 / median(pairwise distance).
    Scale-robust; avoids manual guesswork when not provided.
    """
    d = pairwise_distances(X, metric='euclidean')
    med = np.median(d)
    if med <= 0:
        return 1.0
    return 1.0 / med

# Autoencoder

class AE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32, bottleneck: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck)  # 线性瓶颈，便于可视化
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid()  # 配合 [0,1] 输入，重构更稳定
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


@torch.no_grad()
def encode_dataset(model: AE, loader: DataLoader, device):
    model.eval()
    zs = []
    for (xb,) in loader:
        xb = xb.to(device)
        _, z = model(xb)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def train_autoencoder_pytorch(
    X_minmax: np.ndarray,
    seed: int = 0,
    hidden: int = 32,
    bottleneck: int = 2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    set_seed(seed)
    device = get_device()

    X_tensor = torch.tensor(X_minmax, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = AE(input_dim=X_minmax.shape[1], hidden=hidden, bottleneck=bottleneck).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = crit(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 推理阶段：得到 2D 嵌入
    # 复用同一个 DataLoader 保持顺序一致
    Z = encode_dataset(model, DataLoader(ds, batch_size=1024, shuffle=False), device)
    return Z


# ----------------------------
# Main pipeline for 1(b)
# ----------------------------
def run(seed=25,
        outdir="Homework 1/Latex/Results/Problem_1_b",
        kpca_gamma=None,
        spectral_n_neighbors=10,
        tsne_perplexity=30,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        ae_hidden=32,
        ae_epochs=50,
        ae_batch=128,
        ae_lr=1e-3):
    set_seed(seed)
    ensure_dir(outdir)

    # Load data
    digits = load_digits()
    X = digits.data.astype(float)        # (1797, 64)
    y = digits.target                    # (1797,)

    # Standardize (common for manifold learning)
    X_std = StandardScaler().fit_transform(X)

    # 1) Kernel PCA (RBF)
    gamma = kpca_gamma if kpca_gamma is not None else auto_gamma_rbf(X_std)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma, fit_inverse_transform=False, random_state=seed)
    X_kpca = kpca.fit_transform(X_std)
    plot_embedding(X_kpca, y, f"Kernel PCA (RBF, gamma={gamma:.3g})", f"{outdir}/kpca_2d.png")
    kpca_ari, kpca_nmi, kpca_sil = kmeans_scores(X_kpca, y, seed)

    # 2) Spectral Embedding
    se = SpectralEmbedding(n_components=2, n_neighbors=spectral_n_neighbors, random_state=seed)
    X_se = se.fit_transform(X_std)
    plot_embedding(X_se, y, f"Spectral Embedding (n_neighbors={spectral_n_neighbors})", f"{outdir}/spectral_2d.png")
    se_ari, se_nmi, se_sil = kmeans_scores(X_se, y, seed)

    # 3) Classical MDS (closed-form)
    X_cmds = classical_mds(X_std, n_components=2)
    plot_embedding(X_cmds, y, "Classical MDS", f"{outdir}/classical_mds_2d.png")
    cmds_ari, cmds_nmi, cmds_sil = kmeans_scores(X_cmds, y, seed)

    # 4) Metric MDS (sklearn MDS, metric=True)
    mds_metric = MDS(n_components=2, metric=True, normalized_stress='auto', random_state=seed)
    X_mds = mds_metric.fit_transform(X_std)
    plot_embedding(X_mds, y, "Metric MDS", f"{outdir}/metric_mds_2d.png")
    mds_ari, mds_nmi, mds_sil = kmeans_scores(X_mds, y, seed)

    # 5) t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate='auto', init='pca', random_state=seed)
    X_tsne = tsne.fit_transform(X_std)
    plot_embedding(X_tsne, y, f"t-SNE (perplexity={tsne_perplexity})", f"{outdir}/tsne_2d.png")
    tsne_ari, tsne_nmi, tsne_sil = kmeans_scores(X_tsne, y, seed)

    # 6) UMAP
    umap_model = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=2, random_state=seed)
    X_umap = umap_model.fit_transform(X_std)
    plot_embedding(X_umap, y, f"UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})", f"{outdir}/umap_2d.png")
    umap_ari, umap_nmi, umap_sil = kmeans_scores(X_umap, y, seed)

    # Autoencoder（PyTorch）
    X_minmax = MinMaxScaler().fit_transform(X)
    X_ae = train_autoencoder_pytorch(
        X_minmax,
        seed=seed,
        hidden=ae_hidden,
        bottleneck=2,
        epochs=ae_epochs,
        batch_size=ae_batch,
        lr=ae_lr,
    )
    plot_embedding(X_ae, y, f"Autoencoder (hidden={ae_hidden}, epochs={ae_epochs})", f"{outdir}/autoencoder_2d.png")
    ae_ari, ae_nmi, ae_sil = kmeans_scores(X_ae, y, seed)

    # Save metrics
    import csv
    metrics_path = f"{outdir}/part1b_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Params", "ARI", "NMI", "Silhouette", "Notes"])
        writer.writerow(["Kernel PCA", f"rbf, gamma={gamma:.3g}", f"{kpca_ari:.4f}", f"{kpca_nmi:.4f}", f"{kpca_sil:.4f}", "2D embedding"])
        writer.writerow(["Spectral", f"n_neighbors={spectral_n_neighbors}", f"{se_ari:.4f}", f"{se_nmi:.4f}", f"{se_sil:.4f}", "2D embedding"])
        writer.writerow(["Classical MDS", "-", f"{cmds_ari:.4f}", f"{cmds_nmi:.4f}", f"{cmds_sil:.4f}", "closed-form (eigendecomposition)"])
        writer.writerow(["Metric MDS", "sklearn MDS(metric=True)", f"{mds_ari:.4f}", f"{mds_nmi:.4f}", f"{mds_sil:.4f}", "stress minimization"])
        writer.writerow(["t-SNE", f"perplexity={tsne_perplexity}", f"{tsne_ari:.4f}", f"{tsne_nmi:.4f}", f"{tsne_sil:.4f}", "2D embedding"])
        writer.writerow(["UMAP", f"n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}", f"{umap_ari:.4f}", f"{umap_nmi:.4f}", f"{umap_sil:.4f}", "2D embedding"])
        writer.writerow(["Autoencoder",f"hidden={ae_hidden}, epochs={ae_epochs}, batch={ae_batch}, lr={ae_lr:g}",f"{ae_ari:.4f}", f"{ae_nmi:.4f}", f"{ae_sil:.4f}","2D bottleneck (PyTorch)"])

    # Print a neat summary
    print("\n=== Part 1(b) — Manifold Methods on Digits ===")
    print(f"[Kernel PCA]    ARI={kpca_ari:.4f}  NMI={kpca_nmi:.4f}  Silhouette={kpca_sil:.4f}  (gamma={gamma:.3g})")
    print(f"[Spectral]      ARI={se_ari:.4f}  NMI={se_nmi:.4f}  Silhouette={se_sil:.4f}  (n_neighbors={spectral_n_neighbors})")
    print(f"[Classical MDS] ARI={cmds_ari:.4f}  NMI={cmds_nmi:.4f}  Silhouette={cmds_sil:.4f}")
    print(f"[Metric MDS]    ARI={mds_ari:.4f}  NMI={mds_nmi:.4f}  Silhouette={mds_sil:.4f}")
    print(f"[t-SNE]         ARI={tsne_ari:.4f}  NMI={tsne_nmi:.4f}  Silhouette={tsne_sil:.4f}  (perplexity={tsne_perplexity})")
    print(f"[UMAP]          ARI={umap_ari:.4f}  NMI={umap_nmi:.4f}  Silhouette={umap_sil:.4f}  (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})")
    print(f"\nSaved figures and metrics to: {outdir}/")
    print("Figures:")
    print(" - kpca_2d.png, spectral_2d.png, classical_mds_2d.png, metric_mds_2d.png, tsne_2d.png, umap_2d.png")
    print("Table:")
    print(f" - {Path(metrics_path).name}")
    print(f"[Autoencoder]   ARI={ae_ari:.4f}  NMI={ae_nmi:.4f}  Silhouette={ae_sil:.4f}  (hidden={ae_hidden}, epochs={ae_epochs})")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--outdir", type=str, default="Homework 1/Latex/Results/Problem_1_b", help="output directory")
    parser.add_argument("--kpca_gamma", type=float, default=None, help="RBF gamma for Kernel PCA (auto if None)")
    parser.add_argument("--spectral_n_neighbors", type=int, default=10, help="n_neighbors for Spectral Embedding")
    parser.add_argument("--tsne_perplexity", type=float, default=30, help="perplexity for t-SNE")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="n_neighbors for UMAP")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="min_dist for UMAP")
    parser.add_argument("--ae_hidden", type=int, default=32)
    parser.add_argument("--ae_epochs", type=int, default=50)
    parser.add_argument("--ae_batch", type=int, default=128)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    args = parser.parse_args()

    run(seed=args.seed,
        outdir=args.outdir,
        kpca_gamma=args.kpca_gamma,
        spectral_n_neighbors=args.spectral_n_neighbors,
        tsne_perplexity=args.tsne_perplexity,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        ae_hidden=args.ae_hidden,
        ae_epochs=args.ae_epochs,
        ae_batch=args.ae_batch,
        ae_lr=args.ae_lr)
