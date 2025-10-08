'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-08 17:16:54
LastEditTime: 2025-10-08 18:21:50
FilePath: /Unsupervised-Learning-Homework/Homework 1/Code/Problem_1_a.py
Description: 
    This is the code part of GR5244 Unsupervised Learning Homework 1 Part 1a.
'''
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import csv


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def plot_embedding(X2d, y, title, outpath):
    plt.figure(figsize=(6, 5), dpi=120)
    scatter = plt.scatter(X2d[:, 0], X2d[:, 1], c=y, s=12, cmap='tab10', alpha=0.9, edgecolors='none')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit label")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def plot_components(components, img_shape, n_show, title, outpath):
    n = min(n_show, components.shape[0])
    cols = 10 if n >= 10 else n
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(1.4*cols, 1.4*rows), dpi=120)
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(components[i].reshape(img_shape), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{i+1}", fontsize=8)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def kmeans_scores(X2d, y, seed):
    km = KMeans(n_clusters=10, n_init='auto', random_state=seed)
    labels = km.fit_predict(X2d)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    sil = silhouette_score(X2d, labels)
    return ari, nmi, sil


# ----------------------------
# Main pipeline for 1(a)
# ----------------------------
def run(seed=0, nmf_ks=(10, 15, 20), outdir="Homework 1/Latex"):
    ensure_dir(outdir)

    # Load data
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target
    img_shape = (8, 8)

    # ----------------------------
    # PCA (2D embedding + components)
    # ----------------------------
    pca_2 = PCA(n_components=2, random_state=seed)
    X_pca_2 = pca_2.fit_transform(X)
    plot_embedding(X_pca_2, y, "PCA (2D)", f"{outdir}/pca_2d.png")

    # For components visualization, use more PCs (e.g., 10)
    pca_10 = PCA(n_components=10, random_state=seed).fit(X)
    plot_components(pca_10.components_, img_shape, n_show=10,
                    title="PCA Components (top 10)", outpath=f"{outdir}/pca_components.png")

    pca_ari, pca_nmi, pca_sil = kmeans_scores(X_pca_2, y, seed)

    # ----------------------------
    # NMF (grid over k), then 2D via PCA-on-W for visualization
    # ----------------------------
    X_nonneg = X - X.min() if X.min() < 0 else X
    best_nmf = None
    best_rec = np.inf
    best_k = None

    for k in nmf_ks:
        nmf = NMF(n_components=k, init='nndsvda', random_state=seed, max_iter=1000)
        W = nmf.fit_transform(X_nonneg)
        rec = nmf.reconstruction_err_
        if rec < best_rec:
            best_rec = rec
            best_nmf = nmf
            best_k = k

    # Use the best NMF
    W_best = best_nmf.transform(X_nonneg)  # (n_samples, best_k)
    # Reduce W to 2D for visualization
    nmf_to2 = PCA(n_components=2, random_state=seed)
    X_nmf_2 = nmf_to2.fit_transform(W_best)
    plot_embedding(X_nmf_2, y, f"NMF -> PCA (2D)  [k={best_k}]", f"{outdir}/nmf_2d.png")

    # Visualize NMF basis (H)
    plot_components(best_nmf.components_, img_shape, n_show=10,
                    title=f"NMF Basis (k={best_k}, show 10)", outpath=f"{outdir}/nmf_components.png")

    nmf_ari, nmf_nmi, nmf_sil = kmeans_scores(X_nmf_2, y, seed)

    # ----------------------------
    # ICA (2D embedding + components)
    # ----------------------------
    ica_pipeline = make_pipeline(StandardScaler(with_std=True), FastICA(n_components=2, random_state=seed, max_iter=1000))
    X_ica_2 = ica_pipeline.fit_transform(X)
    plot_embedding(X_ica_2, y, "ICA (2D)", f"{outdir}/ica_2d.png")

    # For components display, fit a separate ICA with more comps (e.g., 10) on standardized X
    scaler = StandardScaler(with_std=True)
    X_std = scaler.fit_transform(X)
    ica_10 = FastICA(n_components=10, random_state=seed, max_iter=1000).fit(X_std)
    plot_components(ica_10.mixing_.T, img_shape, n_show=10,  # mixing_.T ≈ component “images”
                    title="ICA Components (10)", outpath=f"{outdir}/ica_components.png")

    ica_ari, ica_nmi, ica_sil = kmeans_scores(X_ica_2, y, seed)

    # ----------------------------
    # Save metrics
    # ----------------------------
    metrics_path = f"{outdir}/part1a_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Params", "ARI", "NMI", "Silhouette", "Notes"])
        writer.writerow(["PCA", "n_components=2", f"{pca_ari:.4f}", f"{pca_nmi:.4f}", f"{pca_sil:.4f}", "2D embedding"])
        writer.writerow(["NMF -> PCA", f"k={best_k}, then 2D PCA", f"{nmf_ari:.4f}", f"{nmf_nmi:.4f}", f"{nmf_sil:.4f}",
                         f"best reconstruction k among {list(nmf_ks)} (err={best_rec:.4f})"])
        writer.writerow(["ICA", "n_components=2 (with standardization)", f"{ica_ari:.4f}", f"{ica_nmi:.4f}", f"{ica_sil:.4f}", "2D embedding"])

    # Also print a neat summary
    print("\n=== Part 1(a) — Linear Methods on Digits ===")
    print(f"[PCA]       ARI={pca_ari:.4f}  NMI={pca_nmi:.4f}  Silhouette={pca_sil:.4f}")
    print(f"[NMF -> PCA]   ARI={nmf_ari:.4f}  NMI={nmf_nmi:.4f}  Silhouette={nmf_sil:.4f}  (best k={best_k}, recon_err={best_rec:.4f})")
    print(f"[ICA]       ARI={ica_ari:.4f}  NMI={ica_nmi:.4f}  Silhouette={ica_sil:.4f}")
    print(f"\nSaved figures and metrics to: {outdir}/")
    print("Figures:")
    print(" - pca_2d.png, pca_components.png")
    print(" - nmf_2d.png, nmf_components.png")
    print(" - ica_2d.png, ica_components.png")
    print("Table:")
    print(f" - {Path(metrics_path).name}")


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--outdir", type=str, default="Homework 1/Latex/Results/Problem_1_a")
    parser.add_argument("--nmf_ks", type=int, nargs="+", default=[10, 15, 20])
    args = parser.parse_args()
    run(seed=args.seed, nmf_ks=tuple(args.nmf_ks), outdir=args.outdir)

