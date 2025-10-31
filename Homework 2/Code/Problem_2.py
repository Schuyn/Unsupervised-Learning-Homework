'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-30 14:36:11
LastEditTime: 2025-10-30 20:30:35
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_2.py
Description: 
    Apply clustering techniques(KMeans, GMM, Spectral Clustering, Agglomerative Clustering, DBSCAN) to explore the BRCA gene expression data set.
'''
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="n_jobs value 1 overridden to 1 by setting random_state",
                        module=r"umap\.umap_")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="KMeans is known to have a memory leak on Windows with MKL",
                        module=r"sklearn\.cluster\._kmeans")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Could not find the number of physical cores",
                        module=r"joblib\.externals\.loky\.backend\.context")
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from packaging.version import parse as vparse
import sklearn
import umap

def load_brca(data_path: str):
    df = pd.read_csv(data_path, index_col=0)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    clinical_cols = [c for c in ["subtype", "er_status", "pr_status", "her2_status", "node", "metastasis"]
                     if c in df.columns]
    gene_cols = [c for c in df.columns if c not in clinical_cols]

    X = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = StandardScaler().fit_transform(X.values)

    meta = df[clinical_cols].copy()
    return X, meta

def _encode(series):
    cats = series.astype(str).fillna("NA").values
    uniq = sorted(np.unique(cats))
    mapping = {c: i for i, c in enumerate(uniq)}
    idx = np.array([mapping[c] for c in cats], dtype=int)
    return idx, mapping

PAM50_COLOR_MAP = {
    "Luminal A": "#1f77b4",     # blue
    "Luminal B": "#2ca02c",     # green
    "Basal-like": "#ff7f0e",    # orange
    "HER2-enriched": "#d62728", # red
    "Normal-like": "#9467bd",   # purple
}

def scatter_by_labels(X2, labels, title, outpath, xlab="Dim 1", ylab="Dim 2",
                      color_map=None, order=None, s=26):
    from pathlib import Path
    outpath = Path(outpath); outpath.parent.mkdir(parents=True, exist_ok=True)
    lbl = pd.Series(labels).astype(str).fillna("NA")

    # determine unique order
    if order is None:
        uniq = sorted(lbl.unique())
    else:
        # keep only those present
        present = [u for u in order if str(u) in set(lbl)]
        rest = [u for u in lbl.unique() if u not in present]
        uniq = present + sorted(rest)

    # color picking
    colors = []
    if color_map is None:
        # fallback palette (cleaner than default)
        base = ["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2",
                "#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC"]
        while len(base) < len(uniq):
            base = base + base  # simple extend
        colors = base[:len(uniq)]
    else:
        for u in uniq:
            ckey = u if u in color_map else u.lower()
            colors.append(color_map.get(ckey, "#999999"))

    # map labels -> indices
    idx_map = {u:i for i,u in enumerate(uniq)}
    idx = lbl.map(idx_map).values

    plt.figure(figsize=(7.2, 5.6), dpi=130)
    plt.scatter(X2[:,0], X2[:,1], c=[colors[i] for i in idx],
                s=s, alpha=0.95, edgecolors='none')
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title, pad=8)

    # legend
    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                          markersize=6, markerfacecolor=colors[i],
                          markeredgecolor='none') for i in range(len(uniq))]
    if len(uniq) <= 25:
        plt.legend(handles, uniq, bbox_to_anchor=(1.02,1), loc='upper left',
                   fontsize=9, frameon=False)
    plt.grid(True, linewidth=0.3, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight'); plt.close()

def plot_pam50(X2, pam50_series, title, outpath, xlab, ylab):
    order = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched", "Normal-like"]
    scatter_by_labels(
        X2, pam50_series, title, outpath,
        xlab=xlab, ylab=ylab,
        color_map=PAM50_COLOR_MAP, order=order, s=28
    )

def compute_embeddings(X, seed=0, n_pcs_feat=30):
    # PCA for visualization (2D) and as preprocessor for UMAP
    X_pca2 = PCA(n_components=2, random_state=seed).fit_transform(X)
    X_pca30 = PCA(n_components=min(n_pcs_feat, X.shape[1]), random_state=seed).fit_transform(X)

    # UMAP on PCA-30
    u10 = umap.UMAP(n_components=10, n_neighbors=10, min_dist=0.0,
                    random_state=seed, metric="euclidean")
    X_umap10 = u10.fit_transform(X_pca30)

    u2 = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.0,
                   random_state=seed, metric="euclidean")
    X_umap2 = u2.fit_transform(X_pca30)

    return X_pca2, X_umap10, X_umap2

def clustering_all(X_umap10, X_umap2, outdir: Path, seed=0):
    k = 5  # PAM50 count

    # KMeans
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "KMeans (K=5) — UMAP view", outdir / "cluster__kmeans_k5_umap2.png",
                      "UMAP Dim 1", "UMAP Dim 2")

    # Spectral (nearest neighbors affinity)
    labels = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                n_neighbors=10, assign_labels="kmeans",
                                random_state=seed, n_init=10).fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "Spectral (NN=10, K=5) — UMAP view",
                      outdir / "cluster__spectral_nn10_k5_umap2.png", "UMAP Dim 1", "UMAP Dim 2")

    # GMM
    labels = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=seed, n_init=5).fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "GMM (K=5, full) — UMAP view", outdir / "cluster__gmm_k5_umap2.png",
                      "UMAP Dim 1", "UMAP Dim 2")

    # DBSCAN
    labels = DBSCAN(eps=0.5, min_samples=10, metric="euclidean").fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "DBSCAN (eps=0.5, min=10) — UMAP view",
                      outdir / "cluster__dbscan_eps0p5_min10_umap2.png", "UMAP Dim 1", "UMAP Dim 2")

    # Hierarchical (ward / average / complete). Single excluded by design.
    new_api = vparse(sklearn.__version__) >= vparse("1.2")

    # ward
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "Agglomerative — ward (K=5) — UMAP view",
                      outdir / "cluster__agg_ward_k5_umap2.png", "UMAP Dim 1", "UMAP Dim 2")

    # average
    if new_api:
        model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean")
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean")
    labels = model.fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "Agglomerative — average (K=5) — UMAP view",
                      outdir / "cluster__agg_average_k5_umap2.png", "UMAP Dim 1", "UMAP Dim 2")

    # complete
    if new_api:
        model = AgglomerativeClustering(n_clusters=k, linkage="complete", metric="euclidean")
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="complete", affinity="euclidean")
    labels = model.fit_predict(X_umap10)
    scatter_by_labels(X_umap2, labels, "Agglomerative — complete (K=5) — UMAP view",
                      outdir / "cluster__agg_complete_k5_umap2.png", "UMAP Dim 1", "UMAP Dim 2")

def run(
    data_path="Homework 2/Code/Data/BRCA_data.csv",
    outdir="Homework 2/Code/Result/Problem_2",
    seed=25
):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data
    X, meta = load_brca(data_path)
    pam50 = meta["subtype"] if "subtype" in meta.columns else pd.Series(["NA"] * X.shape[0])

    # embeddings (PCA-2 and UMAP-2 are the only two embedding figures we save)
    X_pca2, X_umap10, X_umap2 = compute_embeddings(X, seed=seed, n_pcs_feat=30)
    plot_pam50(X_pca2, pam50, "PAM50 — PCA (2D)",
               outdir / "pam50__pca2.png", "PCA Dim 1", "PCA Dim 2")
    plot_pam50(X_umap2, pam50, "PAM50 — UMAP (2D)",
               outdir / "pam50__umap2.png", "UMAP Dim 1", "UMAP Dim 2")
    clustering_all(X_umap10, X_umap2, outdir, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Homework 2/Code/Data/BRCA_data.csv")
    parser.add_argument("--outdir", type=str, default="Homework 2/Code/Result/Problem_2")
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    run(data_path=args.data_path, outdir=args.outdir, seed=args.seed)

