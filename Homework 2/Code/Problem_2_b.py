'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-31 02:09:44
LastEditTime: 2025-10-31 02:50:01
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_2_b.py
Description: 
    Validation.
'''
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from packaging.version import parse as vparse
import sklearn
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state

from Problem_2_a import load_brca, compute_embeddings

def _kmeans_fit_predict(X, k, seed):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)

def _gmm_fit_predict(X, k, seed):
    gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5)
    return gm.fit_predict(X), gm

def _spectral_fit_predict(X_pca30, k, seed):
    return SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors",
        n_neighbors=10, assign_labels="kmeans",
        random_state=seed, n_init=10
    ).fit_predict(X_pca30)

def _agglo_fit_predict(X, k, linkage):
    new_api = vparse(sklearn.__version__) >= vparse("1.2")
    if linkage == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        if new_api:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric="euclidean")
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity="euclidean")
    return model.fit_predict(X)

def _dbscan_fit_predict(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)

def _silhouette_safe(X, labels):
    labs = np.asarray(labels)
    if len(np.unique(labs)) <= 1 or np.all(labs == -1):
        return np.nan
    try:
        return silhouette_score(X, labs)
    except Exception:
        return np.nan

def _bootstrap_stability(X, fit_fn, n_boot=10, seed=0):
    rng = check_random_state(seed)
    full_labels = fit_fn(X)
    if len(np.unique(full_labels)) <= 1:
        return np.nan
    n = X.shape[0]
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        boot_labels = fit_fn(X[idx])
        if len(np.unique(boot_labels)) <= 1:
            aris.append(np.nan)
        else:
            aris.append(adjusted_rand_score(full_labels[idx], boot_labels))
    aris = np.array(aris, dtype=float)
    return float(np.nanmedian(aris))

def _generalizability_test_silhouette(X, labels, tag, k=None, seed=0):
    """
    80/20 split: fit clusterer on train, assign test labels via:
      - kmeans/gmm: model predict
      - spectral/agg/dbscan: kNN(label transfer) from train to test
    Return test silhouette.
    """
    rng = check_random_state(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, Xte = X[tr], X[te]

    # fit on train
    if tag == "kmeans":
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xtr)
        yte = km.predict(Xte)
    elif tag == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5).fit(Xtr)
        yte = gm.predict(Xte)
    else:
        # label transfer by kNN with train labels from full-data 'labels'
        ytr = np.asarray(labels)[tr]
        knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
        knn.fit(Xtr, ytr)
        yte = knn.predict(Xte)

    return _silhouette_safe(Xte, yte)

# ----------------- validation core -----------------
def validate_all(
    X_pca30, X_umap10,
    K_grid=range(2, 10),
    dbscan_eps=(0.3, 0.4, 0.5, 0.6, 0.7),
    dbscan_min_samples=(5, 10, 15),
    n_boot=10,
    seed=25
):
    rng = check_random_state(seed)
    records = []

    # --- K-based methods ---
    for K in K_grid:
        # KMeans (fit/view on UMAP10)
        km_labels = _kmeans_fit_predict(X_umap10, K, seed)
        sil = _silhouette_safe(X_umap10, km_labels)
        stab = _bootstrap_stability(X_umap10, lambda Z: _kmeans_fit_predict(Z, K, rng.randint(1e9)), n_boot, seed)
        gen = _generalizability_test_silhouette(X_umap10, km_labels, "kmeans", k=K, seed=seed)
        records.append(("kmeans", K, np.nan, np.nan, sil, stab, gen))

        # GMM (UMAP10)
        gmm_labels, _ = _gmm_fit_predict(X_umap10, K, seed)
        sil = _silhouette_safe(X_umap10, gmm_labels)
        stab = _bootstrap_stability(X_umap10, lambda Z: _gmm_fit_predict(Z, K, rng.randint(1e9))[0], n_boot, seed)
        gen = _generalizability_test_silhouette(X_umap10, gmm_labels, "gmm", k=K, seed=seed)
        records.append(("gmm", K, np.nan, np.nan, sil, stab, gen))

        # Spectral (fit on PCA30)
        sp_labels = _spectral_fit_predict(X_pca30, K, seed)
        sil = _silhouette_safe(X_pca30, sp_labels)
        stab = _bootstrap_stability(X_pca30, lambda Z: _spectral_fit_predict(Z, K, rng.randint(1e9)), n_boot, seed)
        gen = _generalizability_test_silhouette(X_pca30, sp_labels, "spectral", k=K, seed=seed)
        records.append(("spectral_pca", K, np.nan, np.nan, sil, stab, gen))

        # Agglomerative (UMAP10): ward/average/complete
        for lk in ("ward", "average", "complete"):
            ag_labels = _agglo_fit_predict(X_umap10, K, lk)
            sil = _silhouette_safe(X_umap10, ag_labels)
            stab = _bootstrap_stability(X_umap10, lambda Z, _lk=lk: _agglo_fit_predict(Z, K, _lk), n_boot, seed)
            gen = _generalizability_test_silhouette(X_umap10, ag_labels, f"agg_{lk}", k=K, seed=seed)
            records.append((f"agg_{lk}", K, np.nan, np.nan, sil, stab, gen))

    # --- DBSCAN grid (UMAP10) ---
    for eps in dbscan_eps:
        for m in dbscan_min_samples:
            db_labels = _dbscan_fit_predict(X_umap10, eps, m)
            sil = _silhouette_safe(X_umap10, db_labels)
            stab = _bootstrap_stability(
                X_umap10, lambda Z, _e=eps, _m=m: _dbscan_fit_predict(Z, _e, _m), n_boot, seed
            )
            gen  = _generalizability_test_silhouette(X_umap10, db_labels, "dbscan", k=None, seed=seed)
            records.append(("dbscan", np.nan, eps, m, sil, stab, gen))

    cols = ["method", "K", "eps", "min_samples", "silhouette", "stability", "generalizability"]
    return pd.DataFrame.from_records(records, columns=cols)

def pick_best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for meth in df["method"].unique():
        sub = df[df["method"] == meth].copy()
        if sub.empty: 
            continue
        # rank by (silhouette, stability, generalizability)
        sub["_rank"] = list(zip(-sub["silhouette"].fillna(-1e9),
                                -sub["stability"].fillna(-1e9),
                                -sub["generalizability"].fillna(-1e9)))
        sub = sub.sort_values("_rank").drop(columns=["_rank"])
        out.append(sub.iloc[[0]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def run(
    data_path="Homework 2/Code/Data/BRCA_data.csv",
    outdir="Homework 2/Code/Result/Problem_2",
    seed=25
):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data + embeddings (reuse 2a)
    X, _ = load_brca(data_path)
    # (pca2 not needed here)
    _, X_pca30, X_umap10, _ = compute_embeddings(X, seed=seed, n_pcs_feat=30)

    # validation
    df = validate_all(
        X_pca30, X_umap10,
        K_grid=range(2, 10),
        dbscan_eps=(0.3, 0.4, 0.5, 0.6, 0.7),
        dbscan_min_samples=(5, 10, 15),
        n_boot=10,
        seed=seed
    )

    # save all + best-per-method
    all_csv = outdir / "problem2b_validation.csv"
    best_csv = outdir / "problem2b_best.csv"
    df.to_csv(all_csv, index=False)
    pick_best_per_method(df).to_csv(best_csv, index=False)
    
    dfk = df[df["K"].notna()].copy()
    dfk["K"] = dfk["K"].astype(int)

    # unify method labels for legend
    label_map = {
        "kmeans": "KMeans",
        "gmm": "GMM",
        "spectral_pca": "Spectral (PCA)",
        "agg_ward": "Agg-Ward",
        "agg_average": "Agg-Average",
        "agg_complete": "Agg-Complete",
    }
    dfk["Method"] = dfk["method"].map(label_map)

    palette = {
        "KMeans": "#1f77b4",
        "GMM": "#2ca02c",
        "Spectral (PCA)": "#ff7f0e",
        "Agg-Ward": "#9467bd",
        "Agg-Average": "#d62728",
        "Agg-Complete": "#8c564b",
    }

    # ---- plotting helper ----
    def plot_metric(metric, ylabel):
        plt.figure(figsize=(7,5), dpi=130)
        sns.lineplot(data=dfk, x="K", y=metric, hue="Method", marker="o", palette=palette)
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs K across clustering methods")
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
        plt.grid(True, linewidth=0.4, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"validation_{metric.lower()}_vs_K.png", bbox_inches="tight")
        plt.close()

    # ---- draw three figures ----
    plot_metric("silhouette", "Silhouette Score")
    plot_metric("stability", "Stability (Bootstrap ARI)")
    plot_metric("generalizability", "Generalizability (Test Silhouette)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Homework 2/Code/Data/BRCA_data.csv")
    parser.add_argument("--outdir", type=str, default="Homework 2/Latex/Figures")
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    run(data_path=args.data_path, outdir=args.outdir, seed=args.seed)
