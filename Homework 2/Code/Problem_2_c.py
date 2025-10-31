'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-31 02:57:20
LastEditTime: 2025-10-31 02:57:22
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_2_c.py
Description: 
    plot the best model.
'''
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from Problem_2_a import load_brca, compute_embeddings, scatter_by_labels

def best_clustering_labels(X_umap10, k=2):
    model = AgglomerativeClustering(n_clusters=k, linkage="average")
    return model.fit_predict(X_umap10)

def run(
    data_path="Homework 2/Code/Data/BRCA_data.csv",
    outdir="Homework 2/Code/Result/Problem_2",
    seed=25
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, meta = load_brca(data_path)
    X_pca2, X_pca30, X_umap10, X_umap2 = compute_embeddings(X, seed=seed, n_pcs_feat=30)

    pam50 = meta["subtype"].astype(str).fillna("NA").values
    labels_best = best_clustering_labels(X_umap10, k=2)

    ari = adjusted_rand_score(pam50, labels_best)
    nmi = normalized_mutual_info_score(pam50, labels_best)
    print(f"ARI = {ari:.3f}, NMI = {nmi:.3f}")

    # Define consistent color map
    PAM50_COLOR_MAP = {
        "Luminal A": "#1f77b4",
        "Luminal B": "#2ca02c",
        "Basal-like": "#ff7f0e",
        "HER2-enriched": "#d62728",
        "Normal-like": "#9467bd",
    }

    # Best clustering (Agg-Average, K=2)
    cmap_best = {str(i): c for i, c in enumerate(["#4C78A8", "#E45756"])}
    scatter_by_labels(
        X_umap2, labels_best.astype(str),
        title="Best Clustering (Agglomerative Average, K=2)",
        outpath=outdir / "2c_bestcluster_umap2.png",
        color_map=cmap_best,
        legend_title="Cluster"
    )

    # Combined overlay (color by PAM50, edge by cluster)
    df_plot = pd.DataFrame({
        "x": X_umap2[:,0], "y": X_umap2[:,1],
        "Cluster": labels_best.astype(str),
        "Subtype": pam50
    })

    plt.figure(figsize=(7,5.5), dpi=130)
    sns.scatterplot(
        data=df_plot, x="x", y="y",
        hue="Subtype", style="Cluster",
        palette=PAM50_COLOR_MAP, alpha=0.9, s=28
    )
    plt.title(f"Overlay of PAM50 and Best Clustering (ARI={ari:.2f}, NMI={nmi:.2f})")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
    plt.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "2c_overlay_umap2.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Homework 2/Code/Data/BRCA_data.csv")
    parser.add_argument("--outdir", type=str, default="Homework 2/Latex/Figures")
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    run(data_path=args.data_path, outdir=args.outdir, seed=args.seed)