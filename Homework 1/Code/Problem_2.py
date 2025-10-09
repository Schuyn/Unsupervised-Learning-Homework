'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-09 13:41:38
LastEditors: Schuyn 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-10-09 14:43:31
FilePath: /Unsupervised-Learning-Homework/Homework 1/Code/Problem_2.py
Description: 
    This data set consists of gene expression measurements for n = 445 breast cancer tumors and p = 353 genes taken from The Cancer Genome Atlas(TCGA). 
    This subset of genes was selected based on whether they contain known somatic mutations in cancer. 
    Additionally, this data contains clinical data on the 
        (i) Subtype (denotes 5 PAM50 subtypes including Basal-like, Luminal A, Luminal B, HER2-enriched, and Normal-like), 
        (ii) ER-Status(estrogen-receptor status), 
        (iii) PR-Status (progesterone-receptor status), 
        (iv) HER2-Status (human epidermal growth factor receptor 2 status), 
        (v) Node (number of lymph nodes involved), and (vi)Metastasis (indicator for whether the cancer has metastasized).
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import csv
import umap

# === Embedding and evaluation pipeline (UMAP included) ===
def run_embedding(X, method='pca', n_components=2, seed=0, **kwargs):
    """
    Apply a dimension reduction method and return 2D embedding.
    Assumes UMAP is available (imported at top).
    """
    method = method.lower()
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=seed)
        name = "PCA"
    elif method == 'nmf':
        model = NMF(n_components=n_components, init='nndsvda', random_state=seed, max_iter=1000)
        name = "NMF"
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=seed, **kwargs)
        name = "tSNE"
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=seed, **kwargs)
        name = "UMAP"
    elif method == 'spectral':
        model = SpectralEmbedding(n_components=n_components, random_state=seed)
        name = "Spectral"
    else:
        raise ValueError(f"Unknown method: {method}")
    X_embedded = model.fit_transform(X)
    return X_embedded, name


def _encode_categories(series):
    """Encode categorical labels to integer indices and return (indices, mapping)."""
    cats = series.astype(str).fillna("NA").values
    uniq = sorted(np.unique(cats))
    mapping = {c: i for i, c in enumerate(uniq)}
    idx = np.array([mapping[c] for c in cats], dtype=int)
    return idx, mapping

def plot_embedding(X2d, meta, hue_col, title, outpath):
    """Plot and save a 2D embedding colored by a clinical variable."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    labels = meta[hue_col].astype(str).fillna("NA")
    idx, mapping = _encode_categories(labels)
    cmap = plt.cm.get_cmap('tab10', len(mapping))

    plt.figure(figsize=(6.4, 5.2), dpi=120)
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=idx, s=14, cmap=cmap, alpha=0.85, edgecolors='none')
    plt.title(f"{title} embedding colored by {hue_col}")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")

    # manual legend
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                          markerfacecolor=cmap(i), markeredgecolor='none')
               for i in range(len(mapping))]
    labels_sorted = list(mapping.keys())
    plt.legend(handles, labels_sorted, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def evaluate_embedding(X2d, subtype_series, seed=0):
    """KMeans(k=5) on the embedding; compute ARI/NMI/Silhouette using Subtype as reference."""
    mask = subtype_series.notna()
    X_eval = X2d[mask.values]
    y_ref = subtype_series[mask].astype(str).values

    km = KMeans(n_clusters=5, random_state=seed, n_init=10)
    pred = km.fit_predict(X_eval)

    ari = adjusted_rand_score(y_ref, pred)
    nmi = normalized_mutual_info_score(y_ref, pred)
    sil = silhouette_score(X_eval, pred)
    return ari, nmi, sil


def run_problem2_pipeline(
    data_path="Homework 1/Data/BRCA_data.csv",
    outdir="Homework 1/Latex/Results/Problem_2",
    seed=0
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess (writes TXT report under outdir)
    X, y_clinical, gene_cols = load_and_preprocess_brca(
        data_path=data_path,
        report_path=str(outdir / "preprocessing_report.txt")
    )

    # 2) Methods to run (UMAP included by assumption)
    methods = [
        ("pca",      dict()),
        ("nmf",      dict()),
        ("spectral", dict()),
        ("tsne",     dict(perplexity=30, learning_rate='auto', init='pca')),
        ("umap",     dict(n_neighbors=15, min_dist=0.1)),
    ]

    # 3) Clinical hues to color by (only those that exist)
    hues = [c for c in ["subtype", "er_status", "pr_status", "her2_status"] if c in y_clinical.columns]

    # 4) Run, plot, evaluate
    metrics_rows = []
    best_by_ari = (None, -1.0)

    for m, kwargs in methods:
        if m == "nmf":
            # NMF 只能用非负数据，传入原始（已填补但未标准化）的矩阵
            X_nonneg = np.maximum(X, 0)  # 保证所有值非负
            X2d, name = run_embedding(X_nonneg, method=m, seed=seed, **kwargs)
        else:
            X2d, name = run_embedding(X, method=m, seed=seed, **kwargs)

        # Evaluation (Subtype as reference, if present)
        if "subtype" in y_clinical.columns:
            ari, nmi, sil = evaluate_embedding(X2d, y_clinical["subtype"], seed=seed)
        else:
            ari = nmi = sil = np.nan

        metrics_rows.append([name, f"{ari:.4f}", f"{nmi:.4f}", f"{sil:.4f}"])
        if not np.isnan(ari) and ari > best_by_ari[1]:
            best_by_ari = (name, ari)

        # Save embedding as CSV (optional, useful for debugging/report)
        emb_path = outdir / f"{name}_embedding.csv"
        np.savetxt(emb_path, X2d, delimiter=",")

        # Plots colored by clinical variables
        for hue in hues:
            fig_path = outdir / f"{name}_{hue}.png"
            plot_embedding(X2d, y_clinical, hue_col=hue, title=name, outpath=fig_path)

    # 5) Save metrics summary
    metrics_path = outdir / "metrics_summary.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "ARI (Subtype)", "NMI (Subtype)", "Silhouette (KMeans=5)"])
        writer.writerows(metrics_rows)

    # Console summary
    print("\n=== Problem 2: embedding metrics (Subtype as reference) ===")
    for r in metrics_rows:
        print("{:<10s}  ARI={}  NMI={}  Sil={}".format(*r))
    if best_by_ari[0] is not None:
        print(f"\nBest by ARI: {best_by_ari[0]} (ARI={best_by_ari[1]:.4f})")

    print(f"\nSaved figures & tables to: {outdir}")

def load_and_preprocess_brca(data_path: str, var_threshold: float = 1e-4, report_path: str = None):
    # 1) Load with sample IDs in index
    df = pd.read_csv(data_path, index_col=0)
    n_samples_raw, n_cols_raw = df.shape
    print(f"Raw data shape: {n_samples_raw} samples × {n_cols_raw} columns")

    # 2) Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )

    # 3) Detect clinical columns robustly
    possible_clinical = ["subtype", "er_status", "pr_status", "her2_status", "node", "metastasis"]
    clinical_cols = [c for c in possible_clinical if c in df.columns]
    if not clinical_cols:
        raise ValueError("No clinical columns detected after normalization. "
                         "Please check the CSV headers.")
    print(f"Detected clinical columns: {clinical_cols}")

    # 4) Split gene vs clinical
    gene_cols = [c for c in df.columns if c not in clinical_cols]
    # Force numeric on genes; coerce errors to NaN
    df[gene_cols] = df[gene_cols].apply(pd.to_numeric, errors="coerce")

    X_df = df[gene_cols].copy()
    y_clinical = df[clinical_cols].copy()

    # 5) Missing value stats and imputation
    n_missing_total = int(X_df.isna().sum().sum())
    n_rows_with_na = int(X_df.isna().any(axis=1).sum())
    n_genes_with_na = int(X_df.isna().any(axis=0).sum())
    print(f"Missing values in gene matrix: {n_missing_total} "
          f"(rows with any NA: {n_rows_with_na}, genes with any NA: {n_genes_with_na})")

    # Impute clinical: categorical by mode, numeric by mean
    for col in y_clinical.columns:
        if y_clinical[col].dtype == "O":
            mode_vals = y_clinical[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            y_clinical[col] = y_clinical[col].fillna(fill_val)
        else:
            y_clinical[col] = y_clinical[col].fillna(y_clinical[col].mean())

    # Impute genes by column median (robust)
    X_df = X_df.fillna(X_df.median())

    # 6) Standardize (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # 7) Build and optionally save summary report
    summary_lines = [
        "=== BRCA Data Preprocessing Summary ===",
        f"Original shape: {n_samples_raw} samples × {n_cols_raw} columns",
        f"Detected clinical columns: {clinical_cols}",
        f"Initial gene columns: {len(gene_cols)}",
        f"Total missing values filled (genes): {n_missing_total}",
        f"Rows with any NA (genes): {n_rows_with_na}",
        f"Genes with any NA: {n_genes_with_na}",
        f"Final standardized data shape: {X_scaled.shape}",
    ]

    # --- Clinical summary by type ---
    summary_text = ["Clinical variable summary:"]
    for col in y_clinical.columns:
        if y_clinical[col].dtype == "O":
            counts = y_clinical[col].value_counts(dropna=False)
            summary_text.append(f"\n{col} (categorical):")
            summary_text.append(str(counts))
        else:
            desc = y_clinical[col].describe()
            summary_text.append(f"\n{col} (numeric):")
            summary_text.append(str(desc))

    report_text = "\n".join(summary_lines + ["\n".join(summary_text)])
    print("\n" + report_text)

    if report_path is not None:
        report_dir = Path(report_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nPreprocessing summary saved to {report_path}")

    return X_scaled, y_clinical, gene_cols

def run_embedding(X, method='pca', n_components=2, seed=0, **kwargs):
    """返回 2D 嵌入 (n_samples, 2) 和方法名"""
    method = method.lower()
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=seed)
        name = 'PCA'
    elif method == 'nmf':
        # 直接 2D NMF 仅用于可视化；你也可以改成 NMF(k)->PCA(2)
        model = NMF(n_components=n_components, init='nndsvda', random_state=seed, max_iter=1000)
        name = 'NMF'
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=seed, **kwargs)
        name = 'tSNE'
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=seed, **kwargs)
        name = 'UMAP'
    elif method == 'spectral':
        model = SpectralEmbedding(n_components=n_components, random_state=seed)
        name = 'Spectral'
    else:
        raise ValueError(f"Unknown method: {method}")

    X_emb = model.fit_transform(X)
    return X_emb, name


def _encode_categories(series):
    """把分类标签编码成 0..K-1，返回 colors_idx 和 类别->颜色索引的映射"""
    cats = series.astype(str).fillna("NA").values
    uniq = sorted(np.unique(cats))
    mapping = {c: i for i, c in enumerate(uniq)}
    idx = np.array([mapping[c] for c in cats], dtype=int)
    return idx, mapping


def plot_embedding(X2d, meta, hue_col, title, outpath):
    """仅用 matplotlib 画散点（避免 seaborn 依赖问题）"""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    labels = meta[hue_col].astype(str).fillna("NA")
    idx, mapping = _encode_categories(labels)
    cmap = plt.cm.get_cmap('tab10', len(mapping))

    plt.figure(figsize=(6.2, 5.2), dpi=120)
    sc = plt.scatter(X2d[:,0], X2d[:,1], c=idx, s=14, cmap=cmap, alpha=0.85, edgecolors='none')
    plt.title(f"{title} embedding colored by {hue_col}")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    # 手工图例
    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                          markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
               for i in range(len(mapping))]
    labels_sorted = list(mapping.keys())
    plt.legend(handles, labels_sorted, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def evaluate_embedding(X2d, subtype_series, seed=0):
    """在嵌入上做 KMeans(k=5)，计算 ARI/NMI/Silhouette（参考 Subtype）"""
    # 只用有 Subtype 的样本
    mask = subtype_series.notna()
    X_eval = X2d[mask.values]
    y_ref = subtype_series[mask].astype(str).values

    # KMeans 聚类
    km = KMeans(n_clusters=5, random_state=seed, n_init=10)
    pred = km.fit_predict(X_eval)

    ari = adjusted_rand_score(y_ref, pred)
    nmi = normalized_mutual_info_score(y_ref, pred)
    sil = silhouette_score(X_eval, pred)

    return ari, nmi, sil



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Homework 1/Data/BRCA_data.csv")
    parser.add_argument("--outdir", type=str, default="Homework 1/Latex/Results/Problem_2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_problem2_pipeline(
        data_path=args.data_path,
        outdir=args.outdir,
        seed=args.seed
    )