'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-09 13:41:38
LastEditors: Schuyn 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-10-09 14:29:11
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


if __name__ == "__main__":
    data_path = "Homework 1/Data/BRCA_data.csv"
    report_path = "Homework 1/Latex/Results/Problem_2/preprocessing_report.txt"
    load_and_preprocess_brca(data_path, report_path=report_path)