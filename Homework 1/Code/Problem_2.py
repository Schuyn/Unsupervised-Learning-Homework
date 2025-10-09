'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-09 13:41:38
LastEditors: Schuyn 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-10-09 14:10:19
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

def load_and_preprocess_brca(data_path: str, var_threshold: float = 1e-4, report_path: str = None):
    # --- 1) Load data: treat the 1st column as sample ID index ---
    df = pd.read_csv(data_path, index_col=0)
    print(f"Raw data shape: {df.shape[0]} samples × {df.shape[1]} columns")


if __name__ == "__main__":
    data_path = "Homework 1/Data/BRCA_data.csv"
    report_path = "Homework 1/Latex/Results/Problem_2/preprocessing_report.txt"
    load_and_preprocess_brca(data_path, report_path=report_path)