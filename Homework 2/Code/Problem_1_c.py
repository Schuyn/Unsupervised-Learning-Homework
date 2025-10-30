'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 21:47:48
LastEditTime: 2025-10-30 12:09:25
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_c.py
Description: 
    Fit a Gaussian mixture model to the author data.
'''
import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from Problem_1_b import load_authors_counts_rda

rda_path = os.path.join("Homework 2", "Code", "Data", "authors.rda")
X, y_true, feature_names = load_authors_counts_rda(rda_path)
n, p = X.shape
print(f"Loaded authors data: {n} samples × {p} features")

# Standardize for GMM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Fit Gaussian Mixture Model (K=4)
K = 4
gmm = GaussianMixture(
    n_components=K,
    covariance_type="diag",
    random_state=25,
    max_iter=500,
    init_params="kmeans",
    reg_covar=1e-6,
)
gmm.fit(X_scaled)

# soft assignments and certainty
gamma = gmm.predict_proba(X_scaled)
hard_labels = gamma.argmax(axis=1)
certainty = gamma.max(axis=1)
uncertainty = 1 - certainty

result_dir = os.path.join("Homework 2", "Code", "Result")
os.makedirs(result_dir, exist_ok=True)

pd.DataFrame({"pi_k": gmm.weights_}).to_csv(os.path.join(result_dir, "gmm_pi.csv"), index=False)
pd.DataFrame(gmm.means_, columns=feature_names).to_csv(os.path.join(result_dir, "gmm_means.csv"), index_label="cluster")
pd.DataFrame(gamma, columns=[f"cluster_{k}" for k in range(K)]).to_csv(os.path.join(result_dir, "gmm_gamma.csv"), index=False)
pd.DataFrame({"hard_label": hard_labels, "certainty": certainty}).to_csv(os.path.join(result_dir, "gmm_labels.csv"), index=False)

top_m = 10
print("\nMixture Weights (π_k):")
for k, val in enumerate(gmm.weights_):
    print(f"  Cluster {k}: {val:.4f}")

print("\nTop Words per Cluster (highest mean features):")
for k in range(K):
    top_idx = np.argsort(-gmm.means_[k])[:top_m]
    top_words = [feature_names[j] for j in top_idx]
    print(f"  Cluster {k}: {', '.join(top_words)}")

low_idx = np.argsort(certainty)[:10]
print("\nChapters with Lowest Cluster Certainty:")
for i in low_idx:
    print(f"  Chapter {i}: certainty={certainty[i]:.3f}")

pd.DataFrame({
    "chapter_id": np.arange(len(certainty)),
    "hard_label": hard_labels,
    "certainty": certainty
}).to_csv(os.path.join(result_dir, "gmm_chapter_certainty.csv"), index=False)
