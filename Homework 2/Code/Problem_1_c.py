'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 21:47:48
LastEditTime: 2025-10-29 21:56:01
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_c.py
Description: 
    Fit a Gaussian mixture model to the author data.
'''
import os
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

from Problem_1_b import load_authors_counts_rda

rda_path = os.path.join("Homework 2", "Code", "Data", "authors.rda")
X_full, y_true, feature_names = load_authors_counts_rda(rda_path)
X = X_full.copy()

# Verify data loaded
result_dir = os.path.join("Homework 2", "Code", "Result")
poisson_labels = pd.read_csv(os.path.join(result_dir, "poisson_labels.csv"))["hard_label"].to_numpy()
poisson_pi = pd.read_csv(os.path.join(result_dir, "poisson_pi.csv"))["pi_k"].to_numpy()

# Fit Gaussian Mixture Model
K = 4
X_scaled = StandardScaler().fit_transform(X)

gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=25)
gmm.fit(X_scaled)

gmm_labels = gmm.predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)

print("\nGaussian Mixture converged in", gmm.n_iter_, "iterations")
print("GMM mixture weights:", np.round(gmm.weights_, 4))

# Save GMM results
gmm_dir = os.path.join(result_dir)
os.makedirs(gmm_dir, exist_ok=True)
pd.DataFrame(gmm.means_, columns=feature_names).to_csv(os.path.join(gmm_dir, "gmm_means.csv"), index_label="cluster")
pd.DataFrame(gmm.covariances_, columns=feature_names).to_csv(os.path.join(gmm_dir, "gmm_cov.csv"), index_label="cluster")
pd.DataFrame(gmm_probs, columns=[f"cluster_{k}" for k in range(K)]).to_csv(os.path.join(gmm_dir, "gmm_probs.csv"), index=False)
pd.DataFrame({"hard_label": gmm_labels}).to_csv(os.path.join(gmm_dir, "gmm_labels.csv"), index=False)
print(f"GMM results saved in: {gmm_dir}")

# Comparison between Poisson and Gaussian mixture results
print("\n=== Comparison Between Poisson and Gaussian Mixtures ===")
print("Mixture weights:")
for k in range(K):
    print(f"  Cluster {k}: Poisson={poisson_pi[k]:.4f} | Gaussian={gmm.weights_[k]:.4f}")

# Alignment between cluster assignments (Poisson vs GMM)
conf = confusion_matrix(poisson_labels, gmm_labels)
print("\nConfusion matrix (Poisson vs GMM hard labels):\n", conf)

# Validation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_true)

ari_poisson = adjusted_rand_score(y_encoded, poisson_labels)
ari_gmm = adjusted_rand_score(y_encoded, gmm_labels)
nmi_poisson = normalized_mutual_info_score(y_encoded, poisson_labels)
nmi_gmm = normalized_mutual_info_score(y_encoded, gmm_labels)

print("\n=== External Validation Using Hidden Author Labels ===")
print(f"Poisson mixture : ARI={ari_poisson:.4f}, NMI={nmi_poisson:.4f}")
print(f"Gaussian mixture: ARI={ari_gmm:.4f}, NMI={nmi_gmm:.4f}")

metrics_df = pd.DataFrame({
    "Model": ["Poisson", "Gaussian"],
    "ARI": [ari_poisson, ari_gmm],
    "NMI": [nmi_poisson, nmi_gmm],
})
metrics_df.to_csv(os.path.join(result_dir, "model_comparison_metrics.csv"), index=False)
print(f"✅ Model comparison metrics saved in: {result_dir}")
