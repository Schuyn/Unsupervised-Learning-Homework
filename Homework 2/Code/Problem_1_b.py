'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 17:59:43
LastEditTime: 2025-10-30 13:20:28
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_b.py
Description: 
    EM algorithm for a mixture of Poisson distributions and fit to the author data with K = 4 clusters.
'''
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH = r"Homework 2\Code\Data\authors.csv"
RESULT_DIR = r"Homework 2\Code\Result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================= EM for Poisson Mixture ======================
class PoissonMixtureResult:
    __slots__ = ("pi", "lmbda", "gamma", "loglik_hist", "converged", "n_iter")
    def __init__(self, pi, lmbda, gamma, loglik_hist, converged, n_iter):
        self.pi = pi
        self.lmbda = lmbda
        self.gamma = gamma
        self.loglik_hist = loglik_hist
        self.converged = converged
        self.n_iter = n_iter

def _softmax_logspace(logW, axis=1):
    m = np.max(logW, axis=axis, keepdims=True)
    W = np.exp(logW - m)
    W /= W.sum(axis=axis, keepdims=True)
    return W

def _logsumexp(A, axis=1):
    m = np.max(A, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

def _init_params(X, K, random_state=None):
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    gamma = rng.dirichlet(alpha=np.ones(K), size=n)
    Nk = gamma.sum(axis=0) + 1e-16
    pi = Nk / Nk.sum()
    lmbda = (gamma.T @ X) / Nk[:, None]
    lmbda = np.clip(lmbda, 1e-8, None)
    return pi, lmbda, gamma

def poisson_mixture_em(X, K, tol=1e-6, max_iter=500, random_state=None, verbose=False):
    X = np.asarray(X, dtype=float)
    if np.any(X < 0) or (np.abs(X - np.round(X)) > 1e-10).any():
        raise ValueError("X must be nonnegative integer counts.")
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    pi, lmbda, gamma = _init_params(X, K, random_state=rng)
    loglik_hist = []
    eps = 1e-12

    for it in range(1, max_iter + 1):
        log_pi = np.log(np.clip(pi, eps, 1.0))
        log_lambda = np.log(np.clip(lmbda, eps, None))
        log_px_given_k = X @ log_lambda.T - np.sum(lmbda, axis=1)[None, :]
        log_w = log_pi[None, :] + log_px_given_k
        gamma = _softmax_logspace(log_w, axis=1)

        Nk = gamma.sum(axis=0) + eps
        pi = Nk / n
        lmbda = (gamma.T @ X) / Nk[:, None]
        lmbda = np.clip(lmbda, 1e-12, None)

        ll_vec = _logsumexp(log_w, axis=1)
        ll = float(ll_vec.sum())
        loglik_hist.append(ll)

        if it > 1:
            ll_prev = loglik_hist[-2]
            denom = max(1.0, abs(ll_prev))
            if (ll - ll_prev) / denom < tol:
                return PoissonMixtureResult(pi, lmbda, gamma, loglik_hist, True, it)

    return PoissonMixtureResult(pi, lmbda, gamma, loglik_hist, False, max_iter)

def poisson_mixture_predict_proba(X, result):
    X = np.asarray(X, dtype=float)
    log_pi = np.log(np.clip(result.pi, 1e-12, 1.0))
    log_lambda = np.log(np.clip(result.lmbda, 1e-12, None))
    log_px_given_k = X @ log_lambda.T - np.sum(result.lmbda, axis=1)[None, :]
    log_w = log_pi[None, :] + log_px_given_k
    return _softmax_logspace(log_w, axis=1)

def poisson_mixture_predict(X, result):
    return np.argmax(poisson_mixture_predict_proba(X, result), axis=1)

# ======================= Data Load & Validation ======================
df = pd.read_csv(DATA_PATH)

# author column handling: first column is author, header may be "" or parsed as-is
cols = list(df.columns)
if len(cols) == 0:
    raise ValueError("CSV appears to have no columns.")
# Prefer an explicit empty-string header if present; else fall back to first column
if "" in cols:
    author_col = ""
else:
    author_col = cols[0]

# identify BookID column (last column named "BookID" per spec)
bookid_candidates = [c for c in cols if c.strip().lower() in {"bookid", "book_id"}]
bookid_col = bookid_candidates[-1] if bookid_candidates else cols[-1]  # fallback to last col

# Extract labels (authors) for validation, but DO NOT use in training
y_names, y = np.unique(df[author_col].astype(str).values, return_inverse=True)

# Feature columns = all columns except author + BookID
feat_cols = [c for c in cols if c not in {author_col, bookid_col}]
X = df[feat_cols].to_numpy(dtype=float)

# ======================= Fit & Validate ==============================
K = len(y_names)
res = poisson_mixture_em(X, K=K, tol=1e-6, max_iter=1000, random_state=0, verbose=False)
hard = poisson_mixture_predict(X, res)

# Contingency table (true authors x predicted clusters)
cont = np.zeros((K, K), dtype=int)
for yi, hi in zip(y, hard):
    cont[yi, hi] += 1

# Purity metric
cluster_max = cont.max(axis=0)
purity = float(cluster_max.sum() / len(y))

# Cluster -> dominant author mapping
cluster_to_author = {}
for k in range(K):
    col = cont[:, k]
    j = int(np.argmax(col))
    cluster_to_author[k] = y_names[j]

# ======================= Save Results ================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# parameters
np.save(os.path.join(RESULT_DIR, f"pi_{ts}.npy"), res.pi)
np.save(os.path.join(RESULT_DIR, f"lambda_{ts}.npy"), res.lmbda)

# responsibilities
gamma_df = pd.DataFrame(res.gamma, columns=[f"gamma_{k}" for k in range(K)])
gamma_df.to_csv(os.path.join(RESULT_DIR, f"responsibilities_{ts}.csv"), index=False)

# hard labels + ground truth
pred_df = pd.DataFrame({
    "author_true": y_names[y],
    "cluster_pred": hard
})
pred_df.to_csv(os.path.join(RESULT_DIR, f"predictions_{ts}.csv"), index=False)

# contingency table
cont_df = pd.DataFrame(cont, index=[f"author={a}" for a in y_names], columns=[f"cluster={k}" for k in range(K)])
cont_df.to_csv(os.path.join(RESULT_DIR, f"contingency_{ts}.csv"))

# metrics + mapping
metrics = {
    "converged": bool(res.converged),
    "n_iter": int(res.n_iter),
    "final_loglik": float(res.loglik_hist[-1]),
    "purity": purity,
    "n_authors": int(K),
    "authors": y_names.tolist(),
    "feature_columns": feat_cols,
    "cluster_sizes": np.bincount(hard, minlength=K).astype(int).tolist(),
    "cluster_to_author_map": {str(k): str(v) for k, v in cluster_to_author.items()},
}
with open(os.path.join(RESULT_DIR, f"metrics_{ts}.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
