'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 17:59:43
LastEditTime: 2025-10-30 14:06:41
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_b.py
Description: 
    EM algorithm for a mixture of Poisson distributions and fit to the author data with K = 4 clusters.
'''
import os
import json
import numpy as np
import pandas as pd

DATA_PATH = r"Homework 2\Code\Data\authors.csv"
RESULT_DIR = r"Homework 2\Code\Result"
os.makedirs(RESULT_DIR, exist_ok=True)

def load_author_data(path):
    df = pd.read_csv(path)
    cols = list(df.columns)

    author_col = "" if "" in cols else cols[0]

    # Drop BookID column if exists
    df = df.drop(columns=cols[-1])

    y_names, y = np.unique(df[author_col].astype(str).values, return_inverse=True)
    feat_cols = [c for c in df.columns if c != author_col]
    X = df[feat_cols].to_numpy(dtype=float)

    return X, y, y_names, feat_cols 

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

# Main function
if __name__ == "__main__":
    X, y, y_names, feat_cols = load_author_data(DATA_PATH)

    K = len(y_names)
    res = poisson_mixture_em(X, K=K, tol=1e-6, max_iter=1000, random_state=0, verbose=False)
    hard = poisson_mixture_predict(X, res)

    cont = np.zeros((K, K), dtype=int)
    for yi, hi in zip(y, hard):
        cont[yi, hi] += 1
    purity = float(cont.max(axis=0).sum() / len(y))
    cluster_to_author = {int(k): str(y_names[int(np.argmax(cont[:, k]))]) for k in range(K)}

    threshold = 0.6
    max_resp = res.gamma.max(axis=1)
    low_certainty_idx = np.where(max_resp < threshold)[0].tolist()

    low_certainty_info = []
    for i in low_certainty_idx:
        low_certainty_info.append({
            "chapter_index": int(i),
            "author_true": str(y_names[y[i]]),
            "pred_cluster": int(hard[i]),
            "max_gamma": float(max_resp[i]),
            "responsibilities": [float(v) for v in res.gamma[i]]
        })

    out_path = os.path.join(RESULT_DIR, f"poisson_mixture_result.json")
    result_json = {
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "final_loglik": float(res.loglik_hist[-1]),
        "n_authors": int(K),
        "authors": y_names.tolist(),
        "feature_columns": feat_cols,
        "em": {
            "pi": res.pi.astype(float).tolist(),
            "lambda": res.lmbda.astype(float).tolist(),
            "loglik_hist": [float(v) for v in res.loglik_hist]
        },
        "predictions": {
            "cluster_pred": [int(v) for v in hard],
            "responsibilities": res.gamma.astype(float).tolist()
        },
        "validation": {
            "purity": float(purity),
            "contingency": cont.astype(int).tolist(),
            "cluster_sizes": np.bincount(hard, minlength=K).astype(int).tolist(),
            "cluster_to_author_map": {str(k): v for k, v in cluster_to_author.items()},
            "low_certainty_chapters": low_certainty_info,
            "threshold": float(threshold)
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)