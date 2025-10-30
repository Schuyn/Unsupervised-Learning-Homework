'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 21:47:48
LastEditTime: 2025-10-30 14:15:24
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_c.py
Description: 
    Fit a Gaussian mixture model to the author data.
'''
import os
import json
import numpy as np
import pandas as pd

DATA_PATH = r"Homework 2\Code\Data\authors.csv"
RESULT_DIR = r"Homework 2\Code\Result"
os.makedirs(RESULT_DIR, exist_ok=True)

from Problem_1_b import load_author_data

class GMMResult:
    __slots__ = ("pi", "mu", "var", "gamma", "loglik_hist", "converged", "n_iter")
    def __init__(self, pi, mu, var, gamma, loglik_hist, converged, n_iter):
        self.pi = pi          # (K,)
        self.mu = mu          # (K, p)
        self.var = var        # (K, p)  diagonal covariances
        self.gamma = gamma    # (n, K)
        self.loglik_hist = loglik_hist
        self.converged = converged
        self.n_iter = n_iter

def _log_gauss_diag(X, mu, var):
    eps = 1e-10
    var = np.clip(var, eps, None)
    n, p = X.shape
    K = mu.shape[0]
    log_det = np.sum(np.log(var), axis=1)
    inv_var = 1.0 / var
    quad = ((X[:, None, :] - mu[None, :, :]) ** 2 * inv_var[None, :, :]).sum(axis=2)
    return -0.5 * (p * np.log(2.0 * np.pi) + log_det[None, :] + quad)

def _softmax_logspace(logW, axis=1):
    m = np.max(logW, axis=axis, keepdims=True)
    W = np.exp(logW - m)
    W /= W.sum(axis=axis, keepdims=True)
    return W

def _logsumexp(A, axis=1):
    m = np.max(A, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

def _init_gmm_params(X, K, random_state=None):
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    mu = np.empty((K, p))
    idx0 = rng.integers(0, n)
    mu[0] = X[idx0]
    d2 = np.sum((X - mu[0])**2, axis=1) + 1e-12
    for k in range(1, K):
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        mu[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X - mu[k])**2, axis=1) + 1e-12)
    dist2 = ((X[:, None, :] - mu[None, :, :])**2).sum(axis=2)
    gamma = np.zeros((n, K))
    gamma[np.arange(n), np.argmin(dist2, axis=1)] = 1.0
    Nk = gamma.sum(axis=0) + 1e-12
    pi = Nk / n
    var = (gamma.T @ (X**2)) / Nk[:, None] - ( (gamma.T @ X) / Nk[:, None] )**2
    var = np.clip(var, 1e-6, None)
    return pi, mu, var, gamma

def gaussian_mixture_em(
    X, K, tol=1e-6, max_iter=500, random_state=None, verbose=False
):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    eps = 1e-12

    pi, mu, var, gamma = _init_gmm_params(X, K, random_state=random_state)
    loglik_hist = []

    for it in range(1, max_iter + 1):
        # E-step
        log_pi = np.log(np.clip(pi, eps, 1.0))  # (K,)
        log_px_given_k = _log_gauss_diag(X, mu, var)  # (n,K)
        log_w = log_pi[None, :] + log_px_given_k
        gamma = _softmax_logspace(log_w, axis=1)      # (n,K)

        # M-step
        Nk = gamma.sum(axis=0) + eps                  # (K,)
        pi = Nk / n
        mu = (gamma.T @ X) / Nk[:, None]              # (K,p)
        # diagonal covariance
        Ex2 = (gamma.T @ (X**2)) / Nk[:, None]        # (K,p)
        var = Ex2 - mu**2
        var = np.clip(var, 1e-6, None)

        # Log-likelihood
        ll = float(_logsumexp(log_w, axis=1).sum())
        loglik_hist.append(ll)

        if verbose and (it == 1 or it % 10 == 0):
            print(f"Iter {it:4d}  loglik={ll:.6f}")

        if it > 1:
            ll_prev = loglik_hist[-2]
            denom = max(1.0, abs(ll_prev))
            if (ll - ll_prev) / denom < tol:
                if verbose:
                    print(f"Converged at iter {it}  Δrel={(ll-ll_prev)/denom:.3e}")
                return GMMResult(pi, mu, var, gamma, loglik_hist, True, it)

    if verbose:
        print("Reached max_iter without convergence.")
    return GMMResult(pi, mu, var, gamma, loglik_hist, False, max_iter)

def gmm_predict_proba(X, result):
    log_pi = np.log(np.clip(result.pi, 1e-12, 1.0))
    log_px_given_k = _log_gauss_diag(X, result.mu, result.var)
    log_w = log_pi[None, :] + log_px_given_k
    return _softmax_logspace(log_w, axis=1)

def gmm_predict(X, result):
    return np.argmax(gmm_predict_proba(X, result), axis=1)

# Main function
if __name__ == "__main__":
    X, y, y_names, feat_cols = load_author_data(DATA_PATH)
    K = len(y_names)

    res = gaussian_mixture_em(
        X, K=K, tol=1e-6, max_iter=1000, random_state=0, verbose=False
    )
    hard = gmm_predict(X, res)

        # 3) Validation（使用 author 标签，仅用于评估）
    cont = np.zeros((K, K), dtype=int)
    for yi, hi in zip(y, hard):
        cont[yi, hi] += 1
    purity = float(cont.max(axis=0).sum() / len(y))
    cluster_to_author = {
        str(k): str(y_names[int(np.argmax(cont[:, k]))]) for k in range(K)
    }

    # 3.1) Identify low-certainty chapters
    threshold = 0.6
    max_resp = res.gamma.max(axis=1)
    low_idx = np.where(max_resp < threshold)[0].tolist()
    low_certainty_info = []
    for i in low_idx:
        low_certainty_info.append({
            "chapter_index": int(i),
            "author_true": str(y_names[y[i]]),
            "pred_cluster": int(hard[i]),
            "max_gamma": float(max_resp[i]),
            "responsibilities": [float(v) for v in res.gamma[i]]
        })

    # 4) Save all results into JSON
    out_path = os.path.join(RESULT_DIR, f"gmm_result.json")

    result_json = {
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "final_loglik": float(res.loglik_hist[-1]),
        "n_authors": int(K),
        "authors": y_names.tolist(),
        "feature_columns": feat_cols,
        "em": {
            "pi": res.pi.astype(float).tolist(),
            "mu": res.mu.astype(float).tolist(),
            "var_diag": res.var.astype(float).tolist(),
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

