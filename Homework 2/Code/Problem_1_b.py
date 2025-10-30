'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 17:59:43
LastEditTime: 2025-10-29 21:45:41
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_b.py
Description: 
    Write a python function implementing your EM algorithm for a mixture of Poisson distributions.
'''
import numpy as np
from scipy.special import logsumexp, gammaln
from sklearn.cluster import KMeans

# EM for Mixture of Poissons
def em_poisson_mixture(
    X,
    K,
    max_iter=300,
    tol=1e-6,
    init="kmeans",
    random_state=42,
    verbose=False,
    lambda_floor=1e-8,
):
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if init == "kmeans":
        # KMeans on raw counts (or small jitter to break ties)
        km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        pi = np.bincount(labels, minlength=K) / n
        Lambda = np.zeros((K, p), dtype=np.float64)
        for k in range(K):
            wk = (labels == k)
            if wk.any():
                Lambda[k] = X[wk].mean(axis=0)
            else:
                # empty cluster: random small positive rates
                Lambda[k] = X.mean(axis=0) * rng.uniform(0.5, 1.5, size=p)
        Lambda = np.clip(Lambda, lambda_floor, None)
    elif init == "random":
        pi = rng.dirichlet(np.ones(K))
        # random positive rates around global mean
        gmean = X.mean(axis=0) + lambda_floor
        Lambda = gmean * rng.uniform(0.5, 1.5, size=(K, p))
        Lambda = np.clip(Lambda, lambda_floor, None)
    elif isinstance(init, dict) and "pi" in init and "Lambda" in init:
        pi = np.asarray(init["pi"], dtype=np.float64)
        Lambda = np.asarray(init["Lambda"], dtype=np.float64)
        assert pi.shape == (K,)
        assert Lambda.shape == (K, p)
        pi = np.clip(pi, 1e-12, None)
        pi = pi / pi.sum()
        Lambda = np.clip(Lambda, lambda_floor, None)
    else:
        raise ValueError("Invalid init. Use 'kmeans', 'random', or {'pi':..., 'Lambda':...}.")

    def obs_loglik(pi, Lambda):
        logp = np.zeros(n)
        for i in range(n):
            # shape (K,)
            lp_k = np.log(pi) + (X[i] * np.log(Lambda) - Lambda).sum(axis=1)
            logp[i] = logsumexp(lp_k)
        return logp.sum()

    ll_hist = []
    prev_ll = -np.inf

    for it in range(1, max_iter + 1):
        log_pi = np.log(pi + 1e-300)
        log_rate = np.log(Lambda)
        
        # E-step: compute responsibilities gamma_{ik} = P(Z_i=k | X_i)
        log_p_x_given_k = X @ log_rate.T - Lambda.sum(axis=1)  # (n, K)
        log_w = log_p_x_given_k + log_pi  # (n, K)
        # normalize to get gamma
        log_norm = logsumexp(log_w, axis=1, keepdims=True)      # (n, 1)
        gamma = np.exp(log_w - log_norm)                        # (n, K)

        # M-step: update pi, Lambda
        Nk = gamma.sum(axis=0) + 1e-300                         # (K,)
        pi = Nk / n

        # lambda_{kj} = sum_i gamma_{ik} x_{ij} / sum_i gamma_{ik}
        # numerator: (K, p)
        Lambda = (gamma.T @ X) / Nk[:, None]
        Lambda = np.clip(Lambda, lambda_floor, None)

        # monitor log-likelihood
        ll = obs_loglik(pi, Lambda)
        ll_hist.append(ll)
        if verbose:
            print(f"Iter {it:3d}: loglik = {ll:.6f}")

        # convergence check
        if it > 1:
            rel_impr = (ll_hist[-1] - ll_hist[-2]) / (abs(ll_hist[-2]) + 1e-12)
            if rel_impr < tol:
                if verbose:
                    print(f"Converged at iter {it} (relative improvement {rel_impr:.3e}).")
                break
        prev_ll = ll

    return pi, Lambda, gamma, ll_hist


def load_authors_counts_rda(path_rda, drop_cols=("Book.ID", "BookID", "book_id", "bookID", "Book ID")):
    import pyreadr

    res = pyreadr.read_r(path_rda)
    # take the first object
    obj = next(iter(res.values()))
    # Convert to pandas DataFrame if it's not already
    import pandas as pd
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)

    df = obj.copy()

    # Identify label column (common names)
    label_col_candidates = ["Author", "author", "label", "Label", "AUTHORS"]
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break

    # Drop non-count columns (book id etc.)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Separate labels if present
    y = None
    if label_col is not None and label_col in df.columns:
        y = df[label_col].to_numpy()
        df = df.drop(columns=[label_col])

    # Remaining should be counts
    X = df.to_numpy(dtype=np.float64)
    feature_names = list(df.columns)
    return X, y, feature_names

# Run on authors.rda (K=4)
if __name__ == "__main__":
    import os
    import pandas as pd

    rda_path = os.path.join("Homework 2", "Code", "Data", "authors.rda")
    X, y, feature_names = load_authors_counts_rda(rda_path)

    # Fit Poisson mixture (K=4 as in the homework)
    K = 4
    pi, Lambda, gamma, ll_hist = em_poisson_mixture(
        X, K=K, max_iter=1000, tol=1e-7, init="kmeans", verbose=True, random_state=25
    )

    # Soft labels and hard assignments
    soft = gamma
    hard = soft.argmax(axis=1)

    # Print brief summary
    print("\nMixture weights (pi):")
    print(np.round(pi, 6))
    print("\nTop words per cluster (by highest lambda):")
    top_m = 10
    for k in range(K):
        idx = np.argsort(-Lambda[k])[:top_m]
        words = [feature_names[j] for j in idx]
        print(f"Cluster {k}: {words}")
        
    result_dir = os.path.join("Homework 2", "Code", "Result")
    os.makedirs(result_dir, exist_ok=True)
    pi_path = os.path.join(result_dir, "poisson_pi.csv")
    pd.DataFrame({"pi_k": pi}).to_csv(pi_path, index=False)
    lambda_path = os.path.join(result_dir, "poisson_lambda.csv")
    pd.DataFrame(Lambda, columns=feature_names).to_csv(lambda_path, index_label="cluster")
    
    gamma_path = os.path.join(result_dir, "poisson_gamma.csv")
    pd.DataFrame(gamma, columns=[f"cluster_{k}" for k in range(gamma.shape[1])]).to_csv(gamma_path, index=False)

    labels_path = os.path.join(result_dir, "poisson_labels.csv")
    pd.DataFrame({"hard_label": hard}).to_csv(labels_path, index=False)
    
    print(f"All Poisson mixture results saved in: {result_dir}")