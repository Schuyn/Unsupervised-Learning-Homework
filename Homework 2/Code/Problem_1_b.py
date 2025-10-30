'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-10-29 17:59:43
LastEditTime: 2025-10-30 12:47:18
FilePath: /Unsupervised-Learning-Homework/Homework 2/Code/Problem_1_b.py
Description: 
    EM algorithm for a mixture of Poisson distributions and fit to the author data with K = 4 clusters.
'''

import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

# def load_authors_counts_rda(path_rda):
#     import pyreadr

#     res = pyreadr.read_r(path_rda)
#     obj = next(iter(res.values()))
#     if not isinstance(obj, pd.DataFrame):
#         obj = pd.DataFrame(obj)
#     df = obj.copy()

#     # Detect Author and Book ID columns
#     label_col = next((c for c in ["Author", "author", "AUTHORS"] if c in df.columns), None)
#     bookid_col = next((c for c in ["Book.ID", "BookID", "book_id", "Book ID"] if c in df.columns), None)

#     if bookid_col:
#         df = df.drop(columns=[bookid_col])

#     y = None
#     if label_col:
#         y_raw = df[label_col]
#         df = df.drop(columns=[label_col])

#         # robust 1D conversion
#         if isinstance(y_raw, pd.Series):
#             y = y_raw.to_numpy()
#         elif isinstance(y_raw, pd.DataFrame):
#             y = y_raw.iloc[:, 0].to_numpy()
#         else:
#             y = np.array(y_raw)

#         # flatten & cast to str (avoid categorical weirdness)
#         y = np.ravel(y).astype(str)

#     X = df.to_numpy(dtype=np.float64)
#     feature_names = list(df.columns)
    
#     if y is not None and y.shape[0] != X.shape[0]:
#         # handle rare case: y is nested object of length 1 containing the vector
#         if y.shape[0] == 1 and hasattr(y, "item"):
#             maybe = y.item()
#             if isinstance(maybe, (list, np.ndarray)):
#                 y = np.ravel(np.array(maybe)).astype(str)
                
#     return X, y, feature_names


def load_authors_counts_rda(path_rda):
    import pyreadr
    import re

    # ---- read R object ----
    res = pyreadr.read_r(path_rda)
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    df = obj.copy()

    # standardize column names for matching (keep original for output)
    def norm(s): 
        return re.sub(r'\s+', '', str(s)).lower()

    cols_norm = {c: norm(c) for c in df.columns}
    inv_norm = {v: k for k, v in cols_norm.items()}

    # if author is in index, pull it out
    idx_name = df.index.name
    if idx_name and norm(idx_name) in {"author","authors","label"}:
        df = df.reset_index()

    # detect author column (case/space-insensitive)
    author_keys = ["author","authors","label"]
    author_col = None
    for k in author_keys:
        if k in cols_norm.values():
            author_col = inv_norm[k]; break

    # detect and drop book id column(s)
    bookid_keys = {"book.id","bookid","book_id","book id"}
    for k in bookid_keys:
        if k in cols_norm.values():
            df.drop(columns=[inv_norm[k]], inplace=True, errors="ignore")

    # extract y robustly
    y = None
    if author_col is not None and author_col in df.columns:
        y_raw = df[author_col]
        # drop author column from feature frame
        df = df.drop(columns=[author_col])

        # robust conversion to 1D numpy array of str
        if isinstance(y_raw, pd.Series):
            y = y_raw.to_numpy()
        elif isinstance(y_raw, pd.DataFrame):
            y = y_raw.iloc[:, 0].to_numpy()
        else:
            y = np.array(y_raw)

        # unwrap nested object like array([list([...])], dtype=object)
        # keep unwrapping until y is 1D with length == n_rows
        max_unwrap = 5
        unwrap_cnt = 0
        while True:
            y = np.ravel(y)
            if y.dtype == object and y.size == 1 and hasattr(y[0], "__iter__") and not isinstance(y[0], (str, bytes)):
                y = np.array(list(y[0]))
                unwrap_cnt += 1
                if unwrap_cnt > max_unwrap:
                    break
            else:
                break

        y = np.ravel(y).astype(str)
    else:
        # no author column; leave y=None
        y = None

    # keep only numeric (counts) columns; coerce non-numeric to NaN->0
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)

    X = df.to_numpy(dtype=np.float64)
    feature_names = list(df.columns)

    # final sanity check
    if y is not None and y.shape[0] != X.shape[0]:
        raise ValueError(f"[load_authors_counts_rda] label length {y.shape[0]} != X rows {X.shape[0]}")

    return X, y, feature_names


# EM Algorithm for Poisson Mixture
def em_poisson_mixture(
    X, K, 
    max_iter=300, 
    tol=1e-6, 
    init="kmeans", 
    random_state=25, 
    verbose=False
):  
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    if init == "kmeans":
        km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        pi = np.bincount(labels, minlength=K) / n
        Lambda = np.zeros((K, p))
        for k in range(K):
            wk = (labels == k)
            Lambda[k] = X[wk].mean(axis=0) if wk.any() else rng.uniform(0.5, 1.5, size=p)
    else:
        pi = rng.dirichlet(np.ones(K))
        Lambda = np.maximum(rng.random((K, p)) * X.mean(axis=0), 1e-6)

    def loglik(pi, Lambda):
        logp = np.zeros(n)
        for i in range(n):
            logp[i] = logsumexp(np.log(pi) + (X[i] * np.log(Lambda) - Lambda).sum(axis=1))
        return logp.sum()

    ll_hist = []
    for it in range(max_iter):
        # E-step
        log_pi = np.log(pi + 1e-12)
        log_rate = np.log(Lambda + 1e-12)
        log_w = X @ log_rate.T - Lambda.sum(axis=1) + log_pi
        log_norm = logsumexp(log_w, axis=1, keepdims=True)
        gamma = np.exp(log_w - log_norm)

        # M-step
        Nk = gamma.sum(axis=0)
        pi = Nk / n
        Lambda = (gamma.T @ X) / (Nk[:, None] + 1e-12)

        ll = loglik(pi, Lambda)
        ll_hist.append(ll)
        if it > 0 and abs(ll - ll_hist[-2]) < tol * abs(ll_hist[-2]):
            if verbose:
                print(f"Converged at iter {it+1}, loglik={ll:.3f}")
            break
        if verbose:
            print(f"Iter {it+1:3d}: loglik={ll:.6f}")

    return pi, Lambda, gamma, ll_hist

if __name__ == "__main__":
    rda_path = os.path.join("Homework 2", "Code", "Data", "authors.rda")
    X, y_true, feature_names = load_authors_counts_rda(rda_path)
    print(f"Loaded data: {X.shape[0]} chapters × {X.shape[1]} stop words")

    K = 4
    pi, Lambda, gamma, ll_hist = em_poisson_mixture(X, K=K, max_iter=500, tol=1e-7, verbose=True)

    hard_labels = gamma.argmax(axis=1)
    certainty = gamma.max(axis=1)
    uncertainty = 1 - certainty

    result_dir = os.path.join("Homework 2", "Code", "Result")
    os.makedirs(result_dir, exist_ok=True)

    pd.DataFrame({"pi_k": pi}).to_csv(os.path.join(result_dir, "poisson_pi.csv"), index=False)
    pd.DataFrame(Lambda, columns=feature_names).to_csv(os.path.join(result_dir, "poisson_lambda.csv"), index_label="cluster")
    pd.DataFrame(gamma, columns=[f"cluster_{k}" for k in range(K)]).to_csv(os.path.join(result_dir, "poisson_gamma.csv"), index=False)
    pd.DataFrame({"hard_label": hard_labels, "certainty": certainty}).to_csv(os.path.join(result_dir, "poisson_labels.csv"), index=False)

    top_m = 10
    print("\nMixture Weights (π_k):")
    for k, val in enumerate(pi):
        print(f"  Cluster {k}: {val:.4f}")

    print("\nTop Words per Cluster:")
    for k in range(K):
        top_idx = np.argsort(-Lambda[k])[:top_m]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  Cluster {k}: {', '.join(top_words)}")

    # Low-certainty chapters
    low_idx = np.argsort(certainty)[:10]
    print("\nChapters with Lowest Cluster Certainty:")
    for i in low_idx:
        print(f"  Chapter {i}: certainty={certainty[i]:.3f}")

    # Validation
    y_vec = np.ravel(y_true).astype(str)
    assert y_vec.shape[0] == X.shape[0], f"Label length {y_vec.shape[0]} != X rows {X.shape[0]}"
    le = LabelEncoder()
    y_enc = le.fit_transform(y_vec)
    ari = adjusted_rand_score(y_enc, hard_labels)
    nmi = normalized_mutual_info_score(y_enc, hard_labels)

    print("\n=== External Validation (Author labels only for evaluation) ===")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    out_csv = os.path.join(result_dir, "poisson_validation.csv")  # 在 1c 改为 "gmm_validation.csv"
    model_name = "Poisson"  # 在 1c 改为 "Gaussian"
    pd.DataFrame({"Model":[model_name], "ARI":[ari], "NMI":[nmi]}).to_csv(out_csv, index=False)
    
    
    # Save interpretation tables
    pd.DataFrame({
        "chapter_id": np.arange(len(certainty)),
        "hard_label": hard_labels,
        "certainty": certainty
    }).to_csv(os.path.join(result_dir, "poisson_chapter_certainty.csv"), index=False)
