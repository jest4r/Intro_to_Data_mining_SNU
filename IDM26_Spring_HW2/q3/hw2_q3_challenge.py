import sys
import math
from collections import defaultdict

import pandas as pd
from sklearn.decomposition import TruncatedSVD

# --- Tunable settings for the ensemble ---
MIN_OVERLAP = 5
TOPK_USERS = 30
TOPK_ITEMS = 30
EPS = 1e-12

# Latent factor model (implicit MF via TruncatedSVD)
SVD_N_COMPONENTS = 64
SVD_RANDOM_STATE = 42

# Mix weights (sum to 1.0)
W_USER = 0.30
W_ITEM = 0.30
W_POP = 0.10
W_SVD = 0.30


def cosine_similarity_from_dicts(vec_a, vec_b):
    """Cosine similarity computed ONLY on common keys."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for k in common:
        a = vec_a[k]
        b = vec_b[k]
        dot += a * b
        na2 += a * a
        nb2 += b * b
    denom = (math.sqrt(na2) + EPS) * (math.sqrt(nb2) + EPS)
    return dot / denom


def load_ratings(ratings_path):
    user_ratings = defaultdict(dict)
    item_ratings = defaultdict(dict)

    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in f:
            user_id_str, movie_id_str, rating_str, _timestamp_str = line.strip().split(",")
            u = int(user_id_str)
            i = int(movie_id_str)
            r = float(rating_str)
            user_ratings[u][i] = r
            item_ratings[i][u] = r

    user_means = {}
    for u, d in user_ratings.items():
        user_means[u] = (sum(d.values()) / len(d)) if d else 0.0

    item_means = {}
    for i, d in item_ratings.items():
        item_means[i] = (sum(d.values()) / len(d)) if d else 0.0

    # Simple popularity prior: number of ratings per item.
    item_counts = {i: len(d) for i, d in item_ratings.items()}

    return user_ratings, item_ratings, user_means, item_means, item_counts


def mean_center_by_user(user_ratings, user_means):
    centered = {}
    for u, d in user_ratings.items():
        mu = user_means.get(u, 0.0)
        centered[u] = {i: (r - mu) for i, r in d.items()}
    return centered


def transpose_user_to_item(user_dict):
    item_dict = defaultdict(dict)
    for u, d in user_dict.items():
        for i, v in d.items():
            item_dict[i][u] = v
    return item_dict


def get_overlap_count(dict_a, dict_b):
    return len(set(dict_a.keys()) & set(dict_b.keys()))


def _sigmoid(x):
    # Stable-ish sigmoid for scoring.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_user_based(u, i, user_ratings, centered_user_ratings, user_means):
    target_vec = centered_user_ratings.get(u, {})
    mean_u = user_means.get(u, 0.0)

    sims = []
    for v, v_ratings in user_ratings.items():
        if v == u or i not in v_ratings:
            continue
        v_vec = centered_user_ratings.get(v, {})
        if get_overlap_count(target_vec, v_vec) < MIN_OVERLAP:
            continue
        s = cosine_similarity_from_dicts(target_vec, v_vec)
        sims.append((s, v))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:TOPK_USERS]

    numer = 0.0
    denom = 0.0
    for s, v in sims:
        numer += s * (user_ratings[v][i] - user_means.get(v, 0.0))
        denom += abs(s)

    if denom <= EPS:
        return mean_u
    return mean_u + numer / denom


def predict_item_based(u, i, user_ratings, centered_item_ratings, item_means):
    mean_i = item_means.get(i, 0.0)
    i_vec = centered_item_ratings.get(i, {})

    rated_by_u = user_ratings.get(u, {})
    sims = []
    for j in rated_by_u.keys():
        if j == i:
            continue
        j_vec = centered_item_ratings.get(j, {})
        if get_overlap_count(i_vec, j_vec) < MIN_OVERLAP:
            continue
        s = cosine_similarity_from_dicts(i_vec, j_vec)
        sims.append((s, j))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:TOPK_ITEMS]

    numer = 0.0
    denom = 0.0
    for s, j in sims:
        numer += s * (rated_by_u[j] - item_means.get(j, 0.0))
        denom += abs(s)

    if denom <= EPS:
        return mean_i
    return mean_i + numer / denom


def build_svd_model(train_path, n_components=SVD_N_COMPONENTS, random_state=SVD_RANDOM_STATE):
    """Build implicit interaction SVD model using pandas + scikit-learn only.

    Returns:
      - svd: fitted TruncatedSVD
      - user_to_row, item_to_col: id->index mappings
      - user_factors (n_users,k)
      - item_factors (n_items,k)
      - global_bias (float)
    """
    df = pd.read_csv(
        train_path,
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # Implicit preference (binary): interacted => 1
    df["pref"] = 1.0

    users = df["user_id"].unique()
    items = df["movie_id"].unique()

    user_to_row = {u: idx for idx, u in enumerate(users)}
    item_to_col = {i: idx for idx, i in enumerate(items)}

    # Sparse-ish pivot as dense DataFrame (dataset size is typically manageable for this HW).
    X = (
        df.pivot_table(index="user_id", columns="movie_id", values="pref", aggfunc="max", fill_value=0.0)
        .astype("float32")
    )

    # Ensure the mapping matches pivot ordering
    users = X.index.to_numpy()
    items = X.columns.to_numpy()
    user_to_row = {int(u): idx for idx, u in enumerate(users)}
    item_to_col = {int(i): idx for idx, i in enumerate(items)}

    k = min(n_components, min(X.shape) - 1) if min(X.shape) > 1 else 1
    svd = TruncatedSVD(n_components=k, random_state=random_state)

    user_factors = svd.fit_transform(X.values)  # (n_users,k)
    item_factors = svd.components_.T  # (n_items,k)

    # Simple bias so scores are not too extreme.
    global_bias = float(df["pref"].mean())

    return svd, user_to_row, item_to_col, user_factors, item_factors, global_bias


def svd_score(u, i, user_to_row, item_to_col, user_factors, item_factors, global_bias):
    ru = user_to_row.get(u)
    ci = item_to_col.get(i)
    if ru is None or ci is None:
        # Cold-start fallback
        return global_bias

    # Dot product in latent space.
    s = float(user_factors[ru].dot(item_factors[ci]))

    # Map to [0,1] preference score with sigmoid.
    return _sigmoid(s)


def ensemble_score(
    u,
    i,
    user_ratings,
    centered_user_ratings,
    user_means,
    centered_item_ratings,
    item_means,
    item_counts,
    global_mean,
    svd_artifacts,
):
    # Neighborhood predictors (explicit rating scale)
    ub = predict_user_based(u, i, user_ratings, centered_user_ratings, user_means)
    ib = predict_item_based(u, i, user_ratings, centered_item_ratings, item_means)

    # Popularity / bias
    cnt = item_counts.get(i, 0)
    mean_i = item_means.get(i, global_mean)
    pop = 0.7 * mean_i + 0.3 * global_mean + 0.15 * math.log1p(cnt)

    # Convert explicit preds into implicit preference-like score
    neigh_pref = _sigmoid(((0.5 * ub + 0.5 * ib) - 3.0) / 0.7)

    # Latent factor implicit score
    (_svd, user_to_row, item_to_col, user_factors, item_factors, global_bias) = svd_artifacts
    svd_pref = svd_score(u, i, user_to_row, item_to_col, user_factors, item_factors, global_bias)

    # Popularity also as preference
    pop_pref = _sigmoid((pop - 3.0) / 0.9)

    score = W_USER * _sigmoid((ub - 3.0) / 0.7) + W_ITEM * _sigmoid((ib - 3.0) / 0.7) + W_POP * pop_pref + W_SVD * svd_pref

    # Ensure within [0,1]
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


if __name__ == "__main__":
    # Train Data set is argument 1
    train_data = sys.argv[1]
    # Test Data set is argument 2
    test_data = sys.argv[2]

    user_ratings, item_ratings, user_means, item_means, item_counts = load_ratings(train_data)
    centered_user_ratings = mean_center_by_user(user_ratings, user_means)
    centered_item_ratings = transpose_user_to_item(centered_user_ratings)

    # Global mean for cold-start-ish fallback.
    all_rs = [r for d in user_ratings.values() for r in d.values()]
    global_mean = (sum(all_rs) / len(all_rs)) if all_rs else 3.0

    # Train an implicit latent factor model (pandas + scikit-learn).
    svd_artifacts = build_svd_model(train_data, SVD_N_COMPONENTS, SVD_RANDOM_STATE)

    with open("output3b.txt", "w", encoding="utf-8") as out:
        for line in open(test_data, "r", encoding="utf-8"):
            # <USER ID>,<MOVIE ID>,<TIMESTAMP>
            (user_id, movie_id, timestamp) = line.strip().split(",")
            u = int(user_id)
            i = int(movie_id)

            score = ensemble_score(
                u,
                i,
                user_ratings,
                centered_user_ratings,
                user_means,
                centered_item_ratings,
                item_means,
                item_counts,
                global_mean,
                svd_artifacts,
            )

            # <USER ID>,<MOVIE ID>,<SCORE FOR MOVIE>,<TIMESTAMP>
            out.write(f"{u},{i},{score},{timestamp}\n")