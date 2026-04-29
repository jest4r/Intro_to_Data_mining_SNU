import argparse
import random
from collections import defaultdict


def load_ratings(ratings_path):
    """Load ratings.txt into per-user lists of (movie_id, timestamp)."""
    by_user = defaultdict(list)
    all_movies = set()

    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in f:
            user_id_str, movie_id_str, _rating_str, timestamp_str = line.strip().split(",")
            u = int(user_id_str)
            i = int(movie_id_str)
            ts = int(timestamp_str)
            by_user[u].append((i, ts))
            all_movies.add(i)

    return by_user, sorted(all_movies)


def build_user_item_sets(by_user):
    rated = {}
    for u, pairs in by_user.items():
        rated[u] = set(i for i, _ts in pairs)
    return rated


def main():
    ap = argparse.ArgumentParser(description="Create implicit-feedback validation set (Option 1).")
    ap.add_argument("--ratings", default="ratings.txt", help="Path to ratings.txt")
    ap.add_argument("--out_test", default="val_test.txt", help="Output test-like file: user,movie,timestamp")
    ap.add_argument(
        "--out_labels",
        default="ratings_test_labels.txt",
        help="Output labels file: user,movie,true_label,timestamp",
    )
    ap.add_argument("--pos", type=int, default=10000, help="Number of positive examples")
    ap.add_argument("--neg", type=int, default=10000, help="Number of negative examples")
    ap.add_argument(
        "--min_user_ratings",
        type=int,
        default=20,
        help="Only sample positives from users with at least this many ratings",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_user, all_movies = load_ratings(args.ratings)
    user_rated = build_user_item_sets(by_user)

    # Eligible users (enough ratings so holding out interactions is meaningful).
    eligible_users = [u for u, pairs in by_user.items() if len(pairs) >= args.min_user_ratings]
    if not eligible_users:
        raise ValueError("No eligible users found; try lowering --min_user_ratings")

    # --- Sample positives: (u, i, ts) where (u,i) exists in ratings ---
    # We sample from eligible users uniformly by interaction.
    positive_pool = []
    for u in eligible_users:
        positive_pool.extend((u, i, ts) for (i, ts) in by_user[u])

    if len(positive_pool) < args.pos:
        raise ValueError(f"Not enough positive interactions to sample {args.pos} positives.")

    rng.shuffle(positive_pool)
    positives = positive_pool[: args.pos]

    # --- Sample negatives: pick (u, j) where user u has NOT rated j ---
    # Timestamp: reuse a plausible timestamp from that user (randomly chosen from their history)
    negatives = []
    neg_seen = set()  # avoid duplicates

    # Precompute per-user timestamps to draw from
    user_timestamps = {u: [ts for _i, ts in pairs] for u, pairs in by_user.items()}

    # To speed up sampling, keep a list of movies and sample until we get enough.
    all_movies_list = list(all_movies)

    # Sample negatives by cycling through random users.
    while len(negatives) < args.neg:
        u = rng.choice(eligible_users)
        rated_set = user_rated[u]

        # Try a few times to find an unrated movie for this user.
        for _ in range(50):
            j = rng.choice(all_movies_list)
            if j in rated_set:
                continue
            key = (u, j)
            if key in neg_seen:
                continue
            neg_seen.add(key)
            ts = rng.choice(user_timestamps[u])
            negatives.append((u, j, ts))
            break
        else:
            # If failed too many times for this user, just continue and try another user.
            continue

    # Combine and shuffle to avoid biased ordering.
    examples = [(u, i, 1, ts) for (u, i, ts) in positives] + [(u, i, 0, ts) for (u, i, ts) in negatives]
    rng.shuffle(examples)

    # Write outputs: val_test.txt and ratings_test_labels.txt
    with open(args.out_test, "w", encoding="utf-8") as f_test, open(args.out_labels, "w", encoding="utf-8") as f_lab:
        for u, i, y, ts in examples:
            f_test.write(f"{u},{i},{ts}\n")
            f_lab.write(f"{u},{i},{y},{ts}\n")

    print(f"Wrote {len(examples)} rows")
    print(f"  test-like: {args.out_test}")
    print(f"  labels:    {args.out_labels}")


if __name__ == "__main__":
    main()
