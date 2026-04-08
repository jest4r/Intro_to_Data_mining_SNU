import sys
import math
import time
import numpy as np

# HW3-3: TF-IDF + random-hyperplane LSH + cosine distance
# Input format: <ARTICLE_ID><SPACE><TEXT>
# Output: all candidate pairs (from same bucket) whose cosine distance < 0.1
# Format per line: <ID>\t<ID>\t<COSINE_SIMILARITY>
# - TF: raw term count in a document
# - IDF(t) = log(N / df(t))
# - global vocabulary: all unique terms sorted lexicographically
# - LSH: 10 random hyperplanes, np.random.seed(0)
# - hash bit = sign( dot(hyperplane, tfidf_vector) )
# - compare only docs in same bucket

HYPERPLANES = 10
DIST_THRESHOLD = 0.1  # cosine distance


def tokenize(text):
    # Simple whitespace tokenizer; lowercase.
    # (Spec does not mention punctuation handling; keep consistent and deterministic.)
    return [t for t in text.lower().split() if t]


def cosine_similarity_dense(a, b):
    # a,b are 1D numpy arrays
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    start_time = time.time()

    if len(sys.argv) < 2:
        print("Usage: python hw3_3.py <articles_file>")
        return

    path = sys.argv[1]

    # 1) Read documents and term frequencies
    doc_ids = []
    doc_tf = []  # list of dict term->count
    df = {}      # dict term->docfreq

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            sp = line.find(' ')
            if sp == -1:
                doc_id, text = line.strip(), ''
            else:
                doc_id, text = line[:sp], line[sp + 1:]

            terms = tokenize(text)
            tf = {}
            seen = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
                if t not in seen:
                    seen[t] = 1
                    df[t] = df.get(t, 0) + 1

            doc_ids.append(doc_id)
            doc_tf.append(tf)

    N = len(doc_ids)
    if N == 0:
        return

    # 2) Global vocabulary sorted lexicographically
    vocab = list(df.keys())
    vocab.sort()
    V = len(vocab)
    term_to_idx = {}
    for i in range(V):
        term_to_idx[vocab[i]] = i

    # 3) Compute IDF
    idf = np.zeros(V, dtype=np.float64)
    for i in range(V):
        t = vocab[i]
        idf[i] = math.log(N / df[t]) if df[t] > 0 else 0.0

    # 4) Build dense TF-IDF vectors (V can be large; OK for this assignment scale)
    X = np.zeros((N, V), dtype=np.float64)
    for di in range(N):
        tf = doc_tf[di]
        for t, cnt in tf.items():
            j = term_to_idx.get(t)
            if j is not None:
                X[di, j] = float(cnt) * idf[j]

    # 5) Random hyperplane LSH with 10 hyperplanes
    np.random.seed(0)
    planes = np.random.normal(0.0, 1.0, size=(HYPERPLANES, V))

    # bucket key: 10-bit signature packed as int
    buckets = {}
    for di in range(N):
        v = X[di]
        proj = np.dot(planes, v)  # shape (10,)
        key = 0
        for k in range(HYPERPLANES):
            if proj[k] >= 0:
                key |= (1 << k)
        if key in buckets:
            buckets[key].append(di)
        else:
            buckets[key] = [di]

    # 6) Compare only within buckets; output pairs with cosine distance < 0.1
    out = []
    for docs in buckets.values():
        m = len(docs)
        if m < 2:
            continue
        for i in range(m):
            a = docs[i]
            for j in range(i + 1, m):
                b = docs[j]
                sim = cosine_similarity_dense(X[a], X[b])
                dist = 1.0 - sim
                if dist < DIST_THRESHOLD:
                    ida, idb = doc_ids[a], doc_ids[b]
                    out.append((ida, idb, sim))

    # Sort output deterministically
    out.sort(key=lambda x: (x[0], x[1]))
    for a, b, sim in out:
        print(f"{a}\t{b}\t{sim:.6f}")

    end = time.time()
    print(f"Elapsed time: {end - start_time:.4f} seconds")
if __name__ == '__main__':
    main()
