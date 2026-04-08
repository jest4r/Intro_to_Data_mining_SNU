import sys
import numpy as np
import time

# MinHash-based LSH for near-duplicate detection (MMDS Ch. 3.4.3)
# - 3-shingles over alphabetic chars + space
# - ignore non-alphabetic except space
# - lowercase
# - deterministic global shingle list sorted lexicographically
# - hash family: h(x) = (a*x + b) % c, where c is smallest prime >= n
# - np.random.seed(0)
# - LSH with b bands and r rows per band (default b=6, r=20 => 120 hashes)
# - output candidate pairs whose MinHash signature agrees by >= 0.9

K = 3
DEFAULT_B = 6
DEFAULT_R = 20
THRESHOLD = 0.9


def normalize_text(text):
    # Keep alphabetic chars and whitespace; convert to lowercase.
    # Ignore any non-alphabet character except space. For other whitespace, keep as space.
    out = []
    for ch in text:
        if ch.isalpha():
            out.append(ch.lower())
        elif ch == ' ':
            out.append(' ')
        elif ch.isspace():
            out.append(' ')
        # else: ignore
    return ''.join(out)


def shingles_3(text):
    # Return set of 3-shingles from normalized text.
    if len(text) < K:
        return set()
    s = set()
    for i in range(len(text) - K + 1):
        s.add(text[i:i + K])
    return s


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def next_prime_ge(n):
    p = max(2, n)
    while not is_prime(p):
        p += 1
    return p


def minhash_signatures(doc_shingle_rows, num_rows, num_hashes):
    # Build MinHash signatures of length num_hashes for each document.
    c = next_prime_ge(num_rows)

    np.random.seed(0)
    a = np.random.randint(0, c, size=num_hashes, dtype=np.int64)
    b = np.random.randint(0, c, size=num_hashes, dtype=np.int64)

    sigs = {}
    for doc_id, rows in doc_shingle_rows.items():
        if not rows:
            sigs[doc_id] = np.full(num_hashes, c, dtype=np.int64)
            continue
        rows_arr = np.fromiter(rows, dtype=np.int64)
        # Compute all hash values for all rows: (a*row + b) % c
        # Shape: (num_hashes, len(rows))
        hv = (a[:, None] * rows_arr[None, :] + b[:, None]) % c
        sigs[doc_id] = hv.min(axis=1)
    return sigs


def signature_agreement(sig1, sig2):
    return float(np.mean(sig1 == sig2))


def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0


def main():
    start_time = time.time()

    if len(sys.argv) < 2:
        print("Usage: python hw3_2.py <articles_file> [b] [r]")
        return

    path = sys.argv[1]
    b = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_B
    r = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_R
    num_hashes = b * r

    # 1) Read docs, shingle
    doc_text = {}
    doc_shingles = {}
    all_shingles = set()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            # split on first space
            sp = line.find(' ')
            if sp == -1:
                doc_id = line.strip()
                text = ''
            else:
                doc_id = line[:sp]
                text = line[sp + 1:]

            norm = normalize_text(text)
            sh = shingles_3(norm)

            doc_text[doc_id] = norm
            doc_shingles[doc_id] = sh
            all_shingles.update(sh)

    # 2) Global shingle list sorted lexicographically => row indices
    shingle_list = sorted(all_shingles)
    shingle_to_row = {s: i for i, s in enumerate(shingle_list)}
    num_rows = len(shingle_list)

    # 3) Convert each doc's shingle set to row-index set
    doc_rows = {}
    for doc_id, sh in doc_shingles.items():
        rows = set()
        for s in sh:
            rows.add(shingle_to_row[s])
        doc_rows[doc_id] = rows

    # 4) MinHash signatures
    sigs = minhash_signatures(doc_rows, num_rows, num_hashes)

    # 5) LSH banding => candidate pairs
    bands = [{} for _ in range(b)]  # list of dict: band_signature(tuple)->list(doc)
    doc_ids = list(sigs.keys())

    for doc_id in doc_ids:
        sig = sigs[doc_id]
        for bi in range(b):
            start = bi * r
            key = tuple(sig[start:start + r].tolist())
            bucket = bands[bi].get(key)
            if bucket is None:
                bands[bi][key] = [doc_id]
            else:
                bucket.append(doc_id)

    candidates = set()
    for bi in range(b):
        for bucket_docs in bands[bi].values():
            if len(bucket_docs) < 2:
                continue
            # all pairs in bucket
            m = len(bucket_docs)
            for i in range(m):
                for j in range(i + 1, m):
                    a_id = bucket_docs[i]
                    b_id = bucket_docs[j]
                    if a_id < b_id:
                        candidates.add((a_id, b_id))
                    else:
                        candidates.add((b_id, a_id))

    # 6) Filter by signature agreement >= 0.9 and output Jaccard similarity
    # (Assignment wording says "signature components agree by at least 0.9"; we use that as filter.)
    out = []
    for a_id, b_id in sorted(candidates):
        agree = signature_agreement(sigs[a_id], sigs[b_id])
        if agree >= THRESHOLD:
            sim = jaccard(doc_shingles[a_id], doc_shingles[b_id])
            out.append((a_id, b_id, sim))

    for a_id, b_id, sim in out:
        print(f"{a_id}\t{b_id}\t{sim:.6f}")

    end = time.time()
    print(f"Elapsed time: {end - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
