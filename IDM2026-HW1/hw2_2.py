import sys

SUPPORT = 100
MIN_CONF = 0.5
TOP_K = 10


def parse_session(line):
    # One browsing session per line: items are space-delimited.
    return sorted(set(x for x in line.strip().split() if x))


def t_index(i, j, n):
    # Triangular method: map an unordered pair (i<j) into a 1D array index.
    return i * n - (i * (i + 1)) // 2 + (j - i - 1)


def t_unindex(k, n):
    # Inverse mapping of t_index (used to decode pair index back to (i,j)).
    i = 0
    while i < n - 1:
        row = n - i - 1
        if k < row:
            return i, i + 1 + k
        k -= row
        i += 1
    raise ValueError("bad index")


def main():
    

    if len(sys.argv) < 2:
        print("Usage: python hw2_2.py <browsing_file> [support]")
        return

    path = sys.argv[1]
    support = int(sys.argv[2]) if len(sys.argv) > 2 else SUPPORT

    # a) Count the candidate pairs

    # Pass 1: count single items and keep sessions.
    item = {}      
    sessions = []  
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            b = parse_session(line)
            if not b:
                continue
            sessions.append(b)
            for x in b:
                item[x] = item.get(x, 0) + 1

    freq = [x for x in item if item[x] >= support]
    freq.sort()
    idx = {}
    for i in range(len(freq)):
        idx[freq[i]] = i
    n = len(freq)

    # Pass 2: count all pairs of frequent items using one triangular array.
    pairs = [0] * (n * (n - 1) // 2)  
    for b in sessions:
        ids = []
        for x in b:
            if x in idx:
                ids.append(idx[x])
        ids.sort()

        m = len(ids)
        for i in range(m):
            a = ids[i]
            for j in range(i + 1, m):
                pairs[t_index(a, ids[j], n)] += 1

    print(n)
    # b) Generate association rules from frequent pairs:

    total = 0 
    top = []   

    for k in range(len(pairs)):
        s = pairs[k]
        if s < support:
            continue

        i, j = t_unindex(k, n)
        a, b = freq[i], freq[j]

        # a -> b
        c = s / item[a]
        if c >= MIN_CONF:
            total += 1
            top.append((c, s, a, b))

        # b -> a
        c = s / item[b]
        if c >= MIN_CONF:
            total += 1
            top.append((c, s, b, a))

    top.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
    print(total)

    lim = TOP_K if len(top) > TOP_K else len(top)
    for i in range(lim):
        c, s, a, b = top[i]
        print(f"Rule: {a} -> {b}, Confidence: {c:.4f}, Support: {s}")

if __name__ == "__main__":
    main()
