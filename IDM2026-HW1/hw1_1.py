from pyspark import SparkContext
import sys
import time

def main():
    start = time.time()
    sc = SparkContext("local", "hw1_1")

    if len(sys.argv) < 2:
        print("Usage: python hw1_1.py <input_file>")
        sc.stop()
        return
    input_path = sys.argv[1]

    lines = sc.textFile(input_path)
    def parse(line):
        parts = line.strip().split('\t')
        u = int(parts[0])
        friends = list(map(int, parts[1].split(','))) if len(parts) > 1 and parts[1] else []
        return (u, friends)
    adjacency = lines.map(parse)

    edges = adjacency.flatMap(lambda x: [(min(x[0], f), max(x[0], f)) for f in x[1]]).distinct()
    edges_bc = sc.broadcast(set(edges.collect()))

    def friend_pairs(user_friends):
        _, friends = user_friends
        friends = sorted(friends)
        res = []
        for i in range(len(friends)):
            for j in range(i + 1, len(friends)):
                res.append(((friends[i], friends[j]), 1))
        return res

    mutual_pairs = adjacency.flatMap(friend_pairs).reduceByKey(lambda a, b: a + b)

    candidates = mutual_pairs.filter(lambda x: x[0] not in edges_bc.value)

    sorted_results = candidates.map(lambda x: (x[0][0], x[0][1], x[1])) \
                               .sortBy(lambda x: (-x[2], x[0], x[1])) \
                               .take(10)

    for u1, u2, c in sorted_results:
        print(f"{u1}\t{u2}\t{c}")

    sc.stop()
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()