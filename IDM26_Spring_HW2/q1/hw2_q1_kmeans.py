# kmeans_spark.py
#
# Alternative skeleton with more explicit Spark pipeline structure.

import sys
import os
from typing import List, Sequence, Tuple

from pyspark.sql import SparkSession


Vector = List[float]


def parse_point(line: str) -> Vector:
    ######### TODO #########

    return [float(x) for x in line.strip().split()]
    ########################

def squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    ######### TODO #########


    return sum((a - b) ** 2 for a, b in zip(x, y))
    ########################


def closest_centroid_index(point: Sequence[float], centroids: Sequence[Sequence[float]]) -> int:
    ######### TODO #########

    min_index = 0
    min_distance = squared_distance(point, centroids[0])
    for i in range(1, len(centroids)):
        dist = squared_distance(point, centroids[i])
        if dist < min_distance:
            min_distance = dist
            min_index = i
    return min_index
    ########################


def vector_add(x: Sequence[float], y: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(x, y)]


def vector_div(x: Sequence[float], c: float) -> Vector:
    return [a / c for a in x]


def assign_point(point: Vector, centroids: Sequence[Sequence[float]]) -> Tuple[int, Tuple[Vector, int]]:
    """
    Returns:
        (cluster_id, (point, 1))
    """
    cluster_id = closest_centroid_index(point, centroids)
    return (cluster_id, (point, 1))


def merge_cluster_values(
    a: Tuple[Vector, int],
    b: Tuple[Vector, int],
) -> Tuple[Vector, int]:
    """
    Merge two partial aggregates:
        ([sum vector], count)
    """
    ######### TODO #########
    sum_vector = vector_add(a[0], b[0])
    count = a[1] + b[1]

    return (sum_vector, count)
    ########################


def compute_wcss(points_rdd, centroids: List[Vector]) -> float:
    ######### TODO #########
    bc_centroids = points_rdd.context.broadcast(centroids)
    wcss = points_rdd.map(lambda p: squared_distance(p, bc_centroids.value[closest_centroid_index(p, bc_centroids.value)])) \
                     .sum()
    ########################
    bc_centroids.unpersist()
    return float(wcss)

def run_kmeans(points_rdd, k: int, max_iter: int) -> Tuple[List[Vector], float]:
    """
    Run k-means and return:
        (final_centroids, final_wcss)
    """

    # Initialize using the first k points
    ######### TODO #########
    sc = points_rdd.context
    centroids = points_rdd.take(k)
    ########################

    if len(centroids) < k:
        raise ValueError(f"Dataset contains only {len(centroids)} points, but k={k} was requested.")

    for _ in range(max_iter):
        # Broadcast current centroids
        ######### TODO #########
        bc_centroids = sc.broadcast(centroids)
        ########################

        # Assign each point to nearest centroid
        ######### TODO #########
        assigned = points_rdd.map(lambda p: assign_point(p, bc_centroids.value))
        ########################

        # Aggregate cluster sums and counts
        ######### TODO #########
        aggregated = assigned.reduceByKey(lambda a, b: merge_cluster_values(a, b))
        ########################

        # Collect aggregated results to driver
        ######### TODO #########
        cluster_stats = aggregated.collectAsMap()
        ########################

        # Build next centroid list
        # If cluster j is empty, keep the old centroid.
        new_centroids = []
        ######### TODO #########
        for j in range(k):
            if j in cluster_stats:
                sum_vector, count = cluster_stats[j]
                new_centroids.append(vector_div(sum_vector, count))
            else:
                new_centroids.append(centroids[j])
        ########################

        bc_centroids.unpersist()
        centroids = new_centroids

    final_wcss = compute_wcss(points_rdd, centroids)
    return centroids, final_wcss

def save_main_outputs(output_path: str, centroids: List[Vector], final_wcss: float) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "centroids.txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(centroids):
            f.write(f"{i}\t" + " ".join(map(str, c)) + "\n")

    with open(os.path.join(output_path, "wcss.txt"), "w", encoding="utf-8") as f:
        f.write(str(final_wcss) + "\n")

def run_elbow_test(points_rdd, max_iter: int, output_path: str) -> None:
    import matplotlib.pyplot as plt

    k_values = [2, 3, 4, 5, 6, 8, 10]
    results = []

    for k in k_values:
        _, wcss = run_kmeans(points_rdd, k, max_iter)
        results.append((k, wcss))
        print(f"[test-and-plot] k={k}, WCSS={wcss}")

    os.makedirs(output_path, exist_ok=True)

    txt_path = os.path.join(output_path, "elbow_wcss.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, wcss in results:
            f.write(f"{k} {wcss}\n")

    ks = [k for k, _ in results]
    wcss_values = [w for _, w in results]

    plt.figure(figsize=(7, 5))
    plt.plot(ks, wcss_values, marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("Elbow Plot for k-Means")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "elbow_plot.png"), dpi=200)
    plt.close()

def main():
    if len(sys.argv) not in (5, 6):
        print(
            "Usage: spark-submit kmeans_spark.py <data_path> <k> <max_iter> <output_path> [--test-and-plot]"
        )
        sys.exit(1)

    data_path = sys.argv[1]
    k = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    output_path = sys.argv[4]
    test_and_plot = len(sys.argv) == 6 and sys.argv[5] == "--test-and-plot"

    if len(sys.argv) == 6 and sys.argv[5] != "--test-and-plot":
        print("Unknown option:", sys.argv[5])
        sys.exit(1)

    spark = SparkSession.builder.appName("KMeansSpark").getOrCreate()
    sc = spark.sparkContext

    # Make points RDD
    ######### TODO #########
    points = sc.textFile(data_path).map(parse_point)
    ########################

    _ = points.count()

    centroids, final_wcss = run_kmeans(points, k, max_iter)
    save_main_outputs(output_path, centroids, final_wcss)

    print(f"[main] k={k}, WCSS={final_wcss}")

    if test_and_plot:
        run_elbow_test(points, max_iter, output_path)

    spark.stop()


if __name__ == "__main__":
    main()