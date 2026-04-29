# visualize_kmeans.py
#
# Visualization for distributed k-means results

import sys
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


Vector = List[float]


def parse_point(line: str) -> Vector:
    ######### TODO #########
    return [float(x) for x in line.strip().split()]
    ########################


def parse_centroid_line(line: str) -> Tuple[int, Vector]:
    """
    Parse one centroid line of the form:
        0\t1.0 2.0 3.0 ...
    into:
        (0, [1.0, 2.0, 3.0, ...])
    """
    ######### TODO #########
    parts = line.strip().split("\t")
    cluster_id = int(parts[0])
    vec = [float(x) for x in parts[1].split()]
    ########################
    return cluster_id, vec

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


def load_points(data_path: str, max_points: int = 5000) -> List[Vector]:
    """
    Load at most max_points data points for visualization.
    """
    points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_points:
                break
            line = line.strip()
            if not line:
                continue
            points.append(parse_point(line))
    return points


def load_centroids(centroids_path: str) -> List[Vector]:
    """
    Load centroids from centroids.txt and return them ordered by cluster id.
    """
    centroid_dict = {}
    ######### TODO #########
    with open(centroids_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cluster_id, vec = parse_centroid_line(line)
            centroid_dict[cluster_id] = vec
    ########################
    return [centroid_dict[i] for i in sorted(centroid_dict.keys())]


def main():
    if len(sys.argv) != 4:
        print("Usage: python visualize_kmeans.py <data_path> <centroids_path> <output_png>")
        sys.exit(1)

    data_path = sys.argv[1]
    centroids_path = sys.argv[2]
    output_png = sys.argv[3]

    points = load_points(data_path, max_points=5000)
    centroids = load_centroids(centroids_path)

    assignments = [closest_centroid_index(p, centroids) for p in points]

    # Visualization features:
    x_idx = 1
    y_idx = 4

    plt.figure(figsize=(8, 6))

    # Plot points colored by cluster assignment
    ######### TODO #########
    for cluster_id in set(assignments):
        cluster_points = [p for p, a in zip(points, assignments) if a == cluster_id]
        xs = [p[x_idx] for p in cluster_points]
        ys = [p[y_idx] for p in cluster_points]
        plt.scatter(xs, ys, s=20, label=f"Cluster {cluster_id}")
        
    ########################

    # Plot centroids
    centroid_xs = [c[x_idx] for c in centroids]
    centroid_ys = [c[y_idx] for c in centroids]
    plt.scatter(centroid_xs, centroid_ys, s=200, marker="X", label="Centroids")

    plt.xlabel("trip_distance")
    plt.ylabel("fare_amount")
    plt.title("k-Means Clusters (trip_distance vs fare_amount)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)


if __name__ == "__main__":
    main()