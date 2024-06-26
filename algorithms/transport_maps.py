import numpy as np


def triangular_parallel_transport_distance(
        points: np.array, root: int, target: int, graph_distances: np.ndarray, midpoints: np.ndarray):
    first_midpoint = midpoints[points, root]
    second_midpoint = midpoints[points, target]
    return graph_distances[first_midpoint, second_midpoint]


def triangular_wasserstein_distance(
        root: int, target: int, graph_distances: np.ndarray, midpoints: np.ndarray, scale: float):
    print("Graph_distances:", graph_distances.shape)
    print("Root neighbours:", len(root_neighbours))
    return triangular_parallel_transport_distance(root_neighbours, root, target, graph_distances, midpoints).mean()
