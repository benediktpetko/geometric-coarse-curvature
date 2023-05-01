import numpy as np
from typing import Tuple


def shortest_path_distances(edge_weights: np.ndarray) -> np.ndarray:
    """
    Compute the shortest path distances between all pairs of points in a graph.

    Also known as the Floyd-Warshall algorithm.

    Parameters
    ----------
    edge_weights : np.ndarray
        A matrix of edge weights. If there is no edge between two points, the
        corresponding entry should be np.inf.

    Returns
    -------
    np.ndarray
        A matrix of shortest path distances between all pairs of points.
    """
    num_points = edge_weights.shape[0]
    distances = edge_weights.copy()
    for k in range(num_points):
        for i in range(num_points):
            for j in range(num_points):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    return distances


def shortest_path_distances_and_midpoints(edge_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the shortest path distances between all pairs of points in a graph
    and keep track of the midpoints on the shortest paths.

    Also known as the Floyd-Warshall algorithm.

    Parameters
    ----------
    edge_weights : np.ndarray
        A matrix of edge weights. If there is no edge between two points, the
        corresponding entry should be np.inf.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A matrix of shortest path distances between all pairs of points, and
        a matrix of midpoints on the shortest paths.
    """
    num_points = edge_weights.shape[0]
    distances = edge_weights.copy()
    midpoints = np.zeros((num_points, num_points), dtype=int)
    for k in range(num_points):
        for i in range(num_points):
            for j in range(num_points):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
                    midpoints[i, j] = k
    return distances, midpoints

