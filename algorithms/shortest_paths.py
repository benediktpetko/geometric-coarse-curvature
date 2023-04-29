import numpy as np


def shortest_path_distances(edge_weights: np.ndarray) -> np.ndarray:
    """
    Compute the shortest path distances between all pairs of points in a graph.

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
                distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
    return distances


def find_shortest_path(edge_weights: np.ndarray, start: int, end: int) -> list:
    """
    Find the shortest path between two points in a graph.

    Parameters
    ----------
    edge_weights : np.ndarray
        A matrix of edge weights. If there is no edge between two points, the
        corresponding entry should be np.inf.
    start : int
        The index of the starting point.
    end : int
        The index of the ending point.

    Returns
    -------
    list
        A list of indices of the points in the shortest path.
    """
    num_points = edge_weights.shape[0]
    distances = edge_weights.copy()
    for k in range(num_points):
        for i in range(num_points):
            for j in range(num_points):
                distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
    path = [start]
    while path[-1] != end:
        path.append(np.argmin(distances[path[-1], :]))
    return path


def find_shortest_path_midpoint(edge_weights: np.ndarray, start: int, end: int) -> int:
    """
    Find the midpoint of the shortest path between two points in a graph.

    Parameters
    ----------
    edge_weights : np.ndarray
        A matrix of edge weights. If there is no edge between two points, the
        corresponding entry should be np.inf.
    start : int
        The index of the starting point.
    end : int
        The index of the ending point.

    Returns
    -------
    int
        The index of the midpoint of the shortest path.
    """
    num_points = edge_weights.shape[0]
    distances = edge_weights.copy()
    for k in range(num_points):
        for i in range(num_points):
            for j in range(num_points):
                distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
    path = [start]
    while path[-1] != end:
        path.append(np.argmin(distances[path[-1], :]))
    return path[len(path) // 2]
