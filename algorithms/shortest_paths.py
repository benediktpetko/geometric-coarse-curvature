import numpy as np
from typing import Tuple
import heapq
from tqdm import tqdm


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


def floyd_warshall(weight_matrix):
    num_nodes = len(weight_matrix)
    distance = weight_matrix.copy()

    # for k in tqdm(range(num_nodes)):
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]

    return distance


def dijkstra_single_source(weight_matrix, source):
    num_nodes = len(weight_matrix)
    distance = np.full(num_nodes, np.inf)
    distance[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:
        dist_u, u = heapq.heappop(priority_queue)

        if dist_u > distance[u]:
            continue

        for v in range(num_nodes):
            if weight_matrix[u][v] != np.inf and distance[u] + weight_matrix[u][v] < distance[v]:
                distance[v] = distance[u] + weight_matrix[u][v]
                heapq.heappush(priority_queue, (distance[v], v))

    return distance


def dijkstra_pairwise(weight_matrix):
    num_nodes = len(weight_matrix)
    distances = np.full((num_nodes, num_nodes), np.inf)

    # for source in tqdm(range(num_nodes)):
    for source in range(num_nodes):
        distances[source] = dijkstra_single_source(weight_matrix, source)

    return distances


def find_midpoint_index(source, endpoint, distances, connectivity=0):
    ### variant 1
    # num_nodes = len(distances)
    # dist = np.inf
    # midpoint = None
    # for node in range(num_nodes):
    #     if max(distances[source, node], distances[node, endpoint]) < dist:
    #         dist = max(distances[source, node], distances[node, endpoint])
    #         midpoint = node
    # return node
    ### variant 2
    # num_nodes = len(distances)
    # dist = distances[source, endpoint]
    # midpoint_dist = distances[source, endpoint] / 2
    # for node in range(num_nodes):
    #     max_midpoint_deviation = max(
    #         np.abs(distances[source, node] - midpoint_dist),
    #         np.abs(distances[node, endpoint] - midpoint_dist)
    #     )
    #     if (max_midpoint_deviation < connectivity and
    #       np.abs(distances[source, node] + distances[node, endpoint] - dist) < connectivity / 100):
    #         return node
    ### variant 3
    num_nodes = len(distances)
    dist = distances[source, endpoint]
    if not dist < np.inf:
        return None
    midpoint = None
    min_midpoint_deviation = np.inf
    for node in range(num_nodes):
        if not distances[source, node] < np.inf:
            continue
        midpoint_deviation = np.abs(distances[source, node] - dist / 2)
        if (np.abs(distances[source, node] + distances[node, endpoint] - dist) < connectivity / 100
                and midpoint_deviation < min_midpoint_deviation):
            min_midpoint_deviation = midpoint_deviation
            midpoint = node
    return midpoint
