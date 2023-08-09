import numpy as np
from typing import Tuple
import heapq


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

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]

    return distance


def bellman_ford(weight_matrix, source):
    num_nodes = len(weight_matrix)
    distance = np.full(num_nodes, np.inf)
    distance[source] = 0

    for _ in range(num_nodes - 1):
        for u in range(num_nodes):
            for v in range(num_nodes):
                if weight_matrix[u][v] != np.inf and distance[u] + weight_matrix[u][v] < distance[v]:
                    distance[v] = distance[u] + weight_matrix[u][v]

    return distance


def dijkstra(weight_matrix, source):
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


def johnson(weight_matrix):
    num_nodes = len(weight_matrix)
    dummy_node = num_nodes

    # Add a dummy row and column with zeros to the weight matrix
    augmented_weight_matrix = np.vstack([weight_matrix, np.zeros(num_nodes)])
    augmented_weight_matrix = np.hstack([augmented_weight_matrix, np.zeros((num_nodes + 1, 1))])

    # Run Bellman-Ford from the dummy node to compute potential values
    potentials = bellman_ford(augmented_weight_matrix, dummy_node)

    # Reweight the graph using the computed potentials
    reweighted_weight_matrix = weight_matrix + (potentials[:-1] - potentials[-1])

    # Run Dijkstra's algorithm from each node to compute shortest paths
    shortest_distances = np.zeros((num_nodes, num_nodes))

    for u in range(num_nodes):
        shortest_paths = dijkstra(reweighted_weight_matrix, u)
        shortest_distances[u] = shortest_paths[:-1] - shortest_paths[-1]

    return shortest_distances


def find_midpoint_index(source, endpoint, distances):
    num_nodes = len(distances)
    dist = np.inf
    midpoint = None
    for node in range(num_nodes):
        if max(distances[source, node], distances[node, endpoint]) < dist:
            dist = max(distances[source, node], distances[node, endpoint])
            midpoint = node
    return midpoint
