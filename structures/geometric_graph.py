import numpy as np
import logging

from tqdm import tqdm
from algorithms.shortest_paths import *
from structures.point_cloud import PointCloud


class GeometricGraph(PointCloud):
    """
    Class for rooted geometric graphs.

    A geometric graph is a point cloud with points connected if they're less than
    the connectivity parameter apart. A root is a fixed point at which we compute the coarse curvature.
    """
    def __init__(self, point_cloud: PointCloud, root: np.ndarray, connectivity: float):
        """
        :param root: coordinates of the root point, appended to the point cloud
        :param connectivity: only points within this distance are connected
        """
        super().__init__(points=point_cloud.points.copy())
        self.midpoint_indices = None
        self.midpoint_distances = None
        self.coarse_curvature = None
        self.graph_distances = None
        self.connectivity = connectivity
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)
        self.edge_weights = None
        self.logger = logging.Logger("Geometric graph")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        self.logger.addHandler(handler)
        handler.setFormatter(formatter)

    def _compute_edge_weights(self, scale: float = np.inf):
        """
        Compute the edge weights of the graph, ignoring points further than the parameter neighbourhood away.
        The neighbourhood parameter is fixed and must be larger than double of any choice of scale parameter.
        :param neighbourhood:
        :return:
        """
        points_subset_idx = np.argwhere(self.ambient_distances[0, :] < 2 * scale).flatten()
        subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        self.logger.info(f"Kept {len(points_subset_idx)} points "
                         f"in a fixed interconnectivity neighbourhood.")
        self.edge_weights = self.ambient_distances[subset_idx_mesh].copy()
        self.edge_weights[self.edge_weights > self.connectivity] = np.inf

    def _compute_graph_distances(self, scale: float = np.inf, algorithm='floyd-warshall'):
        """
        Compute the graph distances and midpoints between points around the root,
        ignoring points further than 2*scale away.
        """
        points_subset_idx = np.argwhere(self.ambient_distances[0, :] < 2 * scale).flatten()
        subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        self.logger.info(f"Kept {len(points_subset_idx)} points at given random walk scale.")
        self.edge_weights = self.ambient_distances[subset_idx_mesh].copy()
        self.edge_weights[self.edge_weights > self.connectivity] = np.inf
        self.logger.info("Computing graph distances...")
        if algorithm == 'floyd-warshall':
            self.graph_distances = floyd_warshall(self.edge_weights)
        elif algorithm == 'dijkstra':
            self.graph_distances = dijkstra_pairwise(self.edge_weights)
        else:
            raise NotImplementedError("Algorithm not implemented.")

    def _compute_midpoint_indices(self, scale: float = np.inf, target: int = None):
        num_points = len(self.graph_distances)
        self.midpoint_indices = np.zeros((num_points, 2), dtype=int)
        midpoint_feasible = np.zeros(num_points, dtype=bool)
        self.logger.info(f"The target has index {target}")
        self.logger.info("Computing midpoint indices...")
        for i in tqdm(range(num_points)):
            first_midpoint_index = find_midpoint_index(
                i, 0, self.graph_distances, self.connectivity
            )
            second_midpoint_index = find_midpoint_index(
                i, target, self.graph_distances, self.connectivity
            )
            if first_midpoint_index is not None and second_midpoint_index is not None:
                self.midpoint_indices[i, 0] = first_midpoint_index
                self.midpoint_indices[i, 1] = second_midpoint_index
                midpoint_feasible[i] = True
        subset_idx_mesh = np.ix_(midpoint_feasible, midpoint_feasible)
        self.midpoint_indices = self.midpoint_indices[midpoint_feasible]
        self.logger.info(f"Kept only {len(self.midpoint_indices)} points due to connectivity constraints.\n"
                         f"Connectivity ratio: {len(self.midpoint_indices) / num_points}")

    def _generate_random_target(self, scale: float = np.inf):
        self.logger.info("Generating target point.")
        targets = np.argwhere((7/8 * scale < np.abs(self.graph_distances[0, :])) *
                             (2 * scale > np.abs(self.graph_distances[0, :])))
        if targets.size == 0:
            self.logger.warning("Couldn't find a target point at given scale.")
        target = targets[0]
        self.distance_to_target = self.graph_distances[0, target]
        return int(target)

    def compute_coarse_curvature(self, scale: float, target: int = None, method="triangular", algorithm="floyd-warshall") -> float:
        """
        Compute the coarse curvature of the graph at the root in the direction of the target.
        If no target is provided, a random target is chosen.

        Methods implemented: ``triangular``.
        """
        # compute necessary data for the graph
        self._compute_ambient_distances(self.root, scale)
        self._compute_edge_weights(scale)
        self._compute_graph_distances(scale, algorithm)
        if target is None:
            target = self._generate_random_target(scale)
        self._compute_midpoint_indices(scale, target)

        num_points = len(self.midpoint_indices)
        self.logger.info("Computing coarse curvature...")
        if method == "triangular":
            self.midpoint_distances = [self.graph_distances[self.midpoint_indices[i, 0], self.midpoint_indices[i, 1]]
                 for i in range(num_points)]
            wasserstein_distance = 1 / num_points * sum(self.midpoint_distances)
            coarse_curvature = 1 - 2 * wasserstein_distance / self.distance_to_target
            self.logger.info(f"Coarse curvature is {coarse_curvature}.")
            return coarse_curvature
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")
