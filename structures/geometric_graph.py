import numpy as np

from algorithms.shortest_paths import johnson_algorithm, find_midpoint
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
        self.midpoints = None
        self.coarse_curvature = None
        self.graph_distances = None
        self.connectivity = connectivity
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)
        self.edge_weights = None

    def _compute_edge_weights(self, scale: float = np.inf):
        """
        Compute the edge weights of the graph, ignoring points further than the parameter neighbourhood away.
        The neighbourhood parameter is fixed and must be larger than double of any choice of scale parameter.
        :param neighbourhood:
        :return:
        """
        if self.ambient_distances is None:
            self.compute_ambient_distances()
        points_subset_idx = np.argwhere(self.ambient_distances[0, :] < 2 * scale).flatten()
        subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        print(f"Kept {len(points_subset_idx)} points in a fixed neigbhourhood.")
        self.edge_weights = self.ambient_distances[subset_idx_mesh].copy()
        self.edge_weights[self.edge_weights > self.connectivity] = np.inf

    def _compute_graph_distances(self, scale: float = np.inf):
        """
        Compute the graph distances and midpoints between points around the root,
        ignoring points further than 2*scale away.
        """
        if self.edge_weights is None:
            self._compute_edge_weights(scale)
        points_subset_idx = np.argwhere(self.ambient_distances[0, :] < 2 * scale).flatten()
        subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        print(f"Kept {len(points_subset_idx)} points at given scale.")
        self.edge_weights = self.ambient_distances[subset_idx_mesh].copy()
        self.edge_weights[self.edge_weights > self.connectivity] = np.inf
        self.graph_distances = johnson_algorithm(self.edge_weights)

    def _compute_midpoints(self, scale: float = np.inf, target: int = None):
        if self.graph_distances is None:
            self._compute_graph_distances(scale)
        num_points = len(self.graph_distances)
        self.midpoints = np.zeros((num_points, 2), dtype=int)
        for i in range(num_points):
            self.midpoints[i, 0] = find_midpoint(i, self.root, self.graph_distances)
            self.midpoints[i, 1] = find_midpoint(i, target, self.graph_distances)

    def _generate_random_target(self, scale: float = np.inf):
        if self.graph_distances is None:
            self._compute_graph_distances(scale)
        target = np.argwhere((scale / 2 < self.graph_distances[0, :]) *
                             (2 * scale > self.graph_distances[0, :]))[0]
        return target

    def compute_coarse_curvature(self, scale: float, target: int = None, method="triangular") -> float:
        """
        Compute the coarse curvature of the graph at the root in the direction of the target.
        If no target is provided, a random target is chosen.

        Methods implemented: ``triangular``.
        """
        if target is None:
            target = self._generate_random_target(scale)
        if self.midpoints is None:
            self._compute_midpoints(scale, target)
        num_points = len(self.graph_distances)
        if method == "triangular":
            wasserstein_distance = 1 / num_points * sum([self.graph_distances(self.midpoints[i,0], self.midpoints[i,1])])
            coarse_curvature = 1 - wasserstein_distance
            return coarse_curvature
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

    # def compute_graph_distances_and_midpoints(self, scale: float = np.inf):
    #     """
    #     Compute the graph distances and midpoints between points around the root,
    #     ignoring points further than 3*scale away.
    #     """
    #     if not self.ambient_distances:
    #         self.compute_ambient_distances()
    #     points_subset_idx = np.argwhere(self.ambient_distances[0, :] < 2*scale).flatten()
    #     subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
    #     print(f"Kept {len(points_subset_idx)} points.")
    #     self.edge_weights = self.ambient_distances[subset_idx_mesh].copy()
    #     self.edge_weights[self.edge_weights > self.connectivity] = np.inf
    #     self.graph_distances, self.midpoints = johnson_algorithm_with_midpoints(edge_weights)

