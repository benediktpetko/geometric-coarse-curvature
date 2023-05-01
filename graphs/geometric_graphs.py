import numpy as np

from algorithms.shortest_paths import shortest_path_distances_and_midpoints
from algorithms.transport_maps import triangular_wasserstein_distance


class PointCloud:
    """
    Base class for points in a Euclidean space.
    """
    def __init__(self, points: np.ndarray):
        self.ambient_distances = None
        self.points = points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]

    def compute_ambient_distances(self):
        self.ambient_distances = np.linalg.norm(
            self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :], axis=2)

    def __str__(self):
        return f"PointCloud with {self.num_points} points in {self.ambient_dim} dimensions. \n" + \
               f"Points: \n {self.points[:10, :]}"


class GeometricGraph(PointCloud):
    """
    Class for rooted geometric graphs.

    A geometric graph is a point cloud with points connected if they're less than
    the connectivity parameter apart. A root is a point at which we compute the coarse curvature.
    """

    def __init__(self, point_cloud: PointCloud, root: np.ndarray, connectivity: float):
        """
        :param root: coordinates of the root point, flat vector appended to point cloud
        :param connectivity: only points within this distance are connected
        """
        super().__init__(points=point_cloud.points)
        self.midpoints = None
        self.ricci_curvature = None
        self.graph_distances = None
        self.connectivity = connectivity
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)

    def compute_graph_distances_and_midpoints(self, scale: float = np.inf):
        """
        Compute the graph distances and midpoints between points around the root at a certain scale.
        """
        if not self.ambient_distances:
            self.compute_ambient_distances()
        edge_weights = self.ambient_distances.copy()
        edge_weights[edge_weights > self.connectivity] = np.inf
        self.graph_distances, self.midpoints = shortest_path_distances_and_midpoints(edge_weights)

    def compute_ricci_curvature(self, target: int, scale: float, method="triangular") -> float:
        """
        Compute the coarse curvature of the graph at the root in the direction of the target.

        Methods implemented: ``triangular``.
        """
        if self.graph_distances is None:
            self.compute_graph_distances_and_midpoints()
        if method == "triangular":
            self.ricci_curvature = 1 - triangular_wasserstein_distance(
                0, target, self.graph_distances, self.midpoints, scale) / self.graph_distances[0, target]
            return self.ricci_curvature
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")
