import numpy as np

from algorithms.shortest_paths import shortest_path_distances


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
    the connectivity parameter apart.
    """

    def __init__(self, root: np.array, connectivity: float):
        """
        :param root: coordinates of the root point, flat vector appended to point cloud
        :param connectivity: only points within this distance are connected
        """
        super().__init__(points=np.array([]))
        self.graph_distances = None
        self.connectivity = connectivity
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)

    def compute_graph_distances(self):
        if not self.ambient_distances:
            self.compute_ambient_distances()
        edge_weights = self.ambient_distances.copy()
        edge_weights[edge_weights > self.connectivity] = np.inf
        self.graph_distances = shortest_path_distances(edge_weights)

    def compute_curvature(self, scale: float, method="triangular"):
        """
        Compute the coarse curvature of the graph at the root.
        """
        if not self.graph_distances:
            self.compute_graph_distances()
