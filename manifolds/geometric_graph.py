import numpy as np

from algorithms.shortest_paths import shortest_path_distances


class PointCloud:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]

    def compute_ambient_distances(self):
        self.ambient_distances = np.linalg.norm(self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :], axis=2)


class GeometricGraph(PointCloud):
    def __init__(self, connectivity: np.float):
        super().__init__(points=np.array([]))
        self.graph_distances = None
        self.connectivity = connectivity

    def compute_graph_distances(self):
        if not self.ambient_distances:
            self.compute_ambient_distances()
        edge_weights = self.ambient_distances.copy()
        edge_weights[edge_weights > self.connectivity] = np.inf
        self.graph_distances = shortest_path_distances(edge_weights)

    def compute_curvature(self, point: int, scale: float, method="triangular"):
        pass
