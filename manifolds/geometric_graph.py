import numpy as np


class PointCloud:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]


class GeometricGraph(PointCloud):
    def __init__(self, connectivity: np.float):
        super().__init__(points=np.array([]))

    def compute_curvature(self, point: int, scale: float, method="triangular"):
        pass
