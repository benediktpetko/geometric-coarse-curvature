import numpy as np


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
