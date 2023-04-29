import numpy as np

from manifold import RiemannianManifold, EmbeddedManifold
from geometric_graph import PointCloud


class Hypersphere(EmbeddedManifold):
    def __init__(self, ambient_dim: int, radius=1, dim=2):
        super().__init__(dim, ambient_dim)
        self.radius = radius
        self.dim = dim

    def poisson_sample(self, intensity: float, nbhood_radius: float):
        if dim != 2:
            raise NotImplementedError
        points = np.array([])
        return PointCloud(points=points)
