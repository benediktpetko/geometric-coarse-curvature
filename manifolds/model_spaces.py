import numpy as np
import matplotlib.pyplot as plt

from manifold import RiemannianManifold, EmbeddedManifold
from geometric_graph import PointCloud


class Hypersphere(EmbeddedManifold):
    def __init__(self, radius=1, dim=2):
        super().__init__(dim, dim+1)
        self.radius = radius
        self.dim = dim

    # def poisson_sample(self, intensity: float, nbhood_radius: float):
    #     if dim != 2:
    #         raise NotImplementedError
    #     points = np.array([])
    #     return PointCloud(points=points)
    def poisson_sample(self, intensity: float):
        num_points = np.random.poisson(lam=intensity)
        points = np.random.normal(size=(num_points, self.dim))
        points = self.radius * points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return PointCloud(points=points)

    def plot(self, points: np.ndarray):
        pass


class HyperbolicSpace(EmbeddedManifold):
    def __init__(self, const=1, dim=2):
        super().__init__(dim, dim+1)
        self.const = const
        self.dim = dim


if __name__=="__main__":
    sphere = Hypersphere()
    print(sphere.poisson_sample(10))