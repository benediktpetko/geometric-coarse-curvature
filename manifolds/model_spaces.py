import numpy as np
import matplotlib.pyplot as plt

from manifolds.base import EmbeddedManifold
from graphs.geometric_graphs import PointCloud


class Hypersphere(EmbeddedManifold):
    def __init__(self, radius=1, dim=2):
        super().__init__(dim, dim + 1)
        self.radius = radius
        self.dim = dim

    def poisson_sample(self, intensity: float):
        num_points = np.random.poisson(lam=intensity)
        points = np.random.normal(size=(num_points, self.ambient_dim))
        points = self.radius * points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        return PointCloud(points=points)

    def plot(self, points: np.ndarray):
        if self.dim == 1:
            plt.scatter(points[:, 0], points[:, 1])
            plt.show()
        elif self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])

            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = self.radius * np.cos(u) * np.sin(v)
            y = self.radius * np.sin(u) * np.sin(v)
            z = self.radius * np.cos(v)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.5, linewidth=0)

            plt.tight_layout()
            plt.show()
        else:
            raise NotImplementedError("Can't plot a hypersphere in dimensions > 2.")


class HyperbolicSpace(EmbeddedManifold):
    def __init__(self, const=1, dim=2):
        super().__init__(dim, dim + 1)
        self.const = const
        self.dim = dim


if __name__ == "__main__":
    sphere = Hypersphere(radius=5)
    print(sphere.poisson_sample(10))
    sphere.plot(sphere.poisson_sample(1000).points)
