import numpy as np
import matplotlib.pyplot as plt
import logging

from manifolds.base import EmbeddedManifold
from structures.point_cloud import PointCloud


class Hypersphere(EmbeddedManifold):
    def __init__(self, radius=1, dim=2):
        super().__init__(dim, dim + 1)
        self.radius = radius
        self.dim = dim
        self.logger = logging.Logger("Manifold")
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("logfile.log")
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    def poisson_sample(self, intensity: float, noise: float = 0):
        """
        Generate Poisson sample, optionally with ambient noise.
        :param intensity:
        :param noise:
        :return:
        """
        num_points = np.random.poisson(lam=intensity)
        points = np.random.normal(size=(num_points, self.ambient_dim))
        points = self.radius * points / np.linalg.norm(points, axis=1)[:, np.newaxis]
        radial_noise = np.random.uniform(size=num_points)
        radial_noise = (2 * noise / self.radius * radial_noise ** (1 / 3) - noise / self.radius).reshape(-1, 1)
        noisy_points = points * (1 + radial_noise / self.radius)
        # self.logger.info(f"Sampled {len(points)} points.")
        return PointCloud(points=points, noisy_points=noisy_points)

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
