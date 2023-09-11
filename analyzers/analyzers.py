import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns

from structures.geometric_graph import GeometricGraph
from abc import ABC

class Analyzer:
    pass

class CoarseRicciCurvatureAnalyzer:
    """
    Compare the curvature at the root of a geometric graph to the curvature of the manifold
    at the corresponding point.
    """
    def __init__(self, manifold, root):
        self.logger = logging.Logger("Analyzer")
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("logfile.log")
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.results = []
        self.manifold = manifold
        self.root = root
        self.geometric_graph = None
        self.point_cloud = None
        self.sample_curvatures = []

    def analyze(self, connectivities, scales, intensities, num_runs=1,
                method="triangular", algorithm='floyd-warshall'):
        for i in range(len(intensities)):
            self.logger.info(f"Point density factor: {intensities[i] * connectivities[i] ** self.manifold.dim}")
            self.logger.info(f"Geodesic approximation factor: {scales[i] / connectivities[i]}")
            self.sample_curvatures.append([])
            for _ in range(num_runs):
                self.point_cloud = self.manifold.poisson_sample(intensities[i])
                self.geometric_graph = GeometricGraph(self.point_cloud, self.root, connectivities[i])
                try:
                    ricci_curvature = 2 * (self.manifold.dim + 2) / (scales[i] ** 2) * \
                                  self.geometric_graph.compute_coarse_curvature(scales[i], method=method, algorithm=algorithm)
                except IndexError:
                    continue
                self.logger.info(f"Estimated Ricci curvature: {ricci_curvature}")
                self.sample_curvatures[i].append(ricci_curvature)
            result = sum(self.sample_curvatures[i]) / len(self.sample_curvatures[i])
            self.results.append(result)
            self.logger.info(f"Scale: {scales[i]}, curvature: {result}")
            self.logger.info(f"Estimate from {len(self.sample_curvatures[i])} samples: {result}")
        plt.plot(range(len(intensities)), self.results)
        plt.show()
        self.logger.info(f"Curvatures at root: {self.results}")


class CoarseExtrinsicCurvatureAnalyzer:
    def __init__(self, manifold, root):
        self.logger = logging.Logger("Analyzer")
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("logfile.log")
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.results = []
        self.manifold = manifold
        self.root = root
        self.geometric_graph = None
        self.point_cloud = None
        self.sample_curvatures = []

    def analyze(self, scales, intensities, noise=0, num_runs=1, regime="small-noise"):
        for i in range(len(intensities)):
            self.sample_curvatures.append([])
            for _ in range(num_runs):
                self.point_cloud = self.manifold.poisson_sample(intensities[i], noise=noise)
                self.point_cloud.root = self.root
                extrinsic_curvature = 1 / (scales[i] ** 2) * self.point_cloud.compute_coarse_curvature(scales[i])
                self.logger.info(f"Estimated extrinsic curvature: {extrinsic_curvature}")
                self.sample_curvatures[i].append(extrinsic_curvature)
            result = sum(self.sample_curvatures[i]) / len(self.sample_curvatures[i])
            self.results.append(result)
            self.logger.info(f"Scale: {scales[i]}, curvature: {result}")
            self.logger.info(f"Estimate from {len(self.sample_curvatures[i])} samples: {result}")
        self.logger.info(f"Curvatures at root: {self.results}")


class DisplayMidpointDistances:
    @staticmethod
    def plot(analyzer: CoarseRicciCurvatureAnalyzer):
        fig, ax = plt.subplots()
        sns.histplot(analyzer.geometric_graph.midpoint_distances, ax=ax)
        ax.axvline(analyzer.geometric_graph.distance_to_target / 2, color='r')
        fig.show()
        fig.savefig("../plots/midpoints_dist.png")