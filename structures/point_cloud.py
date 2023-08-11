import numpy as np
import logging


class PointCloud:
    """
    Base class for points in a Euclidean space.
    """
    def __init__(self, points: np.ndarray):
        self.ambient_distances = None
        self.points = points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]
        self.logger = logging.Logger("Point cloud")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        self.logger.addHandler(handler)
        handler.setFormatter(formatter)

    def _compute_ambient_distances(self, root, scale):
        """
        Computes ambient distances near a root point at given scale.
        :param root:
        :param scale:
        :return:
        """
        self.logger.info("Computing pairwise ambient distances...")
        distances_from_root = np.linalg.norm(root - self.points, axis=1)
        points_subset_idx = np.argwhere(distances_from_root < 2 * scale).flatten()
        # subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        points_subset = self.points[points_subset_idx]
        self.ambient_distances = np.linalg.norm(
            points_subset[:, np.newaxis, :] - points_subset[np.newaxis, :, :], axis=2
        )
        self.logger.info(f"Kept {len(self.ambient_distances)} points from a fixed ambient neighbourhood.")

    # def compute_coarse_curvature(self, root, tangent_scale, normal_scale):
    #     pass

    def __str__(self):
        return f"PointCloud with {self.num_points} points in {self.ambient_dim} dimensions. \n" + \
               f"Points: \n {self.points[:10, :]}"
