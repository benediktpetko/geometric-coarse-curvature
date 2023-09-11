import numpy as np
import logging
import ot


class PointCloud:
    """
    Base class for points in a Euclidean space.
    """
    def __init__(self, points: np.ndarray, root=None):
        self.root = root
        self.ambient_distances = None
        self.points = points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]
        self.logger = logging.Logger("Point cloud")
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("logfile.log")
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    def _compute_ambient_distances(self, root, scale):
        """
        Computes ambient distances near a root point at given scale.
        :param root:
        :param scale:
        :return:
        """
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)
        self.logger.info("Computing pairwise ambient distances...")
        distances_from_root = np.linalg.norm(root - self.points, axis=1)
        points_subset_idx = np.argwhere(distances_from_root < 2 * scale).flatten()
        # subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        points_subset = self.points[points_subset_idx]
        self.ambient_distances = np.linalg.norm(
            points_subset[:, np.newaxis, :] - points_subset[np.newaxis, :, :], axis=2
        )
        self.logger.info(f"Kept {len(self.ambient_distances)} points from a fixed ambient neighbourhood.")

    def _generate_random_target(self, scale: float = np.inf):
        self.logger.info("Generating target point.")
        targets = np.argwhere((1 / 2 * scale < np.abs(self.ambient_distances[0, :])) *
                              (scale > np.abs(self.ambient_distances[0, :])))
        if targets.size == 0:
            self.logger.warning("Couldn't find a target point at given scale.")
        target = targets[0]
        self.distance_to_target = self.ambient_distances[0, target]
        return int(target)

    def compute_coarse_curvature(self, scale: float, target: int = None):
        self._compute_ambient_distances(self.root, scale)
        if target is None:
            target = self._generate_random_target(scale)
        source_nbhood = np.argwhere(self.ambient_distances[0, :] < scale).ravel()
        target_nbhood = np.argwhere(self.ambient_distances[target, :] < scale).ravel()
        num_source = len(source_nbhood)
        num_target = len(target_nbhood)
        subset_idx_mesh = np.ix_(source_nbhood, target_nbhood)
        W = ot.emd2(np.ones(num_source) / num_source, np.ones(num_target) / num_target,
                    self.ambient_distances[subset_idx_mesh])
        return 1 - W / self.distance_to_target

    def __str__(self):
        return f"PointCloud with {self.num_points} points in {self.ambient_dim} dimensions. \n" + \
               f"Points: \n {self.points[:10, :]}"
