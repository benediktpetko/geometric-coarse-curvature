import numpy as np
import logging
import ot


class PointCloud:
    """
    Base class for points in a Euclidean space.
    """
    def __init__(self, points: np.ndarray, noisy_points=None, root=None):
        self.root = root
        self.ambient_distances = None
        self.noisy_ambient_distances = None
        self.points = points
        self.noisy_points = noisy_points
        self.num_points = points.shape[0]
        self.ambient_dim = points.shape[1]
        self.logger = logging.Logger("Point cloud")
        self.points_subset = None
        self.noisy_points_subset = None
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("logfile.log")
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    def _compute_ambient_distances(self, root, scale, scale_multiple=2):
        """
        Computes ambient distances near a root point at given scale.
        :param root:
        :param scale:
        :return:
        """
        self.root = root
        self.points = np.concatenate((self.root[np.newaxis, :], self.points), axis=0)
        if self.noisy_points is not None:
            self.noisy_points = np.concatenate((self.root[np.newaxis, :], self.noisy_points), axis=0)
        # self.("Computing pairwise ambient distances...")
        distances_from_root = np.linalg.norm(root - self.points, axis=1)
        points_subset_idx = np.argwhere(distances_from_root < scale_multiple * scale).flatten()
        # subset_idx_mesh = np.ix_(points_subset_idx, points_subset_idx)
        self.points_subset = self.points[points_subset_idx]
        self.ambient_distances = np.linalg.norm(
            self.points_subset[:, np.newaxis, :] - self.points_subset[np.newaxis, :, :], axis=2
        )
        if self.noisy_points is not None:
            self.noisy_points_subset = self.noisy_points[points_subset_idx]
            self.noisy_ambient_distances = np.linalg.norm(
                self.noisy_points_subset[:, np.newaxis, :] - self.noisy_points_subset[np.newaxis, :, :], axis=2
        )
        # self.logger.info(f"Kept {len(self.ambient_distances)} points from a fixed ambient neighbourhood.")

    def _generate_random_target(self, scale: float = np.inf):
        # self.logger.info("Generating target point.")
        targets = np.argwhere((2 * scale < self.ambient_distances[0, :]) *
                              (3 * scale > self.ambient_distances[0, :]))
        # if targets.size == 0:
        #     self.logger.warning("Couldn't find a target point at given scale.")
        target = int(targets[0])
        self.root_to_target_distance = self.ambient_distances[0, target]
        # self.distances_from_target = np.linalg.norm(self.points[target] - self.points, axis=1)
        return target

    def compute_coarse_curvature(self, scale: float, target: int = None):
        self._compute_ambient_distances(self.root, scale, scale_multiple=4)
        if target is None:
            target = self._generate_random_target(scale)
        source_nbhood = np.argwhere(self.ambient_distances[0, :] < scale).ravel()
        target_nbhood = np.argwhere(self.ambient_distances[target, :] < scale).ravel()
        num_source = len(source_nbhood)
        num_target = len(target_nbhood)
        subset_idx_mesh = np.ix_(source_nbhood, target_nbhood)
        if self.noisy_points is None:
            W = ot.emd2(np.ones(num_source) / num_source, np.ones(num_target) / num_target,
                        self.ambient_distances[subset_idx_mesh])
        else:
            W = ot.emd2(np.ones(num_source) / num_source, np.ones(num_target) / num_target,
                        self.noisy_ambient_distances[subset_idx_mesh])
        return 1 - W / self.root_to_target_distance

    def __str__(self):
        return f"PointCloud with {self.num_points} points in {self.ambient_dim} dimensions. \n" + \
               f"Points: \n {self.points[:10, :]}"
