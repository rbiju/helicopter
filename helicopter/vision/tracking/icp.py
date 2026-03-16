import numpy as np

from .kabsch import Kabsch


class ICP:
    def __init__(self, distance_threshold: float = 1e-1, etol: float = 5e-3, max_iterations: int = 10):
        self.distance_threshold = distance_threshold
        self.etol = etol
        self.kabsch = Kabsch()
        self.max_iterations = max_iterations

    def get_correspondence(self, reference_points, sample_points):
        distances = sample_points[:, None, np.newaxis] - reference_points[None, :, np.newaxis]
        distances = np.linalg.norm(distances, axis=-1)

        min_distances = np.min(distances, axis=1)
        min_idxs = np.argmin(distances, axis=1)

        reference_idxs = [idx for idx, dist in zip(min_idxs, min_distances) if dist < self.distance_threshold]
        sample_idxs = [i for i, (idx, dist) in enumerate(zip(min_idxs, min_distances)) if dist < self.distance_threshold]

        return reference_idxs, sample_idxs

    def iterate(self, q_old, t_old, sample_points, reference_points):
        reference_points = self.kabsch.apply(q_old, t_old, reference_points)

        q = None
        t = None

        error = np.inf
        iter_count = 0
        while error > self.etol and iter_count < self.max_iterations:
            reference_idxs, sample_idxs = self.get_correspondence(reference_points, sample_points)

            reference_subset = reference_points[reference_idxs]
            sample_subset = sample_points[sample_idxs]

            q, t = self.kabsch.kabsch(reference_subset, sample_subset)
            transformed_ref = q.apply(reference_subset) + t

            diff = transformed_ref[:, None, :] - sample_subset[None, :, :]
            diff = np.linalg.norm(diff, axis=-1)
            error = np.trace(diff)
            iter_count += 1

        return q, t
