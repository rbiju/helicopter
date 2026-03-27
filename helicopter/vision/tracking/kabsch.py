import numpy as np
from scipy.spatial.transform import Rotation

class Kabsch:
    @staticmethod
    def kabsch(q: np.ndarray, p: np.ndarray):
        q_c = q - q.mean(axis=0)
        p_c = p - p.mean(axis=0)

        covar = p_c.T @ q_c
        u, _, vh = np.linalg.svd(covar)

        rotation = vh.T @ u.T

        if np.linalg.det(rotation) < 0:
            I = np.eye(3)
            I[2, 2] = -1.0
            rotation = vh.T @ I @ u.T

        translation = q.mean(axis=0) - rotation @ p.mean(axis=0)

        return Rotation.from_matrix(rotation), translation

    # noinspection PyTypeHints
    @staticmethod
    def apply(quat: Rotation, translation: np.ndarray, points: np.ndarray):
        return quat.apply(points) + translation
