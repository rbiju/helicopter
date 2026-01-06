import numpy as np
import quaternionic


class Kabsch:
    @staticmethod
    def kabsch(q: np.ndarray, p: np.ndarray):
        q_c = q - q.mean(axis=0)
        p_c = p - p.mean(axis=0)

        covar = p_c.T @ q_c
        u, _, vh = np.linalg.svd(covar)

        rotation = vh.T @ u.T
        translation = q.mean(axis=0) - rotation @ p.mean(axis=0)

        return quaternionic.array.from_rotation_matrix(rotation), translation

    # noinspection PyTypeHints
    @staticmethod
    def apply(quat: quaternionic.array, translation: np.ndarray, points: np.ndarray):
        return quat.rotate(points) + translation
