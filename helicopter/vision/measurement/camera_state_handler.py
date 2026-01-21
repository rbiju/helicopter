import numpy as np
from numba import jit
import quaternion


@jit(cache=True)
def kabsch(measured_points: np.ndarray, reference_points: np.ndarray):
    """
    Computes the optimal rotation from measured to reference points, which should capture the camera frame to world frame transformation.
    Args:
        measured_points: points in the camera frame
        reference_points: matching points in the world frame

    Returns:

    """
    n = measured_points.shape[0]
    m_c = np.sum(measured_points, axis=0) / n
    r_c = np.sum(reference_points, axis=0) / n

    measured_points_centered = measured_points - m_c
    reference_points_centered = reference_points - r_c

    covar = measured_points_centered.transpose() @ reference_points_centered
    U, s, Vh = np.linalg.svd(covar)

    rotation_matrix = Vh.T @ U.T

    if np.linalg.det(rotation_matrix) < 0:
        Vh_fixed = Vh.copy()
        Vh_fixed[2, :] *= -1
        rotation_matrix = Vh_fixed.T @ U.T

    translation = r_c - (rotation_matrix @ m_c)

    return rotation_matrix, translation


class CameraStateHandler:
    def __init__(self):
        # state vector
        self.quaternion = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

        self.g = np.array([0., 0., 9.80665])

        print('Compiling')
        self.ransac_visual_pose(np.random.rand(5, 3), np.random.rand(5, 3))

    @property
    def nominal_state(self):
        return np.concatenate([quaternion.as_float_array(self.quaternion.copy()),
                               self.position.copy(),
                               self.velocity.copy(),
                               self.accelerometer_bias.copy(),
                               self.gyro_bias.copy()])

    def set_state_from_nominal(self, nominal_state):
        self.quaternion = quaternion.quaternion(*nominal_state[0:4])
        self.position = nominal_state[4:7]
        self.velocity = nominal_state[7:10]
        self.accelerometer_bias = nominal_state[10:13]
        self.gyro_bias = nominal_state[13:16]

    @staticmethod
    @jit(cache=True)
    def ransac_visual_pose(measured_points: np.ndarray, reference_points: np.ndarray, threshold=15e-3):
        n = measured_points.shape[0]

        dummy_R = np.eye(3, dtype=np.float64)
        dummy_t = np.zeros(3, dtype=np.float64)
        if n < 4:
            return False, dummy_R, dummy_t

        inlier_count = -1
        inlier_mask = np.zeros(n, dtype=np.bool_)

        idx_pool = np.arange(n)
        for _ in range(100):
            np.random.shuffle(idx_pool)
            idxs = idx_pool[:3]

            measured_subset = measured_points[idxs]
            reference_subset = reference_points[idxs]

            r_h, t_h = kabsch(measured_subset, reference_subset)

            projected = (measured_points @ r_h.T) + t_h

            diff = projected - reference_points
            dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2

            inliers = dist_sq < (threshold * threshold)
            count = np.sum(inliers)

            if count > inlier_count:
                inlier_count = count
                inlier_mask = inliers

                if count == len(measured_points):
                    break

        if inlier_count >= 3:
            final_measured = measured_points[inlier_mask]
            final_reference = reference_points[inlier_mask]

            R_final, t_final = kabsch(final_measured, final_reference)

            return True, R_final, t_final
        else:
            return False, dummy_R, dummy_t

    def get_visual_pose(self, measured_points: np.ndarray, reference_points: np.ndarray, quat: quaternion.quaternion):
        success, R, t = self.ransac_visual_pose(measured_points, reference_points)

        if not success:
            return False, None, None

        visual_quat = quaternion.from_rotation_matrix(R)

        dot = (visual_quat.w * quat.w +
               visual_quat.x * quat.x +
               visual_quat.y * quat.y +
               visual_quat.z * quat.z)

        if dot < 0:
            # noinspection PyUnresolvedReferences
            visual_quat = -visual_quat

        return True, visual_quat, t
