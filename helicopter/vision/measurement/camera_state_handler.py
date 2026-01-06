import numpy as np
import quaternion


class CameraStateHandler:
    def __init__(self):
        # state vector
        self.quaternion = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

        self.g = np.array([0., 0., 9.80665])

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

    def get_visual_pose(self, measured_points: np.ndarray, reference_points: np.ndarray):
        m_c = measured_points.mean(axis=0)
        r_c = reference_points.mean(axis=0)

        measured_points_centered = measured_points - m_c
        reference_points_centered = reference_points - r_c

        # Rotation from sample points to reference points captures camera rotation
        covar = measured_points_centered.transpose() @ reference_points_centered
        U, s, Vh = np.linalg.svd(covar)

        rotation_matrix = U @ Vh
        translation = r_c - rotation_matrix @ m_c

        return quaternion.from_rotation_matrix(rotation_matrix), translation
