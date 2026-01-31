import numpy as np
import quaternion

from .compiled import _ransac_visual_pose

from helicopter.vision import PrintHider


class CameraStateHandler:
    def __init__(self):
        # state vector
        self.quaternion = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

        self.g = np.array([0., 0., 9.80665])

        print('Compiling pose estimation functions')
        with PrintHider():
            self.ransac_visual_pose(np.random.rand(5, 3), np.random.rand(5, 3))

        self.last_quaternion = self.quaternion
        self.last_position = self.position

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
    def ransac_visual_pose(measured_points: np.ndarray, reference_points: np.ndarray, threshold=15e-3):
        return _ransac_visual_pose(measured_points, reference_points, threshold)

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
