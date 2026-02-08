import numpy as np
from scipy.spatial.transform import Rotation


class CameraStateHandler:
    def __init__(self):
        self.quaternion = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

        self.g = np.array([0., 0., 9.80665])

    @property
    def nominal_state(self):
        return np.concatenate([
            self.quaternion.as_quat(canonical=True),
            self.position,
            self.velocity,
            self.accelerometer_bias,
            self.gyro_bias
        ], dtype=np.float32)

    def set_state_from_nominal(self, nominal_state):
        self.quaternion = Rotation.from_quat(quat=nominal_state[0:4])
        self.position = nominal_state[4:7]
        self.velocity = nominal_state[7:10]
        self.accelerometer_bias = nominal_state[10:13]
        self.gyro_bias = nominal_state[13:16]
