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
