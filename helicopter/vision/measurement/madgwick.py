import numpy as np
import quaternion

from .compiled import madgwick_step


class MadgwickFilter:
    def __init__(self, beta=0.033):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        print('Compiling Madgwick filter')
        _ = self.update(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 1.0, 0.033)

    def update(self, accel, gyro, dt, beta: float = None):
        if beta is None:
            self.q = madgwick_step(self.q, accel, gyro, dt, self.beta)
        else:
            self.q = madgwick_step(self.q, accel, gyro, dt, beta)
        return quaternion.from_float_array(self.q)
