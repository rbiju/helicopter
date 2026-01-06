import numpy as np
import quaternion


class Helicopter:
    def __init__(self):
        self.m = 40 * 1e-3
        self.g = np.array([0, 0, -9.81])
        self.I = 2e-4

        self.position = np.array([0., 0, 0])
        self.velocity = np.array([0., 0, 0])

        self.angular_velocity = np.array([0., 0, 0])
        self.orientation = quaternion.quaternion(1.0, 0, 0, 0)

        self.rotor_max_thrust = np.array([0, 0, 0.43])
        self.pitch_rotor_max_thrust = np.array([0, 0, 4e-2])

        self.POS_IDX = slice(0, 3)
        self.VEL_IDX = slice(3, 6)
        self.THETA_IDX = slice(6, 9)
        self.OMEGA_IDX = slice(9, 12)
        self.SET_IDX = slice(12, 15)
        self.ERR_IDX = slice(15, 18)

    def diff_eq(self, state_vec: np.ndarray) -> np.ndarray:
        """
        :param state_vec: vector s containing:
        [position, velocity, angles, angular_velocity, setpoint, error_integral]
        :return: vector ds/dt containing the time derivative for every term
        """


