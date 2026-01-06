from dataclasses import dataclass

import math
import numpy as np
import quaternion
import scipy

import pandas as pd


@dataclass
class PIDGains:
    p: float = 0.
    i: float = 0.
    d: float = 0.


class Helicopter:
    def __init__(self, throttle_gains=PIDGains(), pitch_gains=PIDGains(), yaw_gains=PIDGains()):
        self.m = 40.0 * 1e-3
        self.g_world = np.array([0, 0, -9.81])

        self.I_tensor = np.diag([2e-4, 2e-4, 1e-4])
        self.I_inv = np.linalg.inv(self.I_tensor)

        self.rotor_max_thrust = 0.80
        self.tail_rotor_max_thrust = 0.125
        self.tail_moment_arm = 1e-1

        self.thrust_time_constant = 0.05
        self.pitch_time_constant = 0.05
        self.yaw_time_constant = 0.03

        self.drag_x = 0.1
        self.drag_y = 0.4
        self.drag_z = 0.25

        self.Kp_POS, self.Ki_POS, self.Kd_POS = throttle_gains.p, throttle_gains.i, throttle_gains.d
        self.Kp_PITCH, self.Ki_PITCH, self.Kd_PITCH = pitch_gains.p, pitch_gains.i, pitch_gains.d
        self.Kp_YAW, self.Ki_YAW, self.Kd_YAW = yaw_gains.p, yaw_gains.i, yaw_gains.d

        self.spring_constant_gyroscope = np.array([0.25, 0.01, 0.0])
        self.angular_damping = np.array([0.05, 0.025, 1e-4])

        self.loiter_radius = 2e-2
        self.heading_flag = 0.
        self.heading_time = 0.
        self.heading_delay = 1.0

        self.POS_IDX = slice(0, 3)
        self.VEL_IDX = slice(3, 6)
        self.QUAT_IDX = slice(6, 10)
        self.OMEGA_IDX = slice(10, 13)
        self.ERR_IDX = slice(13, 16)
        self.INPUT_IDX = slice(16, 19)

        self.columns = ['time',
                        'x', 'y', 'z',
                        'u', 'v', 'w',
                        'q_w', 'q_i', 'q_j', 'q_k',
                        'omega_x', 'omega_y', 'omega_z',
                        'error_height', 'error_dist', 'error_heading',
                        'thrust', 'pitch', 'yaw']

    def parse_gain(self, gain):
        if gain:
            return gain
        else:
            return PIDGains(0., 0., 0.)

    def rotate_to_body_frame(self, vector_body: np.ndarray, q_body_to_world: quaternion.quaternion) -> np.ndarray:
        return (q_body_to_world.inverse() * quaternion.from_vector_part(vector_body) * q_body_to_world).imag

    def rotate_to_world_frame(self, vector_world: np.ndarray, q_body_to_world: quaternion.quaternion) -> np.ndarray:
        return (q_body_to_world * quaternion.from_vector_part(vector_world) * q_body_to_world.inverse()).imag

    @staticmethod
    def quat2euler(q: quaternion.quaternion) -> np.ndarray:
        w, x, y, z = q.w, q.x, q.y, q.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Clamp to avoid numerical issues with asin domain [-1, 1]
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def diff_eq(self, t: float, state_vec: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        pos = state_vec[self.POS_IDX]
        u, v, w = state_vec[self.VEL_IDX]
        q_raw = state_vec[self.QUAT_IDX]
        p, q, r = state_vec[self.OMEGA_IDX]
        I_height, I_dist, I_heading = state_vec[self.ERR_IDX]
        thrust, pitch, yaw = state_vec[self.INPUT_IDX]

        quat = quaternion.quaternion(*q_raw)
        velocity_body = np.array([u, v, w])
        omega_body = np.array([p, q, r])

        dsdt = np.empty_like(state_vec)
        dsdt.fill(0.)

        # PID commands
        position_error = setpoint - pos

        e_z = position_error[2]

        # distance projected onto the heading line
        pos_err_world = position_error
        pos_err_body = self.rotate_to_body_frame(pos_err_world, quat)
        e_d = pos_err_body[0]
        dist_xy = np.linalg.norm(position_error[:2])
        e_d = e_d * np.clip(dist_xy ** 2, 0.0, 1.0)

        if np.linalg.norm(position_error[:2]) < self.loiter_radius:
            e_h = 0.
            self.heading_time = t
            self.heading_flag = True
        else:
            if t - self.heading_time < self.heading_delay and self.heading_flag:
                e_h = 0.
            else:
                self.heading_flag = False
                psi_target = np.arctan2(position_error[1], position_error[0])
                psi_current = self.quat2euler(quat)[2]
                e_h = psi_target - psi_current
                e_h = np.arctan2(np.sin(e_h), np.cos(e_h))
                e_h = e_h * np.clip(np.linalg.norm(position_error[:2]), 0., 1.0)

        cmd_T_norm = (
                self.Kp_POS * e_z +
                self.Ki_POS * I_height +
                self.Kd_POS * (0.0 - w)
        )
        cmd_thrust_norm = np.clip(cmd_T_norm, 0.0, 1.0)

        cmd_pitch_norm = (
                self.Kp_PITCH * e_d +
                self.Ki_PITCH * I_dist +
                self.Kd_PITCH * (0.0 - u)
        )
        cmd_pitch_norm = np.clip(cmd_pitch_norm, -1.0, 1.0)

        cmd_yaw_norm = (
                self.Kp_YAW * e_h +
                self.Ki_YAW * I_heading +
                self.Kd_YAW * (0.0 - r)
        )
        cmd_yaw_norm = np.clip(cmd_yaw_norm, -1.0, 1.0)

        # Quaternion derivative
        dsdt[self.POS_IDX] = self.rotate_to_world_frame(velocity_body, quat)
        q_dot = 0.5 * quat * quaternion.from_vector_part(omega_body)
        dsdt[self.QUAT_IDX] = quaternion.as_float_array(q_dot)

        # Linear dynamics
        g_body = self.rotate_to_body_frame(self.g_world, quat) * self.m
        F_thrust_main = np.array([0.0, 0.0, thrust * self.rotor_max_thrust])
        F_thrust_tail = np.array([0.0, 0.0, pitch * self.tail_rotor_max_thrust])
        F_drag = -np.array([self.drag_x * u, self.drag_y * v, self.drag_z * w])
        F_net = F_thrust_main + F_thrust_tail + g_body + F_drag
        acc_coriolis = np.cross(omega_body, velocity_body)

        dsdt[self.VEL_IDX] = (F_net / self.m) - acc_coriolis

        # Angular dynamics
        tau_pitch_actuator = pitch * self.tail_rotor_max_thrust * self.tail_moment_arm
        tau_yaw_actuator = (self.I_tensor[2, 2] / self.yaw_time_constant) * yaw
        tau_roll_inertial = acc_coriolis[1] * self.m * 0.06
        tau_actuator = np.array([tau_roll_inertial, tau_pitch_actuator, tau_yaw_actuator])

        # Gyro stabilizer thing
        t_b_angles = self.quat2euler(quat)
        tau_torsional_spring = self.spring_constant_gyroscope * t_b_angles
        tau_damping = self.angular_damping * state_vec[self.OMEGA_IDX]

        tau_net = tau_actuator - tau_torsional_spring - tau_damping

        omega_cross_I_omega = np.cross(omega_body, self.I_tensor @ omega_body)
        dsdt[self.OMEGA_IDX] = self.I_inv @ (tau_net - omega_cross_I_omega)

        dsdt[self.ERR_IDX] = np.array([
            e_z,
            e_d,
            e_h
        ])

        dsdt[self.INPUT_IDX] = np.array([(cmd_thrust_norm - thrust) / self.thrust_time_constant,
                                         (cmd_pitch_norm - pitch) / self.pitch_time_constant,
                                         (cmd_yaw_norm - yaw) / self.yaw_time_constant])

        if pos[2] < 0.:
            state_vec[self.POS_IDX][2] = 0.

            if w < 0:
                dsdt[self.VEL_IDX][2] = 0.0
                state_vec[self.VEL_IDX][2] = 0.0

        return dsdt

    def solve(self, t_span, t_eval, setpoint):
        s = np.zeros(19)
        s[6] = 1.0
        sol = scipy.integrate.solve_ivp(fun=self.diff_eq, method='RK23', t_span=t_span, t_eval=t_eval,
                                        y0=s, args=(setpoint,),
                                        rtol=1e-2, atol=1e-4)

        # noinspection All
        df_data = np.hstack((sol.t[:, np.newaxis], sol.y.T))
        df = pd.DataFrame(df_data, columns=self.columns)

        return df


if __name__ == '__main__':
    _throttle_gains = PIDGains(6.0, 0.8, 0.8)
    _pitch_gains = PIDGains(2.0, 0.05, 1.725)
    _yaw_gains = PIDGains(10, 0.001, 2.0)
    helicopter = Helicopter(
        throttle_gains=_throttle_gains,
        pitch_gains=_pitch_gains,
        yaw_gains=_yaw_gains
    )

    _t_span = (0., 8.)
    _t_eval = np.linspace(_t_span[0], _t_span[1], int((_t_span[1] - _t_span[0]) * 25))
    _setpoint = np.array([1., 1., 1.])
    _df = helicopter.solve(_t_span, _t_eval, _setpoint)

    print('done')
