from dataclasses import dataclass
import math
import time

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
        self.g_world = np.array([0.0, 0.0, -9.81])

        self.I_tensor = np.diag([2e-4, 2e-4, 1e-4])
        self.I_inv = np.linalg.inv(self.I_tensor)

        self.rotor_max_thrust = 0.6
        self.tail_rotor_max_thrust = 0.1
        self.tail_moment_arm = 12e-2
        self.thrust_constant = 0.0

        self.thrust_time_constant = 0.05
        self.pitch_time_constant = 0.05
        self.yaw_time_constant = 0.05

        self.drag_x = 0.01
        self.drag_y = 0.01
        self.drag_z = 0.05
        self.angular_drag = np.array([0.05, 0.025, 1e-4])

        self.Kp_POS, self.Ki_POS, self.Kd_POS = throttle_gains.p, throttle_gains.i, throttle_gains.d
        self.Kp_PITCH, self.Ki_PITCH, self.Kd_PITCH = pitch_gains.p, pitch_gains.i, pitch_gains.d
        self.Kp_YAW, self.Ki_YAW, self.Kd_YAW = yaw_gains.p, yaw_gains.i, yaw_gains.d

        self.spring_constant_gyroscope = np.array([0.05, 0.05, 0.0])
        self.max_yaw_torque = 0.05
        self.coriolis_constant = 0.06

        self.battery_capacity = 1000.0
        self.current_draw_coeff = 10.0
        self.internal_resistance = 0.1

        self.rotor_radius = 0.1
        self.ground_effect_max = 0.48

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
        self.BATTERY_IDX = 19

        self.columns = ['time',
                        'x', 'y', 'z',
                        'u', 'v', 'w',
                        'q_w', 'q_i', 'q_j', 'q_k',
                        'omega_x', 'omega_y', 'omega_z',
                        'error_height', 'error_dist', 'error_heading',
                        'thrust', 'pitch', 'yaw', 'battery']

    @staticmethod
    def rotate_to_body_frame(vector_body: np.ndarray, q_body_to_world: quaternion.quaternion) -> np.ndarray:
        return (q_body_to_world.inverse() * quaternion.from_vector_part(vector_body) * q_body_to_world).imag

    @staticmethod
    def rotate_to_world_frame(vector_world: np.ndarray, q_body_to_world: quaternion.quaternion) -> np.ndarray:
        return (q_body_to_world * quaternion.from_vector_part(vector_world) * q_body_to_world.inverse()).imag

    @staticmethod
    def quat2euler(q: quaternion.quaternion) -> np.ndarray:
        w, x, y, z = q.w, q.x, q.y, q.z
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)
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
        battery = state_vec[self.BATTERY_IDX]

        quat = quaternion.quaternion(*q_raw)
        velocity_body = np.array([u, v, w])
        omega_body = np.array([p, q, r])

        dsdt = np.zeros_like(state_vec)

        position_error = setpoint - pos
        e_z = position_error[2]

        pos_err_world = position_error
        pos_err_body = self.rotate_to_body_frame(pos_err_world, quat)
        e_d = pos_err_body[0]
        dist_xy = np.linalg.norm(position_error[:2])
        e_d = e_d * np.clip(dist_xy ** 2, 0.0, 1.0)

        if dist_xy < self.loiter_radius:
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
                e_h = e_h * np.clip(dist_xy, 0., 1.0)

        cmd_T_norm = self.Kp_POS * e_z + self.Ki_POS * I_height + self.Kd_POS * (0.0 - w)
        cmd_thrust_norm = np.clip(cmd_T_norm, 0.0, 1.0)

        cmd_pitch_norm = self.Kp_PITCH * e_d + self.Ki_PITCH * I_dist + self.Kd_PITCH * (0.0 - u)
        cmd_pitch_norm = np.clip(cmd_pitch_norm, -1.0, 1.0)

        cmd_yaw_norm = self.Kp_YAW * e_h + self.Ki_YAW * I_heading + self.Kd_YAW * (0.0 - r)
        cmd_yaw_norm = np.clip(cmd_yaw_norm, -1.0, 1.0)

        current_draw = (thrust ** 2 * self.current_draw_coeff) + (np.abs(pitch) * self.current_draw_coeff * 0.05)

        v_ocv = 3.0 + (3.2 * battery) - (5.0 * battery ** 2) + (3.0 * battery ** 3)
        v_terminal = v_ocv - (current_draw * self.internal_resistance)
        voltage_efficiency = v_terminal / 4.2

        dsdt[self.BATTERY_IDX] = -current_draw / self.battery_capacity

        dsdt[self.POS_IDX] = self.rotate_to_world_frame(velocity_body, quat)

        q_dot = 0.5 * quat * quaternion.from_vector_part(omega_body)
        dsdt[self.QUAT_IDX] = quaternion.as_float_array(q_dot)

        alt = np.maximum(pos[2], 0.001)
        out_of_bounds = np.abs(pos[0]) > 0.355 or np.abs(pos[1]) > 0.685
        table_dropoff = 0.48 if out_of_bounds else 0.0
        ge_multiplier = 1.0 + self.ground_effect_max * np.exp(-3.0 * ((alt + table_dropoff) / self.rotor_radius))

        g_body = self.rotate_to_body_frame(self.g_world, quat) * self.m

        main_rotor_thrust = (self.thrust_constant + thrust * self.rotor_max_thrust * voltage_efficiency) * ge_multiplier
        tail_rotor_thrust = pitch * self.tail_rotor_max_thrust * voltage_efficiency

        F_thrust_main = np.array([0.0, 0.0, main_rotor_thrust])
        F_thrust_tail = np.array([0.0, 0.0, tail_rotor_thrust])
        F_drag = -np.array([self.drag_x * u, self.drag_y * v, self.drag_z * w])

        F_net_body = F_thrust_main + F_thrust_tail + g_body + F_drag

        acc_coriolis = np.cross(omega_body, velocity_body)

        F_net_world_frame = self.rotate_to_world_frame(F_net_body, quat)
        normal_z_mag = -F_net_world_frame[2] if F_net_world_frame[2] < 0 else 0.0
        normal = np.zeros(3)
        if pos[2] <= 0.0:
            normal = self.rotate_to_body_frame(np.array([0.0, 0.0, normal_z_mag]), quat)

        F_net_body += normal

        dvdt = (F_net_body / self.m) - acc_coriolis

        vel_new_world = self.rotate_to_world_frame(velocity_body + dvdt, quat)
        if pos[2] <= 0.0 and vel_new_world[2] < 0.0:
            vel_new_world[2] = 0.0
            dvdt = self.rotate_to_body_frame(vel_new_world, quat) - velocity_body

        dsdt[self.VEL_IDX] = dvdt

        tau_roll = acc_coriolis[1] * self.m * self.coriolis_constant
        tau_actuator_pitch = tail_rotor_thrust * self.tail_moment_arm
        tau_actuator_yaw = yaw * self.max_yaw_torque * voltage_efficiency
        tau_actuator = np.array([tau_roll, tau_actuator_pitch, tau_actuator_yaw])

        world_up = np.array([0.0, 0.0, 1.0])
        world_up_in_body = self.rotate_to_body_frame(world_up, quat)
        local_up = np.array([0.0, 0.0, 1.0])
        restoring_torque_dir = np.cross(world_up_in_body, local_up)
        tau_gyro = self.spring_constant_gyroscope * restoring_torque_dir

        tau_damping = self.angular_drag * omega_body

        tau_net = tau_actuator - tau_gyro - tau_damping

        omega_cross_I_omega = np.cross(omega_body, self.I_tensor @ omega_body)
        dsdt[self.OMEGA_IDX] = self.I_inv @ (tau_net - omega_cross_I_omega)

        dsdt[self.ERR_IDX] = np.array([e_z, e_d, e_h])

        dsdt[self.INPUT_IDX] = np.array([
            (cmd_thrust_norm - thrust) / self.thrust_time_constant,
            (cmd_pitch_norm - pitch) / self.pitch_time_constant,
            (cmd_yaw_norm - yaw) / self.yaw_time_constant
        ])

        return dsdt

    def solve(self, t_span, t_eval, setpoint):
        s = np.zeros(20)
        s[6] = 1.0
        s[19] = 1.0
        start = time.perf_counter()
        sol = scipy.integrate.solve_ivp(
            fun=self.diff_eq, method='RK23', t_span=t_span, t_eval=t_eval,
            y0=s, args=(setpoint,), rtol=1e-2, atol=1e-4
        )
        end = time.perf_counter()

        df_data = np.hstack((sol.t[:, np.newaxis], sol.y.T))
        df = pd.DataFrame(df_data, columns=self.columns)
        return df, end - start


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
    _, elapsed = helicopter.solve(_t_span, _t_eval, _setpoint)

    print(f'Solve_IVP took {elapsed}')
