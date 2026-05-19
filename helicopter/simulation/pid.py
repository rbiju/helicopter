from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import scipy
import pandas as pd
from scipy.spatial.transform import Rotation

from helicopter.utils import SystemParams


@dataclass
class PIDGains:
    p: float = 0.
    i: float = 0.
    d: float = 0.


class Helicopter:
    def __init__(self, throttle_gains=PIDGains(), pitch_gains=PIDGains(), yaw_gains=PIDGains()):
        self.params = SystemParams.from_file(Path(__file__).parents[2]
                                             / 'assets/simulation_params/blue_syma')

        self.Kp_POS, self.Ki_POS, self.Kd_POS = throttle_gains.p, throttle_gains.i, throttle_gains.d
        self.Kp_PITCH, self.Ki_PITCH, self.Kd_PITCH = pitch_gains.p, pitch_gains.i, pitch_gains.d
        self.Kp_YAW, self.Ki_YAW, self.Kd_YAW = yaw_gains.p, yaw_gains.i, yaw_gains.d

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
                        'q_x', 'q_y', 'q_z', 'q_w',
                        'omega_x', 'omega_y', 'omega_z',
                        'error_height', 'error_dist', 'error_heading',
                        'thrust', 'pitch', 'yaw', 'battery']

    def diff_eq(self, t: float, state_vec: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        pos = state_vec[self.POS_IDX]
        u, v, w = state_vec[self.VEL_IDX]
        q_raw = state_vec[self.QUAT_IDX]
        p, q, r = state_vec[self.OMEGA_IDX]
        I_height, I_dist, I_heading = state_vec[self.ERR_IDX]
        thrust, pitch, yaw = state_vec[self.INPUT_IDX]
        battery = state_vec[self.BATTERY_IDX]

        q_norm = np.linalg.norm(q_raw)
        q_raw_norm = q_raw / q_norm if q_norm > 0 else np.array([0.0, 0.0, 0.0, 1.0])
        rot = Rotation.from_quat(q_raw_norm)

        velocity_body = np.array([u, v, w])
        omega_body = np.array([p, q, r])

        dsdt = np.zeros_like(state_vec)

        position_error = setpoint - pos
        e_z = position_error[2]

        pos_err_world = position_error
        pos_err_body = rot.inv().apply(pos_err_world)
        e_d = pos_err_body[0]
        dist_xy = np.linalg.norm(position_error[:2])
        e_d = e_d * np.clip(dist_xy ** 2, 0.0, 1.0)

        t_b_angles = rot.as_euler('ZYX')[::-1]

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
                psi_current = t_b_angles[2]
                e_h = psi_target - psi_current
                e_h = np.arctan2(np.sin(e_h), np.cos(e_h))
                e_h = e_h * np.clip(dist_xy, 0., 1.0)

        cmd_T_norm = self.Kp_POS * e_z + self.Ki_POS * I_height + self.Kd_POS * (0.0 - w)
        cmd_thrust_norm = np.sqrt(np.clip(cmd_T_norm, 0.0, 1.0))

        cmd_pitch_norm = self.Kp_PITCH * e_d + self.Ki_PITCH * I_dist + self.Kd_PITCH * (0.0 - u)
        cmd_pitch_norm = np.clip(cmd_pitch_norm, -1.0, 1.0)

        cmd_yaw_norm = self.Kp_YAW * e_h + self.Ki_YAW * I_heading + self.Kd_YAW * (0.0 - r)
        cmd_yaw_norm = np.clip(cmd_yaw_norm, -1.0, 1.0)

        current_draw = (thrust ** 2 * self.params.CURRENT_DRAW_COEFF) + (
                    np.abs(pitch) * self.params.CURRENT_DRAW_COEFF * 0.05)

        v_ocv = 3.0 + (3.2 * battery) - (5.0 * battery ** 2) + (3.0 * battery ** 3)
        v_terminal = v_ocv - (current_draw * self.params.INTERNAL_RESISTANCE)
        voltage_efficiency = v_terminal / 4.2

        dsdt[self.BATTERY_IDX] = -current_draw / self.params.BATTERY_CAPACITY

        dsdt[self.POS_IDX] = rot.apply(velocity_body)

        qx, qy, qz, qw = q_raw_norm
        q_dot = 0.5 * np.array([
            qw * p - qz * q + qy * r,
            qz * p + qw * q - qx * r,
            -qy * p + qx * q + qw * r,
            -qx * p - qy * q - qz * r
        ])
        dsdt[self.QUAT_IDX] = q_dot

        alt = np.maximum(pos[2], 0.001)
        out_of_bounds = np.abs(pos[0]) > 0.355 or np.abs(pos[1]) > 0.685
        table_dropoff = 0.48 if out_of_bounds else 0.0
        ge_multiplier = 1.0 + self.params.GROUND_EFFECT_MAX * np.exp(
            -3.0 * ((alt + table_dropoff) / self.params.ROTOR_RADIUS))

        g_body = rot.inv().apply(self.params.G_WORLD) * self.params.MASS

        F_thrust_main = (self.params.THRUST_CONSTANT +
                         thrust * self.params.MAX_THRUST * voltage_efficiency) * ge_multiplier
        F_thrust_tail = pitch * self.params.MAX_TAIL_THRUST * voltage_efficiency
        F_drag = -self.params.DRAG * velocity_body

        F_net_body = F_thrust_main + F_thrust_tail + g_body + F_drag

        acc_coriolis = np.cross(omega_body, velocity_body)

        F_net_world_frame = rot.apply(F_net_body)
        normal_z_mag = -F_net_world_frame[2] if F_net_world_frame[2] < 0 else 0.0
        normal = np.zeros(3)
        if pos[2] <= 0.0:
            normal = rot.inv().apply(np.array([0.0, 0.0, normal_z_mag]))

        F_net_body += normal

        dvdt = (F_net_body / self.params.MASS) - acc_coriolis

        vel_new_world = rot.apply(velocity_body + dvdt)
        if pos[2] <= 0.0 and vel_new_world[2] < 0.0:
            vel_new_world[2] = 0.0
            dvdt = rot.inv().apply(vel_new_world) - velocity_body

        dsdt[self.VEL_IDX] = dvdt

        tau_roll = acc_coriolis[1] * self.params.MASS * self.params.CORIOLIS_CONSTANT
        tau_actuator_pitch = (self.params.MAX_TAIL_THRUST[2] *
                              pitch * voltage_efficiency * self.params.TAIL_MOMENT_ARM)
        tau_actuator_yaw = yaw * self.params.MAX_YAW_TORQUE * voltage_efficiency
        tau_actuator = np.array([tau_roll, tau_actuator_pitch, tau_actuator_yaw])

        world_up = np.array([0.0, 0.0, 1.0])
        world_up_in_body = rot.inv().apply(world_up)
        local_up = np.array([0.0, 0.0, 1.0])
        restoring_torque_dir = np.cross(world_up_in_body, local_up)

        gyro_spring_vector = np.array([self.params.GYRO_SPRING_CONSTANT, self.params.GYRO_SPRING_CONSTANT, 0.0])
        tau_gyro = gyro_spring_vector * restoring_torque_dir

        tau_damping = self.params.ANGULAR_DRAG * omega_body

        tau_net = tau_actuator - tau_gyro - tau_damping

        I_tensor = np.diag(self.params.I_TENSOR_DIAGONAL)
        I_inv = np.diag(1.0 / self.params.I_TENSOR_DIAGONAL)

        omega_cross_I_omega = np.cross(omega_body, I_tensor @ omega_body)
        dsdt[self.OMEGA_IDX] = I_inv @ (tau_net - omega_cross_I_omega)

        dsdt[self.ERR_IDX] = np.array([e_z, e_d, e_h])

        dsdt[self.INPUT_IDX] = np.array([
            (cmd_thrust_norm - thrust) / self.params.ROTOR_TIME_CONSTANT,
            (cmd_pitch_norm - pitch) / self.params.PITCH_TIME_CONSTANT,
            (cmd_yaw_norm - yaw) / self.params.YAW_TIME_CONSTANT
        ])

        return dsdt

    def solve(self, t_span, t_eval, setpoint):
        s = np.zeros(20)
        s[9] = 1.0
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
