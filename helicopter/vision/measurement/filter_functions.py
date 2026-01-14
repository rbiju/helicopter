import numpy as np
import quaternion

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_V = slice(7, 10)
IDX_BA = slice(10, 13)
IDX_BG = slice(13, 16)


def compose_fn(state: np.ndarray, error: np.ndarray):
    new_state = state.copy()

    new_state[4:] = state[4:] + error[3:]

    dq = quaternion.from_rotation_vector(error[:3])

    quat = quaternion.quaternion(*state[:4]) * dq
    new_state[:4] = quaternion.as_float_array(quat.normalized())

    return new_state


def decompose_fn(old_state: np.ndarray, new_state: np.ndarray):
    error = np.empty(old_state.shape[0] - 1)
    error[3:] = new_state[4:] - old_state[4:]

    q_old = quaternion.quaternion(*old_state[:4])
    q_new = quaternion.quaternion(*new_state[:4])

    error_quat = q_old.inverse() * q_new
    error[:3] = quaternion.as_rotation_vector(error_quat)
    return error


def propagate(s: np.ndarray,
              dt,
              accelerometer: np.ndarray,
              gyro: np.ndarray,
              g_world: np.ndarray) -> np.ndarray:
    q_prev = quaternion.quaternion(*s[IDX_Q])
    p_prev = s[IDX_P]
    v_prev = s[IDX_V]
    bg_prev = s[IDX_BG]
    ba_prev = s[IDX_BA]

    gyro_corrected = gyro - s[IDX_BG]
    acc_corrected = accelerometer - s[IDX_BA]

    dq = quaternion.from_rotation_vector(gyro_corrected * dt)
    q_new = q_prev * dq

    acc_quat = quaternion.quaternion(0, *acc_corrected)
    a_world = (q_new * acc_quat * q_new.conjugate()).imag - g_world

    v_new = v_prev + a_world * dt
    p_new = p_prev + v_prev * dt + 0.5 * a_world * dt ** 2

    new_state = s.copy()
    new_state[IDX_Q] = quaternion.as_float_array(q_new)
    new_state[IDX_P] = p_new
    new_state[IDX_V] = v_new

    new_state[IDX_BG] = bg_prev
    new_state[IDX_BA] = ba_prev

    return new_state


def transition_fn(error_state: np.ndarray,
                  dt: float,
                  nominal_state: np.ndarray,
                  propagated_nominal: np.ndarray,
                  accelerometer: np.ndarray,
                  gyro: np.ndarray,
                  g_world: np.ndarray) -> np.ndarray:
    full_state = compose_fn(nominal_state, error_state)
    propagated_full = propagate(full_state, dt, accelerometer, gyro, g_world)
    return decompose_fn(propagated_nominal, propagated_full)


def measurement_fn(error_state: np.ndarray, nominal_state: np.ndarray) -> np.ndarray:
    composed_nominal = compose_fn(nominal_state, error_state)
    return np.concat([composed_nominal[IDX_Q], composed_nominal[IDX_P]])


def hx_mean_fn(sigmas, Wm):
    mean_pos = np.dot(Wm, sigmas[:, 4:])

    q_avg = np.dot(Wm, sigmas[:, :4])

    norm = np.linalg.norm(q_avg)
    if norm > 1e-9:
        mean_q = q_avg / norm
    else:
        mean_q = sigmas[0, :4]

    return np.concatenate([mean_q, mean_pos])


def hx_residual_fn(z_actual, z_predicted):
    res_pos = z_actual[4:] - z_predicted[4:]

    q_act = quaternion.quaternion(*z_actual[:4])
    q_pre = quaternion.quaternion(*z_predicted[:4])

    dq = q_pre.inverse() * q_act
    if dq.w < 0:
        dq = -dq

    res_q = quaternion.as_rotation_vector(dq)

    return np.concatenate([[0.0], res_q, res_pos])
