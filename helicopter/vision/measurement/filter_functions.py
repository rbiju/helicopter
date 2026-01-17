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


def project_to_tangent_space(target_quat,
                             target_pos: np.ndarray,
                             ref_quat: quaternion.quaternion) -> np.ndarray:
    q_diff = ref_quat.conjugate() * target_quat

    if q_diff.w < 0:
        q_diff = -q_diff

    rot_vec = quaternion.as_rotation_vector(q_diff)

    return np.concatenate([rot_vec, target_pos])


def measurement_fn(error_state: np.ndarray,
                   nominal_state: np.ndarray,
                   ref_quat: quaternion.quaternion) -> np.ndarray:
    full_state_hypothesis = compose_fn(nominal_state, error_state)

    q_sigma = quaternion.quaternion(*full_state_hypothesis[IDX_Q])
    p_sigma = full_state_hypothesis[IDX_P]

    return project_to_tangent_space(q_sigma, p_sigma, ref_quat)
