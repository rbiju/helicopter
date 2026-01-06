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

    # small angle approx.
    dq = quaternion.quaternion(1.0, *(error[:3] / 2))
    quat = quaternion.quaternion(*state[:4]) * dq
    new_state[:4] = quaternion.as_float_array(quat.normalized())

    return new_state


def decompose_fn(old_state: np.ndarray, new_state: np.ndarray):
    error = np.empty(old_state.shape[0] - 1)
    error[3:] = new_state[4:] - old_state[4:]

    error_quat = quaternion.quaternion(*new_state[:4]) * quaternion.quaternion(*old_state[:4]).inverse()
    error_vec = quaternion.as_rotation_vector(error_quat)
    error[:3] = error_vec
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

    new_state = s.copy()

    gyro_corrected = gyro - bg_prev
    accel_corrected = accelerometer - ba_prev

    gyro_norm = np.linalg.norm(gyro_corrected)
    theta_half = gyro_norm * dt / 2.

    if gyro_norm > 1e-6:
        axis = gyro_corrected / gyro_norm
        vec = axis * np.sin(theta_half)
        dq = quaternion.quaternion(np.cos(theta_half), *vec)
    else:
        dq = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    q_new = dq * q_prev

    new_state[IDX_Q] = quaternion.as_float_array(q_new.normalized())

    acc_quat = quaternion.quaternion(0, *accel_corrected)
    acc_rotated = (q_new * acc_quat * q_new.conjugate()).imag

    a_world = acc_rotated - g_world

    v_new = v_prev + (a_world * dt)
    p_new = p_prev + (v_prev * dt) + (0.5 * a_world * dt ** 2)

    # Store results
    new_state[IDX_P] = p_new
    new_state[IDX_V] = v_new

    new_state[IDX_BG] = bg_prev
    new_state[IDX_BA] = ba_prev

    return new_state


def transition_fn(error_state: np.ndarray,
                  dt: float,
                  nominal_state: np.ndarray,
                  accelerometer: np.ndarray,
                  gyro: np.ndarray,
                  g_world: np.ndarray) -> np.ndarray:
    full_state = compose_fn(nominal_state, error_state)
    propagated_full = propagate(full_state, dt, accelerometer, gyro, g_world)
    propagated_nominal = propagate(nominal_state, dt, accelerometer, gyro, g_world)
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

    q_act = z_actual[:4]
    q_pre = z_predicted[:4]

    if np.dot(q_act, q_pre) < 0:
        q_act = -q_act

    res_q = q_act - q_pre
    return np.concatenate([res_q, res_pos])

# from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
#
# # UKF tracks 15D error state
# points = MerweScaledSigmaPoints(n=15, alpha=0.1, beta=2., kappa=0.)
# ukf = UnscentedKalmanFilter(
#     dim_x=15,  # Error state dimension
#     dim_z=measurement_dim,
#     dt=dt,
#     hx=hx_error,
#     fx=fx_error,
#     points=points,
#     x_mean_fn=lambda sigmas, Wm: np.dot(Wm, sigmas),  # Simple mean in error space
#     residual_x=lambda a, b: a - b  # Simple subtraction in error space
# )
#
# # Initialize with zero error
# ukf.x = np.zeros(15)
# ukf.P = initial_error_covariance  # 15x15
#
# # Store nominal state separately
# nominal_state = np.array([1, 0, 0, 0,  # quaternion
#                           0, 0, 0,  # position
#                           0, 0, 0,  # velocity
#                           0, 0, 0,  # accel bias
#                           0, 0, 0])  # gyro bias
#
#
# def fx_error(error_state, dt):
#     """Propagate error state through dynamics"""
#     # Compose error with nominal to get full state
#     full_state = compose_fn(nominal_state, error_state)
#
#     # Propagate full state
#     propagated_full = propagate_dynamics(full_state, dt)
#
#     # Decompose back to error state relative to propagated nominal
#     # (You'd also propagate the nominal separately)
#     propagated_nominal = propagate_dynamics(nominal_state, dt)
#     return decompose_fn(propagated_nominal, propagated_full)
#
#
# def hx_error(error_state):
#     """Measurement function in error space"""
#     full_state = compose_fn(nominal_state, error_state)
#     return measurement_function(full_state)
#
#
# # After each update, reset:
# def reset_error_state():
#     # Apply correction to nominal state
#     global nominal_state
#     nominal_state = compose_fn(nominal_state, ukf.x)
#
#     # Reset error state to zero
#     ukf.x = np.zeros(15)
