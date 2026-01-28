import numpy as np
import quaternion

from helicopter.vision.test_scripts.point_correspondence import generate_reference_points, get_sample_points

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_V = slice(7, 10)
IDX_BA = slice(10, 13)
IDX_BG = slice(13, 16)


def propagate(s: np.ndarray,
              dt,
              accelerometer: np.ndarray,
              gyro: np.ndarray,
              g_world: np.ndarray) -> np.ndarray:
    q_prev = quaternion.quaternion(*s[IDX_Q])
    p_prev = s[IDX_P]
    v_prev = s[IDX_V]
    ba_prev = s[IDX_BA]
    bg_prev = s[IDX_BG]

    gyro_corrected = gyro - s[IDX_BG]
    acc_corrected = accelerometer - s[IDX_BA]

    dq = quaternion.from_rotation_vector(gyro_corrected * dt)
    q_new = q_prev * dq

    a_world = (q_prev * quaternion.from_vector_part(acc_corrected) * q_prev.conjugate()).imag - g_world

    v_new = v_prev + a_world * dt
    p_new = p_prev + v_prev * dt + 0.5 * a_world * dt ** 2

    new_state = s.copy()
    new_state[IDX_Q] = quaternion.as_float_array(q_new)
    new_state[IDX_P] = p_new
    new_state[IDX_V] = v_new

    new_state[IDX_BA] = ba_prev
    new_state[IDX_BG] = bg_prev

    return new_state


def test_propagate():
    s = np.zeros(16)
    s[0] = 1.0

    dt = 1.0

    accelerometer = np.array([0, 0, 9.80665])
    gyro = np.array([np.pi / 2, 0, 0])

    g_world = np.array([0, 0, 9.80665])

    new_state = propagate(s, dt, accelerometer, gyro, g_world)

    print(new_state)


def kabsch(p: np.ndarray, q: np.ndarray):
    """
    Computes the optimal rotation from measured to reference points, which should capture the camera frame to world frame transformation.
    Args:
        p: matched points in the world frame
        q: points in the camera frame

    Returns:

    """
    n = p.shape[0]
    m_c = np.sum(p, axis=0) / n
    r_c = np.sum(q, axis=0) / n

    measured_points_centered = p - m_c
    reference_points_centered = q - r_c

    covar = measured_points_centered.transpose() @ reference_points_centered
    U, s, Vh = np.linalg.svd(covar)

    rotation_matrix = Vh.T @ U.T

    if np.linalg.det(rotation_matrix) < 0:
        Vh_fixed = Vh.copy()
        Vh_fixed[2, :] *= -1
        rotation_matrix = Vh_fixed.T @ U.T

    translation = r_c - (rotation_matrix @ m_c)

    return rotation_matrix, translation


def test_visual_pose_estimate():
    rng = np.random.default_rng(42)
    ref = generate_reference_points(rng, num_reference_points=20)

    sample_clean, _noise, sample, quat, t, corr = get_sample_points(rng, ref, noise_scale=0.0, num_sample_points=8)

    quat = quaternion.from_rotation_vector(np.array([0., 0, np.pi / 2]))
    t = np.array([1., 0, 0])

    sample = quaternion.rotate_vectors(quat, sample_clean) + t

    R, t_pred = kabsch(sample, ref[corr])

    q_pred = quaternion.from_rotation_matrix(R)

    pred_reference = quaternion.rotate_vectors(q_pred.inverse(), sample - t_pred)

    print('recovered reference')
    print(pred_reference)

    print('true reference')
    print(ref[corr])

    print(q_pred, t_pred)


if __name__ == '__main__':
    test_visual_pose_estimate()
