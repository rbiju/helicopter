import numpy as np
from numba import jit


@jit(cache=True)
def _kabsch(p: np.ndarray, q: np.ndarray):
    """
    Computes the optimal rotation from measured to reference points, which should capture the camera frame to world frame transformation.
    Args:
        p: reference points
        q: measured points

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


@jit(cache=True)
def _ransac_visual_pose(measured_points: np.ndarray, reference_points: np.ndarray, threshold=15e-3):
    n = measured_points.shape[0]

    dummy_R = np.eye(3, dtype=np.float64)
    dummy_t = np.zeros(3, dtype=np.float64)
    if n < 3:
        print('Not enough points to compute Ransac visual pose')
        return False, dummy_R, dummy_t

    inlier_count = -1
    inlier_mask = np.zeros(n, dtype=np.bool_)

    idx_pool = np.arange(n)
    for _ in range(250):
        np.random.shuffle(idx_pool)
        idxs = idx_pool[:3]

        measured_subset = measured_points[idxs]
        reference_subset = reference_points[idxs]

        r_h, t_h = _kabsch(measured_subset, reference_subset)

        projected = (measured_points @ r_h.T) + t_h

        diff = projected - reference_points
        dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2

        inliers = dist_sq < (threshold * threshold)
        count = np.sum(inliers)

        if count > inlier_count:
            inlier_count = count
            inlier_mask = inliers

            if count == len(measured_points):
                break

    if inlier_count >= 3:
        final_measured = measured_points[inlier_mask]
        final_reference = reference_points[inlier_mask]

        R_final, t_final = _kabsch(final_measured, final_reference)

        return True, R_final, t_final
    else:
        print("Failed to find solution during Ransac")
        return False, dummy_R, dummy_t


@jit(nopython=True, cache=True)
def madgwick_step(q_old_arr: np.ndarray,
                  accel: np.ndarray,
                  gyro: np.ndarray,
                  dt: float,
                  beta: float) -> np.ndarray:
    qw, qx, qy, qz = q_old_arr
    gx, gy, gz = gyro
    ax, ay, az = accel

    qDot_w = 0.5 * (-qx * gx - qy * gy - qz * gz)
    qDot_x = 0.5 * (qw * gx + qy * gz - qz * gy)
    qDot_y = 0.5 * (qw * gy - qx * gz + qz * gx)
    qDot_z = 0.5 * (qw * gz + qx * gy - qy * gx)

    if not (ax == 0.0 and ay == 0.0 and az == 0.0):
        norm_a = 1.0 / np.sqrt(ax * ax + ay * ay + az * az)
        ax *= norm_a;
        ay *= norm_a;
        az *= norm_a

        _2qw, _2qx, _2qy, _2qz = 2.0 * qw, 2.0 * qx, 2.0 * qy, 2.0 * qz
        _4qw, _4qx, _4qy = 4.0 * qw, 4.0 * qx, 4.0 * qy
        _8qx, _8qy = 8.0 * qx, 8.0 * qy

        f0 = 2.0 * (qx * qz - qw * qy) - ax
        f1 = 2.0 * (qw * qx + qy * qz) - ay
        f2 = 2.0 * (0.5 - qx * qx - qy * qy) - az

        step_w = -_2qy * f0 + _2qz * f1
        step_x = _2qz * f0 + _2qw * f1 - _4qx * f2
        step_y = -_2qw * f0 + _2qx * f1 - _4qy * f2
        step_z = _2qx * f0 + _2qy * f1

        norm_step = 1.0 / np.sqrt(step_w ** 2 + step_x ** 2 + step_y ** 2 + step_z ** 2)
        step_w *= norm_step;
        step_x *= norm_step;
        step_y *= norm_step;
        step_z *= norm_step

        qDot_w -= beta * step_w
        qDot_x -= beta * step_x
        qDot_y -= beta * step_y
        qDot_z -= beta * step_z

    qw += qDot_w * dt
    qx += qDot_x * dt
    qy += qDot_y * dt
    qz += qDot_z * dt

    norm_res = 1.0 / np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return np.array([qw * norm_res, qx * norm_res, qy * norm_res, qz * norm_res])
