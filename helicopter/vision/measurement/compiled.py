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
def _ransac_visual_pose(measured_points: np.ndarray, reference_points: np.ndarray, threshold=15e-3, iterations=100):
    n = measured_points.shape[0]

    dummy_R = np.eye(3, dtype=np.float64)
    dummy_t = np.zeros(3, dtype=np.float64)
    dummy_rmse = np.float64(0.0)
    if n < 3:
        print('Not enough points to compute Ransac visual pose')
        return False, dummy_R, dummy_t, dummy_rmse

    inlier_count = -1
    inlier_mask = np.zeros(n, dtype=np.bool_)

    idx_pool = np.arange(n)
    for _ in range(iterations):
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

        transformed_points = (R_final @ measured_points.T).T + t_final
        n_points = transformed_points.shape[0]
        errors = np.empty(n_points, dtype=np.float64)

        for i in range(n_points):
            dx = transformed_points[i, 0] - reference_points[i, 0]
            dy = transformed_points[i, 1] - reference_points[i, 1]
            dz = transformed_points[i, 2] - reference_points[i, 2]

            errors[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

        rmse = np.mean(errors)

        return True, R_final, t_final, rmse
    else:
        print("Failed to find solution during Ransac")
        return False, dummy_R, dummy_t, dummy_rmse
