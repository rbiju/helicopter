import numpy as np
import quaternion


def generate_reference_points(rng, num_reference_points=20):
    # Generate random points on a sphere r = 0.5
    reference_points = rng.random((num_reference_points, 3)) * 2 - 1
    reference_points = 0.5 * reference_points / np.linalg.norm(reference_points, axis=1)[:, np.newaxis]

    return reference_points


def get_sample_points(rng, reference_points: np.ndarray, noise_scale=1e-3, num_sample_points=6):
    # Sample by choosing random camera point and getting N closest points
    camera_point = rng.standard_normal((3,))
    camera_point = 5 * camera_point / np.linalg.norm(camera_point)

    distances_idxs = np.argsort(np.linalg.norm(camera_point[np.newaxis, :] - reference_points, axis=-1))
    subset = reference_points[distances_idxs][:num_sample_points, :]

    # Random rotation
    random_q = rng.standard_normal(4)
    random_q = random_q / np.linalg.norm(random_q)
    random_q = quaternion.quaternion(*random_q)
    subset_rotated = quaternion.rotate_vectors(random_q, subset)

    # Random translation
    random_t = (rng.random(3) * 2 - 1) * 3
    subset_translated = subset_rotated + random_t

    # Pollute with noise
    noise = rng.normal(scale=noise_scale, size=(num_sample_points, 3))
    subset_noise = subset_translated + noise

    return subset, noise, subset_noise, random_q, random_t, distances_idxs[:num_sample_points]
