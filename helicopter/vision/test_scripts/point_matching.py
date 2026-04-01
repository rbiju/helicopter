import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jaxRotation

from helicopter.vision.tracking import TrianglePointMatcher, ICP


if __name__ == '__main__':
    jax.config.update('jax_disable_jit', True)

    measured_points = np.array([[1.768, -0.0079125, -0.43461],
                                [1.7783, 0.021053, -0.44639],
                                [1.7688, 0.00070465, -0.44736],
                                [1.7945, 0.11468, -0.44259],
                                [1.7672, -0.043063, -0.43921],
                                [1.7704, -0.026978, -0.43964],
                                [1.7892, 0.12382, -0.45662]])

    reference_points = np.load('/home/ray/projects/helicopter/assets/point_clouds/green_syma.npy')
    matcher = TrianglePointMatcher(n=1000, k=20)
    r, t = matcher.get_alignment(sample_points=measured_points)

    icp = ICP(distance_threshold=1e-1, etol=0.003, max_iterations=10)

    q_jax = jaxRotation.from_quat(jnp.array(r.as_quat(canonical=True)))
    _, _, q_new, t_new = icp.iterate(q_old=q_jax, t_old=t, sample_points=measured_points,
                                     reference_points=reference_points)

    print('done')
