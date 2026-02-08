import jax.numpy as jnp
from jax import jit
from jax.scipy.spatial.transform import Rotation

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_V = slice(7, 10)
IDX_BA = slice(10, 13)
IDX_BG = slice(13, 16)


@jit
def compose_fn(state, error):
    q_nom = Rotation.from_quat(state[IDX_Q])
    dq = Rotation.from_rotvec(error[:3])
    q_new = q_nom * dq

    new_rest = state[4:] + error[3:]

    return jnp.concatenate([q_new, new_rest])


@jit
def decompose_fn(old_state, new_state):
    q_old = Rotation.from_quat(old_state[IDX_Q])
    q_new = Rotation.from_quat(new_state[IDX_Q])

    error_quat = q_old.inv() * q_new
    error_theta = error_quat.as_rotvec()

    error_rest = new_state[4:] - old_state[4:]

    return jnp.concatenate([error_theta, error_rest])


@jit
def propagate(s, dt, accel, gyro, g_world):
    q_prev = Rotation.from_quat(s[IDX_Q])
    p_prev = s[IDX_P]
    v_prev = s[IDX_V]
    ba_prev = s[IDX_BA]
    bg_prev = s[IDX_BG]

    gyro_corrected = gyro - bg_prev
    acc_corrected = accel - ba_prev

    dq = Rotation.from_rotvec(gyro_corrected * dt)
    q_new = q_prev * dq

    a_world = q_prev.apply(acc_corrected) - g_world

    v_new = v_prev + a_world * dt
    p_new = p_prev + v_prev * dt + 0.5 * a_world * dt**2

    return jnp.concatenate([
        q_new.as_quat(canonical=True),
        p_new,
        v_new,
        ba_prev,
        bg_prev
    ])


@jit
def transition_fn(error_state, dt, nominal_state, propagated_nominal, accel, gyro, g_world):
    full_state = compose_fn(nominal_state, error_state)
    propagated_full = propagate(full_state, dt, accel, gyro, g_world)
    return decompose_fn(propagated_nominal, propagated_full)


@jit
def measurement_fn(error_state, ref_point, nominal_state):
    full_state_hypothesis = compose_fn(nominal_state, error_state)

    q = Rotation.from_quat(full_state_hypothesis[IDX_Q])
    t = full_state_hypothesis[IDX_P]

    return q.inv().apply(ref_point - t)
