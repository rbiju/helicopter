import jax.numpy as jnp
from jax import jit
from jax.scipy.spatial.transform import Rotation

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)

MASS = 0.040
G_WORLD = jnp.array([0, 0, -9.8066])
I_TENSOR = jnp.diag(jnp.array([2e-4, 2e-4, 1e-4]))
I_INVERSE = jnp.diag(1 / (I_TENSOR.diagonal()))
DRAG = jnp.array([0.1,  0.4, 0.25])

MAX_THRUST = 0.80
MAX_TAIL_THRUST = 0.125

TAIL_MOMENT_ARM = 0.12

YAW_TIME_CONSTANT = 0.03

GYRO_SPRING_CONSTANT = jnp.array([0.1, 0.1, 0.0])
ANGULAR_DRAG = jnp.array([0.05, 0.025, 1e-4])
CORIOLIS_CONSTANT = 0.06

@jit
def compose_fn(state, error):
    q_nom = Rotation.from_quat(state[IDX_Q])
    dq = Rotation.from_rotvec(error[:3])
    q_new = q_nom * dq

    new_rest = state[4:] + error[3:]

    return jnp.concatenate([q_new.as_quat(canonical=True), new_rest])


@jit
def decompose_fn(old_state, new_state):
    q_old = Rotation.from_quat(old_state[IDX_Q])
    q_new = Rotation.from_quat(new_state[IDX_Q])

    error_quat = q_old.inv() * q_new
    error_theta = error_quat.as_rotvec()

    error_rest = new_state[4:] - old_state[4:]

    return jnp.concatenate([error_theta, error_rest])


@jit
def propagate(s, dt, throttle, pitch, yaw, trim):
    # Position in world frame
    # Velocity in body frame

    pos_old = s[IDX_P]
    vel_old = s[IDX_V]
    omega_old = s[IDX_O]
    quat_old = Rotation.from_quat(s[IDX_Q])
    battery = s[IDX_BATTERY]

    # Quaternion
    dq = Rotation.from_rotvec(omega_old * dt)
    quat_new = dq * quat_old

    # Position
    vel_rotated = quat_old.apply(vel_old)
    pos_new = pos_old + vel_rotated * dt

    # Velocity
    gravity = quat_new.inv().apply(G_WORLD)
    thrust = throttle * MAX_THRUST * battery + pitch * MAX_TAIL_THRUST * battery
    drag = DRAG * vel_old
    F_net = gravity + thrust - drag
    acc_coriolis = jnp.cross(omega_old, vel_old)

    vel_new = vel_old + ((F_net / MASS) - acc_coriolis) * dt

    # Angular Velocity
    tau_roll = acc_coriolis[1] * MASS * CORIOLIS_CONSTANT
    tau_actuator_pitch = MAX_TAIL_THRUST * pitch * battery * TAIL_MOMENT_ARM
    tau_actuator_yaw = (I_TENSOR[2, 2] / YAW_TIME_CONSTANT) * (yaw + trim) * battery
    tau_actuator = jnp.array([tau_roll, tau_actuator_pitch, tau_actuator_yaw])

    tau_gyro = GYRO_SPRING_CONSTANT * quat_old.as_euler(seq=['X', 'Y', 'Z'], degrees=False)
    tau_damping = ANGULAR_DRAG * omega_old

    tau_net = tau_actuator - tau_gyro - tau_damping

    omega_cross_I_omega = jnp.cross(omega_old, I_TENSOR @ omega_old)
    d_omega = I_INVERSE @ (tau_net - omega_cross_I_omega)

    omega_new = omega_old + d_omega * dt

    # New State
    s_new = jnp.concatenate([quat_new.as_quat(canonical=True), pos_new, vel_new, omega_new])

    return s_new


@jit
def transition_fn(error_state, dt, nominal_state, propagated_nominal, throttle, pitch, yaw, trim):
    full_state = compose_fn(nominal_state, error_state)
    propagated_full = propagate(full_state, dt, throttle, pitch, yaw, trim)
    return decompose_fn(propagated_nominal, propagated_full)


@jit
def measurement_fn(error_state, ref_point, nominal_state):
    full_state_hypothesis = compose_fn(nominal_state, error_state)

    q = Rotation.from_quat(full_state_hypothesis[IDX_Q])
    t = full_state_hypothesis[IDX_P]

    return q.apply(ref_point) + t
