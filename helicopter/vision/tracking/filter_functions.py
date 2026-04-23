import jax.numpy as jnp
from jax import jit
from jax.scipy.spatial.transform import Rotation

from typing import NamedTuple

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)
IDX_TRIM = slice(14, 15)
IDX_ACTUAL_THRUST = slice(15, 16)


class SystemParams(NamedTuple):
    MASS: float
    G_WORLD: jnp.ndarray
    I_TENSOR: jnp.ndarray
    I_INVERSE: jnp.ndarray
    DRAG: jnp.ndarray
    THRUST_CONSTANT: jnp.ndarray
    MAX_THRUST: jnp.ndarray
    GROUND_EFFECT_MAX: float
    ROTOR_RADIUS: float
    MAX_TAIL_THRUST: jnp.ndarray
    TAIL_MOMENT_ARM: float
    YAW_TIME_CONSTANT: float
    GYRO_SPRING_CONSTANT: jnp.ndarray
    ANGULAR_DRAG: jnp.ndarray
    CORIOLIS_CONSTANT: float
    ROTOR_TIME_CONSTANT: float

I_TENSOR = jnp.diag(jnp.array([2e-4, 2e-4, 1e-4]))
SIM_CONSTANTS = SystemParams(
    MASS = 0.0404,
    G_WORLD = jnp.array([0, 0, -9.8066]),
    I_TENSOR = I_TENSOR,
    I_INVERSE = jnp.diag(1 / (I_TENSOR.diagonal())),
    DRAG = jnp.array([0.1,  0.4, 1.35]),
    THRUST_CONSTANT = jnp.array([0.0, 0.0, 0.255]),
    MAX_THRUST = jnp.array([0.0, 0.0, 0.392]),
    MAX_TAIL_THRUST = jnp.array([0.0, 0.0, 0.125]),
    GROUND_EFFECT_MAX=0.2,
    ROTOR_RADIUS=0.1,
    TAIL_MOMENT_ARM = 0.12,
    YAW_TIME_CONSTANT = 0.03,
    GYRO_SPRING_CONSTANT = jnp.array([0.01, 0.01, 0.0]),
    ANGULAR_DRAG = jnp.array([0.05, 0.025, 1e-4]),
    CORIOLIS_CONSTANT = 0.06,
    ROTOR_TIME_CONSTANT=0.25)


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
def propagate(s, dt, commands, ground=False):
    pos_old = s[IDX_P]
    vel_old = s[IDX_V]
    omega_old = s[IDX_O]
    quat_old = Rotation.from_quat(s[IDX_Q])
    battery = s[IDX_BATTERY]
    trim_old = s[IDX_TRIM]
    actual_thrust_old = s[IDX_ACTUAL_THRUST]

    thrust, pitch, yaw = commands

    actual_thrust_new = (actual_thrust_old +
                         ((thrust - actual_thrust_old) / SIM_CONSTANTS.ROTOR_TIME_CONSTANT) * dt)

    # Orientation
    dq = Rotation.from_rotvec(omega_old * dt)
    quat_new = quat_old * dq

    # Velocity
    gravity = quat_new.inv().apply(SIM_CONSTANTS.G_WORLD * SIM_CONSTANTS.MASS)
    alt = jnp.maximum(pos_old[2], 0.001)
    ge_multiplier = (1.0 + SIM_CONSTANTS.GROUND_EFFECT_MAX *
                     jnp.exp(-3.0 * (alt / SIM_CONSTANTS.ROTOR_RADIUS)))

    main_rotor_thrust = (SIM_CONSTANTS.THRUST_CONSTANT + actual_thrust_new *
                         SIM_CONSTANTS.MAX_THRUST * battery) * ge_multiplier
    tail_rotor_thrust = pitch * SIM_CONSTANTS.MAX_TAIL_THRUST * battery
    thrust_vector = main_rotor_thrust + tail_rotor_thrust

    drag = SIM_CONSTANTS.DRAG * vel_old
    F_net_body = gravity + thrust_vector - drag
    F_net_world_frame = quat_new.apply(F_net_body)

    normal_z_mag = jnp.where(F_net_world_frame[2] < 0, -F_net_world_frame[2], 0.0)
    normal = jnp.where(ground,
                       quat_new.inv().apply(jnp.array([0.0, 0.0, normal_z_mag])),
                       jnp.array([0.0, 0.0, 0.0]))

    F_net = F_net_body + normal
    acc_coriolis = jnp.cross(omega_old, vel_old)

    vel_new = vel_old + ((F_net / SIM_CONSTANTS.MASS) - acc_coriolis) * dt

    vel_new_world = quat_new.apply(vel_new)

    is_sinking = ground & (vel_new_world[2] < 0.0)
    vel_new_world = jnp.where(is_sinking, vel_new_world.at[2].set(0.0), vel_new_world)
    vel_new = jnp.where(is_sinking, quat_new.inv().apply(vel_new_world), vel_new)

    # Position
    pos_new = pos_old + vel_new_world * dt
    pos_new = jnp.where(ground & (pos_new[2] <= 0.0), pos_new.at[2].set(0.0), pos_new)

    # Angular Velocity
    tau_roll = acc_coriolis[1] * SIM_CONSTANTS.MASS * SIM_CONSTANTS.CORIOLIS_CONSTANT
    tau_actuator_pitch = (SIM_CONSTANTS.MAX_TAIL_THRUST[2] * pitch * battery * SIM_CONSTANTS.TAIL_MOMENT_ARM).squeeze()
    tau_actuator_yaw = ((SIM_CONSTANTS.I_TENSOR[2, 2] / SIM_CONSTANTS.YAW_TIME_CONSTANT) * jnp.clip(yaw + trim_old, -1.0, 1.0) * battery).squeeze()
    tau_actuator = jnp.array([tau_roll, tau_actuator_pitch, tau_actuator_yaw])

    tau_gyro = SIM_CONSTANTS.GYRO_SPRING_CONSTANT * quat_old.as_euler(seq='XYZ', degrees=False)
    tau_damping = SIM_CONSTANTS.ANGULAR_DRAG * omega_old

    tau_net = tau_actuator - tau_gyro - tau_damping

    omega_cross_I_omega = jnp.cross(omega_old, SIM_CONSTANTS.I_TENSOR @ omega_old)
    d_omega = SIM_CONSTANTS.I_INVERSE @ (tau_net - omega_cross_I_omega)

    omega_new = omega_old + d_omega * dt

    # Battery
    battery_new = battery - (thrust * (dt / 360.))

    s_new = jnp.concatenate([
        quat_new.as_quat(canonical=True),
        pos_new,
        omega_new,
        vel_new,
        battery_new,
        trim_old,
        actual_thrust_new
    ])

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
