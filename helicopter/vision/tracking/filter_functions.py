import jax.numpy as jnp
from jax import jit
from jax.scipy.spatial.transform import Rotation

from helicopter.utils import SystemParams


IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_ACTUAL_COMMANDS = slice(13, 16)
IDX_BATTERY = slice(16, 17)
IDX_TRIM = slice(17, 18)


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
def propagate(s, dt, params: SystemParams, commands, ground=False):
    # Takes about 0.55 milliseconds

    pos_old = s[IDX_P]
    vel_old = s[IDX_V]
    omega_old = s[IDX_O]
    quat_old = Rotation.from_quat(s[IDX_Q])
    battery_old = s[IDX_BATTERY]
    trim_old = s[IDX_TRIM]
    actual_commands_old = s[IDX_ACTUAL_COMMANDS]

    I_TENSOR = jnp.diag(params.I_TENSOR_DIAGONAL)
    I_INVERSE = jnp.diag(1 / params.I_TENSOR_DIAGONAL)
    GYRO_SPRING_VECTOR = jnp.array([params.GYRO_SPRING_CONSTANT, params.GYRO_SPRING_CONSTANT, 0.0])


    # First order model for commands
    actual_thrust_old, actual_pitch_old, actual_yaw_old = actual_commands_old
    thrust, pitch, yaw = commands
    thrust = jnp.sqrt(thrust)

    actual_thrust_new = (actual_thrust_old +
                         ((thrust - actual_thrust_old) / params.ROTOR_TIME_CONSTANT) * dt)

    actual_pitch_new = (actual_pitch_old +
                         ((pitch - actual_pitch_old) / params.PITCH_TIME_CONSTANT) * dt)

    # Mirrors remote control logic for adding yaw + trim
    yaw_cmd = jnp.clip(yaw + ((trim_old - 0.5) / 3), -1.0, 1.0)
    actual_yaw_new = (actual_yaw_old +
                      ((yaw_cmd - actual_yaw_old) / params.YAW_TIME_CONSTANT) * dt).squeeze()

    # Battery
    current_draw = (thrust ** 2 * params.CURRENT_DRAW_COEFF) + \
               (jnp.abs(actual_pitch_new) * params.CURRENT_DRAW_COEFF * 0.05)

    battery_new = battery_old - (current_draw / params.BATTERY_CAPACITY) * dt
    battery_new = jnp.clip(battery_new, 0.0, 1.0)

    v_ocv = 3.0 + (3.2 * battery_new) - (5.0 * battery_new**2) + (3.0 * battery_new**3)
    v_terminal = v_ocv - (current_draw * params.INTERNAL_RESISTANCE)
    voltage_efficiency = v_terminal / 4.2

    # Orientation
    dq = Rotation.from_rotvec(omega_old * dt)
    quat_new = quat_old * dq

    # Velocity
    gravity = quat_new.inv().apply(params.G_WORLD * params.MASS)
    alt = jnp.maximum(pos_old[2], 0.001)
    out_of_bounds = jnp.logical_or(jnp.abs(pos_old[0]) > 0.355, jnp.abs(pos_old[1]) > 0.685)
    table_dropoff = jnp.where(out_of_bounds,
                              0.48,
                              0.0)
    ge_multiplier = 1.0 + params.GROUND_EFFECT_MAX * jnp.exp(-3.0 * ((alt + table_dropoff) / params.ROTOR_RADIUS))

    main_rotor_thrust = (params.THRUST_CONSTANT + actual_thrust_new * params.MAX_THRUST * voltage_efficiency) * ge_multiplier
    tail_rotor_thrust = actual_pitch_new * params.MAX_TAIL_THRUST * voltage_efficiency
    thrust_vector = main_rotor_thrust + tail_rotor_thrust

    drag = params.DRAG * vel_old
    F_net_body = gravity + thrust_vector - drag
    F_net_world_frame = quat_new.apply(F_net_body)

    normal_z_mag = jnp.where(F_net_world_frame[2] < 0, -F_net_world_frame[2], 0.0)
    normal = jnp.where(ground,
                       quat_new.inv().apply(jnp.array([0.0, 0.0, normal_z_mag])),
                       jnp.array([0.0, 0.0, 0.0]))

    F_net = F_net_body + normal
    acc_coriolis = jnp.cross(omega_old, vel_old)

    vel_new = vel_old + ((F_net / params.MASS) - acc_coriolis) * dt

    vel_new_world = quat_new.apply(vel_new)

    is_sinking = ground & (vel_new_world[2] < 0.0)
    vel_new_world = jnp.where(is_sinking, vel_new_world.at[2].set(0.0), vel_new_world)
    vel_new = jnp.where(is_sinking, quat_new.inv().apply(vel_new_world), vel_new)

    # Position
    pos_new = pos_old + vel_new_world * dt
    pos_new = jnp.where(ground & (pos_new[2] <= 0.0), pos_new.at[2].set(0.0), pos_new)

    # Angular Velocity
    tau_roll = acc_coriolis[1] * params.MASS * params.CORIOLIS_CONSTANT
    tau_actuator_pitch = (params.MAX_TAIL_THRUST[2] * actual_pitch_new * voltage_efficiency * params.TAIL_MOMENT_ARM).squeeze()
    tau_actuator_yaw = (actual_yaw_new * params.MAX_YAW_TORQUE * voltage_efficiency).squeeze()

    tau_actuator = jnp.array([tau_roll, tau_actuator_pitch, tau_actuator_yaw])

    world_up = jnp.array([0.0, 0.0, 1.0])
    world_up_in_body = quat_old.inv().apply(world_up)
    local_up = jnp.array([0.0, 0.0, 1.0])
    restoring_torque_dir = jnp.cross(world_up_in_body, local_up)
    tau_gyro = GYRO_SPRING_VECTOR * restoring_torque_dir

    tau_damping = params.ANGULAR_DRAG * omega_old

    tau_net = tau_actuator - tau_gyro - tau_damping

    omega_cross_I_omega = jnp.cross(omega_old, I_TENSOR @ omega_old)
    d_omega = I_INVERSE @ (tau_net - omega_cross_I_omega)

    omega_new = omega_old + d_omega * dt

    s_new = jnp.concatenate([
        quat_new.as_quat(canonical=True),
        pos_new,
        omega_new,
        vel_new,
        jnp.array([actual_thrust_new, actual_pitch_new, actual_yaw_new]),
        battery_new,
        trim_old,
    ])

    return s_new


@jit
def transition_fn(error_state, dt, nominal_state, propagated_nominal, params, commands, ground):
    full_state = compose_fn(nominal_state, error_state)
    propagated_full = propagate(full_state, dt, params, commands, ground)
    return decompose_fn(propagated_nominal, propagated_full)


@jit
def measurement_fn(error_state, ref_point, nominal_state):
    full_state_hypothesis = compose_fn(nominal_state, error_state)

    q = Rotation.from_quat(full_state_hypothesis[IDX_Q])
    t = full_state_hypothesis[IDX_P]

    return q.apply(ref_point) + t
