from typing import Tuple
import jax
import jax.numpy as jnp

from flax import struct
from gymnax.environments import environment, spaces

from helicopter.utils import SystemParams
from .sim import simulate


@struct.dataclass
class EnvState:
    state_vector: jnp.ndarray
    noise: jnp.ndarray
    setpoint: jnp.ndarray
    time: float


@struct.dataclass
class EnvParams:
    time_limit: float = 10.0
    simulation_dt: float = 0.004
    command_dt: float = 0.1
    sys_params: SystemParams = struct.field(default_factory=SystemParams)


class FlightEnvironment(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
            self, key: jax.Array, state: EnvState, action: jnp.ndarray, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        n_sim_steps = int(params.command_dt / params.simulation_dt)

        def step_fn(carry, _):
            curr_state_vec, curr_noise, curr_key = carry
            s_concat = jnp.concatenate([curr_state_vec, curr_noise])

            sim_out, next_key = simulate(
                s=s_concat,
                dt=params.simulation_dt,
                params=params.sys_params,
                commands=action,
                key=curr_key
            )

            _next_state_vec = sim_out[:18]
            _next_noise = sim_out[18:]
            return (_next_state_vec, _next_noise, next_key), None

        (next_state_vec, next_noise, step_key), _ = jax.lax.scan(
            step_fn,
            (state.state_vector, state.noise, key),
            xs=None,
            length=n_sim_steps
        )

        next_time = state.time + params.command_dt
        next_state = EnvState(
            state_vector=next_state_vec,
            noise=next_noise,
            setpoint=state.setpoint,
            time=next_time
        )

        pos = next_state_vec[4:7]
        error = jnp.linalg.norm(pos - state.setpoint)
        reward = -(error ** 2)

        out_of_time = next_time >= params.time_limit
        out_of_bounds = (jnp.abs(pos[0]) > 0.355) | (jnp.abs(pos[1]) > 0.6085) | (pos[2] < 0.0) | (pos[2] > 1.0)
        done = out_of_time | out_of_bounds

        info = {"distance": error}
        return self.get_obs(next_state, params), next_state, reward, done, info

    def reset_env(
            self, key: jax.Array, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Sample setpoint
        setpoint = jax.random.uniform(
            subkey1, shape=(3,),
            minval=jnp.array([-0.355, -0.6085, 0.0]),
            maxval=jnp.array([0.355, 0.6085, 1.0])
        )

        pos_sample = jax.random.uniform(
            subkey2, shape=(2,),
            minval=jnp.array([-0.355, -0.6085]),
            maxval=jnp.array([0.355, 0.6085])
        )

        # Initialize default state vector
        init_state_vec = jnp.zeros(18, dtype=jnp.float32)
        init_state_vec = init_state_vec.at[0:4].set(jnp.array([0.0, 0.0, 0.0, 1.0]))  # Quat
        init_state_vec = init_state_vec.at[4:7].set(jnp.array([pos_sample[0], pos_sample[1], 0.0]))  # Pos
        init_state_vec = init_state_vec.at[16].set(1.0)  # Battery
        init_state_vec = init_state_vec.at[17].set(0.5)  # Trim

        state = EnvState(
            state_vector=init_state_vec,
            noise=jnp.zeros(3, dtype=jnp.float32),
            setpoint=setpoint,
            time=0.0
        )

        return self.get_obs(state, params), state

    def get_obs(self, state, params=None, key=None) -> jax.Array:
        return jnp.concatenate([
            state.state_vector,
            state.setpoint
        ])

    @property
    def name(self) -> str:
        return "FlightEnvironment-v0"

    @property
    def num_actions(self) -> int:
        return 3

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([0.0, -1.0, -1.0]),
            high=jnp.array([1.0, 1.0, 1.0]),
            shape=(3,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(21,), dtype=jnp.float32
        )
