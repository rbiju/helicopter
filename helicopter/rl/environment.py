from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

from helicopter.aircraft import Aircraft
from helicopter.utils import SystemParams

from .sim import simulate


class FlightEnvironment(gym.Env):
    def __init__(self, time_limit: float = 10.0 ,
                 params_path: str = 'blue_syma',
                 simulation_dt: float = 0.004,
                 command_dt: float = 0.1,):
        self.time_limit = time_limit
        params_path = Path(__file__).parents[2] / "assets/simulation_params" / params_path
        self._params = SystemParams.from_file(params_path)
        self.simulation_dt = simulation_dt
        self.command_dt = command_dt

        self.observation_space = gym.spaces.Dict(
            {
                "Orientation": gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                "Position": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "Angular Velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "Velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "Commands": gym.spaces.Box(np.array([0.0, -1, -1], dtype=np.float32),
                                                  np.array([1.0, 1, 1], dtype=np.float32),
                                           shape=(3,), dtype=np.float32),
                "Battery": gym.spaces.Box(0, 1, shape=(), dtype=np.float32),
                "Trim": gym.spaces.Box(0, 1, shape=(), dtype=np.float32),
                "Setpoint": gym.spaces.Box(np.array([-0.355, -0.6085, 0], dtype=np.float32),
                                           np.array([0.355, 0.6085, 1], dtype=np.float32),
                                           shape=(3,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Box(np.array([0.0, -1, -1]),
                                                  np.array([1.0, 1, 1]), shape=(3,), dtype=np.float32)

        self.aircraft = Aircraft()
        self._noise = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._setpoint = self.observation_space['Setpoint'].sample()

        self.elapsed_time = 0.0

        seed_val = int(self.np_random.integers(0, 2 ** 32 - 1))
        self.key = jax.random.key(seed_val)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        obs_dict = self.aircraft.state_dict
        obs_dict.update({"Setpoint": self._setpoint})
        return obs_dict

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                (self.aircraft.position - self._setpoint)
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        seed_val = int(self.np_random.integers(0, 2 ** 32 - 1))
        self.key = jax.random.key(seed_val)

        low = self.observation_space['Setpoint'].low
        high = self.observation_space['Setpoint'].high

        self._setpoint = self.np_random.uniform(low, high).astype(np.float32)

        self.aircraft.state_vector = Aircraft.default_state()
        pos_sample = self.np_random.uniform(low, high).astype(np.float32)
        self.aircraft.position = np.array([pos_sample[0], pos_sample[1], 0.0])

        self.elapsed_time = 0.0
        self._noise = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one timestep within the environment.
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        for i in range(int(self.command_dt / self.simulation_dt)):
            state = jnp.concatenate((jnp.array(self.aircraft.state_vector), jnp.array(self._noise)))
            sim_out, key = simulate(s=state,
                               dt=self.simulation_dt,
                               params=self._params,
                               commands=action,
                               key=self.key)
            self.aircraft.state_vector = np.array(sim_out[:Aircraft.STATE_N])
            self._noise = np.array(sim_out[Aircraft.STATE_N:])
            self.elapsed_time += self.simulation_dt
            self.key = key

        out_of_time = self.elapsed_time >= self.time_limit
        out_of_bounds = self.observation_space['Setpoint'].contains(self.aircraft.position)
        terminated = out_of_time or out_of_bounds

        error = np.linalg.norm(self.aircraft.position - self._setpoint)

        truncated = False
        reward = -(error ** 2)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="gymnasium_env/FlightEnvironment-v0",
    entry_point="helicopter.rl.environment:FlightEnvironment",
    max_episode_steps=3000,
)