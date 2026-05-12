from typing import Optional

import numpy as np
import gymnasium as gym

from helicopter.aircraft import Aircraft


class FlightEnvironment(gym.Env):
    def __init__(self, size: int = 5):
        self.size = size

        self._initial_state = Aircraft.default_state()

        self.observation_space = gym.spaces.Dict(
            {
                "quaternion": gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                "position": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "angular_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "actual_commands": gym.spaces.Box(np.array([0.0, -1, -1]),
                                                  np.array([1.0, 1, 1]), shape=(3,), dtype=np.float32),
                "battery": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "trim": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "setpoint": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Box(np.array([0.0, -1, -1]),
                                                  np.array([1.0, 1, 1]), shape=(3,), dtype=np.float32)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        pass

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
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
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._initial_state = Aircraft.default_state()

        # Randomly place target, ensuring it is different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info