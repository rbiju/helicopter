import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SystemParams(NamedTuple):
    MASS: float
    G_WORLD: jnp.ndarray
    I_TENSOR_DIAGONAL: jnp.ndarray
    DRAG: jnp.ndarray
    THRUST_CONSTANT: jnp.ndarray
    MAX_THRUST: jnp.ndarray
    GROUND_EFFECT_MAX: float
    ROTOR_RADIUS: float
    MAX_TAIL_THRUST: jnp.ndarray
    TAIL_MOMENT_ARM: float
    MAX_YAW_TORQUE: float
    GYRO_SPRING_CONSTANT: float
    ANGULAR_DRAG: jnp.ndarray
    CORIOLIS_CONSTANT: float
    ROTOR_TIME_CONSTANT: float
    PITCH_TIME_CONSTANT: float
    YAW_TIME_CONSTANT: float
    CURRENT_DRAW_COEFF: float = 2.5
    BATTERY_CAPACITY: float = 540.
    INTERNAL_RESISTANCE: float = 0.15
    OU_THETA: float = 0.2
    OU_SIGMA: float = 0.03

    @classmethod
    def from_file(cls, filepath: Path):
        with open(filepath / 'parameters.json', 'r') as f:
            data = json.load(f)

        obj = cls(**data)
        _simulation_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, list) else x,
            obj,
            is_leaf=lambda x: isinstance(x, list)
        )

        return _simulation_params

    def __repr__(self):
            args = []
            for key, val in self._asdict().items():
                if hasattr(val, "tolist"):
                    args.append(f"{key}=jnp.array({val.tolist()})")
                else:
                    args.append(f"{key}={repr(val)}")

            formatted_args = ",\n    ".join(args)
            return f"{self.__class__.__name__}(\n    {formatted_args}\n)"