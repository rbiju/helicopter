import ast
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

class RecordingAsset:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.mkdir(parents=True, exist_ok=True)

        self._df = None
        self._initial_state = None
        self._simulation_params = None
        self._param_sweep = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.filepath / 'dataframe.csv')
            df['command'] = df['command'].apply(lambda x: np.array(ast.literal_eval(x)))
            df['position'] = df['position'].apply(lambda x: np.array(ast.literal_eval(x)))
            self._df = df

        return self._df

    @property
    def initial_state(self) -> np.ndarray:
        if self._initial_state is None:
            self._initial_state = np.load(self.filepath / 'initial_state.npy')
        return self._initial_state

    def simulation_params(self, params_class) -> dict:
        if self._simulation_params is None:
            with open(self.filepath / 'parameters.json', 'r') as f:
                data = json.load(f)

            obj = params_class(**data)
            self._simulation_params = jax.tree_util.tree_map(
                lambda x: jnp.array(x) if isinstance(x, list) else x,
                                                             obj,
                is_leaf=lambda x: isinstance(x, list)
            )

        return self._simulation_params

    @property
    def param_sweep(self) -> dict:
        if self._param_sweep is None:
            with open(self.filepath / 'sweep.json', 'r') as f:
                data = json.load(f)
                self._param_sweep = data

        return self._param_sweep

    def write(self, df: pd.DataFrame, initial_state: np.ndarray):
        df.to_csv(self.filepath / 'dataframe.csv', index=False)
        np.save(self.filepath / 'initial_state.npy', initial_state)

    def save_simulation_params(self, params: NamedTuple):
        serializable_state = jax.tree_util.tree_map(
            lambda x: x.tolist() if isinstance(x, jax.Array) else x,
            params
        )
        with open(self.filepath / 'parameters.json', 'w') as f:
            json.dump(serializable_state._asdict(), f)

    def save_simulation_sweep(self, sweep: dict):
        with open(self.filepath / 'sweep.json', 'w') as f:
            json.dump(sweep, f)
