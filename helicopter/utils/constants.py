from dataclasses import dataclass

import numpy as np


@dataclass
class CommandBufferConstants:
    N: int = 3
    dtype = np.float64
