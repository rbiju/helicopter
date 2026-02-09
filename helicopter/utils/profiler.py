import time
import json
from collections import deque

import numpy as np


class Profiler:
    def __init__(self):
        self.events = {}
        self.start_times = {}

    def start(self, name: str):
        self.start_times[name] = time.perf_counter()
        if name not in self.events:
            self.events[name] = deque(maxlen=500)

    def end(self, name: str):
        t = time.perf_counter()
        if name not in self.events:
            raise RuntimeError(f'Profiler for event {name} was not started')
        self.events[name].append(t - self.start_times[name])

    def __repr__(self):
        means = {}
        for name, times in self.events.items():
            if len(times) == 0:
                means[name] = "N/A"
            else:
                times = np.array(list(times))
                means[name] = {}
                means[name]['mean'] = f"{np.mean(times):.6f} sec"
                means[name]['std'] = f"{np.std(times):.6f} sec"
                means[name]['min'] = f"{np.min(times):.6f} sec"
                means[name]['max'] = f"{np.max(times):.6f} sec"
        means = json.dumps(means, indent=4)
        return (f'Profiler: \n'
                f'{means}')
