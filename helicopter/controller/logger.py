from collections import deque
import csv
from pathlib import Path

import numpy as np


class PIDLogger:
    def __init__(self, save_dir='pid'):
        self.save_dir = Path(__file__).parents[2] / 'notebooks/logs' / save_dir

        self.data = deque()
        self.headers = [
            'timestamp',
            'e_t', 'e_p', 'e_y',
            't_p', 't_i', 't_d', 't_accumulator',
            'p_p', 'p_i', 'p_d', 'p_accumulator',
            'y_p', 'y_i', 'y_d', 'y_accumulator'
        ]

    def log(self, timestamp: float, error: np.ndarray, controller_states: list[np.ndarray]):
        row: list = [None] * (1 + 3 + (3*4))

        row[0] = f"{timestamp:.6f}"
        row[1] = f"{error[0]:.6f}"
        row[2] = f"{error[1]:.6f}"
        row[3] = f"{error[2]:.6f}"

        idx = 4
        for arr in controller_states:
            for val in arr:
                row[idx] = f"{val:.6f}"
                idx += 1

        self.data.append(row)

    @staticmethod
    def write_file(save_path: Path, headers, data):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    def save(self):
        print(f"Saving PID logs to {self.save_dir}...")

        log_path = self.save_dir / "log.csv"

        self.write_file(log_path, self.headers, list(self.data))

        print(f"Saved {len(self.data)} PID state entries.")
