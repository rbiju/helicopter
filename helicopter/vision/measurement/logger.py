import csv
import numpy as np


class StateLogger:
    def __init__(self, filepath='debug_log.csv'):
        self.filepath = filepath
        self.data = []
        self.headers = [
            'timestamp', 'event',
            'qw', 'qx', 'qy', 'qz',
            'px', 'py', 'pz',
            'vx', 'vy', 'vz',
            'b_ax', 'b_ay', 'b_az',
            'b_gx', 'b_gy', 'b_gz'
        ]

    def log_state(self, timestamp: float, event: str, state_vector: np.ndarray):
        s = state_vector.flatten()

        row = [
            f"{timestamp:.6f}",
            event,
            f"{s[0]:.6f}", f"{s[1]:.6f}", f"{s[2]:.6f}", f"{s[3]:.6f}",
            f"{s[4]:.6f}", f"{s[5]:.6f}", f"{s[6]:.6f}",
            f"{s[7]:.6f}", f"{s[8]:.6f}", f"{s[9]:.6f}",
            f"{s[10]:.6f}", f"{s[11]:.6f}", f"{s[12]:.6f}",
            f"{s[13]:.6f}", f"{s[14]:.6f}", f"{s[15]:.6f}"
        ]
        self.data.append(row)

    def save(self):
        print(f"Saving log to {self.filepath} with {len(self.data)} entries...")
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)
        print("Log saved.")
