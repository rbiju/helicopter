import quaternion

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class SimulationAnimator:
    def __init__(self):
        self.arrow = None

        self.T_IDX = 0
        self.CMD_IDX = slice(1, 4)
        self.ARROW_IDX = slice(4, 10)

    def arrow_generator(self, df):
        i = 0
        heading = quaternion.quaternion(0., 1.0, 0.0, 0.0)
        while i < len(df):
            series = df.iloc[i]
            x, y, z = series['x'], series['y'], series['z']
            q_w, q_i, q_j, q_k = series['q_w'], series['q_i'], series['q_j'], series['q_k']
            q = quaternion.quaternion(q_w, q_i, q_j, q_k)
            rotated_heading = (q * heading * q.inverse()).imag
            yield (series['time'],
                   series['thrust'], series['pitch'], series['yaw'],
                   x, y, z, *rotated_heading)
            i += 1

    def parse_frame_data(self, output):
        t = output[self.T_IDX]
        pos = output[self.ARROW_IDX][:3]
        cmd = output[self.CMD_IDX]
        arrow_args = output[self.ARROW_IDX]

        return (f"{t:.3f}",
                f"x={pos[0]:.3f} \ny={pos[1]:.3f} \nz={pos[2]:.3f}",
                f"thrust_cmd={cmd[0]:.3f} \npitch_cmd={cmd[1]:.3f} \nyaw_cmd={cmd[2]:.3f}",
                arrow_args)

    def plot_trajectory(self, df: pd.DataFrame, setpoint):
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-1, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.plot(df['x'], df['y'], df['z'], 'b--', linewidth=0.5, alpha=0.5)
        ax.scatter(setpoint[0], setpoint[1], setpoint[2], color='r', marker='*')

        generator = self.arrow_generator(df)
        first_frame = self.parse_frame_data(next(generator))
        self.arrow = ax.quiver(*first_frame[3], color='b', length=0.75, pivot='middle')
        timestamp = ax.text2D(-0.25, 0.0, f"t = {first_frame[0]}", transform=ax.transAxes)
        pos = ax.text2D(-0.25, 1.0, f"{first_frame[1]}", transform=ax.transAxes)
        cmd = ax.text2D(0.9, 1.0, f"{first_frame[1]}", transform=ax.transAxes)

        def update(frame):
            frame = self.parse_frame_data(frame)
            timestamp.set_text(f't = {frame[0]}')
            pos.set_text(f'{frame[1]}')
            cmd.set_text(f'{frame[2]}')

            self.arrow.remove()
            self.arrow = ax.quiver(*frame[3], color='b', length=0.75, pivot='middle')

            return (self.arrow,)

        ani = FuncAnimation(fig, update, frames=generator, interval=40, blit=False, cache_frame_data=False)

        return ani
