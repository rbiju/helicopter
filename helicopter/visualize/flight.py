from collections import deque
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Queue
import time

import numpy as np
from scipy.spatial.transform import Rotation

import viser

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft, FlightState
from helicopter.utils import HelicopterModel, CommandBufferConstants

from .base import Visualizer
from .marker_registry import MUTUALLY_EXCLUSIVE_IDS, MarkerModel, model_registry, GameTableModel


CAM_TO_BODY_MATRIX = np.array([
    [0., -1., 0.],
    [0., 0., -1.],
    [1., 0., 0.]
])
coordinate_transform = Rotation.from_matrix(CAM_TO_BODY_MATRIX)


@HydraConfigurable
class FlightVisualizer(Visualizer):
    def __init__(self, aircraft_sm: SharedMemory,
                 kill_signal: Event,
                 board_pieces: list[int],
                 plot_maxlen: int = 300,
                 fps: float = 30.0):
        super().__init__()
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)
        self.aircraft = None

        self.board_pieces = self.validate_board_pieces(board_pieces)

        self.server.initial_camera.position = (-0.25, -0.5, 0.1)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)

        self.server.scene.add_grid(
            "/grid",
            width=5.0,
            height=5.0,
            position=np.array([0.0, 0.0, -0.15]),
            cell_size=0.1,
            cell_color=(0, 255, 0),
            cell_thickness=0.5,
            section_size=0.40,
            section_thickness=1.0,
            section_color=(0, 255, 0)
        )

        self.land_button = self.server.gui.add_button('Kill Flight')
        self.land_button.on_click(lambda _: self.kill_flight())

        helicopter_mesh = HelicopterModel().mesh()
        self.helicopter_handle = self.add_mesh(helicopter_mesh, '/camera')
        self.models = {}

        self.timestamps = deque(maxlen=plot_maxlen)
        self.plot_handles = {}
        self.plot_data = {}
        self.plot_labels = {'Kinematics': {'Position': ['X', 'Y', 'Z'],
                                           'Velocity': ['X', 'Y', 'Z'],
                                           'Orientation': ['W', 'X', 'Y', 'Z'],
                                           'Angular Velocity': ['X', 'Y', 'Z']},
                            'Control': {'Commands': ['Thrust', 'Pitch', 'Yaw']}}
        for group in self.plot_labels.keys():
            with self.server.gui.add_folder(group):
                for plot_name in self.plot_labels[group].keys():
                    lines = self.plot_labels[group][plot_name]
                    handle = self.server.gui.add_uplot(
                        data=(np.array([], dtype=np.float64),
                              *[np.array([], dtype=np.float64) for _ in range(len(lines))],),
                        series=(viser.uplot.Series(label="timestamp"),
                                *[viser.uplot.Series(label=lines[i],
                                                     stroke=["red", "green", "blue"][i % 3],
                                                     width=1) for i in range(len(lines))]),
                        title=plot_name,
                        scales={
                            "x": viser.uplot.Scale(
                                time=False,
                                auto=True,
                            ),
                            "y": viser.uplot.Scale(range=(-1.5, 2.5)),
                        },
                        legend=viser.uplot.Legend(show=True),
                        aspect=2.0,

                    )
                    self.plot_handles[plot_name] = handle
                    self.plot_data[plot_name].update({line: deque(maxlen=plot_maxlen) for line in lines})

        self.status_badge = self.server.gui.add_markdown("")

        self.camera_quat = Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))
        self.origin_quat = Rotation.from_rotvec(np.array([0.0, 0.0, 0.0]))
        self.origin_position = np.array([0.0, 0.0, 0.0])

        self.point_idxs = []

        self.last_position = np.array([0.0, 0.0, 0.0])

        self.camera_quat = None

        self.path_counter = 0
        self.fps = fps
        self.last_update_time = None

        self.is_running = False
        self.kill_signal = kill_signal

    def kill_flight(self):
        self.kill_signal.set()

    def append_to_plot_data(self, name: str, data: np.ndarray):
        for line, data_point in zip(self.plot_data[name], data):
            self.plot_data[name][line].append(data_point)

    def display_system_state(self, flight_state: FlightState):
        self.status_badge.content = (
            f"<div style='background-color: {flight_state.color}; color: white; "
            f"padding: 12px; text-align: center; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1em; width: 100%; box-sizing: border-box;'>"
            f"{flight_state.name}"
            f"</div>"
        )

    @staticmethod
    def validate_board_pieces(board_pieces: list[int]):
        for marker_set in MUTUALLY_EXCLUSIVE_IDS.keys():
            mut_ex_markers = MUTUALLY_EXCLUSIVE_IDS[marker_set]
            if len(set(mut_ex_markers).intersection(board_pieces)) > 1:
                raise RuntimeError(f'Multiple markers from mutually exclusive set {marker_set} found')

        return board_pieces

    def camera_to_table_space(self, object_rotation: Rotation, object_position: np.ndarray,
                              offset_rotation: Rotation, offset_position: np.ndarray):
        """

        Args:
            object_rotation: camera space marker rotation
            object_position: camera space marker position
            offset_rotation: action to return marker in line with object frame
            offset_position: offset between marker and origin in object frame

        Returns:
            Rotation and translation for object in table space, ready for rendering

        """
        # conversion to world space
        object_world_space_rotation = self.camera_quat * object_rotation * offset_rotation
        object_world_space_position = self.camera_quat.apply(object_position)

        # world to table space
        object_table_space_rotation = self.origin_quat.inv() * object_world_space_rotation
        object_table_space_position = (self.origin_quat.inv().apply(object_world_space_position -
                                                                   self.origin_position) -
                                       object_table_space_rotation.apply(offset_position))

        return object_table_space_rotation, object_table_space_position

    def update_helicopter(self, quat: Rotation, translation: np.ndarray):
        total_rotation = self.camera_quat * quat
        self.helicopter_handle.wxyz = total_rotation.as_quat(canonical=True, scalar_first=True)
        self.helicopter_handle.position = translation

        if np.linalg.norm(translation - self.last_position) > 0.005:
            line = np.vstack([self.last_position, translation])
            self.last_position = translation
            self.server.scene.add_line_segments(
                f"/line_segments/{self.path_counter}",
                points=np.expand_dims(line, 0),
                colors=(255, 255, 255),
                line_width=2.0,
            )
            self.path_counter += 1

    def initialize(self, marker_queue: Queue, origin_queue: Queue, aircraft_lock: Lock):
        if self.aircraft is None:
            self.aircraft = Aircraft(buffer=self.aircraft_buffer, lock=aircraft_lock)

        marker_dict = marker_queue.get()
        for marker_id in marker_dict.keys():
            mesh_obj: MarkerModel = model_registry[marker_id]
            if isinstance(mesh_obj, GameTableModel):
                if marker_id in self.board_pieces:
                    origin_dict = {'id': marker_id, 'position': mesh_obj.marker_offset, 'rotation': mesh_obj.marker_rotation}
                    origin_queue.put(origin_dict)

        time.sleep(1.0)

        origin_dict_tracker = origin_queue.get()
        self.origin_quat = origin_dict_tracker['origin_quat']
        self.origin_position = origin_dict_tracker['origin_position']
        self.camera_quat = origin_dict_tracker['camera_quat']

        for marker_id in marker_dict.keys():
            rotation = marker_dict[marker_id]['rotation']
            position = marker_dict[marker_id]['position']
            if marker_id in self.board_pieces:
                mesh_obj: MarkerModel = model_registry[marker_id]
                mesh = mesh_obj.mesh()
                if isinstance(mesh_obj, GameTableModel):
                    mesh_handle = self.add_mesh(mesh, f'/game_table/{marker_id}')
                else:
                    table_space_rotation, table_space_position = self.camera_to_table_space(object_rotation=rotation,
                                                                                            object_position=position,
                                                                                            offset_rotation=mesh_obj.marker_rotation,
                                                                                            offset_position=mesh_obj.marker_offset)
                    mesh_handle = self.add_mesh(mesh, f'/aruco_mesh/{marker_id}',
                                                position=table_space_position,
                                                orientation=table_space_rotation.as_quat(canonical=True))
                self.models[marker_id] = mesh_handle

        self.update_helicopter(self.aircraft.quaternion, self.aircraft.position)
        print("Visualizer initialized")

    def loop(self, command_sm: SharedMemory, lock: Lock):
        command_buffer = np.ndarray(shape=(CommandBufferConstants.N,),
                                    dtype=CommandBufferConstants.dtype,
                                    buffer=command_sm.buf)
        commands = np.empty_like(command_buffer)
        self.is_running = True
        while self.is_running:
            if self.kill_signal.is_set():
                raise RuntimeError('Visualizer detected kill signal. Shutting down.')

            if self.last_update_time is None:
                self.last_update_time = time.time()

            current_time = self.aircraft.timestamp
            elapsed_time = current_time - self.last_update_time
            if elapsed_time < (1 / self.fps):
                time.sleep(0.001)
                continue
            else:
                self.last_update_time = current_time
                state_dict = self.aircraft.state_dict()
                self.update_helicopter(Rotation.from_quat(state_dict['Orientation']), state_dict['Position'])

                with lock:
                    np.copyto(commands, command_buffer)

                for plot_name in self.plot_handles.keys():
                    plot_handle = self.plot_handles[plot_name]
                    self.timestamps.append(plot_handle.timestamp)
                    self.append_to_plot_data(plot_name, state_dict[plot_name])
                    plot_handle.data = (self.timestamps, *list(self.plot_data[plot_name].values()))

                self.display_system_state(state_dict['Flight State'])

    def cleanup(self):
        self.is_running = False
        self.server.stop()
