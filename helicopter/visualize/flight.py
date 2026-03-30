import time
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Queue

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.utils import HelicopterModel

from .base import Visualizer
from .aruco_registry import ARUCOMarkerModel, aruco_registry


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
                 fps: float = 30.0):
        super().__init__()
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)
        self.aircraft = None

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


    def initialize(self, aruco_queue: Queue, aircraft_lock: Lock):
        if self.aircraft is None:
            self.aircraft = Aircraft(buffer=self.aircraft_buffer, lock=aircraft_lock)

        aruco_dict = aruco_queue.get()
        for marker_id in aruco_dict.keys():
            rotation = aruco_dict[marker_id]['rotation']
            position = aruco_dict[marker_id]['position']

            mesh_obj: ARUCOMarkerModel = aruco_registry[marker_id]
            mesh = mesh_obj.mesh()

            mesh_handle = self.add_mesh(mesh, f'/aruco_mesh/{marker_id}',
                                        position=(position - mesh_obj.marker_offset),
                                        orientation=rotation.as_quat(canonical=True))
            self.models[marker_id] = mesh_handle

        print("Aruco markers initialized")

        self.update_helicopter(self.aircraft.quaternion, self.aircraft.position)


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

    # TODO: plot state vector to uPlot handles using aircraft timestamp
    # TODO: display commands being sent
    def loop(self):
        self.is_running = True
        while self.is_running:
            if self.kill_signal.is_set():
                raise RuntimeError('Visualizer detected kill signal. Shutting down.')

            if self.last_update_time is None:
                self.last_update_time = time.time()

            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            if elapsed_time < (1 / self.fps):
                time.sleep(0.001)
                continue
            else:
                self.last_update_time = current_time
                quat = self.aircraft.quaternion
                translation = self.aircraft.position

                render_quat = self.camera_quat * quat
                self.update_helicopter(render_quat, translation)

    def cleanup(self):
        self.is_running = False
        self.server.stop()
