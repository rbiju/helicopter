import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.utils import HelicopterModel
from .base import Visualizer


class FlightVisualizer(Visualizer):
    def __init__(self):
        super().__init__()

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

        camera_mesh = HelicopterModel().mesh()
        self.helicopter_handle = self.add_mesh(camera_mesh, '/camera')

        self.point_idxs = []

        self.last_position = np.array([0.0, 0.0, 0.0])

    def update_camera(self, quat: Rotation, translation: np.ndarray):
        self.camera_handle.wxyz = quat.as_quat(canonical=True, scalar_first=True)
        self.camera_handle.position = translation

        if np.linalg.norm(translation - self.last_position) > 0.005:
            line = np.vstack([self.last_position, translation])
            self.last_position = translation
            self.server.scene.add_line_segments(
                "/line_segments",
                points=np.expand_dims(line, 0),
                colors=(255, 255, 255),
                line_width=2.0,
            )

    def add_points(self, points: np.ndarray):
        # Safe because points are never deleted from the PointHandler
        for idx in range(len(points)):
            if idx not in self.point_idxs:
                self.point_idxs.append(idx)

                self.server.scene.add_point_cloud(
                    name=f"/points/{idx}",
                    points=np.expand_dims(points[idx], 0),
                    colors=(255, 0, 0),
                    point_size=0.002,
                    point_shape="circle",
                )
