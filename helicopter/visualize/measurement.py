from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import trimesh

from .base import Visualizer


class MeasurementVisualizer(Visualizer):
    def __init__(self):
        super().__init__()

        # TODO: center mesh about centroid
        obj_path = Path(__file__).parent.parent.parent / 'assets/objects/camera/camera.obj'
        camera_mesh = trimesh.load_mesh(obj_path)
        camera_mesh.apply_scale(0.05)

        self.camera_handle = self.server.scene.add_mesh_trimesh(
            "/origin/camera",
            mesh=camera_mesh,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )

        self.point_idxs = []

    def update_camera(self, quat: Rotation, translation: np.ndarray):
        self.camera_handle.wxyz = quat.as_quat(canonical=True, scalar_first=True)
        self.camera_handle.position = translation

    def add_points(self, points: dict[int, np.ndarray]):
        for idx in points.keys():
            if idx not in self.point_idxs:
                self.point_idxs.append(idx)

                self.server.scene.add_point_cloud(
                    name="/origin/points",
                    points=points,
                    colors=np.random.randint(0, 256, 3),
                    point_size=0.005,
                )
