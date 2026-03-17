import numpy as np

import trimesh
import viser


class Visualizer:
    def __init__(self):
        self.server = viser.ViserServer(port=6060)
        self.server.gui.configure_theme(dark_mode=True, show_logo=False)

        self.start = False
        self.stop = False

        self.start_button = self.server.gui.add_button('Start')
        self.stop_button = self.server.gui.add_button('Stop')

        self.start_button.on_click(lambda _: self.set_start_flag())
        self.stop_button.on_click(lambda _: self.set_stop_flag())

        self.origin = self.server.scene.add_frame('/origin',
                                                  wxyz=(1.0, 0.0, 0.0, 0.0),
                                                  position=(0.0, 0.0, 0.0),
                                                  visible=False)

    def add_mesh(self, mesh: trimesh.Trimesh, name: str,
                 orientation: np.ndarray = np.array([1.0, 0, 0, 0]),
                 position: np.ndarray = np.array([0.0, 0.0, 0.0])):
        mesh_handle = self.server.scene.add_mesh_trimesh(
            name=name,
            mesh=mesh,
            wxyz=orientation,
            position=position,
        )
        return mesh_handle

    def set_start_flag(self):
        self.start = True

    def set_stop_flag(self):
        self.stop = True

    def cleanup(self):
        self.server.stop()
