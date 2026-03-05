import viser


class Visualizer:
    def __init__(self):
        self.server = viser.ViserServer()

        self.start = False
        self.stop = False

        self.start_button = self.server.gui.add_button('Start')
        self.stop_button = self.server.gui.add_button('Stop')

        self.start_button.on_click(lambda _: self.set_start_flag())
        self.stop_button.on_click(lambda _: self.set_stop_flag())

        self.origin = self.server.scene.add_frame('/origin',
                                                  wxyz=(1.0, 0.0, 0.0, 0.0),
                                                  position=(0.0, 0.0, 0.0))
    def set_start_flag(self):
        self.start = True

    def set_stop_flag(self):
        self.stop = True
