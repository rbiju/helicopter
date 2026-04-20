from abc import ABC, abstractmethod
import queue
import threading

import sshkeyboard


class KeyListener:
    def __init__(self, key_queue = None):
        if key_queue is None:
            self.queue = queue.Queue()
        else:
            self.queue = key_queue
        self._thread = None

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        sshkeyboard.stop_listening()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("Keyboard Listener stopped.")

    def _run(self):
        sshkeyboard.listen_keyboard(
            on_press=self.queue.put,
            delay_second_char=0.02,
            sleep=0.01
        )


class KeyConsumer(ABC):
    def __init__(self, listener: KeyListener):
        self._listener = listener

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    @abstractmethod
    def process(self):
        raise NotImplementedError


class Quitter(KeyConsumer):
    def __init__(self, listener: KeyListener, quit_key: str = 'q'):
        super().__init__(listener)
        self.quit_key = quit_key
        self.quit = False

    def process(self):
        try:
            while True:
                key = self._listener.queue.get_nowait()
                if key == self.quit_key:
                    self.quit = True
                    print("Quit key detected")
        except queue.Empty:
            pass


class ManualController(KeyConsumer):
    """
    Also implements the Command protocol
    """
    def __init__(self, listener: KeyListener):
        super().__init__(listener)

        self.channel = 128
        self.throttle = 0
        self.pitch = 63
        self.yaw = 63
        self.trim = 63

        self.quit = False

    @staticmethod
    def clip(value: int):
        return min(max(value, 0), 127)

    def process(self):
        while not self._listener.queue.empty():
            key = self._listener.queue.get()
            if key == 'w':
                self.throttle = self.clip(self.throttle + 5)
            elif key == 's':
                self.throttle = self.clip(self.throttle - 5)
            elif key == 'up':
                self.pitch = self.clip(self.pitch - 5)
            elif key == 'down':
                self.pitch = self.clip(self.pitch + 5)
            elif key == 'right':
                self.yaw = self.clip(self.yaw - 5)
            elif key == 'left':
                self.yaw = self.clip(self.yaw + 5)
            elif key == 'd':
                self.trim = self.clip(self.trim - 1)
            elif key == 'a':
                self.trim = self.clip(self.trim + 1)
            elif key == 'q':
                self.quit = True
            elif key == 'r':
                self.reset()
            print(f"c:{self.channel} - t:{self.throttle} - p:{self.pitch} - y:{self.yaw} - tr:{self.trim}")

    def reset(self):
        self.throttle = 0
        self.pitch = 63
        self.yaw = 63
        self.trim = 63

    def convert_to_float(self):
        thrust = self.throttle / 127.
        pitch = (self.pitch - 63.) / 128. * 2
        yaw = (self.yaw - 63.) / 128. * 2
        return [thrust, pitch, yaw]

    def format(self) -> list[int]:
        return [self.channel,
                self.yaw,
                self.pitch,
                self.throttle,
                self.trim]
