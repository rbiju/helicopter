from abc import ABC, abstractmethod
import queue
import threading

import sshkeyboard


class KeyListener:
    def __init__(self):
        self.queue = queue.Queue()
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
            delay_second_char=0.05,
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


class ManualRemoteController(KeyConsumer):
    def __init__(self, listener: KeyListener):
        super().__init__(listener)

        self.throttle = 0
        self.pitch = 63
        self.yaw = 63
        self.trim = 63

    def process(self):
        try:
            while True:
                key = self._listener.queue.get_nowait()
                print(f'Key {key} detected')
        except queue.Empty:
            pass