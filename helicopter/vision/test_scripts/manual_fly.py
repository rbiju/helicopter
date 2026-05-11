import time

from helicopter.remote import SymaRemoteControl, RemoteControlThread, ControlPacket
from helicopter.utils import ArduinoLoader, KeyListener, ManualController

if __name__ == '__main__':
    arduino_loader = ArduinoLoader(sketch_path='py_controller')
    arduino_loader.load()

    listener = KeyListener()
    controller = ManualController(listener)

    rc_thread = RemoteControlThread(rc=SymaRemoteControl())

    listener.start()
    rc_thread.start()
    print('Starting manual flight')
    while not controller.quit:
        controller.process()
        commands = ControlPacket(*controller.convert_to_float())
        rc_thread.update(commands)
        time.sleep(0.01)