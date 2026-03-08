import time

from helicopter.utils import KeyListener, ManualController, SymaRemoteControl

if __name__ == '__main__':
    listener = KeyListener()
    controller = ManualController(listener=listener)
    rc = SymaRemoteControl(port='/dev/ttyUSB0', baudrate=115200)

    try:
        listener.start()
        print('Listening...')

        elapsed_time = 0
        start_time = time.time()
        while elapsed_time < 120.0:
            elapsed_time = time.time() - start_time
            controller.process()
            command = controller.format()
            rc.send_command(command)

    finally:
        controller.reset()
        rc.send_command(controller.format())
        listener.stop()

    print('Done')
