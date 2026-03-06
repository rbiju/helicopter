import time

from helicopter.utils import KeyListener, ManualRemoteController

if __name__ == '__main__':
    listener = KeyListener()
    rc = ManualRemoteController(listener=listener)

    print('Listening...')
    count = 0
    while count < 2000:
        rc.process()
        time.sleep(0.01)

    listener.stop()

    print('Done')