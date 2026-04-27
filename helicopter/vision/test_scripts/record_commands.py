import time
from helicopter.utils import RemoteState, SymaRemoteRecorder

if __name__ == "__main__":
    remote_state = RemoteState(recorder=SymaRemoteRecorder())

    record_iters = 100
    frame_count = 0
    while frame_count < record_iters:
        commands = remote_state.recorder.read_command()
        if len(commands) > 0:
            frame_count += 1
            remote_state.update(commands)
            print(commands)
            print(remote_state.convert_to_float())
        time.sleep(0.001)
