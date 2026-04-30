import time
from helicopter.remote import SymaRemoteRecorder

if __name__ == "__main__":
    recorder = SymaRemoteRecorder()
    record_iters = 100
    frame_count = 0
    while frame_count < record_iters:
        commands = recorder.read_command()
        if len(commands) == 5:
            frame_count += 1
            recorder.remote_state.update(commands)
            print(commands)
            print(recorder.remote_state.get_commands())
        time.sleep(0.001)
