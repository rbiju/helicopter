import time
from helicopter.remote import SymaRemoteRecorder, RecordingPacket
from helicopter.utils import ArduinoLoader

if __name__ == "__main__":
    arduino_loader = ArduinoLoader(sketch_path='py_recorder')
    arduino_loader.load()

    recorder = SymaRemoteRecorder()
    record_iters = 100
    frame_count = 0
    while frame_count < record_iters:
        commands = recorder.read_command()
        if len(commands) == 5:
            frame_count += 1
            commands = RecordingPacket.from_list(commands)
            recorder.remote_state.update(commands)
            print(commands)
            print(recorder.remote_state.convert_to_float())
        time.sleep(0.001)
