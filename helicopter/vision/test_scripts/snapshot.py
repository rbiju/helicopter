import numpy as np

from helicopter.vision import D435i


def snapshot(_camera: D435i):
    depth_image = None
    ir_image = None

    snapped = False
    while not snapped:
        frames = _camera.pipeline.poll_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()
        if depth_frame and ir_frame:
            depth_image, _, ir_image, _, _= _camera.process_frames(frames)

            snapped = True
            print("Snapshot retrieved.")

    print(depth_image.shape)
    print(ir_image.shape)
    np.save('depth_frame.npy', depth_image)
    np.save('ir_frame.npy', ir_image)

    _camera.stop()


if __name__ == '__main__':
    camera = D435i(
                   # video_rate=30,
                   # video_resolution=(1280, 720),
                   projector_power=360.,
                   autoexpose=False,
                   exposure_time=1600)
    snapshot(camera)
