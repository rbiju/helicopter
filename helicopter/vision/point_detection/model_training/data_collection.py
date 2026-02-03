import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from helicopter.vision.d435i import D435i


if __name__ == '__main__':
    camera = D435i(projector_power=360.,
                   toggle_projector=False,
                   autoexpose=False,
                   exposure_time=2400)

    images = []
    for i in tqdm(range(10)):
        frames = camera.pipeline.wait_for_frames()
        depth_image, ts_depth, ir_image, ts_ir, laser_state = camera.process_frames(frames)

        images.append(ir_image.copy())
        print(i)

        time.sleep(0.5)

    camera.stop()

    data_path = "/home/ray/datasets/helicopter/point_detection/temp"
    if not os.path.exists(data_path):
        print(f"Making directory {data_path}")
        os.mkdir(data_path)

    for i, image in tqdm(enumerate(images)):
        img = np.repeat(image[..., np.newaxis], 3, -1)
        cv2.imwrite(f"{data_path}/{i}.png", img)

    print(f'Images saved to {data_path}')
