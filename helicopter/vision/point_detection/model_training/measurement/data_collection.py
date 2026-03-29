import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from helicopter.vision.d435i import D435i


if __name__ == '__main__':
    camera = D435i(projector_power=360.,
                   autoexpose=False,
                   exposure_time=1800)

    print('Starting data collection in 3 seconds...')
    time.sleep(3)

    print('Collecting data')
    camera.start()
    images = []
    for i in tqdm(range(15)):
        frames = camera.pipeline.wait_for_frames()
        video = camera.process_frames(frames)

        images.append(video.ir_image.copy())
        print(i)

        time.sleep(0.5)

    camera.stop()

    data_path = "/home/ray/datasets/helicopter/point_detection/measure/temp"
    if not os.path.exists(data_path):
        print(f"Making directory {data_path}")
        os.mkdir(data_path)

    for i, image in tqdm(enumerate(images)):
        img = np.repeat(image[..., np.newaxis], 3, -1)
        cv2.imwrite(f"{data_path}/{i}.png", img)

    print(f'Images saved to {data_path}')
