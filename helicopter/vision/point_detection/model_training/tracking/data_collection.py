import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from helicopter.vision.d435i import D435i


if __name__ == '__main__':
    camera = D435i(video_resolution=(720, 1280),
                   video_rate=30,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=2400)

    print('Starting detection in 3 seconds...')
    time.sleep(3)

    dataset_name = 'set24'
    data_path = Path("/home/ray/datasets/helicopter/point_detection/tracking") / dataset_name / 'images'
    if not os.path.exists(data_path):
        print(f"Making directory {str(data_path)}")
        os.mkdir(data_path)
    else:
        raise RuntimeError("Directory ", str(data_path), " already exists.")

    print(f'Collecting data for {dataset_name}')
    camera.start()
    images = []
    for i in tqdm(range(25)):
        frames = camera.pipeline.wait_for_frames()
        video = camera.process_frames(frames)

        images.append(video.ir_image.copy())
        time.sleep(1.0)

    camera.stop()

    for i, image in tqdm(enumerate(images)):
        img = np.repeat(image[..., np.newaxis], 3, -1)
        cv2.imwrite(f"{data_path}/{dataset_name}_{i}.png", img)

    print(f'Images saved to {data_path}')
