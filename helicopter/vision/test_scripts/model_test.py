import cv2
from ultralytics import YOLO

from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor
from helicopter.vision.test_scripts.yolo_detect_track import get_refined_keypoints

if __name__ == '__main__':
    imgsz = [720, 1280]
    img = cv2.imread('/home/ray/datasets/helicopter/point_detection/tracking/master/images/val/set27_22.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/track_20260412_0/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(imgsz=imgsz,
                                                             brightness_factor=1.0),
                           imgsz=imgsz,
                           conf=0.1)

    boxes = model(img)
    circles = get_refined_keypoints(img, boxes, margin=2)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    canvas = cv2.drawKeypoints(canvas, circles, None,
                               color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    print('done')
