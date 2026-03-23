from pathlib import Path
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    model_dir = Path("/home/ray/yolo_models/helicopter/track")
    results = model.train(data="/home/ray/datasets/helicopter/point_detection/tracking/tracking.yaml",
                          epochs=100, imgsz=1280, device=0, save_dir=str(model_dir),
                          hsv_v=0.9,
                          flipud=0.5,
                          dropout=0.1,
                          rect=True,
                          translate=0.25,
                          optimizer='MuSGD')

    best = YOLO(str(model_dir / "weights" / "best.pt"))
    best.export(format='engine',
                data="/home/ray/datasets/helicopter/point_detection/tracking/tracking.yaml",
                int8=True,
                imgsz=[720, 1280],
                dynamic=False)
