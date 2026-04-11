from pathlib import Path
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    model_dir = Path("/home/ray/yolo_models/helicopter/track_20260410_0")
    model.train(
        data="/home/ray/datasets/helicopter/point_detection/tracking/tracking.yaml",
        epochs=350,
        imgsz=1280,
        batch=8,
        device=0,
        box=2.0,
        cls=3.0,
        dfl=10.0,
        save_dir=str(model_dir),
        optimizer='MuSGD',
        hsv_s=0.1,
        hsv_v=0.1,
        flipud=0.5,
        fliplr=0.5,
        # scale=0.1,
        perspective=0.0,
        # mosaic=1.0,
        # close_mosaic=25,
        cos_lr=True,
        max_det=20
    )

    best = YOLO(str(model_dir / "weights" / "best.pt"))
    best.export(format='engine',
                data="/home/ray/datasets/helicopter/point_detection/tracking/tracking.yaml",
                simplify=True,
                int8=False,
                half=True,
                imgsz=[736, 1280],
                split='train',
                fraction=1.0,
                dynamic=False)
