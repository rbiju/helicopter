from pathlib import Path
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    model_dir = Path("/home/ray/yolo_models/helicopter/track_20260406_0")
    model.train(
        data="/home/ray/datasets/helicopter/point_detection/tracking/tracking.yaml",
        epochs=500,
        imgsz=1280,
        batch=10,
        device=0,
        box=12.0,
        save_dir=str(model_dir),
        optimizer='MuSGD',
        dropout=0.1,
        hsv_s=0.5,
        hsv_v=0.5,
        flipud=0.5,
        fliplr=0.5,
        degrees=45.0,
        translate=0.3,
        scale=0.5,
        perspective=0.001,
        mosaic=1.0,
        close_mosaic=25,
        mixup=0.1,
        lr0=2e-1,
        cos_lr=True
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
