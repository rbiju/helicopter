from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    results = model.train(data="/home/ray/datasets/helicopter/point_detection/measure/measure.yaml",
                          epochs=300, imgsz=640, device=0, save_dir="/home/ray/yolo_models/helicopter/measure",
                          hsv_v=0.9,
                          flipud=0.5,
                          dropout=0.1,
                          translate=0.25)

    best = YOLO("/home/ray/yolo_models/helicopter/measure/weights/best.pt")
    best.export(format='engine',
                data="/home/ray/datasets/helicopter/point_detection/measure/measure.yaml",
                int8=True,
                imgsz=640,
                dynamic=False)
