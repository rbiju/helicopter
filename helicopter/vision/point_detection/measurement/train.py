from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(data="/home/ray/datasets/helicopter/point_detection/master/measure.yaml",
                      epochs=300, imgsz=640, device=0, save_dir="/home/ray/yolo_models/helicopter/measure",
                      hsv_v=0.75,
                      flipud=0.5)

best = YOLO("/home/ray/yolo_models/helicopter/measure/weights/best.pt")
best.export(format='engine',
            data="/home/ray/datasets/helicopter/point_detection/master/measure.yaml",
            int8=True,
            imgsz=640,
            dynamic=False)
