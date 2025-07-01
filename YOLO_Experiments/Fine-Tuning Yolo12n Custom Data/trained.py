from ultralytics import YOLO
model = YOLO("yolo12n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640)