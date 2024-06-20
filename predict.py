from ultralytics import YOLO

model = YOLO('PlasticModel.pt')

results = model(source=0, show=True, conf=0.3, save=True)