from ultralytics import YOLO

model = YOLO('models/model.pt')

# Load webcam
results = model(0) 