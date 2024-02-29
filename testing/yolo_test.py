from ultralytics import YOLO

model = YOLO('models/model.pt')

results = model(1)

# Results.xyxy[0] contains the detected object info for the first image/frame
# Each detection is [xmin, ymin, xmax, ymax, confidence, class]
for detection in results.xyxy[0]:
    xmin, ymin, xmax, ymax, conf, cls = detection
    print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")