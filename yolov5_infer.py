import torch
import numpy as np
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=False)
model.eval()

# COCO class index for elephants is 21
ELEPHANT_CLASS_ID = 21

def detect_elephants(image: Image.Image):
    # Convert PIL image to format expected by YOLOv5
    results = model(image)

    detections = results.xyxy[0]  # Bounding boxes (x1, y1, x2, y2, conf, class)
    elephant_detections = []

    for det in detections:
        class_id = int(det[5].item())
        if class_id == ELEPHANT_CLASS_ID:
            x1, y1, x2, y2, conf, _ = det.tolist()
            elephant_detections.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": round(conf, 3)
            })

    return len(elephant_detections), elephant_detections
