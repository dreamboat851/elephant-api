import torch
from pathlib import Path
from PIL import Image
import sys

# Append the local YOLOv5 directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load model
device = select_device('')
model = DetectMultiBackend(weights='yolov5s.pt', device=device)
model.model.float().eval()

ELEPHANT_CLASS_ID = 21  # COCO class ID for elephants

def detect_elephants(image: Image.Image):
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)

    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    detections = []

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], image.size).round()
        for *xyxy, conf, cls in pred:
            if int(cls.item()) == ELEPHANT_CLASS_ID:
                x1, y1, x2, y2 = [int(c.item()) for c in xyxy]
                detections.append({
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "confidence": round(conf.item(), 3)
                })

    return len(detections), detections
