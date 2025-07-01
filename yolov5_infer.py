import sys
from pathlib import Path

# Add yolov5 to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

import torch
from PIL import Image
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

device = select_device('')
model = DetectMultiBackend(weights='yolov5s.pt', device=device)
model.model.float().eval()

ELEPHANT_CLASS_ID = 21

def detect_elephants(image: Image.Image):
    import torchvision.transforms as transforms
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
