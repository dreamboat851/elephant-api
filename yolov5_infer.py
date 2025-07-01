import sys
from pathlib import Path
from PIL import Image
import torch

# Add yolov5 to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Constants
ELEPHANT_CLASS_ID = 21  # COCO class ID for elephant

# Setup
device = select_device('')
model = DetectMultiBackend(weights='yolov5s.pt', device=device)
model.model.float().eval()

def detect_elephants(image: Image.Image):
    import torchvision.transforms as transforms

    # Resize to model input size (YOLOv5 default is 640x640)
    resized = image.resize((640, 640))
    transform = transforms.ToTensor()
    img_tensor = transform(resized).unsqueeze(0).to(device)

    # Inference
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Process detections
    detections = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], image.size).round()
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
