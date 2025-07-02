import sys
from pathlib import Path
import numpy as np
# Add yolov5 to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

import torch
from PIL import Image
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Set up device and model
device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

ELEPHANT_CLASS_ID = 20

def detect_elephants(image: Image.Image):
    """
    Simple, production-ready elephant detection.
    Returns coordinates in original image space.
    """
    try:
        # Convert PIL to numpy (RGB)
        img0 = np.array(image)
        
        # Apply letterbox (official YOLOv5 preprocessing)
        img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
        
        # Prepare for model
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]  # add batch dimension
        
        # Run inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        
        detections = []
        
        # Process results
        for det in pred:
            if len(det):
                # Scale boxes back to original image size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Extract elephants
                for *xyxy, conf, cls in det:
                    if int(cls) == ELEPHANT_CLASS_ID:
                        x1, y1, x2, y2 = [int(c.item()) for c in xyxy]
                        detections.append({
                            "x": x1,
                            "y": y1,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "confidence": round(conf.item(), 3)
                        })
        
        return len(detections), detections
    
    except Exception as e:
        print(f"Detection failed: {e}")
        return 0, []
