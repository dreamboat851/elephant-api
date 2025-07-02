import sys
from pathlib import Path
# Add yolov5 to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Set up device and model
device = select_device('')
model = DetectMultiBackend(weights='yolov5s.pt', device=device)
model.model.float().eval()

ELEPHANT_CLASS_ID = 20  # COCO class ID for elephants

def letterbox_pil(image, target_size=640):
    """PIL-compatible letterbox that mimics YOLOv5 behavior exactly."""
    # Calculate scaling factor (maintain aspect ratio)
    scale = min(target_size / image.width, target_size / image.height)
    
    # Calculate new dimensions
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    
    # Resize image
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create padded image with gray background (YOLOv5 default)
    padded = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    
    # Center the resized image
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(resized, (offset_x, offset_y))
    
    return padded

def detect_elephants(image: Image.Image):
    """
    Detect elephants in image. Returns coordinates in ORIGINAL image space.
    """
    try:
        # Apply letterboxing (preserves aspect ratio)
        img_resized = letterbox_pil(image, target_size=640)
        
        # Convert to tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img_resized).unsqueeze(0).to(device)
        
        # Run inference
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        
        detections = []
        if pred is not None and len(pred):
            # Scale coordinates back to original image size
            pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], image.size).round()
            
            # Extract elephant detections
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
    
    except Exception as e:
        print(f"Detection failed: {e}")
        return 0, []
