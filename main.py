from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from yolov5_infer import detect_elephants

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Elephant detection API is online!"}

class ImageRequest(BaseModel):
    image: str  # base64-encoded string with MIME prefix

@app.post("/detect-elephants")
def detect_elephants_api(req: ImageRequest):
    try:
        # Extract base64 string from data URI
        if "," in req.image:
            base64_str = req.image.split(",")[1]
        else:
            base64_str = req.image

        # Decode base64 to image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Run elephant detection
        count, boxes = detect_elephants(image)

        return {
            "count": count,
            "boxes": boxes  # each box includes x, y, w, h, confidence
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
