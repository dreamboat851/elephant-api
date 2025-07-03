from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from yolov5_infer import detect_elephants

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"message": "Elephant detection API is online!"}

class ImageRequest(BaseModel):
    image: str  # base64-encoded image with optional MIME prefix (e.g. "data:image/png;base64,...")

@app.post("/detect-elephants")
def detect_elephants_api(req: ImageRequest):
    try:
        # Extract base64 image string
        base64_str = req.image.split(",")[-1]
        # Decode and load image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Run detection
        count, boxes = detect_elephants(image)
        return {
            "count": count,
            "boxes": boxes  # list of dicts with x, y, w, h, confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
