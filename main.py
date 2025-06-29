from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Elephant detection API is online!"}

class ImageRequest(BaseModel):
    image: str

@app.post("/detect-elephants")
def detect_elephants(req: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image)
        image = Image.open(BytesIO(image_data))

        # Dummy logic for detection
        width, height = image.size
        count = 1  # Replace with real model result
        boxes = [
            {"x": int(width * 0.1), "y": int(height * 0.1), "w": int(width * 0.6), "h": int(height * 0.5)}
        ]

        return {"count": count, "boxes": boxes}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
