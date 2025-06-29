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
        # Strip the MIME type prefix
        if "," in req.image:
            base64_str = req.image.split(",")[1]  # Only get the part after 'base64,'
        else:
            base64_str = req.image  # Fallback

        # Decode base64 image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        # Dummy detection
        width, height = image.size
        count = 1
        boxes = [{"x": int(width * 0.1), "y": int(height * 0.1), "w": int(width * 0.6), "h": int(height * 0.5)}]

        return {"count": count, "boxes": boxes}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
