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
        #print("📥 Received base64 string starts with:", req.image[:50])
        #print("📥 Total length:", len(req.image))
        
        # Strip the MIME prefix if present
        if "," in req.image:
            base64_str = req.image.split(",")[1]
        else:
            base64_str = req.image

        # Decode base64 image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        # Dummy logic
        width, height = image.size
        count = 1
        boxes = [{"x": int(width * 0.1), "y": int(height * 0.1), "w": int(width * 0.6), "h": int(height * 0.5)}]

        return {"count": count, "boxes": boxes}
    except Exception as e:
        print("🚨 ERROR:", str(e))
        raise HTTPException(status_code=400, detail=str(e))
