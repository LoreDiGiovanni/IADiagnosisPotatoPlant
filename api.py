from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

app = FastAPI(docs_url=None, redoc_url=None)
MODEL = tf.keras.models.load_model("models/model.2.h5") 
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    npimage = np.array(Image.open(BytesIO(img)))
    predict = MODEL.predict(np.expand_dims(npimage, 0))
    return {
            "prediction": CLASS_NAMES[np.argmax(predict[0])],
            "confidence": float(np.max(predict[0]))
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


