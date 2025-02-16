from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "preentrenado.keras"
model = load_model(MODEL_PATH)
CLASSES = ['jeans', 'sofa', 'tshirt', 'tv']
target_size = (224, 224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = CLASSES[predicted_class_index]
        confidence = float(np.max(prediction))
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "API de clasificación de imágenes activa"}
