from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import logging
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "preentrenado.keras"
try:
    logger.info("Cargando el modelo desde %s", MODEL_PATH)
    model = load_model(MODEL_PATH)
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error("Error al cargar el modelo: %s", e)
    model = None

CLASSES = ['jeans', 'sofa', 'tshirt', 'tv']
target_size = (224, 224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    try:
        contents = await file.read()
        with Image.open(io.BytesIO(contents)) as img:
            img = img.convert("RGB")
            img = img.resize(target_size)
            image_array = np.array(img, dtype=np.float32) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        predicted_label = CLASSES[predicted_class_index]
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        logger.exception("Error en la predicción: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API de clasificación de imágenes activa"}
