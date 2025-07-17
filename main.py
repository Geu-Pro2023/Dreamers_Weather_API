from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize app
app = FastAPI(
    title="IndabaX_Hackathon South-Sudan 2025",
    description="Weather Condition Classification - Dreamers Team Weather Predictor API",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
MODEL_PATH = "dreamers_weather_model.h5"
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Load model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Preprocess function
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Home/Health check
@app.get("/", tags=["Utility"])
async def root():
    return {"message": "🌤️ Dreamers Weather Predictor API is up and running."}

# List of class labels
@app.get("/labels", tags=["Utility"])
async def get_labels():
    return {"weather_classes": CLASS_NAMES}

# Predict weather from image
@app.post("/predict", response_class=JSONResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        prediction = model.predict(input_tensor)
        predicted_index = np.argmax(prediction)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(prediction))

        return {
            "prediction": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Reload model from disk (e.g. after update)
@app.put("/model/reload", tags=["Admin"])
async def reload_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")

# Clear temporary files (if needed)
@app.delete("/clear", tags=["Admin"])
async def clear_temp_files():
    # In a real app, you'd clear uploads or cache here
    return {"message": "No temporary files to clear (placeholder)."}
