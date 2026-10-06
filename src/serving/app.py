import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from mlflow import MlflowClient

# 1. Environment configuration and MLflow database setup
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"


def load_champion_model() -> YOLO:
    """Dynamically fetches the champion model from the SQLite database and sanitizes the Windows URL."""
    print("🔍 [MLflow] Searching for the 'champion' model in the Model Registry...")
    try:
        client = MlflowClient()
        model_metadata = client.get_model_version_by_alias(
            name="Corrosion_Detection_YOLO_Model",
            alias="champion"
        )

        # Safe extraction of Run ID to build the physical Windows path
        raw_source = model_metadata.source
        run_id_match = re.search(r'([a-f0-9]{32})', raw_source)
        if not run_id_match:
            raise ValueError(f"Invalid Run ID in model metadata: {raw_source}")

        run_id = run_id_match.group(1)
        model_path = os.path.normpath(os.path.join(
            PROJECT_ROOT, "mlruns", "1", run_id, "artifacts", "yolo_weights", "best.pt"
        ))

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Physical model file not found at: {model_path}")

        print(
            f"✅ [MLflow] Model successfully loaded from folder: {model_path}")
        return YOLO(model_path)

    except Exception as e:
        print(f"❌ [Error] Failed to load model from Registry: {e}")
        print("⚠️ Make sure mlflow.db is populated and the 'champion' alias is configured.")
        raise RuntimeError(e)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Manages the application lifecycle (Startup and Shutdown), storing the state cleanly."""
    try:
        # Stores the model instance inside the controlled FastAPI state
        fastapi_app.state.model = load_champion_model()
    except Exception:
        fastapi_app.state.model = None
    yield
    # Resources can be cleaned up here during shutdown if needed
    print("🔌 Shutting down the application server.")

# Initializes FastAPI by attaching the lifecycle manager
app = FastAPI(
    title="Automated Corrosion Detection API",
    description="API for dynamic corrosion detection using MLflow's @champion model",
    version="3.0.0",
    lifespan=lifespan
)


@app.post("/predict", summary="Real-time inference for corrosion detection")
async def predict(request: Request, file: UploadFile = File(...)):
    # Safely retrieves the model from the current request state
    model: YOLO = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Predictive model unavailable or not initialized on the server."
        )

    # Validation of the uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be a valid image."
        )

    try:
        # Reads bytes from the received HTTP file and converts them to OpenCV (BGR) format
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode the received image.")

        # Executes inference on the image, isolating the first result
        results = model.predict(source=img, conf=0.30, verbose=False)[0]
        detections = []

        # Extracts metadata from the detected bounding boxes
        for box in results.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(c, 2) for c in coords]
            })

        return {
            "has_detections": len(detections) > 0,
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(e)}"
        )


@app.post("/reload", summary="Updates the production model without downing the API")
def reload_model(request: Request):
    """Administrative endpoint to reload the current champion in a thread-safe manner"""
    try:
        # Replaces the old instance in the application state with the newly promoted version
        request.app.state.model = load_champion_model()
        return {"status": "success", "message": "Production model dynamically updated!"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
