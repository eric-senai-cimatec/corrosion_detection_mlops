import os
import json
import cv2
from mlflow import MlflowClient
from ultralytics import YOLO
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")

def run_active_learning_inference():
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.makedirs(REVIEW_DIR, exist_ok=True)

    # 1. Loads the Champion model inside the execution context
    client = MlflowClient()
    model_metadata = client.get_model_version_by_alias("Corrosion_Detection_YOLO_Model", "champion")
    clean_path = model_metadata.source.replace("file|///", "").replace("file:///", "").replace("|", "")

    normalized_path = os.path.normpath(clean_path)
    if normalized_path.startswith("home/"):
        normalized_path = "/" + normalized_path

    print(f"🎯 [Automation] Loading model weights from absolute path: {normalized_path}")
    model = YOLO(normalized_path)

    # 2. Scans online images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not os.path.exists(ONLINE_DATA_DIR):
        print("⚠️ ONLINE_DATA_DIR does not exist.")
        return

    images = [f for f in os.listdir(ONLINE_DATA_DIR) if f.lower().endswith(valid_extensions)]

    for img_name in images:
        img_path = os.path.join(ONLINE_DATA_DIR, img_name)

        results = model.predict(source=img_path, conf=0.30)[0]

        img_bgr = cv2.imread(img_path)
        h, w, _ = img_bgr.shape

        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        for box in results.boxes:
            coords = box.xyxy[0].tolist() # CORRIGIDO: Adicionado [0] para extração correta de tensores
            cls_id = int(box.cls[0].item()) if hasattr(box.cls, "__len__") else int(box.cls.item())
            label_name = results.names[cls_id]

            shape = {
                "label": label_name,
                "points": [
                    [coords[0], coords[1]],
                    [coords[2], coords[3]]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_json["shapes"].append(shape)

        shutil.copy2(img_path, os.path.join(REVIEW_DIR, img_name))
        json_path = os.path.join(REVIEW_DIR, os.path.splitext(img_name)[0] + ".json") # CORRIGIDO: Adicionado [0] no splitext
        with open(json_path, "w") as f:
            json.dump(labelme_json, f, indent=2)

        os.remove(img_path)

    print(f"📥 {len(images)} images sent with automatic pre-annotations to: {REVIEW_DIR}")

if __name__ == "__main__":
    run_active_learning_inference()
