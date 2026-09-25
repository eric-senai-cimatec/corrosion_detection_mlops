import os
import json
import cv2
from mlflow import MlflowClient
from ultralytics import YOLO
import shutil


PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")
os.makedirs(REVIEW_DIR, exist_ok=True)

# 1. Carrega o modelo Champion
client = MlflowClient()
model_metadata = client.get_model_version_by_alias(
    "Corrosion_Detection_YOLO_Model", "champion")
clean_path = model_metadata.source.replace(
    "file|///", "").replace("file:///", "").replace("|", "")
model = YOLO(os.path.normpath(clean_path))

# 2. Varre as imagens online
valid_extensions = ('.jpg', '.jpeg', '.png')
images = [f for f in os.listdir(ONLINE_DATA_DIR)
          if f.lower().endswith(valid_extensions)]

for img_name in images:
    img_path = os.path.join(ONLINE_DATA_DIR, img_name)

    # Roda a inferência do YOLO
    results = model.predict(source=img_path, conf=0.25)[0]

    # Carrega dimensões reais da imagem para o JSON do Labelme
    img_bgr = cv2.imread(img_path)
    h, w, _ = img_bgr.shape

    # Estrutura base do arquivo JSON oficial do Labelme
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_name,
        "imageData": None,  # Deixar None força o Labelme a ler o arquivo de imagem local ao lado
        "imageHeight": h,
        "imageWidth": w
    }

    # Converte os boxes do YOLO para o formato de pontos do Labelme [[x1, y1], [x2, y2]]
    for box in results.boxes:
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        cls_id = int(box.cls[0].item())
        label_name = results.names[cls_id]

        shape = {
            "label": label_name,
            "points": [
                [coords[0], coords[1]],  # Canto superior esquerdo
                [coords[2], coords[3]]  # Canto inferior direito
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_json["shapes"].append(shape)

    # Salva a imagem e o JSON de pré-anotação na pasta de revisão humana
    shutil.copy2(img_path, os.path.join(REVIEW_DIR, img_name))
    json_path = os.path.join(
        REVIEW_DIR, os.path.splitext(img_name)[0] + ".json")
    with open(json_path, "w") as f:
        json.dump(labelme_json, f, indent=2)

    # Limpa a pasta temporária original
    os.remove(img_path)

print(f"📥 {len(images)} imagens enviadas com pré-anotações automáticas para: {REVIEW_DIR}")
