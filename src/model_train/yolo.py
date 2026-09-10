import os
import sys
from helper.config import load_config
from ultralytics import YOLO, settings

# 1. Configura os caminhos antes de fazer os imports customizados
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # Define o local do banco de dados na raiz do projeto
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    # Força a ativação do plugin do MLflow na Ultralytics
    settings.update({"mlflow": True})

    # Carrega as configurações do config.yaml
    data_path = load_config('data_path')

    # Inicializa e treina o modelo YOLO
    version = "yolo26n.pt"
    model = YOLO(version)

    model.train(
        data=data_path,
        epochs=10,
        imgsz=640,
        batch=8,
        augment=True,
        workers=4,
        name=f"custom_yolo_model_{version.split('.')[0]}"
    )


if __name__ == '__main__':
    main()
