import os
import mlflow
from ultralytics import YOLO

# Garante que aponta para o mesmo banco de dados SQLite central do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

# Busca dinamicamente o modelo que está com o alias "champion" (produção)
# O MLflow resolve o caminho dos artefatos automaticamente por baixo dos panos!
model_uri = "models:/Corrosion_Detection_YOLO_Model@champion"
model_cache_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

# Carrega no YOLO para fazer predições
model = YOLO(model_cache_path)

# Executa a inferência em uma nova imagem de inspeção
results = model.predict(source=r"C:\Users\eric.santoss\Documents\corrosion_detection_mlops\data\images\test\1_jpg.rf.zmrrEH9XTrsjtb5X52YR.jpg", save=True)
print(results)
