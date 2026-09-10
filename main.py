import os
from mlflow import MlflowClient
from ultralytics import YOLO

# 1. Resolução da raiz do projeto e banco de dados SQLite
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

print("Buscando o modelo 'champion' no Model Registry do MLflow...")

try:
    # 2. Busca os detalhes da versão marcada como 'champion' de forma simples
    client = MlflowClient()
    model_metadata = client.get_model_version_by_alias(
        name="Corrosion_Detection_YOLO_Model",
        alias="champion"
    )

    # 3. Limpeza direta e infalível da URL corrompida do Windows/SQLite
    raw_path = model_metadata.source
    clean_path = raw_path.replace(
        "file|///", "").replace("file:///", "").replace("|", "")
    model_final_path = os.path.normpath(clean_path)

    print(f"Modelo localizado e carregado com sucesso: {model_final_path}")

    # 4. Carrega no YOLO para fazer predições
    model = YOLO(model_final_path)

    # 5. Executa a inferência na imagem de teste
    image_source = r"C:\Users\eric.santoss\Documents\corrosion_detection_mlops\data\images\test\1_jpg.rf.zmrrEH9XTrsjtb5X52YR.jpg"

    results = model.predict(
        source=image_source,
        save=True,      # Salva os resultados visuais em runs/detect/predict
        conf=0.25       # Limiar de confiança
    )

    print("\nPredição concluída! Verifique os resultados na pasta runs/detect/predict.")

except Exception as e:
    print(f"\nErro ao carregar o modelo do Registry: {e}")
    print("Verifique se o seu modelo está registrado e com o alias 'champion' ativo na UI.")
