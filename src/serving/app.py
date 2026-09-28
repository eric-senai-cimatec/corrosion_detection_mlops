import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from mlflow import MlflowClient

# 1. Configuração do ambiente e banco do MLflow
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"


def load_champion_model() -> YOLO:
    """Busca dinamicamente o modelo champion no banco SQLite e limpa a URL do Windows"""
    print("🔍 [MLflow] Buscando o modelo 'champion' no Model Registry...")
    try:
        client = MlflowClient()
        model_metadata = client.get_model_version_by_alias(
            name="Corrosion_Detection_YOLO_Model",
            alias="champion"
        )

        # Extração segura do Run ID para montar o caminho físico do Windows
        raw_source = model_metadata.source
        run_id_match = re.search(r'([a-f0-9]{32})', raw_source)

        if not run_id_match:
            raise ValueError(
                f"Run ID inválido no metadado do modelo: {raw_source}")

        run_id = run_id_match.group(1)
        model_path = os.path.normpath(os.path.join(
            PROJECT_ROOT, "mlruns", "1", run_id, "artifacts", "yolo_weights", "best.pt"
        ))

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Arquivo físico do modelo não encontrado em: {model_path}")

        print(
            f"✅ [MLflow] Modelo carregado com sucesso da pasta: {model_path}")
        return YOLO(model_path)

    except Exception as e:
        print(f"❌ [Erro] Falha ao carregar modelo do Registry: {e}")
        print("⚠️ Certifique-se de que o mlflow.db está populado e o alias 'champion' está configurado.")
        raise RuntimeError(e)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Gerencia o ciclo de vida da aplicação (Startup e Shutdown) armazenando o estado de forma limpa."""
    try:
        # Armazena a instância do modelo dentro do estado controlado do FastAPI
        fastapi_app.state.model = load_champion_model()
    except Exception:
        fastapi_app.state.model = None

    yield
    # Recursos podem ser limpos aqui no shutdown se necessário
    print("🔌 Encerrando o servidor de aplicação.")


# Inicializa o FastAPI acoplando o gerenciador de ciclo de vida
app = FastAPI(
    title="Automated Corrosion Detection API",
    description="API para detecção dinâmica de corrosão usando o modelo @champion do MLflow",
    version="3.0.0",
    lifespan=lifespan
)


@app.post("/predict", summary="Inferência em tempo real para detecção de corrosão")
async def predict(request: Request, file: UploadFile = File(...)):
    # Recupera o modelo de forma segura de dentro do estado da requisição atual
    model: YOLO = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503, detail="Modelo preditivo indisponível ou não inicializado no servidor.")

    # Validação do tipo de arquivo enviado
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="O arquivo enviado precisa ser uma imagem válida.")

    try:
        # Lê os bytes do arquivo recebido via HTTP e converte para formato OpenCV (BGR)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Falha ao decodificar a imagem recebida.")

        # Executa a inferência na imagem isolando o primeiro resultado
        results = model.predict(source=img, conf=0.30, verbose=False)[0]

        detections = []
        # Extrai os metadados das caixas encontradas
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
            status_code=500, detail=f"Erro interno no processamento: {str(e)}")


@app.post("/reload", summary="Atualiza o modelo em produção sem derrubar a API")
def reload_model(request: Request):
    """Endpoint administrativo para recarregar o campeão atual de forma thread-safe"""
    try:
        # Substitui a instância antiga no estado da aplicação pela nova versão promovida
        request.app.state.model = load_champion_model()
        return {"status": "success", "message": "Modelo em produção atualizado dinamicamente!"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Falha ao recarregar modelo: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
