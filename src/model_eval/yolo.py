from ultralytics import YOLO, settings
import os
import sys
import mlflow
from mlflow import MlflowClient
from helper.config import load_config
import yaml

# 1. Configura os caminhos antes de fazer os imports customizados
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 1. Altere a URI para usar o SQLite. Isso habilita o Model Registry local!
    # O arquivo 'mlflow.db' será criado automaticamente na sua raiz.
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    settings.update({"mlflow": True})

    model_path = load_config('model_path')
    data_path = load_config('data_path')

    # Carregar o modelo treinado
    model = YOLO(model_path)

    # Inicia a run exclusiva para o teste
    with mlflow.start_run(run_name="evaluation_test_set") as run:
        # 1. Adiciona Tags para identificar e filtrar facilmente na UI do MLflow
        mlflow.set_tag("pipeline_stage", "testing")
        mlflow.log_param("evaluated_model_path", model_path)

        # 2. Executa a avaliação no split de teste
        metrics = model.val(
            data=data_path,
            split='test'
        )

        # Loga explicitamente os resultados finais estruturados (mAP50, mAP50-95, etc)
        # 3. Extrai e loga as métricas diretamente convertendo para float nativo do Python
        mlflow.log_metrics({
            "metrics_recall": float(metrics.box.r[0]) if hasattr(metrics.box.r, "__len__") else float(metrics.box.r),
            "metrics_precision": float(metrics.box.p[0]) if hasattr(metrics.box.p, "__len__") else float(metrics.box.p),
            "metrics_mAP50": float(metrics.box.map50),
            "metrics_mAP50_95": float(metrics.box.map)
        })

        # 1. Loga o arquivo físico do peso (.pt) do YOLO como um artefato da Run
        mlflow.log_artifact(model_path, artifact_path="yolo_weights")

        # 2. Registra oficialmente esse modelo no catálogo central do MLflow
        # CORREÇÃO: Usamos o caminho absoluto local (URI baseada em arquivo) para registrar pesos puros sem erro
        model_filename = os.path.basename(model_path)
        local_artifact_uri = f"file:///{os.path.join(run.info.artifact_uri, 'yolo_weights', model_filename)}"

        model_details = mlflow.register_model(local_artifact_uri, "Corrosion_Detection_YOLO_Model")

        # ---- ATRIBUIÇÃO DE ALIAS ----
        client = MlflowClient()
        client.set_registered_model_alias(
            name="Corrosion_Detection_YOLO_Model",
            alias="champion",
            version=model_details.version
        )

        # ADICIONE ISSO PARA PREENCHER AS TAGS AUTOMATICAMENTE:
        client.set_registered_model_tag(
            name="Corrosion_Detection_YOLO_Model",
            key="framework",
            value="ultralytics-yolo"
        )
        client.set_registered_model_tag(
            name="Corrosion_Detection_YOLO_Model",
            key="task",
            value="corrosion-detection"
        )

        # 1. Busca o caminho do data.dvc usando a PROJECT_ROOT central
        dvc_file_path = os.path.join(PROJECT_ROOT, "data.dvc")
        dvc_hash = "unknown"

        # 2. Abre o arquivo .dvc e extrai o hash MD5 real do dataset
        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                dvc_data = yaml.safe_load(f)
                # Pega o MD5 do primeiro output rastreado
                dvc_hash = dvc_data.get("outs", [{}])[0].get("md5", "unknown")

        # 3. Salva o hash real na Tag do MLflow!
        client.set_registered_model_tag(
            name="Corrosion_Detection_YOLO_Model",
            key="dataset_version_md5",
            value=dvc_hash
        )


if __name__ == '__main__':
    main()
