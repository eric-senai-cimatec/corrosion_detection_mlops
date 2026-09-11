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
    # Configuração do banco de dados na raiz do projeto
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    settings.update({"mlflow": True})

    model_path = load_config('model_path')
    data_path = load_config('data_path')

    # Carregar o modelo treinado
    model = YOLO(model_path)

    # Inicia a run exclusiva para o teste
    with mlflow.start_run(run_name="evaluation_test_set") as run:
        # Adiciona Tags estruturais para a Run
        mlflow.set_tag("pipeline_stage", "testing")
        mlflow.log_param("evaluated_model_path", model_path)

        # Executa a avaliação no split de teste
        metrics = model.val(
            data=data_path,
            split='test'
        )

        # Converte métricas do YOLO para float nativo
        current_recall = float(metrics.box.r[0]) if hasattr(
            metrics.box.r, "__len__") else float(metrics.box.r)
        current_precision = float(metrics.box.p[0]) if hasattr(
            metrics.box.p, "__len__") else float(metrics.box.p)
        current_mAP50 = float(metrics.box.map50)
        current_mAP50_95 = float(metrics.box.map)

        # Loga no MLflow Tracking
        mlflow.log_metrics({
            "metrics_recall": current_recall,
            "metrics_precision": current_precision,
            "metrics_mAP50": current_mAP50,
            "metrics_mAP50_95": current_mAP50_95
        })

        # Salva o arquivo físico do peso como artefato da Run
        model_filename = os.path.basename(model_path)
        mlflow.log_artifact(model_path, artifact_path="yolo_weights")

        # Registra oficialmente esse modelo no catálogo central do MLflow
        local_artifact_uri = f"file:///{os.path.join(run.info.artifact_uri, 'yolo_weights', model_filename)}"
        model_details = mlflow.register_model(
            local_artifact_uri, "Corrosion_Detection_YOLO_Model")

        # ---- LÓGICA DE NEGÓCIO MULTI-MÉTRICA (CHAMPION VS CHALLENGER VS REJECT) ----
        client = MlflowClient()

        # Definição das Regras de Negócio Básicas (Thresholds)
        MIN_MAP50 = 0.50   # Precisão mínima de localização
        MIN_RECALL = 0.40  # Mínimo de taxa de captura de defeitos real

        print("\n--- VALIDAÇÃO DE REGRAS DE NEGÓCIO ---")
        print(
            f"Métricas obtidas -> mAP50: {current_mAP50:.4f} | Recall: {current_recall:.4f}")

        # Filtro 1: Verificação de Limites Mínimos (Portão de Qualidade)
        if current_mAP50 < MIN_MAP50 or current_recall < MIN_RECALL:
            print(
                f"❌ Modelo REPROVADO nos requisitos mínimos (mAP50 Mín: {MIN_MAP50} / Recall Mín: {MIN_RECALL}).")
            client.set_registered_model_alias(
                name="Corrosion_Detection_YOLO_Model",
                alias="rejected",
                version=model_details.version
            )
        else:
            # Filtro 2: Comparação multi-variável contra o Campeão atual
            try:
                champion_metadata = client.get_model_version_by_alias(
                    name="Corrosion_Detection_YOLO_Model",
                    alias="champion"
                )

                champion_run = client.get_run(champion_metadata.run_id)
                champion_mAP50 = float(
                    champion_run.data.metrics.get("metrics_mAP50", 0.0))
                champion_recall = float(
                    champion_run.data.metrics.get("metrics_recall", 0.0))

                print(
                    f"Campeão atual (Versão {champion_metadata.version}) -> mAP50: {champion_mAP50:.4f} | Recall: {champion_recall:.4f}")

                # Critério de Desempate: O modelo novo é melhor se possuir mAP50 superior E não reduzir o Recall,
                # OU se trouxer um ganho substancial em Recall (foco em segurança) mantendo a estabilidade.
                # Regra adotada: Média aritmética das duas métricas precisa ser estritamente superior.
                current_score = (current_mAP50 + current_recall) / 2
                champion_score = (champion_mAP50 + champion_recall) / 2

                if current_score > champion_score:
                    print(
                        f"🏆 NOVO CAMPEÃO! Score geral superou o anterior ({current_score:.4f} > {champion_score:.4f}).")
                    client.set_registered_model_alias(
                        name="Corrosion_Detection_YOLO_Model",
                        alias="champion",
                        version=model_details.version
                    )
                else:
                    print(
                         "⚔️ CHALLENGER! Modelo passou nos testes, mas score geral inferior ao campeão atual.")
                    client.set_registered_model_alias(
                        name="Corrosion_Detection_YOLO_Model",
                        alias="challenger",
                        version=model_details.version
                    )

            except mlflow.exceptions.MlflowException:
                # Primeiro modelo que cruza a linha de qualidade mínima vira campeão por padrão
                print(
                    "🥇 Primeiro modelo válido aprovado no projeto. Promovido automaticamente a champion!")
                client.set_registered_model_alias(
                    name="Corrosion_Detection_YOLO_Model",
                    alias="champion",
                    version=model_details.version
                )

        # ---- ATRIBUIÇÃO DE METADADOS ADICIONAIS (TAGS) ----
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

        # Vinculação da versão do Dataset via DVC
        dvc_file_path = os.path.join(PROJECT_ROOT, "data.dvc")
        dvc_hash = "unknown"

        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                dvc_data = yaml.safe_load(f)
                dvc_hash = dvc_data.get("outs", [{}])[0].get("md5", "unknown")

        client.set_registered_model_tag(
            name="Corrosion_Detection_YOLO_Model",
            key="dataset_version_md5",
            value=dvc_hash
        )

    print("\nAvaliação de teste concluída e registrada com sucesso no MLflow!")


if __name__ == '__main__':
    main()
