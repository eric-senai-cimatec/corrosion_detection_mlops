import os
import sys
from ultralytics import YOLO, settings

# 1. Configura os caminhos antes de fazer os imports customizados
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 2. Configura a URI local para o SQLite do MLflow
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    # Força a ativação do plugin do MLflow na Ultralytics
    settings.update({"mlflow": True})

    # 3. Localiza dinamicamente o seu arquivo de configuração YAML na raiz
    config_yaml_path = os.path.join(PROJECT_ROOT, "model_args.yaml")

    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(
            f"Arquivo de configuracao nao encontrado em: {config_yaml_path}")

    # 4. Inicializa o modelo base definido no seu YAML (yolo26n.pt)
    # Como o modelo está declarado dentro do YAML, você pode passar a string direto ou fixar
    model = YOLO("yolo26n.pt")

    print(
        f"🚀 Iniciando treinamento do YOLO utilizando as configuracoes de: {config_yaml_path}")

    # 5. A mágica acontece aqui: passamos apenas o caminho do arquivo no parâmetro 'cfg'
    model.train(cfg=config_yaml_path)


if __name__ == '__main__':
    main()
