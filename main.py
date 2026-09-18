import os
import sys
from mlflow import MlflowClient
from ultralytics import YOLO

# 1. Garante a raiz correta baseado na pasta atual do script
# Como main.py está na raiz, o PROJECT_ROOT é o próprio diretório do arquivo.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

# Define o caminho correto das imagens de produção
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")

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

    # 4. Carrega o modelo YOLO campeão
    model = YOLO(model_final_path)

    # 5. Varre a pasta online_data procurando imagens válidas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(ONLINE_DATA_DIR):
        raise FileNotFoundError(
            f"A pasta de produção não foi localizada em: {ONLINE_DATA_DIR}")

    images = [os.path.join(ONLINE_DATA_DIR, f) for f in os.listdir(ONLINE_DATA_DIR) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"Nenhuma nova imagem encontrada na pasta: {ONLINE_DATA_DIR}")
        sys.exit()

    print(f"Iniciando pseudo-rotulagem de {len(images)} imagens do mundo real...")

    # 6. Executa a inferência em lote (batch processing)
    # save_txt=True gera os arquivos .txt com os rótulos automáticos
    # save_conf=False evita salvar a confiança junto no arquivo txt (o YOLO de treino não aceita confiança no txt)
    results = model.predict(
        source=ONLINE_DATA_DIR,
        save=True,          # Salva o resultado visual para auditoria humana rápida
        save_txt=True,      # GERA OS LABELS AUTOMÁTICOS NO FORMATO YOLO (.txt)
        save_conf=False,    # Garante compatibilidade estrita com o formato do dataset de treino
        conf=0.30           # Filtro de confiança: ignora detecções muito fracas e duvidosas
    )

    # O YOLO por padrão salva os arquivos txt dentro de runs/detect/predictX/labels
    # Vamos mover ou confirmar onde eles foram criados para o seu controle
    if results:
        save_dir = results[0].save_dir
        txt_output_dir = os.path.join(save_dir, "labels")
        print("\n🚀 Processo concluído com sucesso!")
        print(f"📸 Imagens visuais salvas em: {save_dir}")
        print(f"📄 Arquivos de anotação (.txt) gerados em: {txt_output_dir}")
        print("\nPróximo passo recomendado de MLOps: Mesclar estas pastas no seu dataset oficial de treino!")

except Exception as e:
    print(f"\nErro ao executar o pipeline de Pseudo-Labeling: {e}")
    print("Verifique se o seu modelo está registrado e com o alias 'champion' ativo na UI.")
