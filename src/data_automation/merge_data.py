import os
import glob
import shutil
import sys

# 1. Definição das raízes do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# 2. Localiza dinamicamente a ÚLTIMA pasta 'predict' criada pelo YOLO
BASE_DETECT_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")

# Busca por 'predict', 'predict2', 'predict3', etc.
predict_folders = glob.glob(os.path.join(BASE_DETECT_DIR, "predict*"))

if not predict_folders:
    print(f"❌ Nenhuma pasta de predição encontrada em: {BASE_DETECT_DIR}")
    sys.exit()

# Ordena as pastas pela data de modificação (a mais recente por último)
latest_predict_dir = max(predict_folders, key=os.path.getmtime)

# 3. Define a pasta de labels com base na última execução encontrada
PREDICT_LABELS_DIR = os.path.join(latest_predict_dir, "labels")

print(f"🔍 Última pasta de inferência detectada: {latest_predict_dir}")
print(f"📄 Buscando labels em: {PREDICT_LABELS_DIR}")

# ---- O resto do seu código de transferência (Destinos, loops e shutil) continua exatamente igual ----
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")


def merge_datasets():
    print("Iniciando a mesclagem dos novos dados pseudo-rotulados...")

    if not os.path.exists(PREDICT_LABELS_DIR):
        print(
            f"❌ Pasta de labels gerados não encontrada: {PREDICT_LABELS_DIR}")
        return

    # Garante que as pastas de destino existam
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    copied_counter = 0

    # Varre as anotações geradas
    for txt_file in os.listdir(PREDICT_LABELS_DIR):
        if txt_file.endswith(".txt"):
            base_name = os.path.splitext(txt_file)[0]

            # Localiza a imagem correspondente na pasta online_data (testando extensões comuns)
            img_file = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                potential_img = base_name + ext
                if os.path.exists(os.path.join(ONLINE_DATA_DIR, potential_img)):
                    img_file = potential_img
                    break

            if img_file:
                # 1. Copia o arquivo .txt para a pasta oficial de labels de treino
                shutil.copy2(
                    os.path.join(PREDICT_LABELS_DIR, txt_file),
                    os.path.join(TRAIN_LABELS_DIR, txt_file)
                )

                # 2. Copia a imagem original para a pasta oficial de imagens de treino
                shutil.copy2(
                    os.path.join(ONLINE_DATA_DIR, img_file),
                    os.path.join(TRAIN_IMAGES_DIR, img_file)
                )

                # 3. Remove a imagem processada da pasta online_data (Limpeza de ambiente)
                os.remove(os.path.join(ONLINE_DATA_DIR, img_file))

                copied_counter += 1

    # Remove o arquivo que não teve detecção (rust8.png) para não acumular lixo
    for residual_file in os.listdir(ONLINE_DATA_DIR):
        if residual_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            os.remove(os.path.join(ONLINE_DATA_DIR, residual_file))

    print(
        f"✅ Sucesso! {copied_counter} novas imagens e labels mesclados à base de treino oficial.")
    print("🧹 Pasta 'online_data' limpa e pronta para receber novos frames.")


if __name__ == "__main__":
    merge_datasets()
