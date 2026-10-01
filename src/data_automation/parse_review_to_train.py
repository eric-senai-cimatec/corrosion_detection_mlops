import os
import json
import shutil
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")

# Mapeamento de classes do seu config.yaml/data.yaml
CLASS_MAPPING = {"corrosion": 0}


def trigger_airflow_phase2():
    """Acorda a esteira de retreino de forma assíncrona usando a API REST do Airflow"""
    url = "http://localhost:8080/api/v1/dags/corrosion_phase2_retrain/dagRuns"
    auth = ("admin", "admin")
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, json={"conf": {}}, headers=headers, auth=auth)
        if response.status_code == 201:
            print(
                "🚀 [MLOps] Esteira de retreino e deploy disparada com sucesso no Airflow!")
        else:
            print(
                f"⚠️ [Airflow API] Servidor respondeu com código inesperado: {response.status_code}")
    except Exception as e:
        print(
            f"❌ Falha crítica de rede ao tentar se conectar com a API do Airflow: {e}")


def main():
    print("🔄 Iniciando conversão de dados do Labelme para formato YOLO...")

    if not os.path.exists(REVIEW_DIR) or not os.listdir(REVIEW_DIR):
        print(
            "⚠️ Pasta 'review_data' vazia ou inexistente. Nenhuma curadoria para processar.")
        return

    # Garante a existência dos diretórios de destino
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    processed_count = 0

    for file in os.listdir(REVIEW_DIR):
        if file.endswith(".json"):
            json_path = os.path.join(REVIEW_DIR, file)
            base_name = os.path.splitext(file)[0]

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_w = data["imageWidth"]
            img_h = data["imageHeight"]
            img_name = data["imagePath"]

            txt_lines = []
            for shape in data["shapes"]:
                if shape["shape_type"] == "rectangle":
                    label = shape["label"]
                    cls_id = CLASS_MAPPING.get(label, 0)

                    # Coordenadas do Labelme [[x1, y1], [x2, y2]]
                    p1, p2 = shape["points"]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]

                    # Converte para formato YOLO (Centro_X, Centro_Y, Largura, Altura) normalizados (0 a 1)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    bbox_w = abs(x2 - x1) / img_w
                    bbox_h = abs(y2 - y1) / img_h

                    txt_lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

            # 1. Salva o TXT oficial na pasta de treino
            with open(os.path.join(TRAIN_LABELS_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))

            # 2. Move a imagem original para a pasta de treino
            shutil.move(os.path.join(REVIEW_DIR, img_name),
                        os.path.join(TRAIN_IMAGES_DIR, img_name))

            # 3. Deleta o JSON para limpar a fila de revisão
            os.remove(json_path)
            processed_count += 1

    print(
        f"✅ Curadoria concluída! {processed_count} imagens convertidas e mescladas ao treino.")

    # DISPARO DO GATILHO: Só acontece após todos os arquivos acima fecharem no disco
    if processed_count > 0:
        trigger_airflow_phase2()


if __name__ == "__main__":
    main()
