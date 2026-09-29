import os
import json
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Definição de caminhos absolutos baseados na raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
PRODUCTION_DIR = os.path.join(PROJECT_ROOT, "online_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "observability")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_image_features(image_dir: str) -> pd.DataFrame:
    """Varre um diretorio de imagens e extrai propriedades fisicas/estruturais."""
    features_list = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(image_dir):
        print(f"⚠️ Pasta nao encontrada: {image_dir}")
        return pd.DataFrame()

    files = [f for f in os.listdir(
        image_dir) if f.lower().endswith(valid_extensions)]

    for file in files:
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Converte para tons de cinza e para o espaço de cores HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Extração de Iluminação e Cores (Média e Desvio Padrão)
        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        hue_mean = float(np.mean(hsv[:, :, 0]))

        # 2. Extração de Contraste e Textura usando GLCM (Matriz de Coocorrência de Tons de Cinza)
        glcm = graycomatrix(gray, distances=[1], angles=[
                            0], levels=256, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])

        features_list.append({
            "file_name": file,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "saturation_mean": saturation_mean,
            "hue_mean": hue_mean,
            "contrast": contrast,
            "homogeneity": homogeneity
        })

    return pd.DataFrame(features_list)


def main():
    print("📊 [Evidently AI] Iniciando extracao de metadados das imagens...")

    # Extrai características das imagens de treino (Base de Referência estável)
    ref_df = extract_image_features(REFERENCE_DIR)

    # Extrai características das imagens novas da webcam (Dados de Produção atuais)
    prod_df = extract_image_features(PRODUCTION_DIR)

    if ref_df.empty or prod_df.empty:
        print("❌ Erro: Uma ou ambas as pastas de imagens estao vazias. Monitoramento abortado.")
        # Salva um JSON falso com drift desativado para nao quebrar a DAG do Airflow
        with open(os.path.join(OUTPUT_DIR, "drift_report.json"), "w") as f:
            json.dump(
                {"metrics": {"dataset_drift": False, "error": "Pastas vazias"}}, f)
        return

    # Remove a coluna de texto do nome do arquivo para o teste estatistico focar apenas nos numeros
    columns_to_analyze = ["brightness_mean", "brightness_std",
                          "saturation_mean", "hue_mean", "contrast", "homogeneity"]

    print("📊 [Evidently AI] Calculando analise estatistica de Data Drift...")

    print("📊 [Evidently AI] Calculando analise estatistica de Data Drift...")

    # 2. Configura o Relatório
    report = Report(metrics=[DataDriftPreset(columns=columns_to_analyze)])

    # O método .run() retorna o objeto de avaliação consolidado (Snapshot)
    my_eval = report.run(
        current_data=prod_df[columns_to_analyze],
        reference_data=ref_df[columns_to_analyze]
    )

    # 3. EXTRAÇÃO CORRIGIDA BASEADA NO PRINT DO SEU AMBIENTE
    report_dict = my_eval.dict()

    dataset_drift_detected = False
    drift_share = 0.0

    # Busca o bloco DriftedColumnsCount que consolida o resumo do dataset
    for metric_entry in report_dict.get("metrics", []):
        metric_name = metric_entry.get("metric_name", "")

        if "DriftedColumnsCount" in metric_name:
            metric_value = metric_entry.get("value", {})

            # Captura a proporção real calculada (Ex: 0.1666)
            drift_share = float(metric_value.get("share", 0.0))

            # Regra de negócio: O threshold configurado no seu ambiente é 0.5 (50%)
            # Se a proporção de colunas com desvio for maior ou igual ao corte, ativa o drift do dataset
            drift_threshold = float(metric_entry.get(
                "config", {}).get("drift_share", 0.5))
            dataset_drift_detected = drift_share >= drift_threshold
            break

    print("\n--- RESULTADO DA OBSERVABILIDADE ---")
    print(f"⚠️ Algum desvio detectado no dataset? {dataset_drift_detected}")
    print(
        f"📈 Proporção de colunas visuais com desvio: {drift_share * 100:.2f}%")

    # 4. Exporta o JSON enxuto que será lido pela DAG do Airflow
    simplified_report = {
        "metrics": {
            "dataset_drift": bool(dataset_drift_detected),
            "share_of_drifted_columns": round(drift_share, 4)
        }
    }

    json_output_path = os.path.join(OUTPUT_DIR, "drift_report.json")
    with open(json_output_path, "w") as f:
        json.dump(simplified_report, f, indent=4)
    print(
        f"💾 Relatorio simplificado JSON salvo com sucesso em: {json_output_path}")

    # 5. Salva o painel visual interativo em HTML usando o objeto de avaliação
    html_output_path = os.path.join(OUTPUT_DIR, "drift_dashboard.html")
    my_eval.save_html(html_output_path)
    print(
        f"🌐 Dashboard grafico HTML gerado com sucesso em: {html_output_path}")


if __name__ == "__main__":
    main()
