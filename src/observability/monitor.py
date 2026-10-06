import os
import json
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Absolute paths definition based on the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
PRODUCTION_DIR = os.path.join(PROJECT_ROOT, "online_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "observability")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_image_features(image_dir: str) -> pd.DataFrame:
    """Scans an image directory and extracts physical/structural properties."""
    features_list = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(image_dir):
        print(f"⚠️ Folder not found: {image_dir}")
        return pd.DataFrame()

    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    for file in files:
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Converts to grayscale and to HSV color space
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Illumination and Color Extraction (Mean and Standard Deviation)
        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        hue_mean = float(np.mean(hsv[:, :, 0]))

        # 2. Contrast and Texture Extraction using GLCM (Gray-Level Co-occurrence Matrix)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
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
    print("📊 [Evidently AI] Starting image metadata extraction...")

    # Extracts features from training images (Stable Reference Baseline)
    ref_df = extract_image_features(REFERENCE_DIR)

    # Extracts features from new webcam images (Current Production Data)
    prod_df = extract_image_features(PRODUCTION_DIR)

    if ref_df.empty or prod_df.empty:
        print("❌ Error: One or both image folders are empty. Monitoring aborted.")
        # Saves a placeholder JSON with drift disabled to avoid breaking the Airflow DAG
        with open(os.path.join(OUTPUT_DIR, "drift_report.json"), "w") as f:
            json.dump(
                {"metrics": {"dataset_drift": False, "error": "Empty folders"}}, f)
        return

    # Removes the text column containing the filename so the statistical test focuses strictly on numbers
    columns_to_analyze = ["brightness_mean", "brightness_std",
                          "saturation_mean", "hue_mean", "contrast", "homogeneity"]

    print("📊 [Evidently AI] Calculating Data Drift statistical analysis...")

    # 2. Configures the Report
    report = Report(metrics=[DataDriftPreset(columns=columns_to_analyze)])

    # The .run() method returns the consolidated evaluation object (Snapshot)
    my_eval = report.run(
        current_data=prod_df[columns_to_analyze],
        reference_data=ref_df[columns_to_analyze]
    )

    # 3. FIXED EXTRACTION BASED ON ENVIRONMENT PRINT
    report_dict = my_eval.dict()

    dataset_drift_detected = False
    drift_share = 0.0

    # Looks for the DriftedColumnsCount block that consolidates the dataset summary
    for metric_entry in report_dict.get("metrics", []):
        metric_name = metric_entry.get("metric_name", "")

        if "DriftedColumnsCount" in metric_name:
            metric_value = metric_entry.get("value", {})

            # Captures the actual calculated proportion (e.g., 0.1666)
            drift_share = float(metric_value.get("share", 0.0))

            # Business rule: The threshold configured in your environment is 0.5 (50%)
            # If the proportion of columns with drift is greater or equal to the cut, activates dataset drift
            drift_threshold = float(metric_entry.get(
                "config", {}).get("drift_share", 0.5))
            dataset_drift_detected = drift_share >= drift_threshold
            break

    print("\n--- OBSERVABILITY RESULT ---")
    print(f"⚠️ Any dataset drift detected? {dataset_drift_detected}")
    print(
        f"📈 Proportion of visual columns with drift: {drift_share * 100:.2f}%")

    # 4. Exports the lean JSON to be read by the Airflow DAG
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
        f"💾 Simplified JSON report successfully saved to: {json_output_path}")

    # 5. Saves the interactive visual HTML dashboard using the evaluation object
    html_output_path = os.path.join(OUTPUT_DIR, "drift_dashboard.html")
    my_eval.save_html(html_output_path)
    print(
        f"🌐 Visual HTML dashboard successfully generated at: {html_output_path}")


if __name__ == "__main__":
    main()
