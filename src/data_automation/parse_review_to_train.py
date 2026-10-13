import os
import json
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")

# Class mapping from your config.yaml/data.yaml
CLASS_MAPPING = {"corrosion": 0}


def trigger_airflow_phase2():
    """Triggers the retraining pipeline directly via Airflow CLI, bypassing HTTP 401 errors"""
    # Locates the exact path of the airflow binary inside your active virtual environment
    venv_bin_dir = os.path.dirname(sys.executable)
    airflow_binary = os.path.join(venv_bin_dir, "airflow")

    # Command: airflow dags trigger corrosion_phase2_retrain
    command = [airflow_binary, "dags", "trigger", "corrosion_phase2_retrain"]

    try:
        print(f"📡 [CLI Trigger] Invoking local Airflow command: {' '.join(command)}")
        
        # Executes the terminal command securely from within Python
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Reinforces the environment path so Airflow knows where its home config sits
            env={**os.environ, "AIRFLOW_HOME": "/home/eric/airflow"}
        )

        print("🚀 [MLOps] SUCCESS! Airflow CLI output:")
        print(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to trigger DAG via Airflow CLI. Exit Code: {e.returncode}")
        print(f"📝 Error Details: {e.stderr.strip()}")
    except Exception as e:
        print(f"❌ Unexpected system failure when launching local trigger: {e}")


def main():
    print("🔄 Starting data conversion from Labelme to YOLO format...")

    if not os.path.exists(REVIEW_DIR) or not os.listdir(REVIEW_DIR):
        print("⚠️ 'review_data' folder is empty or non-existent. No curation data to process.")
        return

    # Ensures that destination directories exist
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

                    # Labelme coordinates [[x1, y1], [x2, y2]]
                    p1, p2 = shape["points"]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]

                    # Converts to YOLO format (X_Center, Y_Center, Width, Height) normalized (0 to 1)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    bbox_w = abs(x2 - x1) / img_w
                    bbox_h = abs(y2 - y1) / img_h

                    txt_lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

            # 1. Saves the official TXT file into the training folder
            with open(os.path.join(TRAIN_LABELS_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))

            # 2. Moves the original image to the training folder
            shutil.move(os.path.join(REVIEW_DIR, img_name),
                        os.path.join(TRAIN_IMAGES_DIR, img_name))

            # 3. Deletes the JSON file to clear the review queue
            os.remove(json_path)
            processed_count += 1

    print(f"✅ Curation completed! {processed_count} images converted and merged into the training dataset.")

    # TRIGGER DISPATCH: Only runs after all files above are completely written to disk
    if processed_count > 0:
        trigger_airflow_phase2()


if __name__ == "__main__":
    main()
