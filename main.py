import os
import sys
from mlflow import MlflowClient
from ultralytics import YOLO

# 1. Ensures the correct project root based on the current script location
# Since main.py is located at the root, PROJECT_ROOT is the directory where this file resides.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

# Defines the correct path for production images
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")

print("Searching for the 'champion' model in the MLflow Model Registry...")

try:
    # 2. Retrieves the details of the model version marked as 'champion'
    client = MlflowClient()
    model_metadata = client.get_model_version_by_alias(
        name="Corrosion_Detection_YOLO_Model",
        alias="champion"
    )

    # 3. Direct and reliable cleanup of a potentially corrupted Windows/SQLite URI
    raw_path = model_metadata.source
    clean_path = raw_path.replace(
        "file|///", "").replace("file:///", "").replace("|", "")
    model_final_path = os.path.normpath(clean_path)

    print(f"Model successfully located and loaded: {model_final_path}")

    # 4. Loads the champion YOLO model
    model = YOLO(model_final_path)

    # 5. Scans the online_data folder for valid images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(ONLINE_DATA_DIR):
        raise FileNotFoundError(
            f"Production folder not found at: {ONLINE_DATA_DIR}")

    images = [os.path.join(ONLINE_DATA_DIR, f) for f in os.listdir(ONLINE_DATA_DIR) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"No new images found in the folder: {ONLINE_DATA_DIR}")
        sys.exit()

    print(f"Starting pseudo-labeling for {len(images)} real-world images...")

    # 6. Runs batch inference
    # save_txt=True generates .txt files with automatic labels
    # save_conf=False avoids saving confidence scores in the txt files
    # since YOLO training datasets do not accept confidence values
    results = model.predict(
        source=ONLINE_DATA_DIR,
        save=True,          # Saves visual prediction results for quick human auditing
        save_txt=True,      # GENERATES AUTOMATIC LABELS IN YOLO FORMAT (.txt)
        save_conf=False,    # Ensures strict compatibility with the training dataset format
        conf=0.30           # Confidence threshold: ignores weak and uncertain detections
    )

    # By default, YOLO saves annotation files in runs/detect/predictX/labels
    # We retrieve and display the output path for tracking purposes
    if results:
        save_dir = results[0].save_dir
        txt_output_dir = os.path.join(save_dir, "labels")
        print("\n🚀 Process completed successfully!")
        print(f"📸 Visual results saved to: {save_dir}")
        print(f"📄 Annotation files (.txt) generated at: {txt_output_dir}")
        print("\nRecommended MLOps next step: Merge these folders into your official training dataset!")

except Exception as e:
    print(f"\nError while executing the pseudo-labeling pipeline: {e}")
    print("Make sure your model is registered and has the 'champion' alias assigned in the MLflow UI.")