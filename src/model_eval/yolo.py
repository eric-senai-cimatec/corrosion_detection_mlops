from ultralytics import YOLO, settings
import os
import sys
import mlflow
from mlflow import MlflowClient
from helper.config import load_config
import yaml
import warnings

# Silences MLflow API deprecation warnings from legacy modules
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Configures the paths before making any custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # Database configuration at the project root
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    settings.update({"mlflow": True})

    # 1. Loads the dataset path (only data_path continues to come from config/yaml)
    config = load_config(os.path.join(PROJECT_ROOT, 'data', 'data.yaml'))
    data_path = config['path']
    
    # 2. AUTOMATIC AND DYNAMIC LOCATION OF THE LATEST TRAINED MODEL
    base_detect_dir = os.path.join(PROJECT_ROOT, "runs", "detect")

    # Lists ONLY actual training folders, ignoring validations (val) and predictions (predict)
    all_train_folders = [
        os.path.join(base_detect_dir, d)
        for d in os.listdir(base_detect_dir)
        if os.path.isdir(os.path.join(base_detect_dir, d))
        and not d.startswith("predict")
        and not d.startswith("val")
    ]

    if not all_train_folders:
        raise FileNotFoundError(
            f"❌ No valid training folder found at: {base_detect_dir}")

    # Finds the training folder modified last by the operating system
    latest_train_folder = max(all_train_folders, key=os.path.getmtime)

    # Builds the final path to the 'best.pt' weights file
    model_path = os.path.normpath(os.path.join(
        latest_train_folder, "weights", "best.pt"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ The weights file 'best.pt' was not found at: {model_path}")

    print(f"📦 [Automation] Latest training located: {latest_train_folder}")
    print(f"🎯 [Automation] Loading weights for test set evaluation: {model_path}")

    # Loads the trained model dynamically
    model = YOLO(model_path)

    # Starts the exclusive run for testing
    with mlflow.start_run(run_name="evaluation_test_set") as run:

        # Adds structural tags to the Run
        mlflow.set_tag("pipeline_stage", "testing")
        mlflow.log_param("evaluated_model_path", model_path)

        # Executes the evaluation on the test split
        metrics = model.val(
            data=data_path,
            split='test'
        )

        # Converts YOLO metrics to native float
        current_recall = float(metrics.box.r[0]) if hasattr(
            metrics.box.r, "__len__") else float(metrics.box.r)
        current_precision = float(metrics.box.p[0]) if hasattr(
            metrics.box.p, "__len__") else float(metrics.box.p)
        current_mAP50 = float(metrics.box.map50)
        current_mAP50_95 = float(metrics.box.map)

        # Logs to MLflow Tracking
        mlflow.log_metrics({
            "metrics_recall": current_recall,
            "metrics_precision": current_precision,
            "metrics_mAP50": current_mAP50,
            "metrics_mAP50_95": current_mAP50_95
        })

        # Saves the physical weights file as a Run artifact
        model_filename = os.path.basename(model_path)
        mlflow.log_artifact(model_path, artifact_path="yolo_weights")

        # Officially registers this model in the MLflow Model Registry
        local_artifact_uri = f"file://{os.path.abspath(os.path.join(run.info.artifact_uri, 'yolo_weights', model_filename))}"
        model_details = mlflow.register_model(
            local_artifact_uri, "Corrosion_Detection_YOLO_Model")

        # ---- MULTI-METRIC BUSINESS LOGIC (CHAMPION VS CHALLENGER) ----
        client = MlflowClient()
        model_name = "Corrosion_Detection_YOLO_Model"
        
        business_cfg = load_config(os.path.join(PROJECT_ROOT, 'business_args.yaml'))
        MIN_MAP50 = float(business_cfg['MIN_MAP50'])
        MIN_RECALL = float(business_cfg['MIN_RECALL'])

        print("\n--- BUSINESS RULES VALIDATION ---")
        print(f"Obtained Metrics -> mAP50: {current_mAP50:.4f} | Recall: {current_recall:.4f}")

        # 1. Check against baseline absolute thresholds
        if current_mAP50 < MIN_MAP50 or current_recall < MIN_RECALL:
            print("❌ REJECTED: Model did not meet the minimum business requirements.")
            sys.exit(0)
            
        print("✅ Passed baseline requirements. Checking tournament against current Champion...")

        # 2. Modern secure checkout using the MLflow Aliases system
        try:
            # Queries the registry for the registered version currently carrying the 'champion' tag
            champion_version = client.get_model_version_by_alias(model_name, "champion")
            
            if champion_version and champion_version.run_id:
                champion_run = client.get_run(champion_version.run_id)
                champion_mAP50 = float(champion_run.data.metrics.get("metrics_mAP50", 0.0))
                print(f"👑 Active Champion found! Version: {champion_version.version} | Champion mAP50: {champion_mAP50:.4f}")
                
                # Tournament condition: challenger must strictly outperform the champion
                if current_mAP50 > champion_mAP50:
                    print("🚀 CHALLENGER WINS! Reassigning 'champion' alias to the new model version.")
                    client.set_registered_model_alias(model_name, "champion", model_details.version)
                else:
                    print("📉 Challenger did not out-perform the current Champion. Keeping active model.")
            else:
                print("✨ No active Champion found with valid Run ID. Promoting current model to first Champion!")
                client.set_registered_model_alias(model_name, "champion", model_details.version)
                
        except Exception:
            # Fallback triggered when the database is fresh and the 'champion' alias does not exist yet
            print("✨ No active Champion found in database. Promoting current model to first Champion!")
            client.set_registered_model_alias(model_name, "champion", model_details.version)

        # ---- ADDITIONAL METADATA ATTRIBUTION (TAGS) ----
        client.set_registered_model_tag(
            name=model_name,
            key="framework",
            value="ultralytics-yolo"
        )
        client.set_registered_model_tag(
            name=model_name,
            key="task",
            value="corrosion-detection"
        )

        # Dataset version tracking via DVC
        dvc_file_path = os.path.join(PROJECT_ROOT, "data.dvc")
        dvc_hash = "unknown"

        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                dvc_data = yaml.safe_load(f)
                dvc_hash = dvc_data.get("outs", [{}])[0].get("md5", "unknown")

        client.set_registered_model_tag(
            name=model_name,
            key="dataset_version_md5",
            value=dvc_hash
        )


if __name__ == '__main__':
    main()
