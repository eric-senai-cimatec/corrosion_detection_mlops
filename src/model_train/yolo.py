import os
import sys
from ultralytics import YOLO, settings

# 1. Configures the paths before making any custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 2. Configures the local URI for the MLflow SQLite database
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    # Forces the activation of the MLflow plugin in Ultralytics
    settings.update({"mlflow": True})

    # 3. Dynamically locates your YAML configuration file at the root level
    config_yaml_path = os.path.join(PROJECT_ROOT, "model_args.yaml")

    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_yaml_path}")

    # 4. Initializes the base model defined in your YAML (yolo26n.pt)
    # Since the model is declared inside the YAML, you can pass the string directly or hardcode it
    model = YOLO("yolo26n.pt")

    print(
        f"🚀 Starting YOLO training using configurations from: {config_yaml_path}")

    # 5. The magic happens here: we pass only the file path to the 'cfg' parameter
    model.train(cfg=config_yaml_path)


if __name__ == '__main__':
    main()
