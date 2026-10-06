from datetime import datetime, timedelta
import os
import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_TZ = pendulum.timezone("America/Sao_Paulo")

default_args = {
    'owner': 'ai_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 9, 30, tzinfo=LOCAL_TZ),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'corrosion_phase2_retrain',
    default_args=default_args,
    description='Phase 2 MLOps: Triggered via post-curation Webhook for retraining and deployment',
    schedule=None,  # IMPORTANT: Disabled! Awaits external trigger exclusively via HTTP API
    catchup=False,
    tags=['mlops', 'retrain', 'deployment']
) as dag:

    # 1. Converts manual annotations from Labelme into normalized TXT files and merges them into the official dataset
    task_merge_curated_data = BashOperator(
        task_id='merge_curated_data',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.data_automation.parse_review_to_train",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # 2. Versions the expanded dataset in DVC and synchronizes it with Google Drive
    task_dvc_versioning = BashOperator(
        task_id='dvc_add_and_push',
        bash_command="dvc add data",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # 3. Triggers YOLO training using the augmentations and hyperparameters from model_config.yaml
    task_train_yolo = BashOperator(
        task_id='train_new_yolo_model',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_train.yolo",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # 4. Locates the latest generated best.pt file and runs the strict governance evaluation (mAP50 + Recall vs Champion)
    task_evaluate_and_gate = BashOperator(
        task_id='model_governance_evaluation',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_eval.yolo",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # 5. CD (Continuous Deployment): Notifies the live API endpoint to dynamic reload the new champion model
    task_live_reload = BashOperator(
        task_id='api_live_reload',
        bash_command='curl -X POST http://127.0.0.1:8000/reload',
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # Linear dependencies of Phase 2 (Heavy computation pipeline)
    task_merge_curated_data >> task_dvc_versioning >> task_train_yolo >> task_evaluate_and_gate >> task_live_reload
