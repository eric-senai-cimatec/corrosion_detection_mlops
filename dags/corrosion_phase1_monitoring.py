from datetime import datetime, timedelta
import os
import json
import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_TZ = pendulum.timezone("America/Sao_Paulo")

default_args = {
    'owner': 'ai_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 9, 30, tzinfo=LOCAL_TZ),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_drift_and_decide():
    json_path = os.path.join(PROJECT_ROOT, "runs",
                             "observability", "drift_report.json")

    if not os.path.exists(json_path):
        print(f"⚠️ File {json_path} not found. Aborting pipeline flow.")
        return 'stop_pipeline'

    with open(json_path, 'r') as f:
        report_data = json.load(f)

    dataset_drifted = report_data.get(
        "metrics", {}).get("dataset_drift", False)
    drift_share = report_data.get("metrics", {}).get(
        "share_of_drifted_columns", 0.0)

    print(
        f"📊 [Airflow] Proportion of columns with drift: {drift_share * 100:.2f}%")

    if dataset_drifted:
        print(
            "🚨 CRITICAL: Data Drift detected. Routing data to Human Curation Queue.")
        return 'generate_pseudo_labels'
    else:
        print("✅ STABLE: Data is under control. Current model kept in serving.")
        return 'stop_pipeline'


with DAG(
    'corrosion_phase1_monitoring',
    default_args=default_args,
    description='Phase 1 MLOps: Runs Observability and prepares data for Labelme if Drift is present',
    schedule='0 8 * * *',  # Scheduled to run every day at 08:00 AM
    catchup=False,
    tags=['mlops', 'observability', 'corrosion']
) as dag:

    # 1. Calculates Data Drift using Evidently AI, comparing webcam data against the training baseline
    task_run_observability = BashOperator(
        task_id='run_observability_monitor',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.observability.monitor",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # 2. Evaluates the condition based on the generated JSON file
    task_verify_drift = BranchPythonOperator(
        task_id='verify_drift_trigger',
        python_callable=check_drift_and_decide
    )

    # Path A: Healthy dataset, terminates the pipeline cleanly
    task_stop_pipeline = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Pipeline finished: no retraining action required today."'
    )

    # Path B: Creates structured pseudo-annotation JSON files inside the review_data/ folder
    task_generate_pseudo_labels = BashOperator(
        task_id='generate_pseudo_labels',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.data_automation.send_to_review",
        env={
            "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
        },
        cwd=PROJECT_ROOT
    )

    # Phase 1 Graph Dependencies
    task_run_observability >> task_verify_drift
    task_verify_drift >> [task_stop_pipeline, task_generate_pseudo_labels]
