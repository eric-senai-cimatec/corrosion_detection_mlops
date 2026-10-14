from datetime import datetime, timedelta
import os
import pendulum
from airflow import DAG
from airflow.utils.email import send_email_smtp
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = "/home/eric/projects/corrosion_detection_mlops"
LOCAL_TZ = pendulum.timezone("America/Sao_Paulo")

def gmail_fail_alert(context): 
    ti = context["task_instance"] 
    dag_id = ti.dag_id
    task_id = ti.task_id 
    run_id = getattr(ti, "run_id", "unknown") 
    execution_time = getattr(ti, "start_date", "unknown") 
    error = str(context.get("exception", "Nenhuma exceção capturada")) 

    base_url = os.getenv("AIRFLOW__WEBSERVER__BASE_URL", "http://localhost:8080") 
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}"

    subject = f"🚨 Tarefa do Airflow Falha: {dag_id}. {task_id}"
    html_content = f""" 
    <h3>🚨 Fall on Airflow Task!</h3> 
    <ul> 
      <li><b>DAG:</b> {dag_id}</li> 
      <li><b>Task:</b> {task_id}</li> 
      <li><b>ID of execution:</b> {run_id}</li> 
      <li><b>Tempo of execution:</b> {execution_time}</li> 
      <li><b>Error:</b> {error}</li>
    </ul> 
    <p><a href="{log_url}">🔗 Show Logs on Airflow</a></p>
    """

    send_email_smtp( 
        to="eric.oliveira.pro@gmail.com", 
        subject=subject, 
        html_content=html_content, 
        conn_id="gmail_connection",
    )

default_args = {
    'owner': 'ai_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 9, 30, tzinfo=LOCAL_TZ),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': gmail_fail_alert,
}

with DAG(
    'corrosion_phase2_retrain',
    default_args=default_args,
    description='Phase 2 MLOps: Triggered via post-curation Webhook for retraining and deployment',
    schedule=None,  # IMPORTANT: Disabled! Awaits external trigger exclusively via HTTP API
    catchup=False,
    tags=['mlops', 'retrain', 'deployment']
) as dag:

    # 1. Converts manual annotations from Labelme into normalized TXT files
    task_merge_curated_data = BashOperator(
        task_id='merge_curated_data',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.data_automation.parse_review_to_train",
        env={"PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"},
        cwd=PROJECT_ROOT
    )

    # 2. Versions the expanded dataset in DVC and synchronizes it with Sharepoint
    task_dvc_versioning = BashOperator(
        task_id='dvc_add_and_push',
        bash_command="""
            cd /home/eric/projects/corrosion_detection_mlops;
            /home/eric/projects/corrosion_detection_mlops/.venv/bin/dvc add data;
            /home/eric/projects/corrosion_detection_mlops/.venv/bin/dvc push;
        """,
        env={"PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"},
        cwd=PROJECT_ROOT
    )

    # 3. Triggers YOLO training using the augmentations and hyperparameters
    task_train_yolo = BashOperator(
        task_id='train_new_yolo_model',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_train.yolo",
        env={"PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"},
        cwd=PROJECT_ROOT
    )

    # 4. Runs the strict governance evaluation (mAP50 + Recall vs Champion)
    task_evaluate_and_gate = BashOperator(
        task_id='model_governance_evaluation',
        bash_command="cd /home/eric/projects/corrosion_detection_mlops && /home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_eval.yolo",
        env={"PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"},
        cwd=PROJECT_ROOT
    )

    # 5. CD (Continuous Deployment): Notifies the live API endpoint to dynamic reload the new champion model
    task_live_reload = BashOperator(
        task_id='api_live_reload',
        bash_command='curl -X POST http://127.0.0.1:8000/reload',
        env={"PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"},
        cwd=PROJECT_ROOT
    )

    # Linear dependencies of Phase 2
    task_merge_curated_data >> task_dvc_versioning >> task_train_yolo >> task_evaluate_and_gate >> task_live_reload
