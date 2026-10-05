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
    description='Fase 2 MLOps: Acionada via Webhook pos-curadoria para retreino e deploy',
    schedule=None,  # IMPORTANTE: Desativado! Aguarda exclusivamente gatilho externo por API HTTP
    catchup=False,
    tags=['mlops', 'retrain', 'deployment']
) as dag:

    # 1. Traduz o trabalho manual do Labelme para TXT normalizado e mescla com a pasta oficial
    task_merge_curated_data = BashOperator(
        task_id='merge_curated_data',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.data_automation.parse_review_to_train",
        env={
        "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
    },
        cwd=PROJECT_ROOT
    )

    # 2. Versiona a base de dados expandida no DVC e sincroniza com o Google Drive
    task_dvc_versioning = BashOperator(
        task_id='dvc_add_and_push',
        bash_command="dvc add data && dvc push",
        env={
        "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
    },
        cwd=PROJECT_ROOT
    )

    # 3. Dispara o treinamento do YOLO com as augmentations e hiperparametros do model_config.yaml
    task_train_yolo = BashOperator(
        task_id='train_new_yolo_model',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_train.yolo",
        env={
        "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
    },
        cwd=PROJECT_ROOT
    )

    # 4. Localiza o ultimo best.pt gerado e roda a avaliacao estrita (mAP50 + Recall vs Campeao)
    task_evaluate_and_gate = BashOperator(
        task_id='model_governance_evaluation',
        bash_command="/home/eric/projects/corrosion_detection_mlops/.venv/bin/python -m src.model_eval.yolo",
        env={
        "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
    },
        cwd=PROJECT_ROOT
    )

    # 5. CD (Continuous Deployment): Notifica o endpoint da API para recarregar o novo champion vivo
    task_live_reload = BashOperator(
        task_id='api_live_reload',
        bash_command='curl -X POST http://127.0.0.1:8000/reload',
        env={
        "PYTHONPATH": "/home/eric/projects/corrosion_detection_mlops"
    },
        cwd=PROJECT_ROOT
    )

    # Dependências lineares da Fase 2 (Esteira de computacao pesada)
    task_merge_curated_data >> task_dvc_versioning >> task_train_yolo >> task_evaluate_and_gate >> task_live_reload
