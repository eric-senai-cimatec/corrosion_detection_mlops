from datetime import datetime, timedelta
import os
import json
import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
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
        print(f"⚠️ Arquivo {json_path} nao localizado. Abortando fluxo.")
        return 'stop_pipeline'

    with open(json_path, 'r') as f:
        report_data = json.load(f)

    dataset_drifted = report_data.get(
        "metrics", {}).get("dataset_drift", False)
    drift_share = report_data.get("metrics", {}).get(
        "share_of_drifted_columns", 0.0)

    print(
        f"📊 [Airflow] Proporcao de colunas com desvio: {drift_share * 100:.2f}%")

    if dataset_drifted:
        print(
            "🚨 CRÍTICO: Data Drift detectado. Encaminhando para Fila de Curadoria Humana.")
        return 'generate_pseudo_labels'
    else:
        print("✅ ESTÁVEL: Dados sob controle. Modelo atual mantido em servimento.")
        return 'stop_pipeline'


with DAG(
    'corrosion_phase1_monitoring',
    default_args=default_args,
    description='Fase 1 MLOps: Roda Observabilidade e prepara dados para o Labelme se houver Drift',
    schedule='0 8 * * *',  # Executa programado todos os dias as 08:00 da manha
    catchup=False,
    tags=['mlops', 'observability', 'corrosion']
) as dag:

    # 1. Calcula o Data Drift com o Evidently AI comparando a webcam com a base de treino
    task_run_observability = BashOperator(
        task_id='run_observability_monitor',
        bash_command=f"{VENV_PYTHON} -m src.observability.monitor",
        cwd=PROJECT_ROOT
    )

    # 2. Avalia a condicional baseada no JSON gerado
    task_verify_drift = BranchPythonOperator(
        task_id='verify_drift_trigger',
        python_callable=check_drift_and_decide
    )

    # Caminho A: Dataset saudavel, encerra o ciclo de forma limpa
    task_stop_pipeline = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Pipeline finalizado: nenhuma acao de retreino necessaria hoje."'
    )

    # Caminho B: Cria os JSONs estruturados de pseudo-anotacao na pasta review_data/
    task_generate_pseudo_labels = BashOperator(
        task_id='generate_pseudo_labels',
        bash_command=f"{VENV_PYTHON} -m src.data_automation.send_to_review",
        cwd=PROJECT_ROOT
    )

    # Dependências do Grafo da Fase 1
    task_run_observability >> task_verify_drift
    task_verify_drift >> [task_stop_pipeline, task_generate_pseudo_labels]
