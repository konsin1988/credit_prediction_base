from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

# Определение DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 12)
}

with DAG(
    'credit_prediction',
    default_args=default_args,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs = 1,
    description='DAG to learn model',
    schedule_interval='@daily'
) as dag:
    # Определение задач
    model_learning = DockerOperator(
        task_id='model_learning',
        api_version='auto',
        auto_remove='success',
        image='cp-learning:latest',
        container_name='cp-learning',
        docker_url="unix:///var/run/docker.sock",
        mount_tmp_dir=False,
        network_mode='container:cp-mlflow',
        dag=dag
    )

    # Order of tasks
model_learning
