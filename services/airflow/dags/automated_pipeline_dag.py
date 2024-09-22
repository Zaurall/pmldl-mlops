from datetime import datetime
from time import sleep
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

def sleep_for_a_while():
    sleep(4) 

dag = DAG(
    'automated_pipeline_dag',
    schedule_interval='*/5 * * * *',
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

datasets_pipeline_task = TriggerDagRunOperator(
    task_id='datasets_pipeline',
    trigger_dag_id='data_prepare_dag',
    dag=dag,
)

wait_timer_task = PythonOperator(
    task_id='wait_timer',
    python_callable=sleep_for_a_while,
    dag=dag,
)

models_pipeline_task = BashOperator(
    task_id='models_pipeline',
    bash_command='ls $PROJECTPATH; ls $PYTHONPATH; cd $PYTHONPATH; python3 models/model_pipeline.py',
    dag=dag,
)

deploy_pipeline_task = BashOperator(
    task_id='deploy_pipeline_task',
    bash_command='cd $PROJECTPATH/code/deployment; timeout 120 docker-compose up && docker-compose down || [[ $? -eq 124 ]]',
    dag=dag,
)

datasets_pipeline_task >> wait_timer_task >> models_pipeline_task >> deploy_pipeline_task