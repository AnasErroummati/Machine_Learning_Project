from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'AUI',
    'depends_on_past': False,
    'start_date': days_ago(31),
    'email': ['aui@aui.ma'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

#instantiates a directed acyclic graph
with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='speech_recognition',
    schedule_interval=timedelta(days=30),
) as dag:

    # instantiate tasks using Operators.
    #BashOperator defines tasks that execute bash scripts. In this case, we run Python scripts for each task.

    ingest_data = BashOperator(
        task_id='ingest_data',
        bash_command='python ingest_data.py',
        dag=dag,
    )

    valid_data = BashOperator(
        task_id='valid_data',
        bash_command='python valid_data.py',
        dag=dag,
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python preprocess_librispeech.py --data_dir drive/MyDrive/ --output_dir output',
        dag=dag,
    )

    train = BashOperator(
        task_id='train',
        depends_on_past=False,
        bash_command='python train_evaluate.py --mode train --data_dir train_set',
        retries=3,
        dag=dag,
    )

    evaluate = BashOperator(
        task_id='evaluate',
        depends_on_past=False,
        bash_command='python train_evaluate.py --mode eval --data_dir eval_set',
        dag=dag,
    )

    serve_commands = "cd RestfullService\ngradle build\ngradle bootrun"
    serve = BashOperator(
        task_id='serve',
        depends_on_past=False,
        bash_command=serve_commands,
        #retries=3,
        dag=dag,
    )

    #sets the ordering of the DAG. The >> directs the 2nd task to run after the 1st task. This means that
    #preprocess_data runs first, then train, then serve.
    ingest_data >> valid_data >> preprocess_data >> train >> evaluate >> serve