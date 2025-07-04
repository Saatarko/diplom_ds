
import os
import sys
from datetime import datetime

from airflow.models import DAG
from airflow.providers.standard.operators.bash import BashOperator

# Add project root to Pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === DAG Configuration ===

with DAG(
    dag_id="agent_learning",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # or e.g., schedule="0 12 * * *"
    catchup=False,
    tags=["agent_learning"],
) as dag:

    agent_learning = BashOperator(
        task_id="build_user_segment_matrix",
        bash_command="cd /home/saatarko/PycharmProjects/diplom_ds && python scripts/agent_learning.py",
        doc_md="**Обучение агента**",
    )