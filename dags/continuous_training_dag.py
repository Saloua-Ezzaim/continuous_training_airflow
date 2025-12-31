"""
DAG Airflow pour le continuous training (version Docker)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys
import yaml
import glob
import shutil

# Configuration des chemins pour Docker
AIRFLOW_HOME = '/opt/airflow'
sys.path.insert(0, os.path.join(AIRFLOW_HOME, 'src'))

from data_processing import DataProcessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

# Charger la configuration
config_path = os.path.join(AIRFLOW_HOME, 'dags', 'config', 'training_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_for_new_data(**context):
    """Verifier la presence de nouvelles donnees"""
    print("Verification de nouvelles donnees...")
    
    new_data_dir = config['paths']['data_new']
    csv_files = glob.glob(os.path.join(new_data_dir, '*.csv'))
    
    if not csv_files:
        print("Aucune nouvelle donnee trouvee")
        raise Exception("Aucune nouvelle donnee - arrêt du pipeline")
    
    print(f"Nouvelles donnees trouvees: {len(csv_files)} fichier(s)")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    context['ti'].xcom_push(key='new_data_files', value=[os.path.basename(f) for f in csv_files])


def merge_new_data(**context):
    """Fusionner les nouvelles donnees avec les donnees existantes"""
    import pandas as pd
    
    print("Fusion des nouvelles donnees...")
    
    new_files = context['ti'].xcom_pull(key='new_data_files', task_ids='check_new_data')
    
    if not new_files:
        raise ValueError("Aucun fichier a fusionner")
    
    data_raw_dir = config['paths']['data_raw']
    data_new_dir = config['paths']['data_new']
    
    original_file = os.path.join(data_raw_dir, 'telco_churn.csv')
    merged_file = os.path.join(data_raw_dir, 'telco_churn_merged.csv')
    
    df_original = pd.read_csv(original_file)
    print(f"  - {len(df_original)} lignes originales")
    
    new_dataframes = []
    for filename in new_files:
        filepath = os.path.join(data_new_dir, filename)
        print(f"Chargement de: {filename}")
        df_new = pd.read_csv(filepath)
        print(f"  - {len(df_new)} lignes")
        new_dataframes.append(df_new)
    
    df_all_new = pd.concat(new_dataframes, ignore_index=True)
    df_merged = pd.concat([df_original, df_all_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates()
    
    print(f"\nDonnees fusionnees: {len(df_merged)} lignes")
    
    df_merged.to_csv(merged_file, index=False)
    print(f"Donnees fusionnees sauvegardees: {merged_file}")
    
    context['ti'].xcom_push(key='merged_data_path', value=merged_file)


def run_preprocessing(**context):
    """Executer le preprocessing"""
    print("Execution du preprocessing...")
    
    merged_path = context['ti'].xcom_pull(key='merged_data_path', task_ids='merge_new_data')
    
    processed_path = os.path.join(config['paths']['data_processed'], 'processed_data.pkl')
    preprocessor_path = os.path.join(config['paths']['models'], 'preprocessor.pkl')
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline(
        merged_path,
        processed_path,
        preprocessor_path
    )
    
    print(f"\nPreprocessing termine:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    
    context['ti'].xcom_push(key='processed_data_path', value=processed_path)
    context['ti'].xcom_push(key='preprocessor_path', value=preprocessor_path)


def run_training(**context):
    """Entrainer le modele"""
    import pickle
    
    print("Entrainement du modele...")
    
    processed_path = context['ti'].xcom_pull(key='processed_data_path', task_ids='run_preprocessing')
    
    print(f"Chargement des donnees: {processed_path}")
    with open(processed_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    model_params = config['model']['params']
    
    trainer = ModelTrainer(model_params)
    model = trainer.train(X_train, y_train)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(config['paths']['models'], f'random_forest_model_{timestamp}.pkl')
    metadata_path = os.path.join(config['paths']['models'], f'random_forest_metadata_{timestamp}.json')
    
    trainer.save_model(model_path, metadata_path)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    context['ti'].xcom_push(key='model_path', value=model_path)
    context['ti'].xcom_push(key='metadata_path', value=metadata_path)
    context['ti'].xcom_push(key='y_test', value=y_test.tolist())
    context['ti'].xcom_push(key='y_pred', value=y_pred.tolist())
    context['ti'].xcom_push(key='y_pred_proba', value=y_pred_proba.tolist())


def run_evaluation(**context):
    """Evaluer le modele"""
    import numpy as np
    
    print("Evaluation du modele...")
    
    y_test = np.array(context['ti'].xcom_pull(key='y_test', task_ids='run_training'))
    y_pred = np.array(context['ti'].xcom_pull(key='y_pred', task_ids='run_training'))
    y_pred_proba = np.array(context['ti'].xcom_pull(key='y_pred_proba', task_ids='run_training'))
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)
    evaluator.detailed_report(y_test, y_pred)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = os.path.join(config['paths']['models'], f'metrics_{timestamp}.json')
    evaluator.save_metrics(metrics, metrics_path)
    
    thresholds = config['metrics_thresholds']
    alerts = []
    
    for metric, threshold in thresholds.items():
        if metric in metrics and metrics[metric] < threshold:
            alerts.append(f"{metric}: {metrics[metric]:.4f} < {threshold}")
    
    if alerts:
        print("\nALERTES - Metriques en dessous des seuils:")
        for alert in alerts:
            print(f"  ! {alert}")
    else:
        print("\nToutes les metriques sont au-dessus des seuils")
    
    context['ti'].xcom_push(key='metrics_path', value=metrics_path)
    context['ti'].xcom_push(key='metrics', value=metrics)


def cleanup_new_data(**context):
    """Nettoyer le dossier new_data"""
    print("Nettoyage du dossier new_data...")
    
    new_files = context['ti'].xcom_pull(key='new_data_files', task_ids='check_new_data')
    
    if not new_files:
        print("Aucun fichier a nettoyer")
        return
    
    data_new_dir = config['paths']['data_new']
    
    for filename in new_files:
        filepath = os.path.join(data_new_dir, filename)
        if os.path.exists(filepath):
            archive_dir = os.path.join(data_new_dir, 'archived')
            os.makedirs(archive_dir, exist_ok=True)
            
            archive_path = os.path.join(archive_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
            shutil.move(filepath, archive_path)
            print(f"  Archive: {filename} -> {os.path.basename(archive_path)}")
    
    print("Nettoyage termine")


with DAG(
    'continuous_training_pipeline',
    default_args=default_args,
    description='Pipeline de continuous training avec Docker',
    schedule_interval='@hourly',
    catchup=False,
    tags=['machine-learning', 'continuous-training', 'docker'],
) as dag:
    
    check_new_data = PythonOperator(
        task_id='check_new_data',
        python_callable=check_for_new_data,
        provide_context=True,
    )
    
    merge_data = PythonOperator(
        task_id='merge_new_data',
        python_callable=merge_new_data,
        provide_context=True,
    )
    
    preprocess = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing,
        provide_context=True,
    )
    
    train = PythonOperator(
        task_id='run_training',
        python_callable=run_training,
        provide_context=True,
    )
    
    evaluate = PythonOperator(
        task_id='run_evaluation',
        python_callable=run_evaluation,
        provide_context=True,
    )
    
    cleanup = PythonOperator(
        task_id='cleanup_new_data',
        python_callable=cleanup_new_data,
        provide_context=True,
    )
    
    check_new_data >> merge_data >> preprocess >> train >> evaluate >> cleanup