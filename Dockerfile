FROM apache/airflow:2.8.0-python3.9

# Passe à l'utilisateur airflow avant d'installer les packages
USER airflow

# Installer les packages supplémentaires
RUN pip install --user --no-cache-dir scikit-learn==1.4.0 pandas==2.1.4 pyyaml==6.0.1 dvc==3.48.0

# Pas besoin de repasser en root
