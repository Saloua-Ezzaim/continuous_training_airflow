Continuous Training with Apache Airflow
📌 Description

Ce projet met en place une pipeline de Continuous Training (entraînement continu) pour un modèle de Machine Learning en utilisant Apache Airflow.
L’objectif est de surveiller les performances du modèle en production et de déclencher automatiquement un ré-entraînement lorsque les performances se dégradent.

Le projet s’inscrit dans une démarche MLOps, combinant automatisation, monitoring, traçabilité des modèles et exposition du modèle via une API connectée à une interface Web.

🎯 Objectifs du projet

Automatiser le cycle de vie du modèle de Machine Learning

Surveiller les métriques de performance (accuracy, précision, etc.)

Détecter la dérive de performance du modèle

Lancer automatiquement le ré-entraînement via Airflow

Centraliser les métriques et résultats

Permettre aux utilisateurs finaux d’obtenir des prédictions via un formulaire Web connecté à une API

🏗️ Architecture du projet

Le projet est organisé autour des composants suivants :

Airflow DAGs : orchestration des tâches (entraînement, évaluation, monitoring)

API REST : liaison entre le modèle ML et l’interface Web, exposition des prédictions et des métriques

Interface Web : formulaire permettant au client de saisir des données et d’obtenir une prédiction du modèle

Models : stockage des modèles entraînés et des fichiers de métriques

Docker : déploiement de l’environnement Airflow

📁 Structure du projet
continuous_training_airflow/
│
├── dags/                # DAGs Airflow
├── api/                 # API pour les prédictions et les métriques
├── web/                 # Interface web (formulaire client)
├── models/              # Modèles entraînés et métriques générées
├── data/                # Données d'entraînement
├── docker-compose.yml   # Déploiement Airflow avec Docker
└── README.md

⚙️ Technologies utilisées

Python

Apache Airflow

Docker & Docker Compose

Machine Learning (Scikit-learn)

API REST

Git & GitHub

🚀 Fonctionnement général

Les données sont analysées périodiquement

Le modèle est entraîné et évalué automatiquement

Les métriques de performance sont sauvegardées

En cas de baisse de performance → retraining automatique via Airflow

Le modèle est exposé via une API REST

Le client remplit un formulaire sur l’interface Web et obtient une prédiction en temps réel

📊 Cas d’usage

Projets MLOps

Surveillance de modèles ML en production

Systèmes de prédiction avec données évolutives

Applications ML accessibles via API et interface utilisateur

👩‍🎓 Contexte académique

Ce projet a été réalisé dans le cadre du Master en Intelligence Artificielle, avec un focus sur les pratiques MLOps et le déploiement de modèles de Machine Learning en production.

👤 Auteurs

Saloua Ezzaim

Ikram Abhih

Karima Er-remyty

Master en Intelligence Artificielle
