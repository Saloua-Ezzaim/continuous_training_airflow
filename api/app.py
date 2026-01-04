"""
API Flask pour les predictions de churn en temps reel
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permettre les requetes cross-origin

# Variables globales pour le modele
model = None
preprocessor = None
feature_names = None


def load_latest_model():
    """Charger le dernier modele entraine et le preprocessor"""
    global model, preprocessor, feature_names
    
    try:
        # Charger le dernier modele RandomForest
        model_files = glob.glob('models/random_forest_model_*.pkl')
        if not model_files:
            print("Aucun modele trouve")
            return False
        
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Chargement du modele: {latest_model}")
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        # Charger le preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                preprocessor = {
                    'scaler': preprocessor_data['scaler'],
                    'label_encoders': preprocessor_data['label_encoders']
                }
        
        # Charger la liste des features
        processed_data_path = 'data/processed/processed_data.pkl'
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
                feature_names = data.get('feature_names', [])
        
        print("Modele charge avec succes")
        return True
        
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {e}")
        return False


def preprocess_input(data):
    """Preprocesser les donnees d'entree pour le RandomForest"""
    try:
        # Convertir en DataFrame
        df = pd.DataFrame([data])

        # Supprimer customerID si present
        df = df.drop(columns=['customerID'], errors='ignore')

        # Remplir TotalCharges si vide
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

        # Ajouter colonnes manquantes avec valeur par defaut
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    if preprocessor and col in preprocessor['label_encoders']:
                        df[col] = preprocessor['label_encoders'][col].classes_[0]
                    else:
                        df[col] = 0
            df = df[feature_names]

        # Encoder les colonnes categoriales
        if preprocessor and 'label_encoders' in preprocessor:
            for col, encoder in preprocessor['label_encoders'].items():
                if col in df.columns:
                    # Remplacer les valeurs inconnues par la premiere classe
                    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    df[col] = encoder.transform(df[col])

        # Normaliser
        if preprocessor and 'scaler' in preprocessor:
            X_scaled = preprocessor['scaler'].transform(df)
        else:
            X_scaled = df.values

        return X_scaled

    except Exception as e:
        print("Erreur lors du preprocessing:", e)
        raise


@app.route('/')
def home():
    return jsonify({
        'message': 'API de prediction de churn',
        'version': '1.0',
        'endpoints': {
            '/health': 'Verifier l\'etat de l\'API',
            '/predict': 'Faire une prediction (POST)',
            '/model-info': 'Informations sur le modele',
            '/reload-model': 'Recharger le dernier modele'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Aucun modele charge'}), 400
    
    metrics_files = glob.glob('models/metrics_*.json')
    latest_metrics = None
    if metrics_files:
        latest_metrics_file = max(metrics_files, key=os.path.getctime)
        with open(latest_metrics_file, 'r') as f:
            import json
            latest_metrics = json.load(f)
    
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else None,
        'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        'metrics': latest_metrics,
        'features': feature_names
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Modele non charge. Utilisez /reload-model'}), 400
        
        data = request.json
        if not data:
            return jsonify({'error': 'Aucune donnee fournie'}), 400
        
        X = preprocess_input(data)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        churn_probability = probability[1]
        
        if churn_probability < 0.3:
            risk_level = 'Faible'
            risk_color = 'green'
        elif churn_probability < 0.7:
            risk_level = 'Moyen'
            risk_color = 'orange'
        else:
            risk_level = 'Eleve'
            risk_color = 'red'
        
        response = {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'prediction_code': int(prediction),
            'probability': {
                'no_churn': float(probability[0]),
                'churn': float(probability[1])
            },
            'churn_probability_percent': f'{churn_probability * 100:.2f}%',
            'risk_level': risk_level,
            'risk_color': risk_color,
            'message': get_message(prediction, churn_probability),
            'recommendations': get_recommendations(prediction, churn_probability),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print("Erreur predict:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/reload-model', methods=['POST'])
def reload_model():
    success = load_latest_model()
    if success:
        return jsonify({
            'message': 'Modele recharge avec succes',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    else:
        return jsonify({'error': 'Echec du chargement du modele'}), 500


def get_message(prediction, probability):
    if prediction == 0:
        return f"Client stable avec {(1-probability)*100:.1f}% de confiance"
    else:
        return f"Risque de churn avec {probability*100:.1f}% de probabilite"


def get_recommendations(prediction, probability):
    recommendations = []
    
    if prediction == 1:
        if probability > 0.8:
            recommendations.extend([
                "Action urgente requise",
                "Contacter le client immediatement",
                "Proposer une offre personnalisee"
            ])
        elif probability > 0.6:
            recommendations.extend([
                "Surveiller le client de pres",
                "Proposer un upgrade ou une promotion"
            ])
        else:
            recommendations.extend([
                "Client a surveiller",
                "Ameliorer la qualite du service"
            ])
    else:
        recommendations.extend([
            "Client fidele",
            "Maintenir la qualite du service"
        ])
    
    return recommendations


print("Demarrage de l'API...")
load_latest_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
