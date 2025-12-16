from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import os
from datetime import datetime
import numpy as np


class ModelTrainer:
    """Classe pour l'entrainement du modele Random Forest"""
    
    def __init__(self, model_params=None):
        if model_params is None:
            # Parametres par defaut
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.model_params = model_params
        self.model = None
        self.training_time = None
        
    def train(self, X_train, y_train):
        
        print("="*60)
        print("DEBUT DE L'ENTRAINEMENT DU MODELE RANDOM FOREST")
        print("="*60)
        
        print(f"\nParametres du modele:")
        for param, value in self.model_params.items():
            print(f"  - {param}: {value}")
        
        print(f"\nDonnees d'entrainement:")
        print(f"  - Nombre d'exemples: {X_train.shape[0]}")
        print(f"  - Nombre de features: {X_train.shape[1]}")
        print(f"  - Distribution des classes: {np.bincount(y_train)}")
        
        # Creer le modele
        self.model = RandomForestClassifier(**self.model_params)
        
        # Entrainer
        print("\nEntrainement en cours...")
        
        import time
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        print(f"Entrainement termine en {self.training_time:.2f} secondes")
        
        # Afficher les features importantes
        print("\nTop 10 features les plus importantes:")
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        print("\n" + "="*60)
        print("ENTRAINEMENT TERMINE AVEC SUCCES")
        print("="*60)
        
        return self.model
    
    def save_model(self, model_filepath, metadata_filepath=None):
        
        if self.model is None:
            raise ValueError("Aucun modele a sauvegarder. Entrainer d'abord le modele.")
        
        print(f"\nSauvegarde du modele: {model_filepath}")
        
        # Creer le dossier si necessaire
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        
        # Sauvegarder le modele
        with open(model_filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print("Modele sauvegarde avec succes")
        
        # Sauvegarder les metadonnees
        if metadata_filepath:
            print(f"Sauvegarde des metadonnees: {metadata_filepath}")
            
            metadata = {
                'model_type': 'RandomForestClassifier',
                'model_params': self.model_params,
                'training_time_seconds': self.training_time,
                'n_estimators': self.model.n_estimators,
                'n_features': self.model.n_features_in_,
                'n_classes': self.model.n_classes_,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print("Metadonnees sauvegardees avec succes")
    
    def load_model(self, model_filepath):
        
        print(f"\nChargement du modele: {model_filepath}")
        
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(f"Modele non trouve: {model_filepath}")
        
        with open(model_filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Modele charge avec succes")
        
        return self.model
    
    def predict(self, X):
        
        if self.model is None:
            raise ValueError("Aucun modele charge. Charger ou entrainer un modele d'abord.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        
        if self.model is None:
            raise ValueError("Aucun modele charge. Charger ou entrainer un modele d'abord.")
        
        return self.model.predict_proba(X)


# Script principal pour tester
if __name__ == '__main__':
    # Charger les donnees preprocessees
    print("Chargement des donnees preprocessees...")
    
    with open('data/processed/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    print(f"Donnees chargees:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - y_train: {y_train.shape}")
    
    # Parametres du modele
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Creer et entrainer le modele
    trainer = ModelTrainer(model_params)
    model = trainer.train(X_train, y_train)
    
    # Sauvegarder le modele
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filepath = f'models/random_forest_model_{timestamp}.pkl'
    metadata_filepath = f'models/random_forest_metadata_{timestamp}.json'
    
    trainer.save_model(model_filepath, metadata_filepath)
    
    print(f"\nFichiers crees:")
    print(f"  - Modele: {model_filepath}")
    print(f"  - Metadonnees: {metadata_filepath}")