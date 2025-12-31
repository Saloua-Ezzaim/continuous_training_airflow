from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import json
import os
from datetime import datetime
import numpy as np
import pickle



class ModelEvaluator:
    """Classe pour l'evaluation du modele"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba=None):
        
        print("="*60)
        print("EVALUATION DU MODELE")
        print("="*60)
        
        # Metriques de base
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred))
        }
        
        # ROC AUC si probabilites disponibles
        if y_pred_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Metriques derivees
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = float(specificity)
        
        # Afficher les resultats
        print(f"\nMetriques de performance:")
        print(f"  - Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  - Precision:   {metrics['precision']:.4f}")
        print(f"  - Recall:      {metrics['recall']:.4f}")
        print(f"  - F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  - Specificity: {metrics['specificity']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  - ROC AUC:     {metrics['roc_auc']:.4f}")
        
        print(f"\nMatrice de confusion:")
        print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
        
        print("\n" + "="*60)
        
        return metrics
    
    @staticmethod
    def detailed_report(y_true, y_pred, target_names=None):
        
        if target_names is None:
            target_names = ['No Churn', 'Churn']
        
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=target_names,
            digits=4
        )
        
        print("\nRapport de classification detaille:")
        print(report)
        
        return report
    
    @staticmethod
    def save_metrics(metrics, filepath):
       
        print(f"\nSauvegarde des metriques: {filepath}")
        
        # Ajouter timestamp
        metrics_with_time = metrics.copy()
        metrics_with_time['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_with_time['evaluation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Creer le dossier si necessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarder
        with open(filepath, 'w') as f:
            json.dump(metrics_with_time, f, indent=4)
        
        print("Metriques sauvegardees avec succes")
    
    @staticmethod
    def load_metrics(filepath):
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier de metriques non trouve: {filepath}")
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    @staticmethod
    def compare_metrics(metrics1, metrics2, model1_name="Model 1", model2_name="Model 2"):
        
        print("="*60)
        print("COMPARAISON DES MODELES")
        print("="*60)
        
        print(f"\n{'Metrique':<15} {model1_name:<15} {model2_name:<15} {'Difference':<15}")
        print("-"*60)
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in key_metrics:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                diff = val2 - val1
                
                # Symbole pour indiquer amelioration ou regression
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                
                print(f"{metric:<15} {val1:<15.4f} {val2:<15.4f} {symbol} {abs(diff):<13.4f}")
        
        print("="*60)


# Script principal pour tester
if __name__ == '__main__':
    import pickle
    
    print("Chargement des donnees et du modele...")
    # Charger les donnees preprocessees
with open('data/processed/processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

X_test = processed_data['X_test']
y_test = processed_data['y_test']

# Trouver le dernier modele
import glob
model_files = glob.glob('models/random_forest_model_*.pkl')

if not model_files:
    print("Erreur: Aucun modele trouve. Entrainer un modele d'abord.")
    exit(1)

latest_model = max(model_files, key=os.path.getctime)
print(f"Modele utilise: {latest_model}")

# Charger le modele
with open(latest_model, 'rb') as f:
    model = pickle.load(f)

# Faire les predictions
print("\nPredictions en cours...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluer
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)

# Rapport detaille
evaluator.detailed_report(y_test, y_pred)

# Sauvegarder les metriques
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metrics_filepath = f'models/metrics_{timestamp}.json'
evaluator.save_metrics(metrics, metrics_filepath)

print(f"\nFichier de metriques cree: {metrics_filepath}")