import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import json
from datetime import datetime


class DataProcessor:
    """Classe pour le preprocessing des donnees"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
       
        print(f"Chargement des donnees depuis: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouve: {filepath}")
            
        df = pd.read_csv(filepath)
        print(f"Donnees chargees: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        return df
    
    def clean_data(self, df):
        """
        Nettoyer les donnees
        
        Args:
            df: DataFrame pandas
            
        Returns:
            DataFrame nettoye
        """
        print("\nNettoyage des donnees...")
        
        # Copie pour eviter les modifications inplace
        df = df.copy()
        
        # Supprimer la colonne customerID (non utile pour la prediction)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            print("- Colonne customerID supprimee")
        
        # Gerer TotalCharges (peut contenir des espaces au lieu de valeurs)
        if 'TotalCharges' in df.columns:
            # Convertir en numeric, les valeurs invalides deviennent NaN
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Remplacer les NaN par la mediane
            if df['TotalCharges'].isnull().sum() > 0:
                median_value = df['TotalCharges'].median()
                df['TotalCharges'].fillna(median_value, inplace=True)
                print(f"- Valeurs manquantes dans TotalCharges remplacees par la mediane: {median_value:.2f}")
        
        # Verifier les valeurs manquantes restantes
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"- Suppression de {missing_count} lignes avec valeurs manquantes")
            df = df.dropna()
        else:
            print("- Aucune valeur manquante")
        
        print(f"Donnees nettoyees: {df.shape[0]} lignes")
        
        return df
    
    def encode_features(self, df, fit=True):
        """
        Encoder les variables categorielles
        
        Args:
            df: DataFrame pandas
            fit: Si True, cree les encoders. Si False, utilise les encoders existants
            
        Returns:
            DataFrame avec variables encodees
        """
        print("\nEncodage des variables categorielles...")
        
        df = df.copy()
        
        # Identifier les colonnes categorielles (type object)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Retirer la colonne cible de la liste si presente
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        print(f"- {len(categorical_cols)} colonnes categorielles trouvees")
        
        # Encoder chaque colonne
        for col in categorical_cols:
            if fit:
                # Creer un nouvel encoder
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Utiliser l'encoder existant
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    print(f"  Warning: Pas d'encoder pour {col}")
        
        # Encoder la variable cible si presente
        if 'Churn' in df.columns:
            if fit:
                self.label_encoders['Churn'] = LabelEncoder()
                df['Churn'] = self.label_encoders['Churn'].fit_transform(df['Churn'])
                print(f"- Variable cible encodee: {dict(zip(self.label_encoders['Churn'].classes_, self.label_encoders['Churn'].transform(self.label_encoders['Churn'].classes_)))}")
            else:
                df['Churn'] = self.label_encoders['Churn'].transform(df['Churn'])
        
        print("Encodage termine")
        
        return df
    
    def split_data(self, df, target_col='Churn', test_size=0.2, random_state=42):
        """
        Separer les donnees en train et test
        
        Args:
            df: DataFrame pandas
            target_col: Nom de la colonne cible
            test_size: Proportion du test set
            random_state: Seed pour la reproductibilite
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nSeparation des donnees (test_size={test_size})...")
        
        if target_col not in df.columns:
            raise ValueError(f"Colonne cible '{target_col}' non trouvee")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"- Train set: {X_train.shape[0]} lignes")
        print(f"- Test set: {X_test.shape[0]} lignes")
        print(f"- Features: {X_train.shape[1]} colonnes")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, fit=True):
        """
        Normaliser les features
        
        Args:
            X_train: Features d'entrainement
            X_test: Features de test
            fit: Si True, fit le scaler. Si False, utilise le scaler existant
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        print("\nNormalisation des features...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("- Scaler fit sur le train set")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Normalisation terminee")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessor(self, filepath):
        """
        Sauvegarder le preprocessor (scaler + encoders)
        
        Args:
            filepath: Chemin pour sauvegarder le preprocessor
        """
        print(f"\nSauvegarde du preprocessor: {filepath}")
        
        preprocessor = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        print("Preprocessor sauvegarde avec succes")
    
    def load_preprocessor(self, filepath):
        """
        Charger un preprocessor sauvegarde
        
        Args:
            filepath: Chemin vers le preprocessor
        """
        print(f"\nChargement du preprocessor: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor non trouve: {filepath}")
        
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        self.scaler = preprocessor['scaler']
        self.label_encoders = preprocessor['label_encoders']
        
        print(f"Preprocessor charge (cree le: {preprocessor.get('timestamp', 'inconnu')})")
    
    def process_pipeline(self, input_filepath, output_filepath, preprocessor_filepath):
        """
        Pipeline complet de preprocessing
        
        Args:
            input_filepath: Chemin vers les donnees brutes
            output_filepath: Chemin pour sauvegarder les donnees preprocessees
            preprocessor_filepath: Chemin pour sauvegarder le preprocessor
            
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        print("="*60)
        print("DEBUT DU PIPELINE DE PREPROCESSING")
        print("="*60)
        
        # 1. Charger les donnees
        df = self.load_data(input_filepath)
        
        # 2. Nettoyer les donnees
        df = self.clean_data(df)
        
        # 3. Encoder les variables categorielles
        df = self.encode_features(df, fit=True)
        
        # 4. Separer train/test
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # 5. Normaliser les features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # 6. Sauvegarder les donnees preprocessees
        print(f"\nSauvegarde des donnees preprocessees: {output_filepath}")
        
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': X_train.columns.tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        with open(output_filepath, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print("Donnees preprocessees sauvegardees")
        
        # 7. Sauvegarder le preprocessor
        self.save_preprocessor(preprocessor_filepath)
        
        print("\n" + "="*60)
        print("PREPROCESSING TERMINE AVEC SUCCES")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Script principal pour tester
if __name__ == '__main__':
    # Chemins
    input_file = 'data/raw/telco_churn.csv'
    output_file = 'data/processed/processed_data.pkl'
    preprocessor_file = 'models/preprocessor.pkl'
    
    # Creer et executer le pipeline
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline(
        input_file, 
        output_file, 
        preprocessor_file
    )
    
    print(f"\nResultats finaux:")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_test: {X_test.shape}")
    print(f"- y_train: {y_train.shape}")
    print(f"- y_test: {y_test.shape}")