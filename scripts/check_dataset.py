import pandas as pd
import os

def check_dataset():
    data_path = 'data/raw/telco_churn.csv'
    
    if not os.path.exists(data_path):
        print("ERREUR: Le fichier n'existe pas!")
        print(f"Chemin recherché: {os.path.abspath(data_path)}")
        return False
    
    print("Fichier trouvé!")
    
    try:
        df = pd.read_csv(data_path)
        print("Dataset chargé avec succès!")
    except Exception as e:
        print(f"ERREUR lors du chargement: {e}")
        return False
    
    print("\n" + "="*60)
    print("INFORMATIONS SUR LE DATASET")
    print("="*60)
    
    print(f"\nDimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    print(f"\nColonnes disponibles:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\nVariable cible (Churn):")
    if 'Churn' in df.columns:
        print(df['Churn'].value_counts())
        print(f"Taux de churn: {(df['Churn'] == 'Yes').mean()*100:.2f}%")
    else:
        print("Colonne 'Churn' non trouvée")
    
    print(f"\nAperçu des données:")
    print(df.head())
    
    print(f"\nTypes de données:")
    print(df.dtypes.value_counts())
    
    print(f"\nValeurs manquantes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Aucune valeur manquante!")
    
    print(f"\nTaille du fichier: {os.path.getsize(data_path) / 1024:.2f} KB")
    
    print("\n" + "="*60)
    print("VERIFICATION TERMINEE AVEC SUCCES!")
    print("="*60)
    
    return True

if __name__ == '__main__':
    os.makedirs('scripts', exist_ok=True)
    check_dataset()
