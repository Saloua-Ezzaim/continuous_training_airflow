"""
Script pour simuler l'arrivee de nouvelles donnees
"""

import pandas as pd
import os
from datetime import datetime
import random


def simulate_new_data(n_samples=100, output_dir='data/new_data'):
    """
    Simuler de nouvelles donnees en prenant un echantillon du dataset original
    
    Args:
        n_samples: Nombre de lignes a generer
        output_dir: Dossier de sortie
    """
    print("Simulation de nouvelles donnees...")
    
    # Charger le dataset original
    original_file = 'data/raw/telco_churn.csv'
    
    if not os.path.exists(original_file):
        print(f"Erreur: Fichier non trouve: {original_file}")
        return
    
    df = pd.read_csv(original_file)
    print(f"Dataset original: {len(df)} lignes")
    
    # Prendre un echantillon aleatoire
    sample = df.sample(n=min(n_samples, len(df)), random_state=random.randint(1, 10000))
    
    # Creer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Generer un nom de fichier avec timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'new_churn_data_{timestamp}.csv')
    
    # Sauvegarder
    sample.to_csv(output_file, index=False)
    
    print(f"Nouvelles donnees generees:")
    print(f"  - Fichier: {output_file}")
    print(f"  - Nombre de lignes: {len(sample)}")
    print(f"  - Taux de churn: {(sample['Churn'] == 'Yes').mean()*100:.2f}%")
    
    return output_file


if __name__ == '__main__':
    import sys
    
    # Nombre de lignes (par defaut 100)
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    simulate_new_data(n_samples)