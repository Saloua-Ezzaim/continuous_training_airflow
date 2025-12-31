"""
Script principal pour generer tous les rapports de monitoring
"""

import os
import sys

# Ajouter le chemin parent au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.visualize_metrics import MetricsVisualizer
from monitoring.create_dashboard import create_html_dashboard


def generate_all_monitoring_reports():
    """Generer tous les rapports de monitoring"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "GENERATION COMPLETE DU MONITORING")
    print("=" * 70 + "\n")
    
    # 1. Generer les visualisations
    print("Etape 1/2: Generation des graphiques...")
    visualizer = MetricsVisualizer()
    visualizer.generate_all_reports()
    print("\n")
    
    # 2. Generer le dashboard HTML
    print("Etape 2/2: Generation du dashboard HTML...")
    dashboard_path = create_html_dashboard()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "TERMINE!")
    print("=" * 70)
    print("\nPour voir le dashboard, ouvrez:")
    print(f"  {os.path.abspath(dashboard_path)}")
    print("\nOu double-cliquez sur le fichier dashboard.html dans le dossier monitoring/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    generate_all_monitoring_reports()