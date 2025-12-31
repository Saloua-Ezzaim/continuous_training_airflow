"""
Script de visualisation des metriques d'entrainement
Cree des graphiques pour suivre l'evolution des performances
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class MetricsVisualizer:
    """Classe pour visualiser les metriques"""
    
    def __init__(self, metrics_dir='models'):
        self.metrics_dir = metrics_dir
        self.metrics_data = []
        
    def load_all_metrics(self):
        """Charger toutes les metriques historiques"""
        print("Chargement des metriques...")
        
        # Trouver tous les fichiers de metriques
        metrics_files = glob.glob(os.path.join(self.metrics_dir, 'metrics_*.json'))
        
        if not metrics_files:
            print("Aucun fichier de metriques trouve")
            return False
        
        print(f"Fichiers trouves: {len(metrics_files)}")
        
        # Charger chaque fichier
        for filepath in sorted(metrics_files):
            try:
                with open(filepath, 'r') as f:
                    metrics = json.load(f)
                    
                # Extraire le timestamp du nom de fichier
                filename = os.path.basename(filepath)
                timestamp_str = filename.replace('metrics_', '').replace('.json', '')
                
                # Convertir en datetime
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    metrics['timestamp'] = timestamp
                    metrics['timestamp_str'] = timestamp_str
                except:
                    metrics['timestamp'] = datetime.now()
                    metrics['timestamp_str'] = timestamp_str
                
                self.metrics_data.append(metrics)
                
            except Exception as e:
                print(f"Erreur lors du chargement de {filepath}: {e}")
        
        print(f"Metriques chargees: {len(self.metrics_data)}")
        return True
    
    def create_metrics_evolution_plot(self, save_path='monitoring/metrics_evolution.png'):
        """Creer un graphique d'evolution des metriques"""
        if not self.metrics_data:
            print("Aucune metrique a visualiser")
            return
        
        print("\nCreation du graphique d'evolution...")
        
        # Convertir en DataFrame
        df = pd.DataFrame(self.metrics_data)
        
        # Trier par timestamp
        df = df.sort_values('timestamp')
        
        # Creer la figure avec 4 sous-graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evolution des Metriques du Modele', fontsize=16, fontweight='bold')
        
        # Metriques a afficher
        metrics_to_plot = [
            ('accuracy', 'Accuracy', axes[0, 0]),
            ('precision', 'Precision', axes[0, 1]),
            ('recall', 'Recall', axes[1, 0]),
            ('f1_score', 'F1-Score', axes[1, 1])
        ]
        
        for metric_name, metric_label, ax in metrics_to_plot:
            if metric_name in df.columns:
                # Ligne d'evolution
                ax.plot(range(len(df)), df[metric_name], 
                       marker='o', linewidth=2, markersize=8, 
                       label=metric_label, color='#2E86AB')
                
                # Ligne de tendance
                z = np.polyfit(range(len(df)), df[metric_name], 1)
                p = np.poly1d(z)
                ax.plot(range(len(df)), p(range(len(df))), 
                       "--", alpha=0.5, color='red', label='Tendance')
                
                # Seuil (si disponible)
                if metric_name == 'accuracy':
                    ax.axhline(y=0.75, color='green', linestyle=':', 
                              alpha=0.5, label='Seuil (75%)')
                elif metric_name == 'f1_score':
                    ax.axhline(y=0.60, color='green', linestyle=':', 
                              alpha=0.5, label='Seuil (60%)')
                
                # Configuration
                ax.set_xlabel('Iteration d\'entrainement', fontsize=10)
                ax.set_ylabel(metric_label, fontsize=10)
                ax.set_title(f'Evolution de {metric_label}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Valeur min/max
                min_val = df[metric_name].min()
                max_val = df[metric_name].max()
                current_val = df[metric_name].iloc[-1]
                
                # Texte avec statistiques
                stats_text = f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nActuel: {current_val:.4f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5), fontsize=9)
        
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegarde: {save_path}")
        
        return save_path
    
    def create_confusion_matrix_plot(self, save_path='monitoring/confusion_matrix_latest.png'):
        """Creer un graphique de la matrice de confusion du dernier modele"""
        if not self.metrics_data:
            print("Aucune metrique a visualiser")
            return
        
        print("\nCreation de la matrice de confusion...")
        
        # Prendre la derniere metrique
        latest_metrics = self.metrics_data[-1]
        
        if 'confusion_matrix' not in latest_metrics:
            print("Pas de matrice de confusion disponible")
            return
        
        cm = latest_metrics['confusion_matrix']
        
        # Creer la figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    ax=ax, cbar_kws={'label': 'Nombre de predictions'})
        
        ax.set_xlabel('Predictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vraies valeurs', fontsize=12, fontweight='bold')
        ax.set_title(f'Matrice de Confusion - {latest_metrics["timestamp_str"]}', 
                    fontsize=14, fontweight='bold')
        
        # Ajouter les pourcentages
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        accuracy = (tn + tp) / total
        text = f'Accuracy: {accuracy:.2%}\nTotal predictions: {total}'
        ax.text(0.5, -0.15, text, ha='center', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardee: {save_path}")
        
        return save_path
    
    def create_metrics_comparison_table(self, save_path='monitoring/metrics_comparison.png'):
        """Creer un tableau comparatif des metriques"""
        if not self.metrics_data:
            print("Aucune metrique a visualiser")
            return
        
        print("\nCreation du tableau comparatif...")
        
        # Convertir en DataFrame
        df = pd.DataFrame(self.metrics_data)
        df = df.sort_values('timestamp')
        
        # Selectionner les colonnes importantes
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_cols = [col for col in metrics_cols if col in df.columns]
        
        if not available_cols:
            print("Pas de metriques disponibles")
            return
        
        # Creer la figure
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparer les donnees du tableau
        table_data = []
        for idx, row in df.iterrows():
            row_data = [row['timestamp_str'][:15]]  # Timestamp court
            for col in available_cols:
                value = row[col]
                row_data.append(f'{value:.4f}')
            table_data.append(row_data)
        
        # Headers
        headers = ['Timestamp'] + [col.replace('_', ' ').title() for col in available_cols]
        
        # Creer le tableau
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.25] + [0.15] * len(available_cols))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style des headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style des lignes (alternance de couleurs)
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor('#E8F4F8')
        
        # Titre
        ax.set_title('Historique des Metriques', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tableau comparatif sauvegarde: {save_path}")
        
        return save_path
    
    def create_roc_curve_comparison(self, save_path='monitoring/roc_evolution.png'):
        """Creer un graphique comparant les courbes ROC"""
        if not self.metrics_data:
            print("Aucune metrique a visualiser")
            return
        
        print("\nCreation du graphique ROC...")
        
        # Convertir en DataFrame
        df = pd.DataFrame(self.metrics_data)
        df = df.sort_values('timestamp')
        
        if 'roc_auc' not in df.columns:
            print("ROC AUC non disponible")
            return
        
        # Creer la figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Evolution du ROC AUC
        ax.plot(range(len(df)), df['roc_auc'], 
               marker='o', linewidth=2, markersize=10,
               color='#2E86AB', label='ROC AUC Score')
        
        # Ligne de reference
        ax.axhline(y=0.5, color='red', linestyle='--', 
                  alpha=0.5, label='Classifieur aleatoire')
        ax.axhline(y=0.75, color='green', linestyle=':', 
                  alpha=0.5, label='Seuil minimum (75%)')
        
        # Configuration
        ax.set_xlabel('Iteration d\'entrainement', fontsize=12)
        ax.set_ylabel('ROC AUC Score', fontsize=12)
        ax.set_title('Evolution du ROC AUC Score', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Annotations
        for i, (idx, row) in enumerate(df.iterrows()):
            if i == 0 or i == len(df) - 1:  # Premier et dernier
                ax.annotate(f'{row["roc_auc"]:.4f}',
                           xy=(i, row['roc_auc']),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique ROC sauvegarde: {save_path}")
        
        return save_path
    
    def generate_summary_report(self, save_path='monitoring/summary_report.txt'):
        """Generer un rapport texte resume"""
        if not self.metrics_data:
            print("Aucune metrique a analyser")
            return
        
        print("\nGeneration du rapport resume...")
        
        # Convertir en DataFrame
        df = pd.DataFrame(self.metrics_data)
        df = df.sort_values('timestamp')
        
        # Ouvrir le fichier
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(" " * 15 + "RAPPORT DE MONITORING DES MODELES\n")
            f.write("=" * 70 + "\n\n")
            
            # Informations generales
            f.write(f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Nombre total d'entrainements: {len(df)}\n")
            f.write(f"Premiere entrainement: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dernier entrainement: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 70 + "\n\n")
            
            # Metriques du dernier modele
            f.write("PERFORMANCES DU DERNIER MODELE\n")
            f.write("-" * 70 + "\n")
            
            latest = df.iloc[-1]
            metrics_info = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score'),
                ('ROC AUC', 'roc_auc'),
                ('Specificity', 'specificity')
            ]
            
            for label, key in metrics_info:
                if key in latest:
                    f.write(f"{label:15s}: {latest[key]:.4f}\n")
            
            # Matrice de confusion
            if 'confusion_matrix' in latest:
                f.write("\nMatrice de Confusion:\n")
                cm = latest['confusion_matrix']
                f.write(f"  TN: {cm[0][0]:5d}  |  FP: {cm[0][1]:5d}\n")
                f.write(f"  FN: {cm[1][0]:5d}  |  TP: {cm[1][1]:5d}\n")
            
            f.write("\n" + "-" * 70 + "\n\n")
            
            # Evolution des metriques
            f.write("EVOLUTION DES METRIQUES\n")
            f.write("-" * 70 + "\n")
            
            for label, key in metrics_info:
                if key in df.columns:
                    values = df[key]
                    f.write(f"\n{label}:\n")
                    f.write(f"  Minimum    : {values.min():.4f}\n")
                    f.write(f"  Maximum    : {values.max():.4f}\n")
                    f.write(f"  Moyenne    : {values.mean():.4f}\n")
                    f.write(f"  Ecart-type : {values.std():.4f}\n")
                    
                    # Tendance
                    if len(values) > 1:
                        first = values.iloc[0]
                        last = values.iloc[-1]
                        change = last - first
                        change_pct = (change / first) * 100
                        trend = "↑" if change > 0 else "↓" if change < 0 else "="
                        f.write(f"  Tendance   : {trend} {change:+.4f} ({change_pct:+.2f}%)\n")
            
            f.write("\n" + "-" * 70 + "\n\n")
            
            # Alertes
            f.write("ALERTES\n")
            f.write("-" * 70 + "\n")
            
            alerts = []
            
            # Verifier les seuils
            if latest.get('accuracy', 1) < 0.75:
                alerts.append(f"⚠ Accuracy en dessous du seuil: {latest['accuracy']:.4f} < 0.75")
            
            if latest.get('f1_score', 1) < 0.60:
                alerts.append(f"⚠ F1-Score en dessous du seuil: {latest['f1_score']:.4f} < 0.60")
            
            if latest.get('roc_auc', 1) < 0.75:
                alerts.append(f"⚠ ROC AUC en dessous du seuil: {latest['roc_auc']:.4f} < 0.75")
            
            # Verifier la degradation
            if len(df) > 1:
                previous = df.iloc[-2]
                for key in ['accuracy', 'f1_score', 'roc_auc']:
                    if key in latest and key in previous:
                        if latest[key] < previous[key] - 0.05:  # Baisse de plus de 5%
                            alerts.append(f"⚠ {key} en baisse significative: "
                                        f"{previous[key]:.4f} → {latest[key]:.4f}")
            
            if alerts:
                for alert in alerts:
                    f.write(f"{alert}\n")
            else:
                f.write("✓ Aucune alerte. Toutes les metriques sont dans les normes.\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Rapport resume sauvegarde: {save_path}")
        return save_path
    
    def generate_all_reports(self):
        """Generer tous les rapports et graphiques"""
        print("\n" + "=" * 70)
        print(" " * 20 + "GENERATION DES RAPPORTS")
        print("=" * 70)
        
        if not self.load_all_metrics():
            print("\nErreur: Impossible de charger les metriques")
            return
        
        import numpy as np  # Import ici pour eviter les erreurs si pas utilise ailleurs
        
        # Generer tous les graphiques
        self.create_metrics_evolution_plot()
        self.create_confusion_matrix_plot()
        self.create_metrics_comparison_table()
        self.create_roc_curve_comparison()
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print(" " * 15 + "TOUS LES RAPPORTS ONT ETE GENERES")
        print("=" * 70)
        print("\nFichiers crees dans le dossier monitoring/:")
        print("  - metrics_evolution.png")
        print("  - confusion_matrix_latest.png")
        print("  - metrics_comparison.png")
        print("  - roc_evolution.png")
        print("  - summary_report.txt")
        print("\n" + "=" * 70)


if __name__ == '__main__':
    # Creer le visualiseur
    visualizer = MetricsVisualizer()
    
    # Generer tous les rapports
    visualizer.generate_all_reports()