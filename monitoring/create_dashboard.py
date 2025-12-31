"""
Creation d'un dashboard HTML interactif
"""

import json
import os
import glob
from datetime import datetime


def create_html_dashboard(output_path='monitoring/dashboard.html'):
    """Creer un dashboard HTML avec les graphiques"""
    
    print("Creation du dashboard HTML...")
    
    # Charger la derniere metrique
    metrics_files = sorted(glob.glob('models/metrics_*.json'))
    
    if not metrics_files:
        print("Aucune metrique trouvee")
        return
    
    with open(metrics_files[-1], 'r') as f:
        latest_metrics = json.load(f)
    
    # Compter le nombre total d'entrainements
    total_trainings = len(metrics_files)
    
    # HTML template
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Continuous Training</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            color: #667eea;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .metric-status {{
            margin-top: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .status-good {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-bad {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .charts-section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .charts-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .chart-container {{
            margin-bottom: 30px;
        }}
        
        .chart-container img {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .info-section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .info-section h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .info-item {{
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .info-item:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #666;
            display: inline-block;
            width: 200px;
        }}
        
        .info-value {{
            color: #333;
        }}
        
        .timestamp {{
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚀 Dashboard de Monitoring</h1>
            <p>Continuous Training Pipeline - Random Forest Model</p>
        </div>
        
        <!-- Metrics Cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{latest_metrics.get('accuracy', 0):.2%}</div>
                <div class="metric-status {'status-good' if latest_metrics.get('accuracy', 0) >= 0.75 else 'status-warning'}">
                    {'✓ Excellent' if latest_metrics.get('accuracy', 0) >= 0.75 else '⚠ Attention'}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{latest_metrics.get('precision', 0):.2%}</div>
                <div class="metric-status status-good">
                    ✓ OK
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{latest_metrics.get('recall', 0):.2%}</div>
                <div class="metric-status status-good">
                    ✓ OK
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{latest_metrics.get('f1_score', 0):.2%}</div>
                <div class="metric-status {'status-good' if latest_metrics.get('f1_score', 0) >= 0.60 else 'status-warning'}">
                    {'✓ Excellent' if latest_metrics.get('f1_score', 0) >= 0.60 else '⚠ Attention'}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">ROC AUC</div>
                <div class="metric-value">{latest_metrics.get('roc_auc', 0):.2%}</div>
                <div class="metric-status {'status-good' if latest_metrics.get('roc_auc', 0) >= 0.75 else 'status-warning'}">
                    {'✓ Excellent' if latest_metrics.get('roc_auc', 0) >= 0.75 else '⚠ Attention'}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Entrainements</div>
                <div class="metric-value">{total_trainings}</div>
                <div class="metric-status status-good">
                    Total
                </div>
            </div>
        </div>
        
        <!-- Charts Section -->
        <div class="charts-section">
            <h2>📊 Visualisations</h2>
            
            <div class="chart-container">
                <h3>Evolution des Metriques</h3>
                <img src="metrics_evolution.png" alt="Evolution des metriques">
            </div>
            
            <div class="chart-container">
                <h3>Matrice de Confusion</h3>
                <img src="confusion_matrix_latest.png" alt="Matrice de confusion">
            </div>
            
            <div class="chart-container">
                <h3>Evolution du ROC AUC</h3>
                <img src="roc_evolution.png" alt="Evolution ROC">
            </div>
        </div>
        
        <!-- Info Section -->
        <div class="info-section">
            <h3>ℹ️ Informations du Dernier Modele</h3>
            <div class="info-item">
                <span class="info-label">Date d'entrainement:</span>
                <span class="info-value">{latest_metrics.get('timestamp', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span class="info-label">True Negatives:</span>
                <span class="info-value">{latest_metrics.get('true_negatives', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span class="info-label">False Positives:</span>
                <span class="info-value">{latest_metrics.get('false_positives', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span class="info-label">False Negatives:</span>
                <span class="info-value">{latest_metrics.get('false_negatives', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span class="info-label">True Positives:</span>
                <span class="info-value">{latest_metrics.get('true_positives', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Specificity:</span>
                <span class="info-value">{latest_metrics.get('specificity', 0):.2%}</span>
            </div>
        </div>
        
        <div class="timestamp">
            Dashboard genere le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard HTML cree: {output_path}")
    return output_path


if __name__ == '__main__':
    create_html_dashboard()