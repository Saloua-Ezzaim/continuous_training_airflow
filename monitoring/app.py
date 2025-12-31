import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import hashlib

# Configuration
st.set_page_config(
    page_title="Airflow Continuous Training Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'last_alert_hash' not in st.session_state:
    st.session_state.last_alert_hash = None
if 'email_sent_count' not in st.session_state:
    st.session_state.email_sent_count = 0
if 'last_email_time' not in st.session_state:
    st.session_state.last_email_time = None
if 'last_success_run' not in st.session_state:
    st.session_state.last_success_run = None

# CSS Professional Style
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {font-size: 2.5rem; font-weight: 700; margin: 0; color: white;}
    .main-header p {font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.95;}
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-danger {background-color: #fee2e2; border-color: #dc2626; color: #7f1d1d;}
    .alert-warning {background-color: #fef3c7; border-color: #f59e0b; color: #78350f;}
    .alert-info {background-color: #dbeafe; border-color: #3b82f6; color: #1e3a8a;}
    .alert-success {background-color: #d1fae5; border-color: #10b981; color: #064e3b;}
    .success-notification {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    .success-icon {
        font-size: 3rem;
        animation: bounce 1s ease-in-out infinite;
    }
    @keyframes slideIn {
        from {transform: translateX(-100%); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }
    @keyframes bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-10px);}
    }
    .email-status {
        background-color: #f0f9ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Airflow Continuous Training Monitor</h1>
        <p>Real-time ML Pipeline Monitoring & Performance Analytics</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Connection Settings")
    airflow_url = st.text_input("Airflow URL", value="http://localhost:8080")
    
    st.markdown("### Authentication")
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", type="password", value="admin")
    
    st.markdown("---")
    st.markdown("### DAG Configuration")
    dag_id = st.text_input("DAG ID", value="continuous_training_dag")
    limit_runs = st.slider("Number of runs", 1, 50, 10)
    metrics_folder = st.text_input("Metrics Folder", value="models")
    
    st.markdown("---")
    st.markdown("### Email Alerts Configuration")
    enable_email_alerts = st.checkbox("Enable Automatic Email Alerts", value=False)
    
    if enable_email_alerts:
        smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        sender_email = st.text_input("Sender Email", value="salouaezzaim175@gmail.com")
        sender_password = st.text_input("Email Password", type="password", value="coox prlj lpxo zejc")
        recipient_email = st.text_input("Recipient Email", value="yara84721@gmail.com")
        
        st.markdown("---")
        st.info("Email alerts will be sent automatically when new issues are detected")
        
        # Test Email Connection
        if st.button("Test Email Connection"):
            if sender_email and sender_password and recipient_email:
                try:
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.quit()
                    st.success("Connection successful")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
            else:
                st.warning("Please fill all email fields")
    
    st.markdown("---")
    st.markdown("### Alert Thresholds")
    accuracy_threshold = st.slider("Accuracy Threshold", 0.0, 1.0, 0.75, 0.01)
    f1_threshold = st.slider("F1-Score Threshold", 0.0, 1.0, 0.60, 0.01)
    
    st.markdown("---")
    st.markdown("### Display Options")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    show_drift_detection = st.checkbox("Show Drift Detection", value=True)
    
    st.markdown("---")
    
    # Email Status Summary
    if enable_email_alerts:
        st.markdown("### Email Status")
        st.metric("Total Emails Sent", st.session_state.email_sent_count)
        if st.session_state.last_email_time:
            st.text(f"Last sent: {st.session_state.last_email_time}")
    
    st.markdown("---")
    refresh_btn = st.button("Refresh Data", use_container_width=True)

# Functions
def load_metrics_from_files(folder_path='models'):
    """Load all metrics from JSON files"""
    try:
        metrics_files = sorted(glob.glob(f'{folder_path}/metrics_*.json'))
        
        if not metrics_files:
            return None, []
        
        with open(metrics_files[-1], 'r') as f:
            latest_metrics = json.load(f)
        
        all_metrics = []
        for file in metrics_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'timestamp' not in data:
                        filename = os.path.basename(file)
                        date_str = filename.replace('metrics_', '').replace('.json', '')
                        data['timestamp'] = date_str
                    all_metrics.append(data)
            except Exception:
                continue
        
        return latest_metrics, all_metrics
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None, []

def send_email_alert(subject, message, smtp_server, smtp_port, sender_email, sender_password, recipient_email):
    """Send email alert with detailed error handling"""
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 20px; color: white; border-radius: 10px;">
                    <h1>Airflow Training Monitor - Alert</h1>
                </div>
                <div style="padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 10px;">
                    <h2 style="color: #dc2626;">Alert Details</h2>
                    <div style="font-size: 16px; line-height: 1.6;">{message}</div>
                    <hr style="margin: 20px 0;">
                    <p style="color: #666; font-size: 14px;">
                        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                        System: Airflow Continuous Training Monitor
                    </p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.set_debuglevel(0)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True, "Email sent successfully"
        
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Authentication error: {str(e)}"
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"General error: {str(e)}"

def get_alert_hash(alerts):
    """Generate a hash from alert messages to detect changes"""
    alert_string = "|".join(sorted([a['message'] for a in alerts]))
    return hashlib.md5(alert_string.encode()).hexdigest()

def send_auto_email(email_alerts, smtp_server, smtp_port, sender_email, sender_password, recipient_email):
    """Automatically send email when new alerts are detected"""
    current_hash = get_alert_hash([{'message': msg} for msg in email_alerts])
    
    # Check if alerts have changed
    if current_hash != st.session_state.last_alert_hash:
        st.session_state.last_alert_hash = current_hash
        
        email_message = "<br><br>".join([f"<strong>Alert {i+1}:</strong><br>{alert}" for i, alert in enumerate(email_alerts)])
        
        success, msg = send_email_alert(
            subject=f"[AUTO] Training Monitor - {len(email_alerts)} issue(s) detected",
            message=email_message,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            sender_password=sender_password,
            recipient_email=recipient_email
        )
        
        if success:
            st.session_state.email_sent_count += 1
            st.session_state.last_email_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return True, msg
        else:
            return False, msg
    
    return None, "No new alerts to send"

@st.cache_data(ttl=30)
def get_dag_runs(airflow_url, username, password, dag_id, limit):
    """Fetch DAG runs from Airflow API"""
    try:
        url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
        params = {"limit": limit, "order_by": "-execution_date"}
        response = requests.get(url, auth=(username, password), params=params, timeout=10)
        if response.status_code == 200:
            return response.json()["dag_runs"]
        return None
    except Exception:
        return None

@st.cache_data(ttl=30)
def get_task_instances(airflow_url, username, password, dag_id, run_id):
    """Fetch task instances from Airflow API"""
    try:
        url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        response = requests.get(url, auth=(username, password), timeout=10)
        if response.status_code == 200:
            return response.json()["task_instances"]
        return []
    except Exception:
        return []

def detect_drift(metrics_history, window=5):
    """Detect drift in metrics"""
    if len(metrics_history) < window:
        return None
    
    df = pd.DataFrame(metrics_history)
    recent = df.tail(window)
    older = df.head(len(df) - window) if len(df) > window else df.head(window)
    
    drift_detected = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        if metric in df.columns:
            recent_mean = recent[metric].mean()
            older_mean = older[metric].mean()
            change = ((recent_mean - older_mean) / older_mean * 100) if older_mean != 0 else 0
            
            if abs(change) > 5:
                drift_detected[metric] = {
                    'change': change,
                    'recent_mean': recent_mean,
                    'older_mean': older_mean
                }
    
    return drift_detected if drift_detected else None

# Main Logic
if refresh_btn or auto_refresh:
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    with st.spinner("Loading data..."):
        latest_metrics, all_metrics = load_metrics_from_files(metrics_folder)
        dag_runs = get_dag_runs(airflow_url, username, password, dag_id, limit_runs)
        
        if latest_metrics or dag_runs:
            # Check for successful runs and show notification
            if dag_runs:
                latest_run = dag_runs[0]
                if latest_run["state"] == "success" and st.session_state.last_success_run != latest_run['dag_run_id']:
                    st.session_state.last_success_run = latest_run['dag_run_id']
                    st.markdown(f"""
                        <div class="success-notification">
                            <div class="success-icon">✓</div>
                            <div style="flex: 1;">
                                <h3 style="margin: 0; color: #065f46;">Training Run Completed Successfully</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #047857;">
                                    Run ID: <strong>{latest_run['dag_run_id']}</strong><br>
                                    Completed at: <strong>{latest_run.get('end_date', 'N/A')}</strong>
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Alerts Section
            st.markdown('<p class="section-header">System Alerts</p>', unsafe_allow_html=True)
            
            alerts = []
            email_alerts = []
            
            # Check metrics
            if latest_metrics:
                accuracy = latest_metrics.get('accuracy', 1.0)
                f1_score = latest_metrics.get('f1_score', 1.0)
                
                if accuracy < accuracy_threshold:
                    alert_msg = f'Low accuracy detected: {accuracy:.4f} (Threshold: {accuracy_threshold:.2f})'
                    alerts.append({'type': 'danger', 'message': f'CRITICAL: {alert_msg}'})
                    email_alerts.append(alert_msg)
                
                if f1_score < f1_threshold:
                    alert_msg = f'Low F1-Score detected: {f1_score:.4f} (Threshold: {f1_threshold:.2f})'
                    alerts.append({'type': 'warning', 'message': f'WARNING: {alert_msg}'})
                    email_alerts.append(alert_msg)
            
            # Display alerts and auto-send email
            if alerts:
                for alert in alerts:
                    st.markdown(f'<div class="alert-box alert-{alert["type"]}">{alert["message"]}</div>', unsafe_allow_html=True)
                
                # Automatic Email Sending
                if enable_email_alerts and email_alerts:
                    if not all([sender_email, sender_password, recipient_email]):
                        st.warning("Email alerts enabled but credentials incomplete. Please configure in sidebar.")
                    else:
                        # Auto-send email
                        email_status, email_msg = send_auto_email(
                            email_alerts,
                            smtp_server,
                            smtp_port,
                            sender_email,
                            sender_password,
                            recipient_email
                        )
                        
                        # Display email status
                        st.markdown('<div class="email-status">', unsafe_allow_html=True)
                        col_status1, col_status2 = st.columns([1, 3])
                        
                        with col_status1:
                            if email_status is True:
                                st.success("✓ Email Sent")
                            elif email_status is False:
                                st.error("✗ Email Failed")
                            else:
                                st.info("○ No Change")
                        
                        with col_status2:
                            st.write(f"**Status:** {email_msg}")
                            st.write(f"**Recipients:** {recipient_email}")
                            st.write(f"**Alert Count:** {len(email_alerts)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-success">System Status: Operational - No alerts detected</div>', unsafe_allow_html=True)
                # Reset hash when no alerts
                if st.session_state.last_alert_hash is not None:
                    st.session_state.last_alert_hash = None
            
            # ML Metrics Section
            if latest_metrics:
                st.markdown('<p class="section-header">Model Performance Metrics</p>', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("Accuracy", f"{latest_metrics.get('accuracy', 0):.4f}")
                col2.metric("Precision", f"{latest_metrics.get('precision', 0):.4f}")
                col3.metric("Recall", f"{latest_metrics.get('recall', 0):.4f}")
                col4.metric("F1-Score", f"{latest_metrics.get('f1_score', 0):.4f}")
                col5.metric("ROC AUC", f"{latest_metrics.get('roc_auc', 0):.4f}")
                col6.metric("Total Trainings", len(all_metrics))
                
                with st.expander("Detailed Model Information"):
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.text(f"Timestamp: {latest_metrics.get('timestamp', 'N/A')}")
                        st.text(f"True Positives: {latest_metrics.get('true_positives', 'N/A')}")
                        st.text(f"True Negatives: {latest_metrics.get('true_negatives', 'N/A')}")
                    with col_d2:
                        st.text(f"False Positives: {latest_metrics.get('false_positives', 'N/A')}")
                        st.text(f"False Negatives: {latest_metrics.get('false_negatives', 'N/A')}")
                        st.text(f"Specificity: {latest_metrics.get('specificity', 0):.4f}")
            
            # Metrics History
            if len(all_metrics) > 1:
                st.markdown('<p class="section-header">Performance Trends</p>', unsafe_allow_html=True)
                
                df_metrics = pd.DataFrame(all_metrics)
                
                fig_evolution = go.Figure()
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                colors = ['#3b82f6', '#10b981', '#f59e0b', '#06b6d4', '#ef4444']
                
                for idx, metric in enumerate(metrics_to_plot):
                    if metric in df_metrics.columns:
                        fig_evolution.add_trace(go.Scatter(
                            x=list(range(1, len(df_metrics) + 1)),
                            y=df_metrics[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title(),
                            line=dict(color=colors[idx], width=2),
                            marker=dict(size=8)
                        ))
                
                fig_evolution.update_layout(
                    title="Metric Evolution Across Training Iterations",
                    xaxis_title="Training Iteration",
                    yaxis_title="Score",
                    height=400,
                    hovermode='x unified',
                    yaxis_range=[0, 1],
                    template='plotly_white'
                )
                st.plotly_chart(fig_evolution, use_container_width=True)
                
                # Comparison
                if len(all_metrics) >= 2:
                    st.markdown('<p class="section-header">Latest vs Previous Comparison</p>', unsafe_allow_html=True)
                    
                    col_comp1, col_comp2 = st.columns(2)
                    
                    latest = all_metrics[-1]
                    previous = all_metrics[-2]
                    
                    comparison_data = []
                    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                        if metric in latest and metric in previous:
                            change = ((latest[metric] - previous[metric]) / previous[metric] * 100) if previous[metric] != 0 else 0
                            comparison_data.append({
                                'Metric': metric.replace('_', ' ').title(),
                                'Latest': latest[metric],
                                'Previous': previous[metric],
                                'Change': change
                            })
                    
                    df_comp = pd.DataFrame(comparison_data)
                    
                    with col_comp1:
                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Bar(name='Latest', x=df_comp['Metric'], y=df_comp['Latest'], marker_color='#3b82f6'))
                        fig_comp.add_trace(go.Bar(name='Previous', x=df_comp['Metric'], y=df_comp['Previous'], marker_color='#64748b'))
                        fig_comp.update_layout(barmode='group', height=350, yaxis_range=[0, 1], title="Score Comparison", template='plotly_white')
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    with col_comp2:
                        fig_change = px.bar(df_comp, x='Metric', y='Change', color='Change',
                                          color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                                          title="Percentage Change")
                        fig_change.update_layout(height=350, showlegend=False, template='plotly_white')
                        st.plotly_chart(fig_change, use_container_width=True)
            
            # Drift Detection
            if show_drift_detection and len(all_metrics) >= 5:
                st.markdown('<p class="section-header">Drift Analysis</p>', unsafe_allow_html=True)
                
                drift_results = detect_drift(all_metrics, window=5)
                
                if drift_results:
                    st.warning("Performance drift detected in recent training iterations")
                    
                    drift_data = []
                    for metric, info in drift_results.items():
                        status = 'Degradation' if info['change'] < 0 else 'Improvement'
                        drift_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Change (%)': f"{info['change']:.2f}%",
                            'Recent Average': f"{info['recent_mean']:.4f}",
                            'Previous Average': f"{info['older_mean']:.4f}",
                            'Status': status
                        })
                    
                    st.dataframe(pd.DataFrame(drift_data), use_container_width=True, hide_index=True)
                    st.info("Recommendation: Review training data and consider model retraining")
                else:
                    st.success("No significant drift detected - Model performance is stable")
            
            # Airflow Statistics
            if dag_runs:
                st.markdown('<p class="section-header">Pipeline Execution Statistics</p>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(dag_runs)
                success = sum(1 for r in dag_runs if r["state"] == "success")
                failed = sum(1 for r in dag_runs if r["state"] == "failed")
                running = sum(1 for r in dag_runs if r["state"] == "running")
                
                col1.metric("Total Runs", total)
                col2.metric("Successful", success, delta=f"{(success/total*100):.1f}%")
                col3.metric("Failed", failed, delta=f"{(failed/total*100):.1f}%", delta_color="inverse")
                col4.metric("Running", running)
                
                with st.expander("Recent Execution History"):
                    runs_data = []
                    for run in dag_runs[:10]:
                        runs_data.append({
                            'Run ID': run['dag_run_id'],
                            'State': run['state'].upper(),
                            'Start Date': run['start_date'],
                            'End Date': run.get('end_date', 'N/A')
                        })
                    st.dataframe(pd.DataFrame(runs_data), use_container_width=True, hide_index=True)
            
            # Footer
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem; color: #64748b; border-top: 1px solid #e2e8f0; margin-top: 3rem;'>
                    <p><strong>Airflow Continuous Training Monitor</strong></p>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """, unsafe_allow_html=True)
        
        else:
            st.error("No data available. Please verify Airflow connection and metrics folder.")

else:
    st.info("Configure connection settings and click 'Refresh Data' to start monitoring")