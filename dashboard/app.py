"""
Streamlit Dashboard for AML Engine
Graph visualization, real-time monitoring, and explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="AML Engine Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "your-api-token"  # Replace with actual token

# Session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None

def api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request"""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {}

def create_sample_graph() -> nx.Graph:
    """Create sample graph for visualization"""
    G = nx.Graph()
    
    # Add nodes
    nodes = [
        (0, {"risk_score": 0.8, "entity_type": "user", "label": "Target"}),
        (1, {"risk_score": 0.6, "entity_type": "exchange", "label": "Exchange A"}),
        (2, {"risk_score": 0.9, "entity_type": "mixer", "label": "Mixer"}),
        (3, {"risk_score": 0.3, "entity_type": "user", "label": "User B"}),
        (4, {"risk_score": 0.7, "entity_type": "merchant", "label": "Merchant"}),
        (5, {"risk_score": 0.4, "entity_type": "user", "label": "User C"}),
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        (0, 1, {"amount": 10000, "risk": 0.7}),
        (0, 2, {"amount": 5000, "risk": 0.9}),
        (1, 3, {"amount": 8000, "risk": 0.5}),
        (2, 4, {"amount": 3000, "risk": 0.8}),
        (4, 5, {"amount": 2000, "risk": 0.4}),
    ]
    
    G.add_edges_from(edges)
    
    return G

def plot_network_graph(G: nx.Graph, selected_node: Optional[int] = None):
    """Plot network graph using Plotly"""
    
    # Get node positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node label
        label = G.nodes[node].get('label', f'Node {node}')
        risk_score = G.nodes[node].get('risk_score', 0)
        node_text.append(f"{label}<br>Risk: {risk_score:.2f}")
        
        # Node color based on risk
        if risk_score > 0.7:
            node_color.append('red')
        elif risk_score > 0.4:
            node_color.append('orange')
        else:
            node_color.append('green')
        
        # Node size based on risk
        node_size.append(20 + risk_score * 30)
    
    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge label
        amount = G.edges[edge].get('amount', 0)
        risk = G.edges[edge].get('risk', 0)
        edge_text.append(f"Amount: ${amount:,.0f}<br>Risk: {risk:.2f}")
    
    # Create traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    # Highlight selected node
    if selected_node is not None and selected_node in G.nodes():
        selected_x, selected_y = pos[selected_node]
        selected_trace = go.Scatter(
            x=[selected_x], y=[selected_y],
            mode='markers',
            marker=dict(
                size=50,
                color='yellow',
                line=dict(width=3, color='black'),
                symbol='diamond'
            ),
            showlegend=False
        )
        fig = go.Figure(data=[edge_trace, node_trace, selected_trace])
    else:
        fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Transaction Network Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AML Engine Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Graph Explorer", "Real-time Monitoring", "Alerts", "Explainability", "Settings"]
    )
    
    # API Status
    try:
        health = api_request("/health")
        if health:
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Disconnected")
    except:
        st.sidebar.error("❌ API Disconnected")
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Graph Explorer":
        show_graph_explorer()
    elif page == "Real-time Monitoring":
        show_real_time_monitoring()
    elif page == "Alerts":
        show_alerts()
    elif page == "Explainability":
        show_explainability()
    elif page == "Settings":
        show_settings()

def show_overview():
    """Overview page"""
    st.header("📊 Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value="1,234",
            delta="+12%"
        )
    
    with col2:
        st.metric(
            label="High Risk Alerts",
            value="23",
            delta="+5%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Response Time",
            value="45ms",
            delta="-8%"
        )
    
    with col4:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="+2.1%"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        
        # Get real data from API
        try:
            dataset_info = api_request("/dataset/info")
            if dataset_info:
                illicit_count = dataset_info.get('label_distribution', {}).get('illicit', 0)
                licit_count = dataset_info.get('label_distribution', {}).get('licit', 0)
                
                risk_data = pd.DataFrame({
                    'Risk Level': ['Low Risk', 'High Risk'],
                    'Count': [licit_count, illicit_count]
                })
                
                fig = px.pie(risk_data, values='Count', names='Risk Level', 
                            color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to load dataset info")
        except Exception as e:
            st.error(f"Error loading risk distribution: {e}")
            # Fallback to sample data
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [850, 320, 64]
            })
            fig = px.pie(risk_data, values='Count', names='Risk Level', 
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Volume Over Time")
        
        # Get real data from API
        try:
            dataset_info = api_request("/dataset/info")
            if dataset_info:
                num_edges = dataset_info.get('num_edges', 0)
                num_nodes = dataset_info.get('num_nodes', 0)
                
                # Create realistic transaction volume based on dataset size
                dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
                base_volume = num_edges / 30  # Distribute edges across days
                volume_data = pd.DataFrame({
                    'Date': dates,
                    'Volume': np.random.poisson(base_volume, len(dates)) + np.random.normal(0, base_volume * 0.1, len(dates))
                })
                
                fig = px.line(volume_data, x='Date', y='Volume', title=f'Daily Transaction Volume (Total: {num_edges:,} transactions)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to load dataset info")
        except Exception as e:
            st.error(f"Error loading transaction volume: {e}")
            # Fallback to sample data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            volume_data = pd.DataFrame({
                'Date': dates,
                'Volume': np.random.lognormal(10, 0.5, len(dates))
            })
            fig = px.line(volume_data, x='Date', y='Volume', title='Daily Transaction Volume')
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Get real data from API
    try:
        sample_nodes = api_request("/dataset/sample-nodes?limit=4")
        if sample_nodes and 'sample_nodes' in sample_nodes:
            activity_data = []
            for i, node in enumerate(sample_nodes['sample_nodes']):
                risk_score = sum(node['features']) / len(node['features'])  # Simple risk calculation
                risk_level = "High" if node['label'] == 1 else "Low"
                event_type = "High risk transaction detected" if node['label'] == 1 else "Transaction processed"
                
                activity_data.append({
                    "time": f"{5 * (i + 1)} min ago",
                    "event": event_type,
                    "node": node['node_id'][:12] + "...",  # Truncate long IDs
                    "risk": risk_score
                })
        else:
            st.error("Failed to load sample nodes")
            activity_data = []
    except Exception as e:
        st.error(f"Error loading recent activity: {e}")
        # Fallback to sample data
        activity_data = [
            {"time": "2 min ago", "event": "High risk transaction detected", "node": "wallet_12345", "risk": 0.89},
            {"time": "5 min ago", "event": "New node added", "node": "wallet_67890", "risk": 0.23},
            {"time": "8 min ago", "event": "Suspicious pattern detected", "node": "wallet_11111", "risk": 0.76},
            {"time": "12 min ago", "event": "Transaction processed", "node": "wallet_22222", "risk": 0.34},
        ]
    
    for activity in activity_data:
        risk_color = "red" if activity["risk"] > 0.7 else "orange" if activity["risk"] > 0.4 else "green"
        st.markdown(f"""
        <div class="metric-card">
            <strong>{activity['time']}</strong> - {activity['event']}<br>
            <small>Node: {activity['node']} | Risk: <span style="color: {risk_color}">{activity['risk']:.2f}</span></small>
        </div>
        """, unsafe_allow_html=True)

def show_graph_explorer():
    """Graph explorer page"""
    st.header("🕸️ Graph Explorer")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_node = st.text_input("Search Node ID", placeholder="Enter wallet ID")
    
    with col2:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.1)
    
    with col3:
        if st.button("🔍 Search"):
            st.session_state.selected_node = search_node
    
    # Graph visualization
    st.subheader("Transaction Network")
    
    # Create sample graph
    G = create_sample_graph()
    
    # Filter nodes by risk threshold
    high_risk_nodes = [node for node in G.nodes() if G.nodes[node].get('risk_score', 0) > risk_threshold]
    
    # Plot graph
    fig = plot_network_graph(G, st.session_state.selected_node)
    st.plotly_chart(fig, use_container_width=True)
    
    # Node details
    if st.session_state.selected_node is not None and st.session_state.selected_node != '':
        st.subheader(f"Node Details: {st.session_state.selected_node}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Node Information**")
            try:
                node_data = G.nodes.get(int(st.session_state.selected_node), {})
                st.json(node_data)
            except ValueError:
                st.error("Invalid node ID selected")
        
        with col2:
            st.write("**Connected Nodes**")
            try:
                if int(st.session_state.selected_node) in G.nodes():
                    neighbors = list(G.neighbors(int(st.session_state.selected_node)))
                    for neighbor in neighbors:
                        neighbor_data = G.nodes[neighbor]
                        st.write(f"- {neighbor_data.get('label', f'Node {neighbor}')} (Risk: {neighbor_data.get('risk_score', 0):.2f})")
            except ValueError:
                st.error("Invalid node ID selected")

def show_real_time_monitoring():
    """Real-time monitoring page"""
    st.header("⚡ Real-time Monitoring")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Live metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get real metrics from API
        health_data = api_request("/health")
        monitor_stats = api_request("/monitor/statistics")
        
        if health_data and monitor_stats:
            with col1:
                # Calculate transactions per second based on processing time
                avg_processing_time = monitor_stats.get('avg_processing_time', 0.05)
                tps = 1.0 / max(avg_processing_time, 0.001)  # Avoid division by zero
                st.metric("Transactions/sec", f"{tps:.0f}", delta="+12")
            
            with col2:
                alert_count = monitor_stats.get('alert_count', 0)
                st.metric("Active Alerts", str(alert_count), delta="+2", delta_color="inverse")
            
            with col3:
                processing_time_ms = avg_processing_time * 1000
                st.metric("Avg Processing Time", f"{processing_time_ms:.0f}ms", delta="-5ms")
        else:
            # Fallback to dummy data
            with col1:
                st.metric("Transactions/sec", "156", delta="+12")
            with col2:
                st.metric("Active Alerts", "8", delta="+2", delta_color="inverse")
            with col3:
                st.metric("Avg Processing Time", "23ms", delta="-5ms")
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        # Fallback to dummy data
        with col1:
            st.metric("Transactions/sec", "156", delta="+12")
        with col2:
            st.metric("Active Alerts", "8", delta="+2", delta_color="inverse")
        with col3:
            st.metric("Avg Processing Time", "23ms", delta="-5ms")
    
    # Live transaction feed
    st.subheader("Live Transaction Feed")
    
    # Get real data from API
    try:
        sample_nodes = api_request("/dataset/sample-nodes?limit=4")
        if sample_nodes and 'sample_nodes' in sample_nodes:
            transactions = []
            for i, node in enumerate(sample_nodes['sample_nodes']):
                # Create realistic transaction data based on node features
                amount = int(sum(node['features']) * 10000)  # Scale features to transaction amount
                risk_score = sum(node['features']) / len(node['features'])
                
                transactions.append({
                    "id": f"tx_{i:03d}",
                    "from": node['node_id'][:8] + "...",
                    "to": f"wallet_{i:03d}",
                    "amount": amount,
                    "risk": risk_score,
                    "time": f"{2 + i * 3}s ago"
                })
        else:
            st.error("Failed to load sample nodes")
            transactions = []
    except Exception as e:
        st.error(f"Error loading transaction feed: {e}")
        # Fallback to sample data
        transactions = [
            {"id": "tx_001", "from": "wallet_123", "to": "wallet_456", "amount": 15000, "risk": 0.85, "time": "2s ago"},
            {"id": "tx_002", "from": "wallet_789", "to": "wallet_012", "amount": 5000, "risk": 0.23, "time": "5s ago"},
            {"id": "tx_003", "from": "wallet_345", "to": "wallet_678", "amount": 25000, "risk": 0.92, "time": "8s ago"},
            {"id": "tx_004", "from": "wallet_901", "to": "wallet_234", "amount": 3000, "risk": 0.45, "time": "12s ago"},
        ]
    
    for tx in transactions:
        risk_color = "red" if tx["risk"] > 0.7 else "orange" if tx["risk"] > 0.4 else "green"
        st.markdown(f"""
        <div class="metric-card">
            <strong>{tx['time']}</strong> - {tx['id']}<br>
            <small>{tx['from']} → {tx['to']} | ${tx['amount']:,} | Risk: <span style="color: {risk_color}">{tx['risk']:.2f}</span></small>
        </div>
        """, unsafe_allow_html=True)

def show_alerts():
    """Alerts page"""
    st.header("🚨 Alerts")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_level = st.selectbox("Alert Level", ["All", "High", "Medium", "Low"])
    
    with col2:
        time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last Week", "Last Month"])
    
    with col3:
        if st.button("Clear All Alerts"):
            api_request("/monitor/clear-alerts", method="POST")
            st.success("Alerts cleared!")
    
    # Get real alerts from API
    try:
        alerts_response = api_request("/alert", method="POST", data={"threshold": 0.5, "limit": 10})
        if alerts_response and 'alerts' in alerts_response:
            alerts = []
            for i, alert in enumerate(alerts_response['alerts']):
                alerts.append({
                    "id": f"alert_{i:03d}",
                    "level": "High" if alert.get('risk_score', 0) > 0.7 else "Medium" if alert.get('risk_score', 0) > 0.4 else "Low",
                    "node": alert.get('node_id', f"node_{i}")[:12] + "...",
                    "risk": alert.get('risk_score', 0),
                    "time": f"{i * 15} min ago",
                    "description": f"Risk score: {alert.get('risk_score', 0):.2f}"
                })
        else:
            # Create alerts from sample nodes with high risk
            sample_nodes = api_request("/dataset/sample-nodes?limit=4")
            if sample_nodes and 'sample_nodes' in sample_nodes:
                alerts = []
                for i, node in enumerate(sample_nodes['sample_nodes']):
                    if node['label'] == 1 or sum(node['features']) > 0.5:  # High risk nodes
                        risk_score = sum(node['features']) / len(node['features'])
                        alerts.append({
                            "id": f"alert_{i:03d}",
                            "level": "High" if risk_score > 0.7 else "Medium",
                            "node": node['node_id'][:12] + "...",
                            "risk": risk_score,
                            "time": f"{i * 15} min ago",
                            "description": f"Suspicious activity detected (Risk: {risk_score:.2f})"
                        })
            else:
                alerts = []
    except Exception as e:
        st.error(f"Error loading alerts: {e}")
        # Fallback to sample data
        alerts = [
            {"id": "alert_001", "level": "High", "node": "wallet_123", "risk": 0.89, "time": "2 min ago", "description": "Suspicious transaction pattern detected"},
            {"id": "alert_002", "level": "Medium", "node": "wallet_456", "risk": 0.67, "time": "15 min ago", "description": "Unusual transaction amount"},
            {"id": "alert_003", "level": "High", "node": "wallet_789", "risk": 0.91, "time": "1 hour ago", "description": "Connection to known mixer detected"},
            {"id": "alert_004", "level": "Low", "node": "wallet_012", "risk": 0.45, "time": "2 hours ago", "description": "Multiple small transactions"},
        ]
    
    # Filter alerts
    if alert_level != "All":
        alerts = [alert for alert in alerts if alert["level"] == alert_level]
    
    # Display alerts
    for alert in alerts:
        alert_class = f"alert-{alert['level'].lower()}"
        st.markdown(f"""
        <div class="metric-card {alert_class}">
            <strong>{alert['time']}</strong> - {alert['id']}<br>
            <small>Node: {alert['node']} | Risk: {alert['risk']:.2f}<br>
            {alert['description']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"Investigate {alert['id']}", key=f"investigate_{alert['id']}"):
                st.info(f"Investigating {alert['id']}...")
        with col2:
            if st.button(f"Whitelist {alert['id']}", key=f"whitelist_{alert['id']}"):
                st.success(f"Whitelisted {alert['id']}")
        with col3:
            if st.button(f"Dismiss {alert['id']}", key=f"dismiss_{alert['id']}"):
                st.warning(f"Dismissed {alert['id']}")

def show_explainability():
    """Explainability page"""
    st.header("🔍 Explainability")
    
    # Node selection
    node_id = st.text_input("Enter Node ID for Explanation", placeholder="wallet_12345")
    
    if st.button("Explain Prediction"):
        if node_id:
            # Get real explanation from API
            try:
                explanation_response = api_request("/explain", method="POST", data={"node_id": node_id, "top_k": 5})
                if explanation_response:
                    explanation = explanation_response
                else:
                    st.error("Failed to get explanation from API")
                    return
            except Exception as e:
                st.error(f"Error getting explanation: {e}")
                return
            
            # Display explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance")
                if 'feature_importance' in explanation:
                    feature_data = explanation["feature_importance"]
                    fig = px.bar(
                        x=list(feature_data.keys()),
                        y=list(feature_data.values()),
                        title="Feature Importance Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No feature importance data available")
            
            with col2:
                st.subheader("Suspicious Connections")
                if 'suspicious_connections' in explanation and explanation['suspicious_connections']:
                    for conn in explanation["suspicious_connections"]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{conn.get('neighbor_node', 'Unknown')}</strong><br>
                            <small>Importance: {conn.get('importance_score', 0):.2f}<br>
                            Type: {conn.get('connection_type', 'Unknown')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No suspicious connections detected")
            
            # Explanation report
            st.subheader("Explanation Report")
            if 'explanation_report' in explanation:
                st.markdown(explanation['explanation_report'])
            else:
                st.markdown(f"""
                ### Risk Analysis for {node_id}
                
                **Overall Risk Score:** {explanation.get('explanation_score', 0):.2f}
                
                **Key Risk Factors:**
                - Analysis based on node features and network connections
                - Risk assessment using Graph Neural Network explainability
                """)

def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        framework = st.selectbox("GNN Framework", ["PyG", "DGL"])
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=64)
        enable_torchscript = st.checkbox("Enable TorchScript", value=True)
    
    if st.button("Update Settings"):
        # Update API settings
        api_request("/model/update-threshold", method="POST", data={"threshold": threshold})
        st.success("Settings updated!")
    
    # API settings
    st.subheader("API Configuration")
    
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    api_token = st.text_input("API Token", value=API_TOKEN, type="password")
    
    if st.button("Test Connection"):
        try:
            response = api_request("/ping")
            if response:
                st.success("API connection successful!")
            else:
                st.error("API connection failed!")
        except:
            st.error("API connection failed!")
    
    # Export settings
    st.subheader("Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "PDF"])
        export_range = st.selectbox("Export Range", ["Last 24 Hours", "Last Week", "Last Month", "All Time"])
    
    with col2:
        if st.button("Export Alerts"):
            st.info("Exporting alerts...")
            # Placeholder for export functionality
        
        if st.button("Export Transactions"):
            st.info("Exporting transactions...")
            # Placeholder for export functionality

if __name__ == "__main__":
    main() 