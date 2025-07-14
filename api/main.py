"""
FastAPI Backend for AML Engine
Real-time inference, alerts, and explainability endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import time
import json
import os
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphSAGE, GAT, TemporalGraphNetwork
from inference.inference import AMLInferenceEngine, RealTimeAMLMonitor
from data.loader import GNNInputLoader
from loguru import logger
from api.metrics import get_metrics, metrics_collector

# Security
security = HTTPBearer()

# Global variables
inference_engine = None
monitor = None
model = None
framework = "pyg"
graph_data = None  # Store actual graph data
node_mapping = None  # Store node ID to index mapping

# Pydantic models
class NodeFeatures(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    creation_time: float
    entity_type: int
    total_volume: float
    tx_count: int
    avg_tx_amount: float

class EdgeFeatures(BaseModel):
    amount: float
    timestamp: float
    direction: int
    tx_type: int
    fee: float
    block_height: int

class PredictionRequest(BaseModel):
    node_id: str
    node_features: NodeFeatures
    neighbor_features: List[NodeFeatures] = []
    edge_features: Optional[List[EdgeFeatures]] = None

class PredictionResponse(BaseModel):
    node_id: str
    prediction: int
    fraud_probability: float
    confidence: float
    inference_time: float
    risk_level: str
    recommendations: List[str]

class AlertRequest(BaseModel):
    threshold: float = Field(0.8, ge=0, le=1)
    limit: int = Field(100, ge=1, le=1000)

class AlertResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total_count: int
    alert_rate: float

class ExplainRequest(BaseModel):
    node_id: str
    top_k: int = Field(5, ge=1, le=20)

class ExplainResponse(BaseModel):
    node_id: str
    explanation_score: float
    suspicious_connections: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    explanation_report: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

async def startup_event():
    """Initialize models and inference engine on startup"""
    global inference_engine, monitor, model, framework, graph_data, node_mapping
    
    logger.info("Starting AML Engine API...")
    
    # Load configuration
    framework = os.getenv("GNN_FRAMEWORK", "pyg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load actual dataset
    logger.info("Loading Elliptic++ dataset...")
    try:
        loader = GNNInputLoader(framework=framework, data_path='./data')
        graph_data = loader.load_elliptic_dataset(
            nodes_file='wallets_features.csv',
            edges_file='AddrAddr_edgelist.csv',
            classes_file='wallets_classes.csv',
            max_nodes=10000  # Load only 10K nodes for faster testing
        )
        node_mapping = graph_data['node_mapping']
        logger.info(f"Loaded dataset: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        # Fallback to dummy data
        graph_data = None
        node_mapping = None
    
    # Initialize model (placeholder - in production, load trained model)
    in_channels = 6
    hidden_channels = 128
    out_channels = 64
    
    if framework == "pyg":
        model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            framework=framework
        )
    else:
        model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            framework=framework
        )
    
    # Initialize inference engine
    inference_engine = AMLInferenceEngine(
        model=model,
        framework=framework,
        device=device,
        use_torchscript=True,
        threshold=0.7
    )
    
    # Initialize real-time monitor
    monitor = RealTimeAMLMonitor(
        inference_engine=inference_engine,
        alert_threshold=0.8
    )
    
    logger.info(f"AML Engine initialized with {framework} framework on {device}")

async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AML Engine API...")

# Create FastAPI app
app = FastAPI(
    title="AML Engine API",
    description="Graph Neural Network-based Anti-Money Laundering Engine",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - replace with proper auth in production"""
    token = credentials.credentials
    # Add proper token validation here
    return {"user_id": "default_user"}

# Utility functions
def convert_node_features(node_features: NodeFeatures) -> torch.Tensor:
    """Convert Pydantic model to tensor"""
    return torch.tensor([
        node_features.risk_score,
        node_features.creation_time,
        node_features.entity_type,
        node_features.total_volume,
        node_features.tx_count,
        node_features.avg_tx_amount
    ], dtype=torch.float32)

def convert_edge_features(edge_features: List[EdgeFeatures]) -> List[torch.Tensor]:
    """Convert Pydantic models to tensors"""
    return [
        torch.tensor([
            ef.amount, ef.timestamp, ef.direction,
            ef.tx_type, ef.fee, ef.block_height
        ], dtype=torch.float32)
        for ef in edge_features
    ]

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "AML Engine API",
        "version": "1.0.0",
        "framework": framework
    }

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "pong", "timestamp": datetime.now().isoformat()}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_info={
            "framework": framework,
            "device": inference_engine.device,
            "threshold": inference_engine.threshold
        },
        performance_metrics=inference_engine.get_performance_metrics()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, user: Dict = Depends(get_current_user)):
    """
    Predict fraud risk for a single node
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Convert features to tensors
        node_features = convert_node_features(request.node_features)
        neighbor_features = [convert_node_features(nf) for nf in request.neighbor_features]
        edge_features = None
        if request.edge_features:
            edge_features = convert_edge_features(request.edge_features)
        
        # Make prediction
        result = inference_engine.predict_single_node(
            node_features=node_features,
            neighbor_indices=list(range(len(neighbor_features))),
            neighbor_features=neighbor_features,
            edge_features=edge_features
        )
        
        # Get risk level and recommendations
        risk_level = inference_engine._get_risk_level(result['fraud_probability'])
        recommendations = inference_engine._get_recommendations(result['fraud_probability'])
        
        return PredictionResponse(
            node_id=request.node_id,
            prediction=result['prediction'],
            fraud_probability=result['fraud_probability'],
            confidence=result['confidence'],
            inference_time=result['inference_time'],
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest], user: Dict = Depends(get_current_user)):
    """
    Predict fraud risk for multiple nodes
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        results = []
        
        for request in requests:
            # Convert features to tensors
            node_features = convert_node_features(request.node_features)
            neighbor_features = [convert_node_features(nf) for nf in request.neighbor_features]
            edge_features = None
            if request.edge_features:
                edge_features = convert_edge_features(request.edge_features)
            
            # Make prediction
            result = inference_engine.predict_single_node(
                node_features=node_features,
                neighbor_indices=list(range(len(neighbor_features))),
                neighbor_features=neighbor_features,
                edge_features=edge_features
            )
            
            # Get risk level and recommendations
            risk_level = inference_engine._get_risk_level(result['fraud_probability'])
            recommendations = inference_engine._get_recommendations(result['fraud_probability'])
            
            results.append(PredictionResponse(
                node_id=request.node_id,
                prediction=result['prediction'],
                fraud_probability=result['fraud_probability'],
                confidence=result['confidence'],
                inference_time=result['inference_time'],
                risk_level=risk_level,
                recommendations=recommendations
            ))
        
        return results
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/alert", response_model=AlertResponse)
async def get_alerts(request: AlertRequest, user: Dict = Depends(get_current_user)):
    """
    Get recent alerts
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        alerts = monitor.get_alerts(limit=request.limit)
        stats = monitor.get_statistics()
        
        return AlertResponse(
            alerts=alerts,
            total_count=stats['alert_count'],
            alert_rate=stats['alert_rate']
        )
    
    except Exception as e:
        logger.error(f"Alert retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")

@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest, user: Dict = Depends(get_current_user)):
    """
    Explain prediction for a node using actual dataset
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        # Convert node_id to index using the actual mapping
        if request.node_id not in node_mapping:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found in dataset")
        
        node_idx = node_mapping[request.node_id]
        
        # Convert to framework-specific format
        if framework == "pyg":
            import torch_geometric
            from torch_geometric.data import Data
            
            # Use actual graph data
            x = torch.tensor(graph_data['node_features'], dtype=torch.float32)
            edge_index = torch.tensor(graph_data['edge_indices'].T, dtype=torch.long)
            edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
            graph_data_pyg = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            # Get explanation using actual data
            explanation_result = inference_engine.explain_prediction(
                node_id=node_idx,
                graph_data=graph_data_pyg,
                top_k=request.top_k
            )
        else:
            import dgl
            # Convert to DGL format
            src_nodes = graph_data['edge_indices'][:, 0]
            dst_nodes = graph_data['edge_indices'][:, 1]
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=graph_data['num_nodes'])
            g.ndata['feat'] = torch.tensor(graph_data['node_features'], dtype=torch.float32)
            g.edata['feat'] = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
            
            explanation_result = inference_engine.explain_prediction(
                node_id=node_idx,
                graph_data=g,
                top_k=request.top_k
            )
        
        return ExplainResponse(
            node_id=request.node_id,
            explanation_score=explanation_result['explanation']['explanation_score'],
            suspicious_connections=explanation_result['suspicious_connections'],
            feature_importance=explanation_result['explanation']['feature_importance'],
            explanation_report=explanation_result['explanation_report']
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/monitor/transaction")
async def process_transaction(transaction_data: Dict, user: Dict = Depends(get_current_user)):
    """
    Process a transaction for real-time monitoring
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        result = monitor.process_transaction(transaction_data)
        return result
    
    except Exception as e:
        logger.error(f"Transaction processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Transaction processing failed: {str(e)}")

@app.get("/monitor/statistics")
async def get_monitor_statistics(user: Dict = Depends(get_current_user)):
    """
    Get monitoring statistics
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        return monitor.get_statistics()
    
    except Exception as e:
        logger.error(f"Statistics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@app.post("/monitor/clear-alerts")
async def clear_alerts(user: Dict = Depends(get_current_user)):
    """
    Clear alert queue
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        monitor.clear_alerts()
        return {"message": "Alerts cleared successfully"}
    
    except Exception as e:
        logger.error(f"Alert clearing error: {e}")
        raise HTTPException(status_code=500, detail=f"Alert clearing failed: {str(e)}")

@app.get("/model/info")
async def get_model_info(user: Dict = Depends(get_current_user)):
    """
    Get model information
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        return {
            "framework": framework,
            "device": inference_engine.device,
            "threshold": inference_engine.threshold,
            "optimization": "torchscript" if inference_engine.use_torchscript else "none",
            "performance_metrics": inference_engine.get_performance_metrics()
        }
    
    except Exception as e:
        logger.error(f"Model info retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

@app.post("/model/update-threshold")
async def update_threshold(threshold: float = Query(..., ge=0, le=1), user: Dict = Depends(get_current_user)):
    """
    Update prediction threshold
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        inference_engine.threshold = threshold
        return {"message": f"Threshold updated to {threshold}"}
    
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold update failed: {str(e)}")

# Background tasks
async def process_transaction_background(transaction_data: Dict):
    """Background task for processing transactions"""
    if monitor is not None:
        try:
            monitor.process_transaction(transaction_data)
        except Exception as e:
            logger.error(f"Background transaction processing error: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    try:
        # Update graph metrics if data is available
        if graph_data is not None:
            metrics_collector.record_graph_load(
                num_nodes=graph_data['num_nodes'],
                num_edges=graph_data['num_edges'],
                load_time=0.1  # Placeholder
            )
        
        return Response(
            content=get_metrics(),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return Response(
            content="# Error collecting metrics\n",
            media_type="text/plain",
            status_code=500
        )

@app.get("/dataset/sample-nodes")
async def get_sample_nodes(limit: int = Query(10, ge=1, le=100), user: Dict = Depends(get_current_user)):
    """
    Get sample node IDs from the loaded dataset for testing
    """
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        # Get sample node IDs
        sample_nodes = list(node_mapping.keys())[:limit]
        
        # Get their features and labels
        sample_data = []
        for node_id in sample_nodes:
            node_idx = node_mapping[node_id]
            features = graph_data['node_features'][node_idx]
            label = graph_data['node_labels'][node_idx]
            
            sample_data.append({
                "node_id": node_id,
                "features": features.tolist(),
                "label": int(label),
                "risk_level": "High" if label == 1 else "Low"
            })
        
        return {
            "sample_nodes": sample_data,
            "total_nodes": len(node_mapping),
            "message": f"Showing {len(sample_data)} sample nodes from {len(node_mapping)} total nodes"
        }
    
    except Exception as e:
        logger.error(f"Sample nodes retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Sample nodes retrieval failed: {str(e)}")

@app.get("/dataset/info")
async def get_dataset_info(user: Dict = Depends(get_current_user)):
    """
    Get information about the loaded dataset
    """
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        # Count labels
        labels = graph_data['node_labels']
        illicit_count = int(np.sum(labels == 1))
        licit_count = int(np.sum(labels == 0))
        
        return {
            "dataset_name": "Elliptic++",
            "num_nodes": graph_data['num_nodes'],
            "num_edges": graph_data['num_edges'],
            "node_features": graph_data['node_features'].shape[1],
            "edge_features": graph_data['edge_features'].shape[1],
            "label_distribution": {
                "illicit": illicit_count,
                "licit": licit_count,
                "illicit_percentage": round(illicit_count / len(labels) * 100, 2)
            },
            "framework": framework
        }
    
    except Exception as e:
        logger.error(f"Dataset info retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset info retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 