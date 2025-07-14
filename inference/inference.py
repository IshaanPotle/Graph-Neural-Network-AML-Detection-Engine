"""
Inference Module for AML Engine
Real-time GNN inference with ONNX/TorchScript acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Any
import time
import json
from pathlib import Path
import pickle

# Framework imports
try:
    import torch_geometric
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

if TYPE_CHECKING:
    if DGL_AVAILABLE:
        from dgl import DGLGraph
    else:
        DGLGraph = None

from loguru import logger


class AMLInferenceEngine:
    """
    Real-time inference engine for AML detection
    Supports ONNX and TorchScript optimization
    """
    
    def __init__(self,
                 model: nn.Module,
                 framework: str = "pyg",
                 device: str = "cpu",
                 use_onnx: bool = False,
                 use_torchscript: bool = False,
                 threshold: float = 0.7):
        """
        Initialize AML inference engine
        
        Args:
            model: Trained GNN model
            framework: Framework ('pyg' or 'dgl')
            device: Device to run on
            use_onnx: Whether to use ONNX optimization
            use_torchscript: Whether to use TorchScript optimization
            threshold: Classification threshold
        """
        self.model = model
        self.framework = framework
        self.device = device
        self.threshold = threshold
        self.use_onnx = use_onnx
        self.use_torchscript = use_torchscript
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize explainer
        from .explain import AMLExplainer
        self.explainer = AMLExplainer(model, framework, device)
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_inferences = 0
        
        # Optimize model if requested
        if use_torchscript:
            self._optimize_torchscript()
        elif use_onnx:
            self._optimize_onnx()
        
        # Initialize embedding cache
        self.embedding_cache = {}
        self.cache_size = 10000
        
        logger.info(f"Initialized AML Inference Engine on {device}")
    
    def _optimize_torchscript(self):
        """Optimize model using TorchScript"""
        try:
            # Create dummy input for tracing
            if self.framework == "pyg":
                dummy_data = Data(
                    x=torch.randn(10, self.model.in_channels),
                    edge_index=torch.randint(0, 10, (2, 20)),
                    edge_attr=torch.randn(20, 6)
                )
                self.model = torch.jit.trace(self.model, dummy_data)
            else:
                # For DGL, we need to handle differently
                dummy_g = dgl.rand_graph(10, 20)
                dummy_g.ndata['feat'] = torch.randn(10, self.model.in_channels)
                dummy_g.ndata['label'] = torch.randint(0, 2, (10,))
                self.model = torch.jit.trace(self.model, dummy_g)
            
            logger.info("Model optimized with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
    
    def _optimize_onnx(self):
        """Optimize model using ONNX (placeholder)"""
        logger.info("ONNX optimization not implemented yet")
    
    def predict_single_node(self, 
                           node_features: torch.Tensor,
                           neighbor_indices: List[int],
                           neighbor_features: List[torch.Tensor],
                           edge_features: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Predict risk for a single node
        
        Args:
            node_features: Node features
            neighbor_indices: List of neighbor node indices
            neighbor_features: List of neighbor features
            edge_features: List of edge features (optional)
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = hash(str(node_features.cpu().numpy().tobytes()))
        cached_embedding = self.get_cached_embedding(cache_key)
        
        if cached_embedding is not None:
            self.cache_hits += 1
            # Use cached embedding for prediction
            with torch.no_grad():
                # Create a simple prediction from cached embedding
                logits = self.model.classifier(cached_embedding.unsqueeze(0))
                probabilities = F.softmax(logits, dim=1)
                fraud_prob = probabilities[0, 1].item()
                prediction = 1 if fraud_prob > self.threshold else 0
        else:
            self.cache_misses += 1
            
            # Create subgraph for the node
            if self.framework == "pyg":
                graph_data = self._create_pyg_subgraph(
                    node_features, neighbor_indices, neighbor_features, edge_features
                )
            else:
                graph_data = self._create_dgl_subgraph(
                    node_features, neighbor_indices, neighbor_features, edge_features
                )
            
            # Make prediction
            with torch.no_grad():
                embeddings, logits = self.model(graph_data)
                probabilities = F.softmax(logits, dim=1)
                fraud_prob = probabilities[0, 1].item()  # Probability of fraud
                prediction = 1 if fraud_prob > self.threshold else 0
            
            # Cache embeddings
            self._cache_embeddings(cache_key, embeddings[0])  # 0 is the target node index
        
        inference_time = time.time() - start_time
        
        # Track performance metrics
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        # Keep only last 1000 inference times for average calculation
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        
        return {
            'prediction': prediction,
            'fraud_probability': fraud_prob,
            'confidence': max(fraud_prob, 1 - fraud_prob),
            'inference_time': inference_time,
            'node_embedding': embeddings[0].cpu().numpy() if 'embeddings' in locals() else cached_embedding.cpu().numpy()
        }
    
    def predict_batch(self, 
                     batch_data: List[Dict]) -> List[Dict]:
        """
        Predict risk for a batch of nodes
        
        Args:
            batch_data: List of node data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        for node_data in batch_data:
            result = self.predict_single_node(
                node_data['node_features'],
                node_data['neighbor_indices'],
                node_data['neighbor_features'],
                node_data.get('edge_features')
            )
            results.append(result)
        
        return results
    
    def _create_pyg_subgraph(self, 
                           node_features: torch.Tensor,
                           neighbor_indices: List[int],
                           neighbor_features: List[torch.Tensor],
                           edge_features: Optional[List[torch.Tensor]] = None) -> Data:
        """Create PyG subgraph for inference"""
        
        # Combine node features
        all_node_features = [node_features] + neighbor_features
        x = torch.stack(all_node_features)
        
        # Create edge indices (target node connects to all neighbors)
        edge_index = []
        for i, neighbor_idx in enumerate(neighbor_indices):
            edge_index.append([0, i + 1])  # Target node (0) to neighbor (i+1)
            edge_index.append([i + 1, 0])  # Bidirectional
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create edge features
        if edge_features:
            edge_attr = torch.stack(edge_features)
        else:
            edge_attr = torch.ones(len(edge_index[0]), 6)  # Default edge features
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    
    def _create_dgl_subgraph(self, 
                           node_features: torch.Tensor,
                           neighbor_indices: List[int],
                           neighbor_features: List[torch.Tensor],
                           edge_features: Optional[List[torch.Tensor]] = None) -> Any:
        """Create DGL subgraph for inference"""
        
        # Create graph
        num_nodes = 1 + len(neighbor_features)
        src_nodes = []
        dst_nodes = []
        
        for i in range(len(neighbor_features)):
            src_nodes.extend([0, i + 1])  # Bidirectional edges
            dst_nodes.extend([i + 1, 0])
        
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        
        # Add node features
        all_node_features = [node_features] + neighbor_features
        g.ndata['feat'] = torch.stack(all_node_features)
        
        # Add edge features
        if edge_features:
            g.edata['feat'] = torch.stack(edge_features)
        
        return g
    
    def _cache_embeddings(self, node_id: int, embedding: torch.Tensor):
        """Cache node embeddings for faster lookup"""
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[node_id] = embedding.cpu()
    
    def get_cached_embedding(self, node_id: int) -> Optional[torch.Tensor]:
        """Get cached embedding for a node"""
        return self.embedding_cache.get(node_id)
    
    def explain_prediction(self, 
                          node_id: int,
                          graph_data,
                          top_k: int = 5) -> Dict:
        """
        Explain prediction for a node
        
        Args:
            node_id: Node ID
            graph_data: Graph data
            top_k: Number of top suspicious connections
            
        Returns:
            Explanation results
        """
        explanation = self.explainer.explain_node(
            data=graph_data, 
            target_node=node_id,
            num_hops=2,
            epochs=50,  # Reduced for faster response
            lr=0.01
        )
        suspicious_connections = self.explainer.get_top_suspicious_connections(
            graph_data, node_id, top_k
        )
        
        return {
            'explanation': explanation,
            'suspicious_connections': suspicious_connections,
            'explanation_report': self.explainer.generate_explanation_report(explanation)
        }
    
    def get_risk_score(self, 
                      node_features: torch.Tensor,
                      neighbor_features: List[torch.Tensor],
                      edge_features: Optional[List[torch.Tensor]] = None) -> float:
        """
        Get risk score for a node
        
        Args:
            node_features: Node features
            neighbor_features: List of neighbor features
            edge_features: List of edge features (optional)
            
        Returns:
            Risk score (0-1)
        """
        # Create dummy neighbor indices
        neighbor_indices = list(range(len(neighbor_features)))
        
        result = self.predict_single_node(
            node_features, neighbor_indices, neighbor_features, edge_features
        )
        
        return result['fraud_probability']
    
    def batch_risk_assessment(self, 
                            nodes_data: List[Dict]) -> List[Dict]:
        """
        Perform batch risk assessment
        
        Args:
            nodes_data: List of node data
            
        Returns:
            List of risk assessment results
        """
        results = []
        
        for node_data in nodes_data:
            risk_score = self.get_risk_score(
                node_data['node_features'],
                node_data['neighbor_features'],
                node_data.get('edge_features')
            )
            
            result = {
                'node_id': node_data['node_id'],
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'recommendations': self._get_recommendations(risk_score)
            }
            results.append(result)
        
        return results
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on score"""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_recommendations(self, risk_score: float) -> List[str]:
        """Get recommendations based on risk score"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "Immediate investigation required",
                "Freeze account temporarily",
                "Enhanced due diligence"
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "Monitor closely",
                "Additional verification",
                "Review transaction patterns"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Regular monitoring",
                "Standard verification"
            ])
        else:
            recommendations.append("Normal processing")
        
        return recommendations
    
    def save_model(self, path: str):
        """Save optimized model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'framework': self.framework,
            'threshold': self.threshold,
            'model_config': {
                'in_channels': getattr(self.model, 'in_channels', None),
                'hidden_channels': getattr(self.model, 'hidden_channels', None),
                'out_channels': getattr(self.model, 'out_channels', None)
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint.get('threshold', self.threshold)
        logger.info(f"Model loaded from {path}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': self.cache_hits / max(self.total_inferences, 1),
            'avg_inference_time': sum(self.inference_times) / max(self.total_inferences, 1),
            'framework': self.framework,
            'device': self.device,
            'optimization': 'torchscript' if self.use_torchscript else 'onnx' if self.use_onnx else 'none'
        }


class RealTimeAMLMonitor:
    """
    Real-time AML monitoring system
    """
    
    def __init__(self, 
                 inference_engine: AMLInferenceEngine,
                 alert_threshold: float = 0.8,
                 batch_size: int = 100):
        """
        Initialize real-time monitor
        
        Args:
            inference_engine: AML inference engine
            alert_threshold: Threshold for alerts
            batch_size: Batch size for processing
        """
        self.inference_engine = inference_engine
        self.alert_threshold = alert_threshold
        self.batch_size = batch_size
        self.alert_queue = []
        self.processed_count = 0
        self.processing_times = []
        
        logger.info(f"Initialized Real-time AML Monitor with threshold {alert_threshold}")
    
    def process_transaction(self, 
                          transaction_data: Dict) -> Dict:
        """
        Process a single transaction
        
        Args:
            transaction_data: Transaction data
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        # Extract node and neighbor information
        node_features = transaction_data['node_features']
        neighbor_features = transaction_data.get('neighbor_features', [])
        edge_features = transaction_data.get('edge_features')
        
        # Make prediction
        result = self.inference_engine.predict_single_node(
            node_features, [], neighbor_features, edge_features
        )
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 1000 processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Check for alerts
        if result['fraud_probability'] > self.alert_threshold:
            alert = {
                'transaction_id': transaction_data.get('transaction_id'),
                'node_id': transaction_data.get('node_id'),
                'risk_score': result['fraud_probability'],
                'timestamp': time.time(),
                'details': result
            }
            self.alert_queue.append(alert)
        
        self.processed_count += 1
        
        return {
            'transaction_id': transaction_data.get('transaction_id'),
            'risk_score': result['fraud_probability'],
            'alert_triggered': result['fraud_probability'] > self.alert_threshold,
            'processing_time': processing_time
        }
    
    def process_batch(self, 
                     transactions: List[Dict]) -> List[Dict]:
        """
        Process a batch of transactions
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List of processing results
        """
        results = []
        
        for transaction in transactions:
            result = self.process_transaction(transaction)
            results.append(result)
        
        return results
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """
        Get recent alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        return self.alert_queue[-limit:]
    
    def clear_alerts(self):
        """Clear alert queue"""
        self.alert_queue.clear()
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'processed_count': self.processed_count,
            'alert_count': len(self.alert_queue),
            'alert_rate': len(self.alert_queue) / max(self.processed_count, 1),
            'avg_processing_time': sum(self.processing_times) / max(self.processed_count, 1),
            'alert_threshold': self.alert_threshold
        } 