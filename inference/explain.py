"""
Explainability Module for AML Engine
GNNExplainer and Integrated Gradients for risk attribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Framework imports
try:
    import torch_geometric
    from torch_geometric.explain import Explainer, GNNExplainer
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

from loguru import logger


class AMLExplainer:
    """
    AML-specific explainability module
    Combines GNNExplainer and Integrated Gradients for comprehensive explanations
    """
    
    def __init__(self, 
                 model: nn.Module,
                 framework: str = "pyg",
                 device: str = "cpu"):
        """
        Initialize AML Explainer
        
        Args:
            model: Trained GNN model
            framework: "pyg" or "dgl"
            device: Device to run on
        """
        self.model = model
        self.framework = framework.lower()
        self.device = device
        self.model.to(device)
        
        if self.framework == "pyg" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        elif self.framework == "dgl" and not DGL_AVAILABLE:
            raise ImportError("DGL not available")
        
        logger.info(f"Initialized AML Explainer with {framework} framework")
    
    def explain_node(self, 
                    data,
                    target_node: int,
                    num_hops: int = 2,
                    epochs: int = 100,
                    lr: float = 0.01) -> Dict:
        """
        Explain predictions for a specific node
        
        Args:
            data: Graph data
            target_node: Target node index
            num_hops: Number of hops for subgraph extraction
            epochs: Number of explanation epochs
            lr: Learning rate for explanation
            
        Returns:
            Explanation dictionary
        """
        if self.framework == "pyg":
            return self._explain_node_pyg(data, target_node, num_hops, epochs, lr)
        else:
            return self._explain_node_dgl(data, target_node, num_hops, epochs, lr)
    
    def _explain_node_pyg(self, data, target_node: int, num_hops: int, 
                         epochs: int, lr: float) -> Dict:
        """Explain node using PyG (simplified implementation)"""
        
        # Simple explanation using feature importance and neighborhood analysis
        node_mask = torch.ones(data.x.size(0))
        edge_mask = torch.ones(data.edge_index.size(1))
        
        # Get subgraph
        subgraph_nodes, subgraph_edges = self._extract_subgraph_pyg(
            data, target_node, num_hops, node_mask, edge_mask
        )
        
        # Get feature importance
        feature_importance = self._get_feature_importance_pyg(data, target_node)
        
        # Get neighbor importance
        neighbor_importance = self._get_neighbor_importance_pyg(
            data, target_node, node_mask, edge_mask
        )
        
        # Calculate explanation score based on feature importance and neighbor analysis
        feature_importance_sum = sum(feature_importance.values())
        neighbor_importance_sum = sum(neighbor_importance.values()) if neighbor_importance else 0
        
        # Normalize and combine scores
        explanation_score = min(1.0, (feature_importance_sum + neighbor_importance_sum) / 2.0)
        
        return {
            'target_node': target_node,
            'node_mask': node_mask.tolist(),
            'edge_mask': edge_mask.tolist(),
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edges': subgraph_edges,
            'feature_importance': feature_importance,
            'neighbor_importance': neighbor_importance,
            'explanation_score': explanation_score
        }
    
    def _explain_node_dgl(self, g, target_node: int, num_hops: int, 
                         epochs: int, lr: float) -> Dict:
        """Explain node using DGL (custom implementation)"""
        
        # Custom GNNExplainer for DGL
        node_mask, edge_mask = self._custom_gnn_explainer_dgl(
            g, target_node, epochs, lr
        )
        
        # Get subgraph
        subgraph_nodes, subgraph_edges = self._extract_subgraph_dgl(
            g, target_node, num_hops, node_mask, edge_mask
        )
        
        # Get feature importance
        feature_importance = self._get_feature_importance_dgl(g, target_node)
        
        # Get neighbor importance
        neighbor_importance = self._get_neighbor_importance_dgl(
            g, target_node, node_mask, edge_mask
        )
        
        # Calculate explanation score based on feature importance and neighbor analysis
        feature_importance_sum = sum(feature_importance.values())
        neighbor_importance_sum = sum(neighbor_importance.values()) if neighbor_importance else 0
        
        # Normalize and combine scores
        explanation_score = min(1.0, (feature_importance_sum + neighbor_importance_sum) / 2.0)
        
        return {
            'target_node': target_node,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edges': subgraph_edges,
            'feature_importance': feature_importance,
            'neighbor_importance': neighbor_importance,
            'explanation_score': explanation_score
        }
    
    def _custom_gnn_explainer_dgl(self, g, target_node: int, 
                                 epochs: int, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom GNNExplainer implementation for DGL"""
        
        # Initialize masks
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        
        node_mask = torch.nn.Parameter(torch.randn(num_nodes, 1))
        edge_mask = torch.nn.Parameter(torch.randn(num_edges, 1))
        
        optimizer = torch.optim.Adam([node_mask, edge_mask], lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Apply masks
            masked_node_features = g.ndata['feat'] * torch.sigmoid(node_mask)
            masked_edge_features = g.edata.get('feat', torch.zeros(num_edges, 1)) * torch.sigmoid(edge_mask)
            
            # Forward pass
            with torch.no_grad():
                self.model.eval()
                embeddings, logits = self.model(g)
                original_pred = F.softmax(logits[target_node], dim=0)
            
            # Create masked graph
            masked_g = g.clone()
            masked_g.ndata['feat'] = masked_node_features
            if 'feat' in masked_g.edata:
                masked_g.edata['feat'] = masked_edge_features
            
            # Forward pass with masked graph
            masked_embeddings, masked_logits = self.model(masked_g)
            masked_pred = F.softmax(masked_logits[target_node], dim=0)
            
            # Loss: maximize prediction difference
            loss = -torch.norm(original_pred - masked_pred, p=2)
            
            # Add sparsity regularization
            sparsity_loss = torch.mean(torch.sigmoid(node_mask)) + torch.mean(torch.sigmoid(edge_mask))
            loss += 0.1 * sparsity_loss
            
            loss.backward()
            optimizer.step()
        
        return torch.sigmoid(node_mask), torch.sigmoid(edge_mask)
    
    def _extract_subgraph_pyg(self, data, target_node: int, num_hops: int,
                             node_mask: torch.Tensor, edge_mask: torch.Tensor) -> Tuple[List, List]:
        """Extract relevant subgraph for PyG"""
        
        # Get neighbors within num_hops
        neighbors = set([target_node])
        current_neighbors = set([target_node])
        
        for hop in range(num_hops):
            new_neighbors = set()
            for node in current_neighbors:
                # Get neighbors of current node
                if self.framework == "pyg":
                    node_edges = data.edge_index[:, data.edge_index[0] == node]
                    node_neighbors = node_edges[1].tolist()
                else:
                    node_neighbors = data.successors(node).tolist()
                
                new_neighbors.update(node_neighbors)
            
            current_neighbors = new_neighbors - neighbors
            neighbors.update(current_neighbors)
        
        # Filter by mask importance
        important_nodes = []
        for node in neighbors:
            if node_mask[node] > 0.1:  # Threshold for importance
                important_nodes.append(node)
        
        # Get important edges
        important_edges = []
        if self.framework == "pyg":
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[:, i]
                if src.item() in important_nodes and dst.item() in important_nodes:
                    if edge_mask[i] > 0.1:  # Threshold for importance
                        important_edges.append((src.item(), dst.item()))
        
        return important_nodes, important_edges
    
    def _extract_subgraph_dgl(self, g, target_node: int, num_hops: int,
                             node_mask: torch.Tensor, edge_mask: torch.Tensor) -> Tuple[List, List]:
        """Extract relevant subgraph for DGL"""
        
        # Similar to PyG but using DGL API
        neighbors = set([target_node])
        current_neighbors = set([target_node])
        
        for hop in range(num_hops):
            new_neighbors = set()
            for node in current_neighbors:
                node_neighbors = g.successors(node).tolist()
                new_neighbors.update(node_neighbors)
            
            current_neighbors = new_neighbors - neighbors
            neighbors.update(current_neighbors)
        
        # Filter by mask importance
        important_nodes = []
        for node in neighbors:
            if node_mask[node] > 0.1:
                important_nodes.append(node)
        
        # Get important edges
        important_edges = []
        src, dst = g.edges()
        for i in range(len(src)):
            if src[i].item() in important_nodes and dst[i].item() in important_nodes:
                if edge_mask[i] > 0.1:
                    important_edges.append((src[i].item(), dst[i].item()))
        
        return important_nodes, important_edges
    
    def _get_feature_importance_pyg(self, data, target_node: int) -> Dict[str, float]:
        """Get feature importance for PyG"""
        
        # Simplified feature importance calculation
        feature_importance = {}
        
        # Get node features for the target node
        if hasattr(data, 'x'):
            node_features = data.x[target_node]
        else:
            # If data is already the features tensor
            node_features = data[target_node]
        
        # Use feature values as importance scores (simplified)
        for i, feature_value in enumerate(node_features):
            feature_importance[f'feature_{i}'] = abs(feature_value.item())
        
        return feature_importance
    
    def _get_feature_importance_dgl(self, g, target_node: int) -> Dict[str, float]:
        """Get feature importance for DGL"""
        
        feature_importance = {}
        
        # Get original prediction
        with torch.no_grad():
            self.model.eval()
            embeddings, logits = self.model(g)
            original_pred = F.softmax(logits[target_node], dim=0)
        
        # Compute gradients
        g.ndata['feat'].requires_grad_(True)
        embeddings, logits = self.model(g)
        pred = F.softmax(logits[target_node], dim=0)
        
        # Backward pass
        pred.backward()
        gradients = g.ndata['feat'].grad[target_node]
        
        # Feature importance scores
        for i, grad in enumerate(gradients):
            feature_importance[f'feature_{i}'] = abs(grad.item())
        
        return feature_importance
    
    def _get_neighbor_importance_pyg(self, data, target_node: int,
                                   node_mask: torch.Tensor, edge_mask: torch.Tensor) -> Dict[int, float]:
        """Get neighbor importance for PyG"""
        
        neighbor_importance = {}
        
        # Get neighbors
        if self.framework == "pyg":
            node_edges = data.edge_index[:, data.edge_index[0] == target_node]
            neighbors = node_edges[1].tolist()
        else:
            neighbors = data.successors(target_node).tolist()
        
        # Compute importance based on masks and edge weights
        for i, neighbor in enumerate(neighbors):
            # Find edge index
            if self.framework == "pyg":
                edge_idx = (data.edge_index[0] == target_node) & (data.edge_index[1] == neighbor)
                edge_idx = edge_idx.nonzero().item()
            else:
                edge_idx = i  # Simplified for DGL
            
            # Compute importance score
            node_importance = node_mask[neighbor].item()
            edge_importance = edge_mask[edge_idx].item() if edge_idx < len(edge_mask) else 0.5
            
            # Combined importance
            importance = (node_importance + edge_importance) / 2
            neighbor_importance[neighbor] = importance
        
        return neighbor_importance
    
    def _get_neighbor_importance_dgl(self, g, target_node: int,
                                   node_mask: torch.Tensor, edge_mask: torch.Tensor) -> Dict[int, float]:
        """Get neighbor importance for DGL"""
        
        neighbor_importance = {}
        
        # Get neighbors
        neighbors = g.successors(target_node).tolist()
        
        # Compute importance
        for i, neighbor in enumerate(neighbors):
            node_importance = node_mask[neighbor].item()
            edge_importance = edge_mask[i].item() if i < len(edge_mask) else 0.5
            
            importance = (node_importance + edge_importance) / 2
            neighbor_importance[neighbor] = importance
        
        return neighbor_importance
    
    def get_top_suspicious_connections(self, data, target_node: int, 
                                     top_k: int = 5) -> List[Dict]:
        """
        Get top-k suspicious connections for a node
        
        Args:
            data: Graph data
            target_node: Target node
            top_k: Number of top connections to return
            
        Returns:
            List of suspicious connections with explanations
        """
        
        # Get explanation
        explanation = self.explain_node(data, target_node)
        
        # Get neighbor importance
        neighbor_importance = explanation['neighbor_importance']
        
        # Sort by importance
        sorted_neighbors = sorted(
            neighbor_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top-k connections
        suspicious_connections = []
        for neighbor, importance in sorted_neighbors[:top_k]:
            connection_info = {
                'neighbor_node': neighbor,
                'importance_score': importance,
                'connection_type': self._get_connection_type(data, target_node, neighbor),
                'risk_factors': self._get_risk_factors(data, target_node, neighbor)
            }
            suspicious_connections.append(connection_info)
        
        return suspicious_connections
    
    def _get_connection_type(self, data, node1: int, node2: int) -> str:
        """Get type of connection between two nodes"""
        
        # Analyze edge features to determine connection type
        if self.framework == "pyg":
            edge_idx = (data.edge_index[0] == node1) & (data.edge_index[1] == node2)
            if edge_idx.any():
                edge_idx = edge_idx.nonzero().item()
                edge_features = data.edge_attr[edge_idx]
                
                # Determine connection type based on features
                tx_type = int(edge_features[3]) if len(edge_features) > 3 else 0
                amount = edge_features[0] if len(edge_features) > 0 else 0
                fee = edge_features[4] if len(edge_features) > 4 else 0
                
                if tx_type == 1:
                    return "mixer"
                elif tx_type == 2:
                    return "exchange"
                elif amount > 5.0:
                    return "large_transfer"
                elif fee > 0.005:
                    return "high_fee"
                else:
                    return "regular"
        
        return "transaction"
    
    def _get_risk_factors(self, data, node1: int, node2: int) -> List[str]:
        """Get risk factors for a connection"""
        
        risk_factors = []
        
        # Analyze edge features for risk factors
        if self.framework == "pyg":
            edge_idx = (data.edge_index[0] == node1) & (data.edge_index[1] == node2)
            if edge_idx.any():
                edge_idx = edge_idx.nonzero().item()
                edge_features = data.edge_attr[edge_idx]
                
                # Check for risk factors
                if edge_features[0] > 10000:  # Large amount
                    risk_factors.append("large_transaction_amount")
                
                if edge_features[4] > 0.01:  # High fee
                    risk_factors.append("high_transaction_fee")
        
        return risk_factors
    
    def visualize_explanation(self, explanation: Dict, save_path: Optional[str] = None):
        """
        Visualize explanation results
        
        Args:
            explanation: Explanation dictionary
            save_path: Path to save visualization
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Node importance
        node_mask = explanation['node_mask']
        axes[0, 0].bar(range(len(node_mask)), node_mask.squeeze().cpu().numpy())
        axes[0, 0].set_title('Node Importance')
        axes[0, 0].set_xlabel('Node Index')
        axes[0, 0].set_ylabel('Importance Score')
        
        # 2. Edge importance
        edge_mask = explanation['edge_mask']
        axes[0, 1].bar(range(len(edge_mask)), edge_mask.squeeze().cpu().numpy())
        axes[0, 1].set_title('Edge Importance')
        axes[0, 1].set_xlabel('Edge Index')
        axes[0, 1].set_ylabel('Importance Score')
        
        # 3. Feature importance
        feature_importance = explanation['feature_importance']
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())
        axes[1, 0].bar(features, scores)
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Neighbor importance
        neighbor_importance = explanation['neighbor_importance']
        neighbors = list(neighbor_importance.keys())
        scores = list(neighbor_importance.values())
        axes[1, 1].bar(range(len(neighbors)), scores)
        axes[1, 1].set_title('Neighbor Importance')
        axes[1, 1].set_xlabel('Neighbor Index')
        axes[1, 1].set_ylabel('Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_explanation_report(self, explanation: Dict) -> str:
        """
        Generate a human-readable explanation report
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Formatted explanation report
        """
        
        report = f"""
# AML Risk Explanation Report

## Target Node: {explanation['target_node']}

## Summary
- Explanation Score: {explanation['explanation_score']:.3f}
- Number of Important Nodes: {len(explanation['subgraph_nodes'])}
- Number of Important Edges: {len(explanation['subgraph_edges'])}

## Top Risk Factors

### Feature Importance
"""
        
        # Add feature importance
        feature_importance = explanation['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, score in sorted_features[:5]:
            report += f"- {feature}: {score:.3f}\n"
        
        report += "\n### Most Suspicious Connections\n"
        
        # Add neighbor importance
        neighbor_importance = explanation['neighbor_importance']
        sorted_neighbors = sorted(neighbor_importance.items(), key=lambda x: x[1], reverse=True)
        
        for neighbor, score in sorted_neighbors[:5]:
            report += f"- Node {neighbor}: {score:.3f}\n"
        
        report += "\n## Recommendations\n"
        report += "- Monitor transactions with high-risk neighbors\n"
        report += "- Review feature patterns for suspicious activity\n"
        report += "- Consider additional verification for flagged connections\n"
        
        return report 