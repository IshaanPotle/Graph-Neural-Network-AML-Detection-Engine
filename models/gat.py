"""
Graph Attention Network (GAT) for AML Engine
Learned edge attention for fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

# Framework imports
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GATv2Conv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

from loguru import logger


class GAT(nn.Module):
    """
    Graph Attention Network for fraud detection
    Supports both PyG and DGL frameworks with learned attention
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 framework: str = "pyg",
                 use_edge_features: bool = True,
                 edge_dim: int = 6,
                 attention_dropout: float = 0.1):
        """
        Initialize GAT model
        
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            framework: "pyg" or "dgl"
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension
            attention_dropout: Attention dropout rate
        """
        super(GAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.framework = framework.lower()
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.attention_dropout = attention_dropout
        
        if self.framework == "pyg" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        elif self.framework == "dgl" and not DGL_AVAILABLE:
            raise ImportError("DGL not available")
        
        self._build_model()
        logger.info(f"Initialized GAT model with {framework} framework, {num_heads} heads")
    
    def _build_model(self):
        """Build the model architecture"""
        if self.framework == "pyg":
            self._build_pyg_model()
        else:
            self._build_dgl_model()
    
    def _build_pyg_model(self):
        """Build PyTorch Geometric GAT model"""
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(
            self.in_channels,
            self.hidden_channels // self.num_heads,
            heads=self.num_heads,
            dropout=self.attention_dropout,
            concat=True
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(GATv2Conv(
                self.hidden_channels,
                self.hidden_channels // self.num_heads,
                heads=self.num_heads,
                dropout=self.attention_dropout,
                concat=True
            ))
        
        # Output layer (single head for final prediction)
        self.convs.append(GATv2Conv(
            self.hidden_channels,
            self.out_channels,
            heads=1,
            dropout=self.attention_dropout,
            concat=False
        ))
        
        # Edge feature encoder
        if self.use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_dim, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels)
            )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 2)  # Binary classification
        )
    
    def _build_dgl_model(self):
        """Build DGL GAT model"""
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(dglnn.GATConv(
            self.in_channels,
            self.hidden_channels // self.num_heads,
            num_heads=self.num_heads,
            feat_drop=self.dropout,
            attn_drop=self.attention_dropout,
            residual=False,
            allow_zero_in_degree=True
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(dglnn.GATConv(
                self.hidden_channels,
                self.hidden_channels // self.num_heads,
                num_heads=self.num_heads,
                feat_drop=self.dropout,
                attn_drop=self.attention_dropout,
                residual=True,
                allow_zero_in_degree=True
            ))
        
        # Output layer
        self.convs.append(dglnn.GATConv(
            self.hidden_channels,
            self.out_channels,
            num_heads=1,
            feat_drop=self.dropout,
            attn_drop=self.attention_dropout,
            residual=False,
            allow_zero_in_degree=True
        ))
        
        # Edge feature encoder
        if self.use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_dim, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels)
            )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 2)  # Binary classification
        )
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: Graph data (PyG Data or DGL Graph)
            
        Returns:
            Node embeddings and classification logits
        """
        if self.framework == "pyg":
            return self._forward_pyg(data)
        else:
            return self._forward_dgl(data)
    
    def _forward_pyg(self, data):
        """Forward pass for PyG"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Process edge features if available
        if self.use_edge_features and edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
            # Note: PyG GAT doesn't directly support edge features
            # We'll use them in a custom way or ignore for now
        
        # GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Classification
        logits = self.classifier(x)
        
        return x, logits
    
    def _forward_dgl(self, g):
        """Forward pass for DGL"""
        h = g.ndata['feat']
        
        # Process edge features if available
        if self.use_edge_features and 'feat' in g.edata:
            edge_features = self.edge_encoder(g.edata['feat'])
            # Note: DGL GAT doesn't directly support edge features
            # We'll use them in a custom way or ignore for now
        
        # GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            h = h.view(h.size(0), -1)  # Flatten multi-head attention
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final layer
        h = self.convs[-1](g, h)
        h = h.squeeze(1)  # Remove head dimension
        
        # Classification
        logits = self.classifier(h)
        
        return h, logits
    
    def get_embeddings(self, data):
        """
        Get node embeddings without classification
        
        Args:
            data: Graph data
            
        Returns:
            Node embeddings
        """
        if self.framework == "pyg":
            x, edge_index = data.x, data.edge_index
            
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            return x
        else:
            g = data
            h = g.ndata['feat']
            
            for i, conv in enumerate(self.convs[:-1]):
                h = conv(g, h)
                h = h.view(h.size(0), -1)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            h = self.convs[-1](g, h)
            h = h.squeeze(1)
            return h
    
    def get_attention_weights(self, data):
        """
        Get attention weights for explainability
        
        Args:
            data: Graph data
            
        Returns:
            Attention weights dictionary
        """
        if self.framework == "pyg":
            return self._get_attention_weights_pyg(data)
        else:
            return self._get_attention_weights_dgl(data)
    
    def _get_attention_weights_pyg(self, data):
        """Get attention weights for PyG"""
        attention_weights = {}
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            # Get attention weights from the layer
            if hasattr(conv, 'get_attention_weights'):
                attn_weights = conv.get_attention_weights(x, edge_index)
                attention_weights[f'layer_{i}'] = attn_weights
            else:
                # For GATv2Conv, we need to manually compute attention
                # This is a simplified version
                attention_weights[f'layer_{i}'] = None
            
            # Forward pass
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return attention_weights
    
    def _get_attention_weights_dgl(self, g):
        """Get attention weights for DGL"""
        attention_weights = {}
        h = g.ndata['feat']
        
        for i, conv in enumerate(self.convs):
            # Get attention weights from the layer
            if hasattr(conv, 'get_attention_weights'):
                attn_weights = conv.get_attention_weights(g, h)
                attention_weights[f'layer_{i}'] = attn_weights
            else:
                # For DGL GAT, we need to manually compute attention
                attention_weights[f'layer_{i}'] = None
            
            # Forward pass
            h = conv(g, h)
            if i < len(self.convs) - 1:
                h = h.view(h.size(0), -1)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = h.squeeze(1)
        
        return attention_weights
    
    def predict(self, data, threshold: float = 0.5):
        """
        Make predictions
        
        Args:
            data: Graph data
            threshold: Classification threshold
            
        Returns:
            Predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            embeddings, logits = self.forward(data)
            probabilities = F.softmax(logits, dim=1)
            predictions = (probabilities[:, 1] > threshold).long()
            
        return predictions, probabilities


class GATTrainer:
    """Training wrapper for GAT model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.2, 0.8]).to(device)  # Higher weight for fraud class
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
        
        logger.info(f"Initialized GAT trainer on device: {device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move to device
            if self.model.framework == "pyg":
                batch = batch.to(self.device)
                labels = batch.y
            else:
                batch = batch.to(self.device)
                labels = batch.ndata['label']
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings, logits = self.model(batch)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                if self.model.framework == "pyg":
                    batch = batch.to(self.device)
                    labels = batch.y
                else:
                    batch = batch.to(self.device)
                    labels = batch.ndata['label']
                
                # Forward pass
                embeddings, logits = self.model(batch)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_predictions, all_labels 