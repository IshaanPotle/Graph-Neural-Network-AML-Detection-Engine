"""
GraphSAGE Model for AML Engine
Inductive learning on transaction graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

# Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
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


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for fraud detection
    Supports both PyG and DGL frameworks
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 framework: str = "pyg",
                 use_edge_features: bool = True,
                 edge_dim: int = 6):
        """
        Initialize GraphSAGE model
        
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (2 for binary classification)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            framework: "pyg" or "dgl"
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension
        """
        super(GraphSAGE, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.framework = framework.lower()
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        
        if self.framework == "pyg" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        elif self.framework == "dgl" and not DGL_AVAILABLE:
            raise ImportError("DGL not available")
        
        self._build_model()
        logger.info(f"Initialized GraphSAGE model with {framework} framework")
    
    def _build_model(self):
        """Build the model architecture"""
        if self.framework == "pyg":
            self._build_pyg_model()
        else:
            self._build_dgl_model()
    
    def _build_pyg_model(self):
        """Build PyTorch Geometric GraphSAGE model"""
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(
            self.in_channels, 
            self.hidden_channels,
            aggr='mean'
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(
                self.hidden_channels,
                self.hidden_channels,
                aggr='mean'
            ))
        
        # Output layer
        self.convs.append(SAGEConv(
            self.hidden_channels,
            self.out_channels,
            aggr='mean'
        ))
        
        # Edge feature encoder (if used)
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
        """Build DGL GraphSAGE model"""
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(dglnn.SAGEConv(
            self.in_channels,
            self.hidden_channels,
            aggregator_type='mean'
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(dglnn.SAGEConv(
                self.hidden_channels,
                self.hidden_channels,
                aggregator_type='mean'
            ))
        
        # Output layer
        self.convs.append(dglnn.SAGEConv(
            self.hidden_channels,
            self.out_channels,
            aggregator_type='mean'
        ))
        
        # Edge feature encoder (if used)
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
    
    def forward(self, data, edge_index=None):
        """
        Forward pass
        
        Args:
            data: Graph data (PyG Data or DGL Graph) or node features
            edge_index: Edge index (optional, for GNNExplainer compatibility)
            
        Returns:
            Node embeddings and classification logits
        """
        if self.framework == "pyg":
            return self._forward_pyg(data, edge_index)
        else:
            return self._forward_dgl(data)
    
    def _forward_pyg(self, data, edge_index=None):
        """Forward pass for PyG"""
        # Handle both data object and separate parameters
        if hasattr(data, 'x'):
            # data is a PyG Data object
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            # data is node features, edge_index is separate
            x, edge_attr = data, None
        
        # Process edge features if available
        if self.use_edge_features and edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
            # Note: PyG SAGEConv doesn't directly support edge features
            # We'll use them in a custom way or ignore for now
        
        # GraphSAGE layers
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
            # Note: DGL SAGEConv doesn't directly support edge features
            # We'll use them in a custom way or ignore for now
        
        # GraphSAGE layers
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final layer
        h = self.convs[-1](g, h)
        
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
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            h = self.convs[-1](g, h)
            return h
    
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
    
    def get_attention_weights(self, data):
        """
        Get attention weights (for compatibility with attention-based models)
        Returns None for GraphSAGE as it doesn't use attention
        """
        return None


class GraphSAGETrainer:
    """Training wrapper for GraphSAGE model"""
    
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
        
        logger.info(f"Initialized GraphSAGE trainer on device: {device}")
    
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