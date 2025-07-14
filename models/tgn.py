"""
Temporal Graph Network (TGN) for AML Engine
Dynamic fraud behavior detection with temporal snapshots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

# Framework imports
try:
    import torch_geometric
    from torch_geometric.nn import GATConv
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


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network for dynamic fraud detection
    Handles temporal snapshots and evolving graph structures
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
                 temporal_window: int = 24,
                 memory_size: int = 1000):
        """
        Initialize TGN model
        
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            framework: "pyg" or "dgl"
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension
            temporal_window: Time window for temporal aggregation
            memory_size: Size of node memory buffer
        """
        super(TemporalGraphNetwork, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.framework = framework.lower()
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.temporal_window = temporal_window
        self.memory_size = memory_size
        
        if self.framework == "pyg" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        elif self.framework == "dgl" and not DGL_AVAILABLE:
            raise ImportError("DGL not available")
        
        self._build_model()
        logger.info(f"Initialized TGN model with {framework} framework")
    
    def _build_model(self):
        """Build the model architecture"""
        if self.framework == "pyg":
            self._build_pyg_model()
        else:
            self._build_dgl_model()
    
    def _build_pyg_model(self):
        """Build PyTorch Geometric TGN model"""
        # Node memory for temporal information
        self.node_memory = nn.Parameter(
            torch.randn(self.memory_size, self.hidden_channels)
        )
        self.memory_time = nn.Parameter(
            torch.zeros(self.memory_size, 1)
        )
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.hidden_channels + 1, self.hidden_channels),  # +1 for time
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # GNN layers for temporal snapshots
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            self.in_channels,
            self.hidden_channels // self.num_heads,
            heads=self.num_heads,
            dropout=self.dropout,
            concat=True
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(GATConv(
                self.hidden_channels,
                self.hidden_channels // self.num_heads,
                heads=self.num_heads,
                dropout=self.dropout,
                concat=True
            ))
        
        # Output layer
        self.convs.append(GATConv(
            self.hidden_channels,
            self.out_channels,
            heads=1,
            dropout=self.dropout,
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
        """Build DGL TGN model"""
        # Node memory for temporal information
        self.node_memory = nn.Parameter(
            torch.randn(self.memory_size, self.hidden_channels)
        )
        self.memory_time = nn.Parameter(
            torch.zeros(self.memory_size, 1)
        )
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.hidden_channels + 1, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # GNN layers for temporal snapshots
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(dglnn.GATConv(
            self.in_channels,
            self.hidden_channels // self.num_heads,
            num_heads=self.num_heads,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
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
                attn_drop=self.dropout,
                residual=True,
                allow_zero_in_degree=True
            ))
        
        # Output layer
        self.convs.append(dglnn.GATConv(
            self.hidden_channels,
            self.out_channels,
            num_heads=1,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
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
    
    def update_node_memory(self, node_ids: torch.Tensor, 
                          embeddings: torch.Tensor, 
                          timestamps: torch.Tensor):
        """
        Update node memory with new temporal information
        
        Args:
            node_ids: Node indices
            embeddings: Node embeddings
            timestamps: Timestamps for the updates
        """
        # Update memory for the given nodes
        for i, node_id in enumerate(node_ids):
            if node_id < self.memory_size:
                # Combine embedding with temporal information
                temporal_input = torch.cat([
                    embeddings[i].unsqueeze(0),
                    timestamps[i].unsqueeze(0)
                ], dim=1)
                
                # Update memory
                self.node_memory.data[node_id] = self.temporal_encoder(temporal_input).squeeze(0)
                self.memory_time.data[node_id] = timestamps[i]
    
    def get_temporal_embeddings(self, node_ids: torch.Tensor, 
                               current_time: float) -> torch.Tensor:
        """
        Get temporal embeddings for nodes
        
        Args:
            node_ids: Node indices
            current_time: Current timestamp
            
        Returns:
            Temporal embeddings
        """
        temporal_embeddings = []
        
        for node_id in node_ids:
            if node_id < self.memory_size:
                # Get memory and time difference
                memory = self.node_memory[node_id]
                time_diff = current_time - self.memory_time[node_id]
                
                # Combine with temporal information
                temporal_input = torch.cat([
                    memory.unsqueeze(0),
                    time_diff.unsqueeze(0)
                ], dim=1)
                
                temporal_embedding = self.temporal_encoder(temporal_input)
                temporal_embeddings.append(temporal_embedding)
            else:
                # For new nodes, use zero embedding
                temporal_embeddings.append(
                    torch.zeros(1, self.hidden_channels, device=node_ids.device)
                )
        
        return torch.cat(temporal_embeddings, dim=0)
    
    def forward(self, data, timestamps: Optional[torch.Tensor] = None):
        """
        Forward pass with temporal information
        
        Args:
            data: Graph data (PyG Data or DGL Graph)
            timestamps: Timestamps for nodes
            
        Returns:
            Node embeddings and classification logits
        """
        if self.framework == "pyg":
            return self._forward_pyg(data, timestamps)
        else:
            return self._forward_dgl(data, timestamps)
    
    def _forward_pyg(self, data, timestamps):
        """Forward pass for PyG with temporal information"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Get temporal embeddings if timestamps provided
        if timestamps is not None:
            node_ids = torch.arange(x.size(0), device=x.device)
            temporal_embeddings = self.get_temporal_embeddings(node_ids, timestamps.mean())
            # Combine with node features
            x = torch.cat([x, temporal_embeddings], dim=1)
        
        # Process edge features if available
        if self.use_edge_features and edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
        
        # GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Update memory if timestamps provided
        if timestamps is not None:
            node_ids = torch.arange(x.size(0), device=x.device)
            self.update_node_memory(node_ids, x, timestamps)
        
        # Classification
        logits = self.classifier(x)
        
        return x, logits
    
    def _forward_dgl(self, g, timestamps):
        """Forward pass for DGL with temporal information"""
        h = g.ndata['feat']
        
        # Get temporal embeddings if timestamps provided
        if timestamps is not None:
            node_ids = torch.arange(h.size(0), device=h.device)
            temporal_embeddings = self.get_temporal_embeddings(node_ids, timestamps.mean())
            # Combine with node features
            h = torch.cat([h, temporal_embeddings], dim=1)
        
        # Process edge features if available
        if self.use_edge_features and 'feat' in g.edata:
            edge_features = self.edge_encoder(g.edata['feat'])
        
        # GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            h = h.view(h.size(0), -1)  # Flatten multi-head attention
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final layer
        h = self.convs[-1](g, h)
        h = h.squeeze(1)  # Remove head dimension
        
        # Update memory if timestamps provided
        if timestamps is not None:
            node_ids = torch.arange(h.size(0), device=h.device)
            self.update_node_memory(node_ids, h, timestamps)
        
        # Classification
        logits = self.classifier(h)
        
        return h, logits
    
    def get_embeddings(self, data, timestamps: Optional[torch.Tensor] = None):
        """
        Get node embeddings without classification
        
        Args:
            data: Graph data
            timestamps: Timestamps for nodes
            
        Returns:
            Node embeddings
        """
        if self.framework == "pyg":
            x, edge_index = data.x, data.edge_index
            
            # Get temporal embeddings if timestamps provided
            if timestamps is not None:
                node_ids = torch.arange(x.size(0), device=x.device)
                temporal_embeddings = self.get_temporal_embeddings(node_ids, timestamps.mean())
                x = torch.cat([x, temporal_embeddings], dim=1)
            
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            return x
        else:
            g = data
            h = g.ndata['feat']
            
            # Get temporal embeddings if timestamps provided
            if timestamps is not None:
                node_ids = torch.arange(h.size(0), device=h.device)
                temporal_embeddings = self.get_temporal_embeddings(node_ids, timestamps.mean())
                h = torch.cat([h, temporal_embeddings], dim=1)
            
            for i, conv in enumerate(self.convs[:-1]):
                h = conv(g, h)
                h = h.view(h.size(0), -1)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            h = self.convs[-1](g, h)
            h = h.squeeze(1)
            return h
    
    def predict(self, data, timestamps: Optional[torch.Tensor] = None, 
               threshold: float = 0.5):
        """
        Make predictions
        
        Args:
            data: Graph data
            timestamps: Timestamps for nodes
            threshold: Classification threshold
            
        Returns:
            Predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            embeddings, logits = self.forward(data, timestamps)
            probabilities = F.softmax(logits, dim=1)
            predictions = (probabilities[:, 1] > threshold).long()
            
        return predictions, probabilities


class TGNTrainer:
    """Training wrapper for TGN model"""
    
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
        
        logger.info(f"Initialized TGN trainer on device: {device}")
    
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
                timestamps = getattr(batch, 'timestamps', None)
            else:
                batch = batch.to(self.device)
                labels = batch.ndata['label']
                timestamps = batch.ndata.get('timestamps', None)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings, logits = self.model(batch, timestamps)
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
                    timestamps = getattr(batch, 'timestamps', None)
                else:
                    batch = batch.to(self.device)
                    labels = batch.ndata['label']
                    timestamps = batch.ndata.get('timestamps', None)
                
                # Forward pass
                embeddings, logits = self.model(batch, timestamps)
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