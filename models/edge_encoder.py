"""
Edge Encoder for AML Engine
Learned edge embeddings for money flow patterns and transaction characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from loguru import logger


class EdgeEncoder(nn.Module):
    """
    Edge Encoder for learning edge embeddings
    Captures money flow patterns, transaction frequency, and amount volatility
    """
    
    def __init__(self,
                 edge_dim: int = 6,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        """
        Initialize Edge Encoder
        
        Args:
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(EdgeEncoder, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        self._build_model()
        logger.info(f"Initialized EdgeEncoder: {edge_dim} -> {output_dim}")
    
    def _build_model(self):
        """Build the edge encoder architecture"""
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        
        # First layer
        self.feature_layers.append(nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.BatchNorm1d(self.hidden_dim)
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.feature_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(self.hidden_dim)
            ))
        
        # Output layer
        self.feature_layers.append(nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ))
        
        # Attention mechanism for edge importance
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.output_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Edge type specific encoders
        self.tx_type_encoder = nn.Embedding(10, self.hidden_dim // 4)  # Transaction types
        self.direction_encoder = nn.Embedding(2, self.hidden_dim // 4)  # In/Out
        
        # Amount volatility encoder
        self.amount_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
        
        # Time-based encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            edge_features: Edge features [num_edges, edge_dim]
            
        Returns:
            Edge embeddings [num_edges, output_dim]
        """
        # Extract different feature types
        amount = edge_features[:, 0:1]  # Amount
        timestamp = edge_features[:, 1:2]  # Timestamp
        direction = edge_features[:, 2:3].long()  # Direction (0/1)
        tx_type = edge_features[:, 3:4].long()  # Transaction type
        fee = edge_features[:, 4:5]  # Fee
        block_height = edge_features[:, 5:6]  # Block height
        
        # Encode different feature types
        tx_type_emb = self.tx_type_encoder(tx_type.squeeze(-1))
        direction_emb = self.direction_encoder(direction.squeeze(-1))
        amount_emb = self.amount_encoder(amount)
        time_emb = self.time_encoder(timestamp)
        
        # Combine all embeddings
        combined_features = torch.cat([
            tx_type_emb,
            direction_emb,
            amount_emb,
            time_emb,
            fee,
            block_height
        ], dim=1)
        
        # Pass through feature extraction layers
        x = combined_features
        for layer in self.feature_layers:
            x = layer(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        return x
    
    def get_edge_importance(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Get edge importance scores
        
        Args:
            edge_features: Edge features
            
        Returns:
            Importance scores [num_edges, 1]
        """
        if not self.use_attention:
            return torch.ones(edge_features.size(0), 1, device=edge_features.device)
        
        # Extract features and get embeddings
        amount = edge_features[:, 0:1]
        timestamp = edge_features[:, 1:2]
        direction = edge_features[:, 2:3].long()
        tx_type = edge_features[:, 3:4].long()
        fee = edge_features[:, 4:5]
        block_height = edge_features[:, 5:6]
        
        # Encode features
        tx_type_emb = self.tx_type_encoder(tx_type.squeeze(-1))
        direction_emb = self.direction_encoder(direction.squeeze(-1))
        amount_emb = self.amount_encoder(amount)
        time_emb = self.time_encoder(timestamp)
        
        # Combine features
        combined_features = torch.cat([
            tx_type_emb,
            direction_emb,
            amount_emb,
            time_emb,
            fee,
            block_height
        ], dim=1)
        
        # Pass through feature layers
        x = combined_features
        for layer in self.feature_layers:
            x = layer(x)
        
        # Get attention weights
        importance = self.attention(x)
        
        return importance


class TemporalEdgeEncoder(nn.Module):
    """
    Temporal Edge Encoder for dynamic edge features
    Captures evolving transaction patterns over time
    """
    
    def __init__(self,
                 edge_dim: int = 6,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 temporal_window: int = 24,
                 dropout: float = 0.2):
        """
        Initialize Temporal Edge Encoder
        
        Args:
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            temporal_window: Time window for temporal aggregation
            dropout: Dropout rate
        """
        super(TemporalEdgeEncoder, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temporal_window = temporal_window
        self.dropout = dropout
        
        self._build_model()
        logger.info(f"Initialized TemporalEdgeEncoder with {temporal_window}h window")
    
    def _build_model(self):
        """Build the temporal edge encoder architecture"""
        
        # Base edge encoder
        self.base_encoder = EdgeEncoder(
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        )
        
        # Temporal aggregation
        self.temporal_aggregator = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, edge_features: torch.Tensor, 
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal information
        
        Args:
            edge_features: Edge features [num_edges, edge_dim]
            timestamps: Timestamps [num_edges]
            
        Returns:
            Temporal edge embeddings [num_edges, output_dim]
        """
        # Get base embeddings
        base_embeddings = self.base_encoder(edge_features)
        
        # Sort by timestamp
        sorted_indices = torch.argsort(timestamps)
        sorted_embeddings = base_embeddings[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]
        
        # Create temporal windows
        temporal_embeddings = []
        for i in range(len(sorted_embeddings)):
            # Get edges within temporal window
            window_start = sorted_timestamps[i] - self.temporal_window * 3600  # hours to seconds
            window_mask = sorted_timestamps >= window_start
            
            if window_mask.sum() > 0:
                window_embeddings = sorted_embeddings[window_mask]
                
                # Pad or truncate to fixed size
                if len(window_embeddings) > self.temporal_window:
                    window_embeddings = window_embeddings[-self.temporal_window:]
                else:
                    padding = torch.zeros(
                        self.temporal_window - len(window_embeddings),
                        self.output_dim,
                        device=window_embeddings.device
                    )
                    window_embeddings = torch.cat([padding, window_embeddings], dim=0)
                
                temporal_embeddings.append(window_embeddings.unsqueeze(0))
            else:
                # No edges in window, use zero embedding
                zero_embedding = torch.zeros(
                    1, self.temporal_window, self.output_dim,
                    device=base_embeddings.device
                )
                temporal_embeddings.append(zero_embedding)
        
        # Stack temporal embeddings
        temporal_embeddings = torch.cat(temporal_embeddings, dim=0)
        
        # Process with LSTM
        lstm_out, _ = self.temporal_aggregator(temporal_embeddings)
        
        # Apply temporal attention
        attention_weights = self.temporal_attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Project to output dimension
        output_embeddings = self.output_projection(attended_output)
        
        # Restore original order
        original_indices = torch.argsort(sorted_indices)
        output_embeddings = output_embeddings[original_indices]
        
        return output_embeddings


class EdgePatternEncoder(nn.Module):
    """
    Edge Pattern Encoder for detecting suspicious transaction patterns
    """
    
    def __init__(self,
                 edge_dim: int = 6,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 pattern_types: List[str] = None):
        """
        Initialize Edge Pattern Encoder
        
        Args:
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            pattern_types: Types of patterns to detect
        """
        super(EdgePatternEncoder, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pattern_types = pattern_types or [
            'high_frequency', 'large_amount', 'unusual_timing',
            'mixer_connection', 'exchange_pattern'
        ]
        
        self._build_model()
        logger.info(f"Initialized EdgePatternEncoder for {len(self.pattern_types)} patterns")
    
    def _build_model(self):
        """Build the pattern encoder architecture"""
        
        # Pattern-specific encoders
        self.pattern_encoders = nn.ModuleDict()
        
        for pattern in self.pattern_types:
            self.pattern_encoders[pattern] = nn.Sequential(
                nn.Linear(self.edge_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Pattern combination layer
        self.pattern_combiner = nn.Sequential(
            nn.Linear(len(self.pattern_types), self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Base edge encoder
        self.base_encoder = EdgeEncoder(
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=0.2
        )
    
    def detect_patterns(self, edge_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect suspicious patterns in edge features
        
        Args:
            edge_features: Edge features [num_edges, edge_dim]
            
        Returns:
            Dictionary of pattern scores
        """
        patterns = {}
        
        for pattern_name, encoder in self.pattern_encoders.items():
            if pattern_name == 'high_frequency':
                # Detect high frequency transactions
                patterns[pattern_name] = encoder(edge_features)
            
            elif pattern_name == 'large_amount':
                # Detect large amount transactions
                patterns[pattern_name] = encoder(edge_features)
            
            elif pattern_name == 'unusual_timing':
                # Detect unusual timing patterns
                patterns[pattern_name] = encoder(edge_features)
            
            elif pattern_name == 'mixer_connection':
                # Detect mixer connections
                patterns[pattern_name] = encoder(edge_features)
            
            elif pattern_name == 'exchange_pattern':
                # Detect exchange patterns
                patterns[pattern_name] = encoder(edge_features)
        
        return patterns
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            edge_features: Edge features [num_edges, edge_dim]
            
        Returns:
            Pattern-aware edge embeddings [num_edges, output_dim]
        """
        # Detect patterns
        patterns = self.detect_patterns(edge_features)
        
        # Combine pattern scores
        pattern_scores = torch.cat(list(patterns.values()), dim=1)
        
        # Get pattern embeddings
        pattern_embeddings = self.pattern_combiner(pattern_scores)
        
        # Get base embeddings
        base_embeddings = self.base_encoder(edge_features)
        
        # Combine base and pattern embeddings
        combined_embeddings = base_embeddings + pattern_embeddings
        
        return combined_embeddings
    
    def get_pattern_scores(self, edge_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get individual pattern scores for explainability
        
        Args:
            edge_features: Edge features
            
        Returns:
            Dictionary of pattern scores
        """
        return self.detect_patterns(edge_features) 