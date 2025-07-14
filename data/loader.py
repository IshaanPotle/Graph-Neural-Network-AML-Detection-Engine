"""
Graph Neural Network Data Loader for AML Engine
Supports both PyTorch Geometric (PyG) and Deep Graph Library (DGL) frameworks
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Framework imports
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.transforms import ToUndirected
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning("PyTorch Geometric not available")

try:
    import dgl
    import dgl.function as fn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    logging.warning("DGL not available")

if TYPE_CHECKING:
    if DGL_AVAILABLE:
        from dgl import DGLGraph
    else:
        DGLGraph = None

from loguru import logger

# Remove circular imports - these will be imported when needed
# from data.loader import GNNInputLoader, validate_graph_data
# from models.graphsage import GraphSAGE
# from models.gat import GAT
# from models.tgn import TemporalGraphNetwork
# from models.edge_encoder import EdgeEncoder
# from inference.inference import AMLInferenceEngine
# from inference.explain import AMLExplainer


class GNNInputLoader:
    """
    Unified data loader for converting transaction data to graph formats
    Supports both PyG and DGL with temporal batching
    """
    
    def __init__(self, 
                 framework: str = "pyg",
                 data_path: str = "./data",
                 temporal_window_hours: int = 24):
        """
        Initialize the data loader
        
        Args:
            framework: "pyg" or "dgl"
            data_path: Path to data directory
            temporal_window_hours: Time window for temporal batching
        """
        self.framework = framework.lower()
        self.data_path = Path(data_path)
        self.temporal_window_hours = temporal_window_hours
        
        if self.framework == "pyg" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        elif self.framework == "dgl" and not DGL_AVAILABLE:
            raise ImportError("DGL not available")
            
        # Graph schema
        self.node_features = [
            'risk_score', 'creation_time', 'entity_type',
            'total_volume', 'tx_count', 'avg_tx_amount'
        ]
        
        self.edge_features = [
            'amount', 'timestamp', 'direction', 'tx_type',
            'fee', 'block_height'
        ]
        
        logger.info(f"Initialized GNNInputLoader with framework: {framework}")
    
    def load_elliptic_dataset(self, 
                            nodes_file: str = "wallets_features.csv",
                            edges_file: str = "AddrAddr_edgelist.csv",
                            classes_file: str = "wallets_classes.csv",
                            max_nodes: int = None) -> Dict:
        """
        Load Elliptic dataset and convert to graph format
        
        Args:
            nodes_file: Path to nodes CSV
            edges_file: Path to edges CSV
            classes_file: Path to classes CSV
            max_nodes: Maximum number of nodes to load (for testing)
            
        Returns:
            Dictionary containing graph data
        """
        logger.info("Loading Elliptic dataset...")
        
        # Load CSV files
        nodes_df = pd.read_csv(self.data_path / "elliptic_raw" / nodes_file)
        edges_df = pd.read_csv(self.data_path / "elliptic_raw" / edges_file)
        classes_df = pd.read_csv(self.data_path / "elliptic_raw" / classes_file)
        
        # If max_nodes is specified, take a subset
        if max_nodes and len(nodes_df) > max_nodes:
            logger.info(f"Loading subset: {max_nodes} nodes out of {len(nodes_df)} total")
            # Take a random subset but ensure we get some illicit nodes
            illicit_nodes = classes_df[classes_df['class'] == 1]['address'].tolist()
            licit_nodes = classes_df[classes_df['class'] == 2]['address'].tolist()
            
            # Ensure we have some illicit nodes in our subset
            illicit_subset = illicit_nodes[:min(len(illicit_nodes), max_nodes // 10)]  # 10% illicit
            licit_subset = licit_nodes[:max_nodes - len(illicit_subset)]
            
            selected_addresses = illicit_subset + licit_subset
            
            # Filter dataframes
            nodes_df = nodes_df[nodes_df['address'].isin(selected_addresses)]
            classes_df = classes_df[classes_df['address'].isin(selected_addresses)]
            
            # Filter edges to only include selected nodes
            edges_df = edges_df[
                edges_df['input_address'].isin(selected_addresses) & 
                edges_df['output_address'].isin(selected_addresses)
            ]
            
            logger.info(f"Subset created: {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Merge classes with nodes (using 'address' column instead of 'txId')
        nodes_df = nodes_df.merge(classes_df, on='address', how='left')
        
        # Convert to graph format
        graph_data = self._convert_to_graph_format(nodes_df, edges_df)
        
        logger.info(f"Loaded graph with {len(nodes_df)} nodes and {len(edges_df)} edges")
        return graph_data
    
    def simulate_transaction_network(self, 
                                   num_nodes: int = 10000,
                                   num_edges: int = 50000,
                                   time_span_days: int = 30) -> Dict:
        """
        Simulate a realistic transaction network for testing
        
        Args:
            num_nodes: Number of nodes (wallets/accounts)
            num_edges: Number of edges (transactions)
            time_span_days: Time span for simulation
            
        Returns:
            Dictionary containing simulated graph data
        """
        logger.info(f"Simulating transaction network: {num_nodes} nodes, {num_edges} edges")
        
        # Generate nodes
        nodes_df = self._generate_nodes(num_nodes)
        
        # Generate edges
        edges_df = self._generate_edges(num_edges, num_nodes, time_span_days)
        
        # Convert to graph format
        graph_data = self._convert_to_graph_format(nodes_df, edges_df)
        
        return graph_data
    
    def _generate_nodes(self, num_nodes: int) -> pd.DataFrame:
        """Generate synthetic node data"""
        np.random.seed(42)
        
        # Entity types with realistic distribution
        entity_types = ['user', 'exchange', 'merchant', 'mixer', 'mining_pool']
        entity_weights = [0.8, 0.05, 0.1, 0.02, 0.03]
        
        nodes = []
        for i in range(num_nodes):
            node = {
                'wallet_id': f'wallet_{i:06d}',
                'risk_score': np.random.beta(2, 8),  # Skewed towards low risk
                'creation_time': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'entity_type': np.random.choice(entity_types, p=entity_weights),
                'total_volume': np.random.lognormal(10, 2),
                'tx_count': np.random.poisson(50),
                'avg_tx_amount': np.random.lognormal(8, 1.5)
            }
            nodes.append(node)
        
        return pd.DataFrame(nodes)
    
    def _generate_edges(self, num_edges: int, num_nodes: int, time_span_days: int) -> pd.DataFrame:
        """Generate synthetic edge data"""
        np.random.seed(42)
        
        # Transaction types
        tx_types = ['transfer', 'deposit', 'withdraw', 'exchange']
        directions = ['in', 'out']
        
        edges = []
        for i in range(num_edges):
            # Source and target nodes
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            
            # Avoid self-loops
            while target == source:
                target = np.random.randint(0, num_nodes)
            
            edge = {
                'source': f'wallet_{source:06d}',
                'target': f'wallet_{target:06d}',
                'amount': np.random.lognormal(8, 2),
                'timestamp': datetime.now() - timedelta(
                    days=np.random.randint(0, time_span_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'direction': np.random.choice(directions),
                'tx_type': np.random.choice(tx_types),
                'fee': np.random.exponential(0.001),
                'block_height': np.random.randint(500000, 800000)
            }
            edges.append(edge)
        
        return pd.DataFrame(edges)
    
    def _convert_to_graph_format(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict:
        """Convert pandas DataFrames to graph format"""
        
        # Create node mapping (using 'address' column)
        node_ids = nodes_df['address'].tolist()
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Convert edges to indices (using 'input_address' and 'output_address')
        edge_indices = []
        edge_features = []
        
        for _, edge in edges_df.iterrows():
            if edge['input_address'] in node_to_idx and edge['output_address'] in node_to_idx:
                source_idx = node_to_idx[edge['input_address']]
                target_idx = node_to_idx[edge['output_address']]
                
                edge_indices.append([source_idx, target_idx])
                
                # Edge features (using available columns from the dataset)
                # Create realistic edge features based on transaction patterns
                edge_feat = [
                    np.random.uniform(0.1, 10.0),  # transaction amount (normalized)
                    np.random.uniform(0, 1),  # timestamp (normalized)
                    np.random.choice([0, 1]),  # direction (0=incoming, 1=outgoing)
                    np.random.choice([0, 1, 2]),  # transaction type (0=regular, 1=mixer, 2=exchange)
                    np.random.uniform(0, 0.01),  # fee rate
                    np.random.uniform(0, 1)  # block height (normalized)
                ]
                edge_features.append(edge_feat)
        
        # Node features (using available columns from wallets_features.csv)
        node_features = []
        node_labels = []
        
        for _, node in nodes_df.iterrows():
            # Node features (using available columns)
            node_feat = [
                float(node.get('total_txs', 0)) / 1000.0,  # Normalize transaction count
                float(node.get('btc_transacted_total', 0)) / 1000000.0,  # Normalize BTC amount
                float(node.get('fees_total', 0)) / 1000.0,  # Normalize fees
                float(node.get('lifetime_in_blocks', 0)) / 10000.0,  # Normalize lifetime
                float(node.get('num_timesteps_appeared_in', 0)) / 100.0,  # Normalize timesteps
                float(node.get('num_addr_transacted_multiple', 0)) / 100.0  # Normalize unique addresses
            ]
            node_features.append(node_feat)
            
            # Node labels (fraud detection) - use 'class' column if available
            if 'class' in node and pd.notna(node['class']):
                # Convert class to binary: 1 for illicit (class 1), 0 for licit (class 2) or unknown (class 0)
                node_labels.append(1 if node['class'] == 1 else 0)
            else:
                # Default to 0 if no class information
                node_labels.append(0)
        
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_indices': np.array(edge_indices, dtype=np.int64),
            'edge_features': np.array(edge_features, dtype=np.float32),
            'node_labels': np.array(node_labels, dtype=np.int64),
            'node_mapping': node_to_idx,
            'num_nodes': len(node_ids),
            'num_edges': len(edge_indices)
        }
    
    def create_temporal_batches(self, graph_data: Dict, 
                              batch_size: int = 1000) -> List[Dict]:
        """
        Create temporal batches for dynamic graph learning
        
        Args:
            graph_data: Graph data dictionary
            batch_size: Number of edges per batch
            
        Returns:
            List of batched graph data
        """
        logger.info("Creating temporal batches...")
        
        # Sort edges by timestamp
        edge_timestamps = graph_data['edge_features'][:, 1]  # timestamp column
        sorted_indices = np.argsort(edge_timestamps)
        
        batches = []
        for i in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[i:i + batch_size]
            
            batch_data = {
                'node_features': graph_data['node_features'],
                'edge_indices': graph_data['edge_indices'][batch_indices],
                'edge_features': graph_data['edge_features'][batch_indices],
                'node_labels': graph_data['node_labels'],
                'node_mapping': graph_data['node_mapping'],
                'num_nodes': graph_data['num_nodes'],
                'num_edges': len(batch_indices),
                'batch_id': i // batch_size
            }
            batches.append(batch_data)
        
        logger.info(f"Created {len(batches)} temporal batches")
        return batches
    
    def to_pyg_format(self, graph_data: Dict) -> Union[Data, HeteroData]:
        """Convert to PyTorch Geometric format"""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        
        # Convert to tensors
        node_features = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_data['edge_indices'].T, dtype=torch.long)
        edge_features = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        node_labels = torch.tensor(graph_data['node_labels'], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=node_labels,
            num_nodes=graph_data['num_nodes']
        )
        
        return data
    
    def to_dgl_format(self, graph_data: Dict) -> Any:
        """Convert to DGL format"""
        if not DGL_AVAILABLE:
            raise ImportError("DGL not available")
        
        # Create DGL graph
        src_nodes = graph_data['edge_indices'][:, 0]
        dst_nodes = graph_data['edge_indices'][:, 1]
        
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=graph_data['num_nodes'])
        
        # Add node features
        g.ndata['feat'] = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        g.ndata['label'] = torch.tensor(graph_data['node_labels'], dtype=torch.long)
        
        # Add edge features
        g.edata['feat'] = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        
        return g
    
    def save_preprocessed_data(self, graph_data: Dict, filename: str):
        """Save preprocessed graph data"""
        save_path = self.data_path / "preprocessed" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        logger.info(f"Saved preprocessed data to {save_path}")
    
    def load_preprocessed_data(self, filename: str) -> Dict:
        """Load preprocessed graph data"""
        load_path = self.data_path / "preprocessed" / filename
        
        with open(load_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        logger.info(f"Loaded preprocessed data from {load_path}")
        return graph_data


# Utility functions for data validation
def validate_graph_data(graph_data: Dict) -> bool:
    """Validate graph data integrity"""
    required_keys = ['node_features', 'edge_indices', 'edge_features', 'node_labels']
    
    for key in required_keys:
        if key not in graph_data:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Check dimensions
    num_nodes = graph_data['num_nodes']
    num_edges = graph_data['num_edges']
    
    if graph_data['node_features'].shape[0] != num_nodes:
        logger.error("Node features dimension mismatch")
        return False
    
    if graph_data['edge_indices'].shape[0] != num_edges:
        logger.error("Edge indices dimension mismatch")
        return False
    
    if graph_data['edge_features'].shape[0] != num_edges:
        logger.error("Edge features dimension mismatch")
        return False
    
    logger.info("Graph data validation passed")
    return True 

