"""
Graph Neural Network for Traffic Prediction
Models road network topology and spatial relationships
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for traffic prediction on road networks.
    """
    
    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        """
        Initialize GNN model.
        
        Args:
            node_features: Number of features per node (road segment)
            edge_features: Number of features per edge (road connection)
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_dim = output_dim
        
        # Graph Convolutional Layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level predictions
        
        Returns:
            Predictions
        """
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Graph-level pooling (if batch provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # Node-level predictions
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction
        x = self.fc(x)
        
        return x


class GNNTrafficPredictor:
    """
    Wrapper class for GNN traffic prediction.
    """
    
    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize GNN predictor.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def build_model(self, output_dim: int = 1):
        """Build GNN model."""
        self.model = GraphNeuralNetwork(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            output_dim=output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info(f"Built GNN model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def create_graph_data(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> Data:
        """
        Create PyTorch Geometric Data object.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_features]
            y: Target values
        
        Returns:
            Data object
        """
        x = torch.FloatTensor(node_features).to(self.device)
        edge_index_tensor = torch.LongTensor(edge_index).to(self.device)
        
        data = Data(x=x, edge_index=edge_index_tensor)
        
        if edge_attr is not None:
            data.edge_attr = torch.FloatTensor(edge_attr).to(self.device)
        
        if y is not None:
            data.y = torch.FloatTensor(y).to(self.device)
        
        return data
    
    def build_road_network_graph(
        self,
        coordinates: np.ndarray,
        k_nearest: int = 5,
        distance_threshold: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph from road network coordinates.
        
        Args:
            coordinates: Node coordinates [num_nodes, 2] (lat, lon)
            k_nearest: Number of nearest neighbors to connect
            distance_threshold: Maximum distance for edges (meters)
        
        Returns:
            Tuple of (edge_index, edge_attributes)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate distances
        nbrs = NearestNeighbors(n_neighbors=k_nearest + 1, metric='haversine').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Build edge list
        edges = []
        edge_attrs = []
        
        for i in range(len(coordinates)):
            for j, neighbor_idx in enumerate(indices[i][1:], 1):  # Skip self
                dist = distances[i][j] * 6371000  # Convert to meters
                if dist <= distance_threshold:
                    edges.append([i, neighbor_idx])
                    edge_attrs.append([dist, 1.0])  # Distance and connection type
        
        edge_index = np.array(edges).T if edges else np.array([[], []], dtype=int)
        edge_attr = np.array(edge_attrs) if edge_attrs else np.array([[]])
        
        return edge_index, edge_attr
    
    def train(
        self,
        data_list: List[Data],
        epochs: int = 200,
        batch_size: int = 16,
        verbose: bool = True
    ):
        """
        Train GNN model.
        
        Args:
            data_list: List of Data objects
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Print training progress
        """
        if self.model is None:
            # Infer output dimension from data
            output_dim = data_list[0].y.shape[-1] if hasattr(data_list[0], 'y') and data_list[0].y is not None else 1
            self.build_model(output_dim=output_dim)
        
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, data: Data) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Data object
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data.x, data.edge_index)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'node_features': self.node_features,
                'edge_features': self.edge_features,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout
            }
        }, filepath)
        logger.info(f"Saved GNN model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint['config']
        
        self.build_model(output_dim=1)  # Will be adjusted based on data
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded GNN model from {filepath}")

