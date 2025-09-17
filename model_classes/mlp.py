import torch
import torch.nn as nn
from typing import List, Dict, Any


class MLP(nn.Module):
    """
    Modello Multi-Layer Perceptron per classificazione con regolarizzazione avanzata
    """
    
    def __init__(self, input_dim: int, num_classes: int, cfg: Dict[str, Any]):
        super().__init__()
        
        hidden_dims = cfg['model']['mlp']['hidden']
        dropout_rate = cfg['model']['mlp']['dropout']
        
        layers = []
        
        # Input layer con BatchNorm
        prev_dim = input_dim
        
        # Hidden layers con BatchNorm per migliore convergenza
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # BatchNorm per stabilità
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)