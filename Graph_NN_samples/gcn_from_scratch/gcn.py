import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    A mathematically complete Graph Convolutional Network (GCN) layer implemented from scratch.
    Equation: H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} )
    """
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable weight parameter and bias
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix of shape (num_nodes, in_features)
        adj: Adjacency matrix of shape (num_nodes, num_nodes)
        """
        # 1. Add self-loops to adjacency matrix
        num_nodes = adj.size(0)
        adj_tilde = adj + torch.eye(num_nodes, device=adj.device)
        
        # 2. Compute degree matrix D_tilde and its inverse square root
        deg = torch.sum(adj_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # 3. Symmetric normalization: D^{-1/2} * A_tilde * D^{-1/2}
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        
        # 4. Linear projection: H * W
        support = torch.mm(x, self.weight)
        
        # 5. Graph propagation: norm_adj * support
        output = torch.mm(norm_adj, support)
        
        return output + self.bias


class DeepGCN(nn.Module):
    """
    An advanced, deeply stackable Graph Convolutional Network.
    Supports:
      - Variable depth (any number of GCN layers)
      - Residual shortcut connections to alleviate over-smoothing
      - Dropout pathways for regularization
      - Optional LayerNorm for training stability in deep stacks
    """
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, 
                 num_layers: int = 5, dropout: float = 0.2, use_residuals: bool = True,
                 use_layernorm: bool = True):
        super(DeepGCN, self).__init__()
        self.num_layers = num_layers
        self.use_residuals = use_residuals
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        
        # 1. Stack of GCN Layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
            
        # Output layer
        self.layers.append(GCNLayer(hidden_features, num_classes))
        
        # 2. LayerNorm stacks
        if self.use_layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_features) for _ in range(num_layers - 1)
            ])
            
        # 3. Dimension alignment projections for residual shortcuts (if input dims differ)
        if self.use_residuals:
            self.project_input = nn.Linear(in_features, hidden_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        
        # Process GCN layers (excluding final output layer)
        for i in range(self.num_layers - 1):
            h_prev = h
            
            # GCN transformation + activation
            h_gcn = F.relu(self.layers[i](h, adj))
            h_gcn = self.dropout(h_gcn)
            
            # Apply residual connection
            if self.use_residuals:
                if i == 0:
                    # Map input features to hidden dimension for the first residual shortcut
                    h = h_gcn + self.project_input(h_prev)
                else:
                    h = h_gcn + h_prev
            else:
                h = h_gcn
                
            # Apply LayerNorm
            if self.use_layernorm:
                h = self.norms[i](h)
                
        # Final linear prediction layer (no activation, no residual)
        out = self.layers[-1](h, adj)
        return out


# Keep classic GCN definition for backward compatibility
class GCN(nn.Module):
    """Classic 2-layer Graph Convolutional Network."""
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        out = self.gcn2(h, adj)
        return out
