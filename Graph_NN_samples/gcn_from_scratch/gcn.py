import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) layer implemented from scratch.
    Equation: H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} )
    """
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable weight parameter
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
        # 1. Add self-loops: A_tilde = A + I
        num_nodes = adj.size(0)
        adj_tilde = adj + torch.eye(num_nodes, device=adj.device)
        
        # 2. Compute degree matrix D_tilde and D_tilde^{-1/2}
        deg = torch.sum(adj_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # 3. Symmetric normalization: D^{-1/2} A D^{-1/2}
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        
        # 4. Feature transformation: H * W
        support = torch.mm(x, self.weight)
        
        # 5. Neighborhood aggregation: A_norm * (H * W)
        output = torch.mm(norm_adj, support)
        
        return output + self.bias

class GCN(nn.Module):
    """A 2-layer Graph Convolutional Network."""
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        out = self.gcn2(h, adj)
        return out

if __name__ == "__main__":
    # Test GCN implementation with a mock graph (5 nodes, 3-dimensional features)
    num_nodes = 5
    in_features = 3
    hidden_features = 4
    num_classes = 2
    
    # Random node features
    x = torch.randn(num_nodes, in_features)
    
    # Fully connected adjacency matrix for testing
    adj = torch.tensor([
        [0., 1., 1., 0., 0.],
        [1., 0., 1., 1., 0.],
        [1., 1., 0., 0., 0.],
        [0., 1., 0., 0., 1.],
        [0., 0., 0., 1., 0.]
    ])
    
    model = GCN(in_features, hidden_features, num_classes)
    output = model(x, adj)
    print("Output shapes (nodes, classes):", output.shape)
    print("GCN Output:\n", output)
