import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# =====================================================================
# 1. LOGISTIC REGRESSION PIPELINE
# =====================================================================

def build_logistic_regression_pipeline(numerical_cols, categorical_cols):
    """
    Constructs a robust Scikit-Learn pipeline for Logistic Regression.
    Includes feature scaling for rankings and one-hot encoding for categorical variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    ])
    
    return pipeline


# =====================================================================
# 2. PYTORCH GRAPH CONVOLUTIONAL NETWORK (GCN) FROM SCRATCH
# =====================================================================

class GCNLayer(nn.Module):
    """
    A custom Graph Convolutional Network (GCN) Layer implemented in pure PyTorch.
    Implements the classic message-passing formula: H^(l+1) = D^(-1/2) * A_tilde * D^(-1/2) * H^(l) * W^(l)
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, norm_adj):
        """
        Args:
            x: Node feature matrix of shape [num_nodes, in_features]
            norm_adj: Normalized adjacency matrix with self-loops, shape [num_nodes, num_nodes]
        """
        # Linear transformation (W * H)
        support = self.linear(x)
        # Message passing / aggregation (D^-1/2 * A_tilde * D^-1/2 * Support)
        out = torch.mm(norm_adj, support)
        return out


class TeamGCN(nn.Module):
    """
    Graph Neural Network for Cricket Match Outcome Prediction (Edge Classification).
    
    Learns team representations dynamically via GCN message passing, and combines them
    with match-specific features (venue, toss, dew) to classify whether the team batting first wins.
    """
    def __init__(self, num_teams, embed_dim, hidden_dim, out_dim, num_edge_features):
        super(TeamGCN, self).__init__()
        # Initial team representation is a learned embedding layer
        self.team_embeddings = nn.Embedding(num_teams, embed_dim)
        
        # 2 GCN layers to aggregate neighbor (opponent) information
        self.gcn1 = GCNLayer(embed_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        
        # MLP classifier head for predicting match outcome
        # Inputs: Concatenated embeddings of Team 1 and Team 2, plus edge features
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2 + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Single raw logit output
        )
        
    def forward(self, norm_adj, team_1_idx, team_2_idx, edge_features):
        """
        Args:
            norm_adj: Normalized adjacency matrix, shape [num_nodes, num_nodes]
            team_1_idx: Integer indices of the teams batting first, shape [batch_size]
            team_2_idx: Integer indices of the teams batting second, shape [batch_size]
            edge_features: Match-specific features, shape [batch_size, num_edge_features]
        """
        num_nodes = norm_adj.size(0)
        device = norm_adj.device
        all_nodes = torch.arange(num_nodes, device=device)
        
        # 1. Fetch initial embeddings for all teams
        x = self.team_embeddings(all_nodes) # [num_teams, embed_dim]
        
        # 2. Graph Message Passing
        h = F.relu(self.gcn1(x, norm_adj))   # [num_teams, hidden_dim]
        h = self.gcn2(h, norm_adj)            # [num_teams, out_dim]
        
        # 3. Retrieve representations of the playing teams for this batch
        h1 = h[team_1_idx] # [batch_size, out_dim]
        h2 = h[team_2_idx] # [batch_size, out_dim]
        
        # 4. Concatenate team representations with match-specific (edge) features
        edge_repr = torch.cat([h1, h2, edge_features], dim=1) # [batch_size, out_dim * 2 + num_edge_features]
        
        # 5. Pass through Classifier MLP
        logits = self.classifier(edge_repr).squeeze(1)
        return logits


# =====================================================================
# 3. ADJACENCY MATRIX GENERATOR UTILITY
# =====================================================================

def build_normalized_adj(num_teams, matches_df, device):
    """
    Builds the symmetrically normalized adjacency matrix (with self-loops)
    from a list of matches.
    
    A_tilde = A + I
    norm_adj = D^(-1/2) * A_tilde * D^(-1/2)
    """
    # Initialize adjacency matrix
    A = torch.zeros((num_teams, num_teams), device=device)
    
    # Fill adjacency based on matches (undirected edges: who played whom)
    for _, row in matches_df.iterrows():
        t1, t2 = int(row["team_1_idx"]), int(row["team_2_idx"])
        A[t1, t2] += 1.0
        A[t2, t1] += 1.0
        
    # Add self-loops
    I = torch.eye(num_teams, device=device)
    A_tilde = A + I
    
    # Calculate degree matrix D_tilde
    d = torch.sum(A_tilde, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0 # handle potential division-by-zero
    
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Symmetrically normalize
    norm_adj = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
    
    return norm_adj
