import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from typing import List, Tuple

# =====================================================================
# 1. GRAPH CONVOLUTIONAL NETWORK (GCN) DEFINITION
# =====================================================================

class GCNLayer(nn.Module):
    """
    A custom Graph Convolutional Network (GCN) Layer.
    Formula: H^(l+1) = D^(-1/2) * A_tilde * D^(-1/2) * H^(l) * W^(l)
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, norm_adj):
        support = self.linear(x)
        out = torch.mm(norm_adj, support)
        return out

class KarateGCN(nn.Module):
    """
    2-Layer GCN to map initial node features to a 2D space.
    The 2D output will serve both as the embedding for visualization
    and the logits for classification.
    """
    def __init__(self, num_nodes, hidden_dim=8):
        super(KarateGCN, self).__init__()
        self.features = torch.eye(num_nodes)
        self.gcn1 = GCNLayer(num_nodes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, 2)
        
    def forward(self, norm_adj):
        x = self.features
        h1 = torch.tanh(self.gcn1(x, norm_adj))
        h2 = self.gcn2(h1, norm_adj)
        return h2

# =====================================================================
# 2. DEEPWALK (RANDOM WALK EMBEDDING) FROM SCRATCH IN PYTORCH
# =====================================================================

def generate_random_walks(G: nx.Graph, num_walks: int = 15, walk_length: int = 10) -> List[List[int]]:
    """Generates random walks of fixed length starting from each node in the graph."""
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = list(G.neighbors(curr))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    return walks

def build_skip_gram_dataset(walks: List[List[int]], window_size: int = 2) -> List[Tuple[int, int]]:
    """Constructs skip-gram (center, context) token-node pairs from random walks."""
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            # Grab local context tokens within window bounds
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((center, walk[j]))
    return pairs

class DeepWalkModel(nn.Module):
    """
    PyTorch Skip-Gram embedding model with negative sampling approximation.
    Maps nodes to a 2D space for structural comparison.
    """
    def __init__(self, num_nodes: int, embedding_dim: int = 2):
        super(DeepWalkModel, self).__init__()
        self.in_embed = nn.Embedding(num_nodes, embedding_dim)
        self.out_embed = nn.Embedding(num_nodes, embedding_dim)
        
    def forward(self, center: torch.Tensor, context: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Get target dot products
        c_emb = self.in_embed(center)       # [batch_size, dim]
        v_emb = self.out_embed(context)      # [batch_size, dim]
        pos_loss = torch.log(torch.sigmoid(torch.sum(c_emb * v_emb, dim=1)) + 1e-8)
        
        # Get negative dot products
        n_emb = self.out_embed(negative)     # [batch_size, num_neg, dim]
        neg_loss = torch.log(torch.sigmoid(-torch.bmm(n_emb, c_emb.unsqueeze(2)).squeeze(2)) + 1e-8)
        
        return -torch.mean(pos_loss + torch.sum(neg_loss, dim=1))

def train_deepwalk(G: nx.Graph, num_nodes: int, epochs: int = 40) -> np.ndarray:
    """Trains DeepWalk models to extract 2D structural embeddings."""
    print("Generating random walks & preparing Skip-Gram pairs...")
    walks = generate_random_walks(G, num_walks=10, walk_length=12)
    dataset = build_skip_gram_dataset(walks, window_size=2)
    
    centers = torch.tensor([pair[0] for pair in dataset], dtype=torch.long)
    contexts = torch.tensor([pair[1] for pair in dataset], dtype=torch.long)
    
    model = DeepWalkModel(num_nodes, embedding_dim=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    batch_size = 256
    num_neg = 5
    print(f"Training DeepWalk for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(centers.size(0))
        epoch_loss = 0
        
        for i in range(0, centers.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            b_center = centers[indices]
            b_context = contexts[indices]
            
            # Generate negative samples
            b_neg = torch.randint(0, num_nodes, (b_center.size(0), num_neg))
            
            optimizer.zero_grad()
            loss = model(b_center, b_context, b_neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"    DeepWalk Epoch {epoch:02d}/{epochs} | Avg Loss: {epoch_loss / (centers.size(0)/batch_size):.4f}")
            
    # Extract final learned node vectors
    with torch.no_grad():
        embeddings = model.in_embed.weight.numpy()
    return embeddings

# =====================================================================
# 3. UTILITY FUNCTIONS
# =====================================================================

def build_normalized_adj(G):
    """Builds the symmetrically normalized adjacency matrix with self-loops."""
    num_nodes = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    
    I = torch.eye(num_nodes)
    A_tilde = A_tensor + I
    
    d = torch.sum(A_tilde, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
    return norm_adj

# =====================================================================
# 4. MAIN SCRIPT
# =====================================================================

def main():
    print("=" * 60)
    print("        GCN VS. DEEPWALK COMPARATIVE BENCHMARK ON KARATE CLUB")
    print("=" * 60)
    
    # Load Zachary's Karate Club Graph
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()
    
    # Ground Truth labels representing the club split ('Mr. Hi' = 0, 'Officer' = 1)
    labels = []
    for i in range(num_nodes):
        faction = G.nodes[i]['club']
        labels.append(0 if faction == 'Mr. Hi' else 1)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # 1. Train from-scratch DeepWalk Baseline
    dw_embeddings = train_deepwalk(G, num_nodes)
    
    # 2. Train semi-supervised GCN Model
    norm_adj = build_normalized_adj(G)
    gcn_model = KarateGCN(num_nodes=num_nodes, hidden_dim=8)
    criterion = nn.CrossEntropyLoss()
    gcn_optimizer = optim.Adam(gcn_model.parameters(), lr=0.08)
    
    # Semi-supervised split: Train on ONLY node 0 and node 33 (faction leaders)
    train_indices = torch.tensor([0, 33], dtype=torch.long)
    train_labels = labels[train_indices]
    
    print("\nTraining GCN on ONLY 2 nodes (0 and 33)...")
    epochs = 200
    for epoch in range(1, epochs + 1):
        gcn_model.train()
        gcn_optimizer.zero_grad()
        
        embeddings = gcn_model(norm_adj)
        train_logits = embeddings[train_indices]
        loss = criterion(train_logits, train_labels)
        
        loss.backward()
        gcn_optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            preds = embeddings.argmax(dim=1)
            correct = (preds == labels).sum().item()
            acc = correct / num_nodes * 100
            print(f"    GCN Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} | Overall Inference Acc: {acc:.2f}%")
            
    # Extract final GCN results
    gcn_model.eval()
    with torch.no_grad():
        gcn_embeddings = gcn_model(norm_adj).numpy()
        gcn_preds = gcn_model(norm_adj).argmax(dim=1).numpy()
        
    # =====================================================================
    # 5. HIGH-FIDELITY SIDE-BY-SIDE VISUALIZATION
    # =====================================================================
    print("\nSaving high-fidelity comparison visualization...")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), facecolor='#121214')
    
    # Modern styling choices
    blue_color = '#00E5FF' # Cyan
    orange_color = '#FF9100' # Bright Orange
    colors = [blue_color if l == 0 else orange_color for l in labels.numpy()]
    
    for ax in axs:
        ax.set_facecolor('#121214')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#37474F')
            
    # Subplot 1: Graph Structure Network
    axs[0].set_title("1. Original Structural Layout", color='white', fontsize=14, fontweight='bold', pad=15)
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [700 if n in [0, 33] else 250 for n in G.nodes()]
    node_edges = ['white' if n in [0, 33] else '#37474F' for n in G.nodes()]
    line_widths = [2.5 if n in [0, 33] else 0.5 for n in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, edge_color='#37474F', alpha=0.6, ax=axs[0])
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, 
                           edgecolors=node_edges, linewidths=line_widths, ax=axs[0])
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=8, font_weight='bold', ax=axs[0])
    axs[0].axis('off')
    
    # Subplot 2: DeepWalk Embeddings
    axs[1].set_title("2. DeepWalk 2D Embeddings\n(Unsupervised Structural)", color='white', fontsize=14, fontweight='bold', pad=15)
    axs[1].scatter(dw_embeddings[:, 0], dw_embeddings[:, 1], s=250, c=colors, edgecolors='#37474F', zorder=5)
    for i in range(num_nodes):
        axs[1].annotate(str(i), (dw_embeddings[i, 0], dw_embeddings[i, 1]), 
                        fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=10)
    axs[1].grid(True, linestyle=':', alpha=0.3, color='gray')
    axs[1].set_xlabel("Embedding Dim 1", color='white')
    axs[1].set_ylabel("Embedding Dim 2", color='white')
    
    # Subplot 3: GCN Embeddings
    axs[2].set_title("3. GCN 2D Embeddings\n(Semi-Supervised Class-Driven)", color='white', fontsize=14, fontweight='bold', pad=15)
    axs[2].scatter(gcn_embeddings[:, 0], gcn_embeddings[:, 1], s=250, c=colors, edgecolors='#37474F', zorder=5)
    for i in range(num_nodes):
        axs[2].annotate(str(i), (gcn_embeddings[i, 0], gcn_embeddings[i, 1]), 
                        fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=10)
    axs[2].grid(True, linestyle=':', alpha=0.3, color='gray')
    axs[2].set_xlabel("Latent Dim 1", color='white')
    axs[2].set_ylabel("Latent Dim 2", color='white')
    
    plt.suptitle("Zachary's Karate Club Embedding Representation Benchmark", color='white', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', 'karate_club_comparison.png')
    plt.savefig(plot_path, facecolor='#121214', edgecolor='none', dpi=150)
    print(f"[SUCCESS] Saved beautiful comparative visualization to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
