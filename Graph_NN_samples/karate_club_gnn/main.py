import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, num_nodes, hidden_dim=4):
        super(KarateGCN, self).__init__()
        # Initial node features: we use an Identity matrix (one-hot encoding for each node)
        self.features = torch.eye(num_nodes)
        
        # Layer 1: Map num_nodes (34) -> hidden_dim (4)
        self.gcn1 = GCNLayer(num_nodes, hidden_dim)
        
        # Layer 2: Map hidden_dim (4) -> 2D Space
        self.gcn2 = GCNLayer(hidden_dim, 2)
        
    def forward(self, norm_adj):
        x = self.features
        h1 = torch.tanh(self.gcn1(x, norm_adj))
        h2 = self.gcn2(h1, norm_adj) # Shape: [34, 2]
        return h2

# =====================================================================
# 2. UTILITY FUNCTIONS
# =====================================================================

def build_normalized_adj(G):
    """Builds the symmetrically normalized adjacency matrix with self-loops."""
    num_nodes = G.number_of_nodes()
    # Adjacency matrix
    A = nx.to_numpy_array(G)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    
    # Add self-loops (A_tilde = A + I)
    I = torch.eye(num_nodes)
    A_tilde = A_tensor + I
    
    # Calculate degree matrix D_tilde
    d = torch.sum(A_tilde, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Symmetrically normalize: D^(-1/2) * A_tilde * D^(-1/2)
    norm_adj = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
    return norm_adj

# =====================================================================
# 3. MAIN SCRIPT
# =====================================================================

def main():
    print("Loading Zachary's Karate Club Graph...")
    # Load the graph
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()
    
    # Extract Ground Truth Labels
    # 'Mr. Hi' -> Faction 0, 'Officer' -> Faction 1
    labels = []
    for i in range(num_nodes):
        faction = G.nodes[i]['club']
        labels.append(0 if faction == 'Mr. Hi' else 1)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Build Adjacency Matrix
    norm_adj = build_normalized_adj(G)
    
    # Initialize Model
    model = KarateGCN(num_nodes=num_nodes, hidden_dim=8)
    criterion = nn.CrossEntropyLoss()
    # Using a slightly higher learning rate and momentum for fast convergence on this small graph
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # SEMI-SUPERVISED SETUP:
    # We only train on Node 0 (Mr. Hi) and Node 33 (Officer).
    # We mask out the other 32 nodes during training.
    train_indices = torch.tensor([0, 33], dtype=torch.long)
    train_labels = labels[train_indices]
    
    print("\nTraining GCN on ONLY 2 nodes (Node 0 and Node 33)...")
    epochs = 200
    
    # To animate/track the embeddings over time, we'll store the outputs
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (all nodes get an embedding)
        embeddings = model(norm_adj)
        
        # Compute loss ONLY on the 2 labeled training nodes
        train_logits = embeddings[train_indices]
        loss = criterion(train_logits, train_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            # Calculate accuracy on ALL nodes to see how well it's inferring
            preds = embeddings.argmax(dim=1)
            correct = (preds == labels).sum().item()
            acc = correct / num_nodes * 100
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} | Overall Accuracy: {acc:.2f}%")
            
    # Extract final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(norm_adj).numpy()
        final_preds = model(norm_adj).argmax(dim=1).numpy()
        
    # =====================================================================
    # 4. VISUALIZATION
    # =====================================================================
    print("\nGenerating visualizations...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original Graph Structure
    axs[0].set_title("Zachary's Karate Club: Graph Structure", fontsize=14, fontweight='bold')
    pos = nx.spring_layout(G, seed=42)
    # Color nodes by their true faction
    colors = ['#1f77b4' if l == 0 else '#ff7f0e' for l in labels.numpy()]
    
    # Highlight the two training nodes
    node_sizes = [800 if n in [0, 33] else 300 for n in G.nodes()]
    node_edges = ['black' if n in [0, 33] else 'none' for n in G.nodes()]
    line_widths = [3 if n in [0, 33] else 0 for n in G.nodes()]
    
    nx.draw(G, pos, node_color=colors, with_labels=True, 
            node_size=node_sizes, edgecolors=node_edges, linewidths=line_widths,
            font_color='white', font_weight='bold', ax=axs[0])
    axs[0].text(0.5, -0.1, "Outlined nodes (0, 33) were the ONLY training labels.", 
                ha='center', va='center', transform=axs[0].transAxes, fontsize=10, fontstyle='italic')

    # Plot 2: Learned 2D Embeddings from GCN
    axs[1].set_title("Learned GCN 2D Embeddings", fontsize=14, fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # Scatter plot of embeddings, colored by the GCN's *predictions*
    pred_colors = ['#1f77b4' if p == 0 else '#ff7f0e' for p in final_preds]
    
    axs[1].scatter(final_embeddings[:, 0], final_embeddings[:, 1], 
                   s=300, c=pred_colors, edgecolors='black', zorder=5)
                   
    # Annotate node numbers on the embeddings
    for i in range(num_nodes):
        axs[1].annotate(str(i), (final_embeddings[i, 0], final_embeddings[i, 1]), 
                        fontsize=9, fontweight='bold', color='white',
                        ha='center', va='center', zorder=10)
                        
    axs[1].set_xlabel("Latent Dimension 1")
    axs[1].set_ylabel("Latent Dimension 2")
    axs[1].text(0.5, -0.1, "Nodes cleanly separated purely by graph connections!", 
                ha='center', va='center', transform=axs[1].transAxes, fontsize=10, fontstyle='italic')

    plt.tight_layout()
    # Save the plot
    import os
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', 'karate_club_gnn_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved successfully to {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
