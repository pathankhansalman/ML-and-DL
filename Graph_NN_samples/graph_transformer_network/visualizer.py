import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def draw_token_graph(tokens: List[str], attention_matrix: np.ndarray, save_path: str, threshold: float = 0.05):
    """
    Constructs a directed token-graph and saves a beautiful visual representation.
    
    Args:
        tokens: List of string characters or subwords.
        attention_matrix: 2D numpy array [seq_len, seq_len] representing attention scores.
                          Row i represents attention from token i to all other tokens.
        save_path: Output file path for the plot.
        threshold: Ignore edges with weight below this threshold to prevent visual clutter.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    seq_len = len(tokens)
    G = nx.DiGraph()
    
    # 1. Add nodes with their labels
    # We will append index to tokens to ensure unique node identifiers
    node_ids = [f"{i}: '{t}'" for i, t in enumerate(tokens)]
    for node_id, t in zip(node_ids, tokens):
        G.add_node(node_id, label=t)
        
    # 2. Add edges above threshold
    for i in range(seq_len):
        for j in range(seq_len):
            weight = attention_matrix[i, j]
            if weight >= threshold:
                G.add_edge(node_ids[i], node_ids[j], weight=weight)
                
    # 3. Choose layout
    # Shell or spring layout with circular or timeline flow
    # A circular layout is gorgeous for visualizing self-attention graphs
    pos = nx.circular_layout(G)
    
    # Setup Figure and Style
    plt.figure(figsize=(10, 8), facecolor='#121214')
    ax = plt.gca()
    ax.set_facecolor('#121214')
    
    # Node Styling
    node_color = '#7C4DFF'  # Cool deep violet
    node_size = 1400
    
    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_color, 
        node_size=node_size, 
        edgecolors='#B0BEC5', 
        linewidths=1.5,
        alpha=0.9,
        ax=ax
    )
    
    # Label Styling
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_color='white', 
        font_weight='bold', 
        ax=ax
    )
    
    # Edge Styling based on attention weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if len(weights) > 0:
        # Scale weights to visual edge widths
        max_w = max(weights)
        min_w = min(weights)
        norm_weights = [1.0 + 4.0 * (w - min_w) / (max_w - min_w + 1e-8) for w in weights]
        
        # Draw directed edges
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=edges, 
            width=norm_weights, 
            edge_color=weights, 
            edge_cmap=plt.cm.cool, 
            arrowstyle='-|>', 
            arrowsize=18, 
            connectionstyle='arc3,rad=0.15',  # Curved arrows look elegant
            alpha=0.75,
            ax=ax
        )
    
    plt.title("Token-Graph Attention Adjacency Network", color='white', fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(save_path, facecolor='#121214', edgecolor='none', dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved beautiful token graph visualization to: {save_path}")
