import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

class SimpleTokenizer:
    """A lightweight character-level tokenizer for demonstration purposes."""
    def __init__(self, text_corpus: str):
        # Create a sorted list of unique characters in the corpus
        self.chars = sorted(list(set(text_corpus)))
        # Add special tokens
        if '<pad>' not in self.chars:
            self.chars.append('<pad>')
        if '<unk>' not in self.chars:
            self.chars.append('<unk>')
        
        self.vocab_size = len(self.chars)
        self.char_to_id = {char: idx for idx, char in enumerate(self.chars)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        self.pad_id = self.char_to_id['<pad>']
        self.unk_id = self.char_to_id['<unk>']

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char.get(idx, '<unk>') for idx in ids if idx != self.pad_id])


class GraphTransformerLayer(nn.Module):
    """
    A single layer of the Graph Transformer.
    Self-attention is formulated as directed graph message passing where:
      - Nodes = Tokens
      - Directed Edges = Attention links between Q and K
      - Edge Weights = Softmax-normalized attention scores
      - Message = Value vector
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Shape [batch_size, seq_len, d_model]
        mask: Shape [batch_size, 1, seq_len, seq_len] or similar
        Returns:
            - output embeddings: [batch_size, seq_len, d_model]
            - attention/adjacency matrix: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Graph Adjacency/Attention weights computation
        # Score = (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Edge weights (Softmax normalizes row-wise, i.e., each node's attention over all others)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 3. Message Passing / Aggregation
        # New node states are weighted sums of the Value vectors (messages) along active edges
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attention_output = self.out_linear(context)
        
        # Residual + Norm
        x = self.norm1(x + self.dropout(attention_output))
        
        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attention_weights


class GraphTransformerModel(nn.Module):
    """Full lightweight model capable of both generation (causal) and encoding (bidirectional)."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, 
                 n_layers: int = 2, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, causal: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        token_ids: Shape [batch_size, seq_len]
        causal: If True, uses lower-triangular causal mask to prevent nodes from seeing future tokens.
        Returns:
            - logits: [batch_size, seq_len, vocab_size]
            - attention_maps: List of [batch_size, n_heads, seq_len, seq_len] per layer
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Embed tokens and positions
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        x = self.token_embeddings(token_ids) + self.pos_embeddings(positions)
        x = self.dropout(x)
        
        # Generate Causal/Temporal mask if required
        mask = None
        if causal:
            # Lower triangular mask: only allow edges from prior tokens (nodes)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
            
        attention_maps = []
        for layer in self.layers:
            x, att_map = layer(x, mask)
            attention_maps.append(att_map)
            
        logits = self.lm_head(x)
        return logits, attention_maps

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encodes sequence bidirectionally (allowing fully connected token-graph relationships).
        Returns a single pooled sentence embedding.
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = token_ids.shape
            device = token_ids.device
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            x = self.token_embeddings(token_ids) + self.pos_embeddings(positions)
            
            for layer in self.layers:
                x, _ = layer(x, mask=None)  # No mask = bidirectional
                
            # Mean pool token embeddings across the sequence to get a global sequence graph representation
            pooled = x.mean(dim=1)
            return pooled


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Computes the cosine similarity between two vector embeddings."""
    dot_product = torch.dot(emb1, emb2)
    norm1 = torch.norm(emb1)
    norm2 = torch.norm(emb2)
    similarity = dot_product / (norm1 * norm2 + 1e-8)
    return similarity.item()
