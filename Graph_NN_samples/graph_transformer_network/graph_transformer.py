import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

class BPETokenizer:
    """A lightweight, from-scratch Byte Pair Encoding (BPE) Subword Tokenizer."""
    def __init__(self, text_corpus: str, num_merges: int = 40):
        self.special_tokens = ['<pad>', '<unk>', ' ', '</w>']
        unique_chars = sorted(list(set(text_corpus)))
        # Remove duplicates if space exists in corpus already
        unique_chars = [c for c in unique_chars if c not in self.special_tokens]
        self.vocab = self.special_tokens + unique_chars
        self.vocab_size = len(self.vocab)
        
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        
        self.pad_id = self.token_to_id['<pad>']
        self.unk_id = self.token_to_id['<unk>']
        self.merges = {}
        
        self._train_bpe(text_corpus, num_merges)
        
    def _train_bpe(self, corpus: str, num_merges: int):
        words = corpus.split(' ')
        tokenized_words = [list(w) + ['</w>'] for w in words]
        
        if '</w>' not in self.token_to_id:
            self._add_to_vocab('</w>')
            
        for _ in range(num_merges):
            pairs = {}
            for word in tokenized_words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break
                
            merged_token = best_pair[0] + best_pair[1]
            self._add_to_vocab(merged_token)
            self.merges[best_pair] = merged_token
            
            new_tokenized_words = []
            for word in tokenized_words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_tokenized_words.append(new_word)
            tokenized_words = new_tokenized_words

    def _add_to_vocab(self, token: str):
        if token not in self.token_to_id:
            new_id = len(self.vocab)
            self.vocab.append(token)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token
            self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        words = text.split(' ')
        encoded_ids = []
        
        for w_idx, w in enumerate(words):
            word_tokens = list(w) + ['</w>']
            for pair, merged in self.merges.items():
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                word_tokens = new_word
            
            for token in word_tokens:
                encoded_ids.append(self.token_to_id.get(token, self.unk_id))
            
            if w_idx < len(words) - 1:
                if ' ' not in self.token_to_id:
                    self._add_to_vocab(' ')
                encoded_ids.append(self.token_to_id[' '])
                
        return encoded_ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(idx, '<unk>') for idx in ids if idx != self.pad_id]
        decoded = "".join(tokens).replace('</w>', ' ')
        return decoded


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
