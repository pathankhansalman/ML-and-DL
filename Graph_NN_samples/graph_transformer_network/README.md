# Graph-Based Transformer Architecture

A lightweight, high-fidelity implementation of a self-attention model formulated as a **directed graph message-passing network**. 

In standard transformer architectures, attention matrix coefficients can be mathematically interpreted as directed edge weights on a fully connected graph of token-nodes. This project implements that concept explicitly, using PyTorch for the model mechanics and NetworkX for graph extraction and structural analysis.

---

## 🌌 Mathematical Conception

1. **Tokens as Nodes ($V$)**: Every token (character or subword) in an input sequence represents a unique vertex/node in the graph.
2. **QK Pairs as Edges ($E$)**: The query vector $Q_i$ of node $i$ interacts with the key vector $K_j$ of node $j$ to compute attention scores:
   $$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$
   $A_{ij}$ is the weight of the directed edge $i \to j$.
3. **Value vectors as Messages ($M$)**: The state update of node $i$ is the aggregation of messages $V_j$ along these directed attention edges:
   $$H'_i = \sum_{j} A_{ij} V_j$$

---

## 📂 Repository Structure

* **`graph_transformer.py`**:
  * `SimpleTokenizer`: Lightweight tokenizer preparing sequences.
  * `GraphTransformerLayer`: Graph attention layer performing directed message passing.
  * `GraphTransformerModel`: Fully functional model stack featuring causal masking (autoregressive generation) and bidirectional encoding.
* **`visualizer.py`**:
  * `draw_token_graph`: Utility converting attention matrices into a circular `networkx.DiGraph` rendering curved, weight-sensitive relationships on a sleek dark background.
* **`run_demo.py`**:
  * Trains the Graph Transformer on text corpus, generates text, extracts node embeddings, performs semantic cosine similarities, and saves the visualization.
* **`plots/`**:
  * Stores the generated high-quality attention network graph plots (`token_graph.png`).

---

## ⚡ Quick Start & Execution

Ensure your environment has the required scientific packages installed (`torch`, `networkx`, `matplotlib`):

```bash
pip install -r requirements.txt
```

Run the demonstration script:

```bash
python run_demo.py
```

---

## 📊 Core Demo Results

### 1. Training Next Token Prediction
The network converges rapidly under cross-entropy loss by minimizing sequence prediction error over 120 epochs:
* **Initial Loss**: `~3.60`
* **Final Loss**: `~0.02`

### 2. Autoregressive Generation Output
* **Prompt**: `"each token is "`
* **Generated output**: `"each token is phents attention weights. each token is a node in the graph."`

### 3. Sentence Graph Cosine Similarity
Extracting bidirectional node states across complete graph sequences yields dense semantic embeddings:
* **High Similarity**: Between transformer/graph topics (`~0.88`).
* **Low Similarity**: Between unrelated topics (e.g. graph vs cooking pasta, `~0.77`).
