import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from graph_transformer import SimpleTokenizer, GraphTransformerModel, compute_cosine_similarity
from visualizer import draw_token_graph
import os

def main():
    print("=" * 60)
    print("      GRAPH-BASED TRANSFORMER ARCHITECTURE DEMONSTRATION      ")
    print("=" * 60)

    # 1. Prepare training corpus
    corpus = (
        "the graph represents attention weights. "
        "each token is a node in the graph. "
        "two tokens having a qk pair have a directed edge. "
        "the value vector is the message passed along the edge. "
        "we can perform next token prediction with a causal graph. "
        "we can perform sequence encoding with a bidirectional graph. "
        "cosine similarity compares graph-based sentence representations. "
    )
    print(f"\n[1] Prepared training corpus ({len(corpus)} characters).")
    
    # 2. Tokenizer setup
    tokenizer = SimpleTokenizer(corpus)
    print(f"    Vocabulary Size: {tokenizer.vocab_size} characters/tokens.")
    
    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    
    model = GraphTransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    # 4. Prepare training data
    # Create input-target pairs for next-token prediction
    input_text = corpus[:-1]
    target_text = corpus[1:]
    
    input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long, device=device)
    target_ids = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long, device=device)
    
    # 5. Train the model
    print("\n[2] Training the Graph Transformer model (Next Token Prediction)...")
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 120
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(input_ids, causal=True)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs:03d} | Loss: {loss.item():.4f}")
            
    print("    Training completed successfully.")

    # 6. Text Generation / Next Token Prediction
    print("\n[3] Autoregressive Text Generation starting from prompt...")
    model.eval()
    prompt = "each token is "
    generated_ids = tokenizer.encode(prompt)
    
    # Generate 50 tokens
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits, _ = model(inp, causal=True)
            # Sample next token from the last position's logits
            next_token_logits = logits[0, -1, :]
            # Apply softmax with temperature
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token)
            
    generated_text = tokenizer.decode(generated_ids)
    print(f"    Prompt: '{prompt}'")
    print(f"    Generated: '{generated_text}'")

    # 7. Sequence Encoding and Cosine Similarity
    print("\n[4] Bidirectional Sequence Encoding & Cosine Similarity...")
    sentences = [
        "each token is a node in the graph.",
        "two tokens having a qk pair have a directed edge.",
        "completely unrelated sentence about cooking tasty pasta."
    ]
    
    embeddings = []
    for s in sentences:
        ids = torch.tensor([tokenizer.encode(s)], dtype=torch.long, device=device)
        emb = model.encode(ids)[0]
        embeddings.append(emb)
        
    sim_1_2 = compute_cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = compute_cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"    Sentence 1: '{sentences[0]}'")
    print(f"    Sentence 2: '{sentences[1]}'")
    print(f"    Sentence 3: '{sentences[2]}'")
    print(f"    -> Cosine Similarity (S1 <-> S2) [Graph/Transformer topics]: {sim_1_2:.4f}")
    print(f"    -> Cosine Similarity (S1 <-> S3) [Graph topic <-> Pasta topic]: {sim_1_3:.4f}")
    
    # 8. Token Graph Extraction and Visualization
    print("\n[5] Extracting Token Graph Attention Matrix and Plotting...")
    sample_sentence = "each token is a node."
    sample_ids = torch.tensor([tokenizer.encode(sample_sentence)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, attention_maps = model(sample_ids, causal=False)
        # Average attention over all heads of the first layer
        # attention_maps[0] shape: [1, n_heads, seq_len, seq_len]
        attn_matrix = attention_maps[0][0].mean(dim=0).cpu().numpy()
        
    # Split token characters for node labels
    tokens = list(sample_sentence)
    
    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "token_graph.png")
    
    draw_token_graph(tokens, attn_matrix, plot_path, threshold=0.03)
    
    print("\n" + "=" * 60)
    print("Demo execution finished. Graph plots saved successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
