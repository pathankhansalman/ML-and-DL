import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import ActivationCollector
from dataset import load_or_create_dataset

class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model, dictionary_size, k=15):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, dictionary_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        self.decoder = nn.Linear(dictionary_size, d_model, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x_centered = x - self.b_dec
        hidden_pre_act = self.encoder(x_centered) + self.encoder_bias
        positive_acts = torch.relu(hidden_pre_act)
        
        # Keep only top-k positive activations
        values, indices = torch.topk(positive_acts, self.k, dim=-1)
        feature_acts = torch.zeros_like(positive_acts)
        feature_acts.scatter_(-1, indices, values)
        
        reconstruction = self.decoder(feature_acts) + self.b_dec
        return reconstruction, feature_acts

def train_overnight(steps=5000, lr=1e-3, k=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model & Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # 2. Load a larger dataset of 100 sentences from dataset.py
    # (Since this is a demo, we will use our cached local dataset.json text)
    print("Loading text data to harvest activations...")
    dataset = load_or_create_dataset(use_huggingface=False)
    texts = [item["text"] for item in dataset]

    # 3. Collect activations at Layer 10 MLP output
    print("Extracting activations from Layer 10 MLP...")
    collector = ActivationCollector()
    target_layer = model.model.layers[10].mlp
    handle = target_layer.register_forward_hook(collector.hook)
    
    with torch.no_grad():
        # Only run on the first 50 sentences to prevent memory issues on CPU/GPU
        for text in texts[:50]:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
    handle.remove()

    # Flatten activations: Shape [Total Tokens, Hidden Dim]
    activations = torch.cat([act.view(-1, act.shape[-1]) for act in collector.activations], dim=0).float()
    print(f"Total activations collected: {activations.shape[0]} tokens, Hidden Dim: {activations.shape[1]}")

    # 4. Initialize Top-K SAE
    d_model = activations.shape[-1]
    dict_size = d_model * 4  # 4x expansion factor
    sae = TopKSparseAutoencoder(d_model, dict_size, k=k).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Batch size for training
    batch_size = 128
    num_samples = activations.shape[0]

    print(f"\n--- Starting Training ({steps} steps) ---")
    sae.train()
    
    for step in range(1, steps + 1):
        # Sample a random batch of activations
        indices = torch.randint(0, num_samples, (batch_size,))
        batch = activations[indices].to(device)
        
        optimizer.zero_grad()
        reconstructed, feature_acts = sae(batch)
        
        # Loss is pure reconstruction MSE
        loss = nn.MSELoss()(reconstructed, batch)
        loss.backward()
        optimizer.step()

        if step % (10 if steps <= 50 else 100) == 0 or step == 1:
            l0 = (feature_acts > 1e-4).float().sum(dim=-1).mean().item()
            print(f"Step {step:04d}/{steps} | Loss: {loss.item():.4f} | L0 Sparsity: {l0:.1f} features/token")

    # Evaluate final reconstruction
    sae.eval()
    with torch.no_grad():
        reconstructed, feature_acts = sae(activations.to(device))
        final_mse = nn.MSELoss()(reconstructed, activations.to(device)).item()
        variance_explained = 1 - (final_mse / activations.var().item())
        print("\n--- Final Metrics ---")
        print(f"Final Reconstruction MSE: {final_mse:.4f}")
        print(f"Variance Explained: {variance_explained * 100:.2f}%")

    # 5. Save model weights
    save_path = "mats_probing/qwen_sae_weights.pt"
    torch.save(sae.state_dict(), save_path)
    print(f"Saved trained SAE weights to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Run a quick 5-step verification test")
    args = parser.parse_args()
    
    if args.toy:
        train_overnight(steps=5, k=5)
    else:
        train_overnight(steps=5000, k=15)
