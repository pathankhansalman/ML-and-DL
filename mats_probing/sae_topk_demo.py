import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the Top-K Sparse Autoencoder (SAE)
class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model, dictionary_size, k=20):
        super().__init__()
        self.k = k # The exact number of features allowed to be active at once
        
        self.encoder = nn.Linear(d_model, dictionary_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        
        self.decoder = nn.Linear(dictionary_size, d_model, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. Center the inputs
        x_centered = x - self.b_dec
        
        # 2. Get pre-activations
        hidden_pre_act = self.encoder(x_centered) + self.encoder_bias
        
        # 3. Apply ReLU (keep only positive activations)
        positive_acts = torch.relu(hidden_pre_act)
        
        # 4. Top-K Selection: keep only the largest K values, set everything else to 0
        # values: [batch, k], indices: [batch, k]
        values, indices = torch.topk(positive_acts, self.k, dim=-1)
        
        # Create a sparse tensor of zeros
        feature_acts = torch.zeros_like(positive_acts)
        # Scatter the top-k values back into the sparse tensor
        feature_acts.scatter_(-1, indices, values)
        
        # 5. Decode to reconstruct
        reconstruction = self.decoder(feature_acts) + self.b_dec
        
        return reconstruction, feature_acts

# Helper to extract activations
class ActivationCollector:
    def __init__(self):
        self.activations = []

    def hook(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.activations.append(hidden_states.detach())

def run_topk_sae_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Qwen-0.5B
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # 3. Collect activations
    texts = [
        "The capital of France is Paris.",
        "AI safety and mechanistic interpretability are crucial fields.",
        "Python is an excellent language for machine learning.",
        "Neural networks learn representations through optimization."
    ]
    
    collector = ActivationCollector()
    target_layer = model.model.layers[10].mlp
    handle = target_layer.register_forward_hook(collector.hook)
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
    handle.remove()

    activations = torch.cat([act.view(-1, act.shape[-1]) for act in collector.activations], dim=0).float()
    print(f"Collected activations shape: {activations.shape}")

    # 4. Instantiate Top-K SAE (setting k=5)
    d_model = activations.shape[-1]
    dict_size = d_model * 2 
    k_value = 5 # Only 5 active features per token allowed
    
    sae = TopKSparseAutoencoder(d_model, dict_size, k=k_value).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)

    print(f"\n--- Training Top-K SAE (k={k_value}) for 10 Steps ---")
    sae.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, feature_acts = sae(activations)
        
        # Loss is ONLY Reconstruction MSE (no L1 penalty!)
        loss = nn.MSELoss()(reconstructed, activations)
        
        loss.backward()
        optimizer.step()

        # Check sparsity (should be exactly K)
        l0_sparsity = (feature_acts > 1e-4).float().sum(dim=-1).mean().item()
        print(f"Step {step:02d} | Recon MSE Loss: {loss.item():.4f} | L0 Sparsity: {l0_sparsity:.1f} features/token")

    # 5. Evaluate final metrics
    sae.eval()
    with torch.no_grad():
        reconstructed, feature_acts = sae(activations)
        final_mse = nn.MSELoss()(reconstructed, activations).item()
        variance_explained = 1 - (final_mse / activations.var().item())
        print("\n--- Final Metrics ---")
        print(f"Final Reconstruction MSE: {final_mse:.4f}")
        print(f"Variance Explained: {variance_explained * 100:.2f}% (Should be positive and increasing!)")

if __name__ == "__main__":
    run_topk_sae_demo()
