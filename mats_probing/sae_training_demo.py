import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the Sparse Autoencoder (SAE) in PyTorch
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dictionary_size):
        super().__init__()
        # Encoder: projects activation to a higher dimension
        self.encoder = nn.Linear(d_model, dictionary_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        
        # Decoder: projects back to the original model dimension
        self.decoder = nn.Linear(dictionary_size, d_model, bias=False)
        # Initialize decoder weights close to encoder weights transpose
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        
        # Pre-encoder bias
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Center the input
        x_centered = x - self.b_dec
        
        # Encode: TopHalf activations = ReLU(x * W_enc + b_enc)
        hidden_pre_act = self.encoder(x_centered) + self.encoder_bias
        feature_acts = torch.relu(hidden_pre_act)
        
        # Decode: reconstruct = hidden * W_dec + b_dec
        reconstruction = self.decoder(feature_acts) + self.b_dec
        
        return reconstruction, feature_acts

# Helper to extract MLP activations
class ActivationCollector:
    def __init__(self):
        self.activations = []

    def hook(self, module, input, output):
        # Qwen MLP layer output is the first element
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.activations.append(hidden_states.detach())

def run_sae_training_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Qwen-0.5B to collect activations
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # 3. Collect activations from a few sample sentences
    texts = [
        "The capital of France is Paris.",
        "AI safety and mechanistic interpretability are crucial fields.",
        "Python is an excellent language for machine learning.",
        "Neural networks learn representations through optimization."
    ]
    
    collector = ActivationCollector()
    # Let's target the MLP output of Layer 10
    target_layer = model.model.layers[10].mlp
    handle = target_layer.register_forward_hook(collector.hook)
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
    handle.remove()

    # Concatenate all token activations: Shape [Num Tokens, Hidden Dim]
    # Qwen-0.5B MLP output dimension is 896
    activations = torch.cat([act.view(-1, act.shape[-1]) for act in collector.activations], dim=0).float()
    print(f"Collected activations tensor shape: {activations.shape}")

    # 4. Instantiate SAE
    d_model = activations.shape[-1]
    # Standard dictionary size expansion is 4x to 16x. We use 2x (1792) for a tiny demo.
    dict_size = d_model * 2 
    sae = SparseAutoencoder(d_model, dict_size).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)

    # L1 penalty scaling factor (higher sparsity penalty = fewer features fire)
    l1_coefficient = 0.05

    print("\n--- Starting Toy Training Run (10 Steps) ---")
    sae.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        
        # Forward pass through SAE
        reconstructed, feature_acts = sae(activations)
        
        # Loss 1: Reconstruction MSE loss
        recon_loss = nn.MSELoss()(reconstructed, activations)
        
        # Loss 2: L1 Sparsity loss (sum of absolute values of activations)
        l1_loss = feature_acts.abs().sum(dim=-1).mean()
        
        # Total Loss
        total_loss = recon_loss + l1_coefficient * l1_loss
        
        total_loss.backward()
        optimizer.step()

        # Calculate metrics
        # L0 = average number of non-zero features per token
        l0_sparsity = (feature_acts > 1e-4).float().sum(dim=-1).mean().item()
        
        print(f"Step {step:02d} | Total Loss: {total_loss.item():.4f} | Recon MSE: {recon_loss.item():.4f} | L0 Sparsity: {l0_sparsity:.1f} features/token")

    # 5. Evaluate the final reconstruction quality
    sae.eval()
    with torch.no_grad():
        reconstructed, feature_acts = sae(activations)
        final_mse = nn.MSELoss()(reconstructed, activations).item()
        variance_explained = 1 - (final_mse / activations.var().item())
        print("\n--- Final Metrics ---")
        print(f"Final Reconstruction MSE: {final_mse:.4f}")
        print(f"Variance Explained: {variance_explained * 100:.2f}% (Higher is better, meaning closer to original model performance)")

if __name__ == "__main__":
    run_sae_training_demo()
