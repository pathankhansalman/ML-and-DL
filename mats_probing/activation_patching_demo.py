import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# --- 1. DEFINE A SIMPLE TOY SEQUENCE MODEL ---
# This model simulates two layers processing sequence tokens of dimension 4.
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        # Final output layer mapping back to 1 scalar logit per token
        self.unembed = nn.Linear(4, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        h1 = torch.relu(self.layer1(x))
        h2 = torch.relu(self.layer2(h1))
        logits = self.unembed(h2)
        return logits

# Initialize the model and put it in evaluation mode
model = ToyModel()
model.eval()

# --- 2. DEFINE INPUTS ---
# We have a Clean prompt (the task we want) and a Corrupted prompt (the control)
# Sequence length = 3 tokens, Embedding dimension = 4
clean_input = torch.randn(1, 3, 4)
corrupted_input = torch.randn(1, 3, 4)

print("--- Step 1: Running clean and corrupted baselines ---")
with torch.no_grad():
    clean_logits = model(clean_input)
    corrupted_logits = model(corrupted_input)

print(f"Clean model output at final token: {clean_logits[0, -1].item():.4f}")
print(f"Corrupted model output at final token: {corrupted_logits[0, -1].item():.4f}")
print("-" * 50)

# --- 3. CACHE CORRUPTED ACTIVATIONS ---
# We want to patch the output of layer1. 
# Let's save the activations of layer1 from the CORRUPTED run.
corrupted_activations = {}

def cache_hook(module, inputs, output):
    # Save a copy of the activations
    corrupted_activations["layer1"] = output.clone()
    return output

# Temporarily register hook to capture corrupted activations
cache_handle = model.layer1.register_forward_hook(cache_hook)
with torch.no_grad():
    _ = model(corrupted_input)
cache_handle.remove()  # Remove it immediately so it doesn't affect future runs

# --- 4. PREPARE THE PATCHING HOOK ---
# We want to patch only the activation at Token Index 1
token_to_patch = 1
saved_corrupted_vector = corrupted_activations["layer1"][0, token_to_patch]

def patch_hook(module, inputs, output):
    # output shape: [batch, seq_len, dim] -> [1, 3, 4]
    # Overwrite only the vector at token_to_patch with the corrupted vector
    patched_output = output.clone()
    patched_output[0, token_to_patch] = saved_corrupted_vector
    return patched_output

# --- 5. RUN THE PATCHED CLEAN RUN ---
# Register the patch hook
patch_handle = model.layer1.register_forward_hook(patch_hook)

with torch.no_grad():
    patched_logits = model(clean_input)

# Clean up hook
patch_handle.remove()

print("--- Step 2: Running the patched model ---")
print(f"Patched model output at final token: {patched_logits[0, -1].item():.4f}")

# Calculate how much we damaged the model (0% means like Clean, 100% means like Corrupted)
clean_val = clean_logits[0, -1].item()
corr_val = corrupted_logits[0, -1].item()
patched_val = patched_logits[0, -1].item()

damage_effect = (clean_val - patched_val) / (clean_val - corr_val) * 100
print(f"\nDamage effect: {damage_effect:.2f}%")
if damage_effect > 50:
    print("Result: This component is highly causal for the prediction!")
else:
    print("Result: This component has low causal effect for the prediction.")
print("-" * 50)
