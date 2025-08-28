import torch
import torch.optim as optim
import torch.nn as nn
from gcn import GCN

def train():
    # Simple Karate-club like synthetic node classification task
    num_nodes = 34
    in_features = 10
    hidden_features = 16
    num_classes = 2
    
    # Generate mock features and adjacency matrix
    torch.manual_seed(42)
    x = torch.randn(num_nodes, in_features)
    
    # Random adjacency with self-loops
    adj = (torch.rand(num_nodes, num_nodes) > 0.85).float()
    adj = (adj + adj.t() > 0).float() # Make symmetric
    
    # Mock labels (two communities)
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    model = GCN(in_features, hidden_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting mock training of GCN...")
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(x, adj)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.1f}%")

if __name__ == "__main__":
    train()
