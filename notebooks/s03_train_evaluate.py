import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, roc_auc_score, auc
from sklearn.manifold import TSNE
import time
import json
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import MODELS, CHARTS

class Autoencoder(nn.Module):
    def __init__(self, input_dim=29, bottleneck_dim=16):
        super(Autoencoder, self).__init__()
        
        # Encoder: 29 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU(True)
        )
        
        # Decoder: 16 -> 32 -> 64 -> 29
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def stop_7_architecture(input_dim=29, bottleneck_dim=16):
    print("\n[STOP 7] Initializing Autoencoder Architecture (Bottleneck=16)...")
    model = Autoencoder(input_dim, bottleneck_dim)
    print(f"Network Initialized: Encoder(29->{bottleneck_dim}) & Decoder({bottleneck_dim}->29)")
    return model

def stop_9_training(model, train_loader, epochs=50, device="cpu"):
    print("\n[STOP 9] Starting Unsupervised Training (Normal Only)...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    model.to(device)
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for data in progress_bar:
            inputs = data.to(device)
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Reconstruction Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), MODELS / "model.pkl")
    return train_losses

def stop_10_threshold_tuning(model, val_loader, device="cpu"):
    print("\n[STOP 10] Tuning Reconstruction Error Threshold on Validation Set...")
    model.eval()
    errors = []
    labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            outputs, _ = model(data)
            
            # Per-sample MSE
            mse = torch.mean((outputs - data)**2, dim=1).cpu().numpy()
            errors.extend(mse)
            labels.extend(target.cpu().numpy())
            
    # Try different thresholds to find max F1
    precisions, recalls, thresholds = precision_recall_curve(labels, errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Best Threshold (F1 Max): {best_threshold:.6f}")
    
    # Save threshold
    with open(MODELS / "threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold)}, f)
        
    return best_threshold

def stop_12_latent_viz(model, test_X, test_y, device="cpu"):
    print("\n[STOP 12] Visualizing Latent Space (t-SNE Projection)...")
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        inputs = torch.FloatTensor(test_X).to(device)
        _, latent = model(inputs)
    
    z = latent.cpu().numpy()
    
    # Use PCA followed by t-SNE for speed on large datasets
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    z_2d = tsne.fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[test_y == 0, 0], z_2d[test_y == 0, 1], label='Normal', alpha=0.3, s=5, c='blue')
    plt.scatter(z_2d[test_y == 1, 0], z_2d[test_y == 1, 1], label='Fraud', alpha=0.8, s=15, c='red')
    plt.title("Latent Space Projection (t-SNE)")
    plt.legend()
    plt.savefig(CHARTS / "latent_space.png")
    print(f"Latent viz saved to {CHARTS}/latent_space.png")

def run_training_pipeline(train_loader, val_loader, test_data, epochs=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = stop_7_architecture()
    losses = stop_9_training(model, train_loader, epochs=epochs, device=device)
    threshold = stop_10_threshold_tuning(model, val_loader, device=device)
    
    test_X, test_y = test_data
    stop_12_latent_viz(model, test_X, test_y, device=device)
    
    print("\n[STOP 13] Autoencoder Model Ready for Inference.")
    print("\n[TRAINING & EVAL MODULE COMPLETE]")
    return threshold

if __name__ == "__main__":
    # Mock loaders call for testing
    pass
