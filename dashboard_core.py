import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
from path_utils import MODELS

# Re-import Autoencoder class manually if not using external file 
class Autoencoder(nn.Module):
    def __init__(self, input_dim=29, bottleneck_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, bottleneck_dim), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(True),
            nn.Linear(32, 64), nn.ReLU(True),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def load_inference_artifacts():
    """Load model, scaler, and threshold"""
    model = Autoencoder()
    model_path = MODELS / "model.pkl"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    scaler = joblib.load(MODELS / "scaler.pkl")
    
    with open(MODELS / "threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
        
    return model, scaler, threshold

def analyze_transaction(v_features, amount):
    """
    v_features: list/array of 28 floats (V1-V28)
    amount: float
    """
    model, scaler, threshold = load_inference_artifacts()
    
    # Preprocess
    amount_log = np.log1p(amount)
    data = np.array(v_features + [amount_log]).reshape(1, -1)
    data_scaled = scaler.transform(data)
    
    # Inference
    with torch.no_grad():
        input_tensor = torch.FloatTensor(data_scaled)
        output_tensor, _ = model(input_tensor)
        
        # Reconstruction Error (MSE)
        mse = torch.mean((output_tensor - input_tensor)**2).item()
        
    is_fraud = mse > threshold
    return {
        "reconstruction_error": mse,
        "threshold": threshold,
        "verdict": "FRAUDULENT" if is_fraud else "NORMAL",
        "anomaly_score": (mse / threshold) * 100 # Multiplier for visual UI
    }
