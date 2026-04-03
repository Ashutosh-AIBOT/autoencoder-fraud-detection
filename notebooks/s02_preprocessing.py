import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_RAW, DATA_PROCESSED, MODELS

def stop_4_isolation(df, auto_mode=False):
    print("\n[STOP 4] Isolating Normal Transactions for Training...")
    # Training: 80% of normal transactions
    # Validation: 10% normal + some fraud
    # Test: 10% normal + remaining fraud
    
    normal_df = df[df['Class'] == 0]
    fraud_df = df[df['Class'] == 1]
    
    # Split normal data
    train_norm, test_norm = train_test_split(normal_df, test_size=0.2, random_state=42)
    val_norm, test_norm = train_test_split(test_norm, test_size=0.5, random_state=42)
    
    # Split fraud data for validation and test 
    val_fraud, test_fraud = train_test_split(fraud_df, test_size=0.5, random_state=42)
    
    # Combine
    train_normal_only = train_norm.drop('Class', axis=1)
    val_data = pd.concat([val_norm, val_fraud])
    test_data = pd.concat([test_norm, test_fraud])
    
    print(f"Train (Normal Only): {len(train_normal_only):,}")
    print(f"Val (Mix): {len(val_data):,}")
    print(f"Test (Mix): {len(test_data):,}")
    
    return train_normal_only, val_data, test_data

def stop_5_scaling(train_df, val_df, test_df, auto_mode=False):
    print("\n[STOP 5] Performing Log-Transform and Feature Scaling...")
    # Drop Time
    train_df = train_df.drop('Time', axis=1)
    val_X = val_df.drop(['Time', 'Class'], axis=1)
    test_X = test_df.drop(['Time', 'Class'], axis=1)
    
    # Log transform Amount (handles skewness)
    train_df['Amount'] = np.log1p(train_df['Amount'])
    val_X['Amount'] = np.log1p(val_X['Amount'])
    test_X['Amount'] = np.log1p(test_X['Amount'])
    
    # Fit scaler ONLY on normal train data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)
    
    # Save Scaler
    joblib.dump(scaler, MODELS / "scaler.pkl")
    print(f"Scaler saved to {MODELS}/scaler.pkl")
    
    return train_scaled, val_scaled, val_df['Class'].values, test_scaled, test_df['Class'].values

def stop_6_dataloaders(train_X, val_X, val_y, test_X, test_y, batch_size=256):
    print("\n[STOP 6] Creating PyTorch DataLoaders (Unsupervised Style)...")
    
    # Tensors
    train_tensor = torch.FloatTensor(train_X)
    val_tensor = torch.FloatTensor(val_X)
    test_tensor = torch.FloatTensor(test_X)
    
    # Labels for val/test (to tune threshold)
    val_y_tensor = torch.FloatTensor(val_y)
    test_y_tensor = torch.FloatTensor(test_y)
    
    # DataLoaders (Train has ONLY X)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_y_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_tensor, test_y_tensor), batch_size=batch_size)
    
    print(f"Batch Size: {batch_size} | Training Batches: {len(train_loader)}")
    return train_loader, val_loader, test_loader

def run_preprocessing(auto_mode=False):
    print("Initializng Autoencoder Preprocessing Pipeline...")
    df = pd.read_csv(DATA_RAW / "creditcard.csv")
    
    train_norm, val_data, test_data = stop_4_isolation(df, auto_mode)
    train_X, val_X, val_y, test_X, test_y = stop_5_scaling(train_norm, val_data, test_data, auto_mode)
    train_loader, val_loader, test_loader = stop_6_dataloaders(train_X, val_X, val_y, test_X, test_y)
    
    print("\n[PREPROCESSING MODULE COMPLETE]")
    return train_loader, val_loader, test_loader, (val_X, val_y), (test_X, test_y)

if __name__ == "__main__":
    run_preprocessing(auto_mode=False)
