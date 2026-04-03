import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from notebooks.s01_eda import run_eda
from notebooks.s02_preprocessing import run_preprocessing
from notebooks.s03_train_evaluate import run_training_pipeline

def run_full_ae_pipeline(epochs=5, auto_mode=False):
    print("\n" + "="*60)
    print("      AUTOENCODER ANOMALY DETECTION MASTER PIPELINE       ")
    print("="*60 + "\n")
    
    # 1. EDA Module
    run_eda(auto_mode=auto_mode)
    
    # 2. Preprocessing Module
    train_loader, val_loader, test_loader, val_data, test_data = run_preprocessing(auto_mode=auto_mode)
    
    # 3. Training & Evaluation Module
    print(f"\n[PIPELINE] Starting Unsupervised Training ({epochs} epochs)...")
    threshold = run_training_pipeline(train_loader, val_loader, test_data, epochs=epochs)
    
    print("\n" + "="*60)
    print("         AUTOENCODER PIPELINE EXECUTION COMPLETE           ")
    print("="*60 + "\n")
    print(f"Optimal Threshold: {threshold:.6f}")
    print("Next Step: Launch Dashboard via 'streamlit run app.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Anomaly Pipeline Runner")
    parser.add_argument("--auto", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    run_full_ae_pipeline(epochs=args.epochs, auto_mode=args.auto)
