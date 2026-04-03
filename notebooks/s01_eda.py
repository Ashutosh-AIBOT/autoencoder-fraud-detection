import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_RAW, CHARTS

def print_stop(stop_num, title, explanation, auto_mode=False):
    print(f"\n{'='*60}")
    print(f"STOP {stop_num}: {title}")
    print(f"{'='*60}")
    print(f"\nEXPLANATION:\n{explanation}")
    print(f"\n{'='*60}\n")
    if not auto_mode:
        input("Press Enter to continue to the next STOP...")

def stop_1_class_dist(df, auto_mode=False):
    counts = df['Class'].value_counts()
    normal = counts.get(0, 0)
    fraud = counts.get(1, 0)
    fraud_rate = (fraud / len(df)) * 100
    
    explanation = (
        f"Dataset loaded with {len(df):,} transactions.\n"
        f"Normal: {normal:,} | Fraud: {fraud:,} | Fraud Rate: {fraud_rate:.2f}%\n\n"
        "This extreme imbalance (99.83% normal) is the reason we use an Autoencoder.\n"
        "Instead of teaching the model to 'find fraud', we teach it to 'know normal'.\n"
        "Anything it doesn't recognize as normal is flagged as an anomaly."
    )
    print_stop(1, "Class Distribution & AE Justification", explanation, auto_mode)

def stop_2_feature_analysis(df, auto_mode=False):
    # Plot V14 distribution for normal vs fraud
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Class'] == 0], x='V14', label='Normal', fill=True)
    sns.kdeplot(data=df[df['Class'] == 1], x='V14', label='Fraud', fill=True)
    plt.title("V14 Feature Distribution (Normal vs Fraud)")
    plt.legend()
    plt.savefig(CHARTS / "v14_distribution.png")
    
    explanation = (
        "PCA components (V1-V28) often show clear separation for fraud in specific dimensions.\n"
        "For example, V14 is highly discriminative. The Autoencoder will learn these joint\n"
        "distributions to detect outliers across all 29 feature dimensions simultaneously."
    )
    print_stop(2, "Feature Discriminative Analysis", explanation, auto_mode)

def stop_3_bottleneck_concept(auto_mode=False):
    explanation = (
        "RECONSTRUCTION CONCEPT:\n"
        "1. Input (29 feats) → Encoder → Bottleneck (16 latent units) → Decoder → Output (29 feats).\n"
        "2. The 'Bottleneck' forces the model to compress information.\n"
        "3. Since 99.8% of data is 'normal', the model gets very good at reconstructing normal patterns.\n"
        "4. Fraudulent transactions have different patterns; when they pass through the bottleneck,\n"
        "   the Decoder fails to reconstruct them accurately, leading to a HIGH Reconstruction Error (MSE)."
    )
    print_stop(3, "The Information Bottleneck Principle", explanation, auto_mode)

def run_eda(auto_mode=False):
    print("Initializing Autoencoder EDA...")
    csv_path = DATA_RAW / "creditcard.csv"
    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    stop_1_class_dist(df, auto_mode)
    stop_2_feature_analysis(df, auto_mode)
    stop_3_bottleneck_concept(auto_mode)
    print("\n[EDA MODULE COMPLETE]")

if __name__ == "__main__":
    run_eda(auto_mode=False)
