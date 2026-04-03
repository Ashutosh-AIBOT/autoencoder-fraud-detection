---
title: Autoencoder Fraud Detection
emoji: 🔐
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# 🔐 Autoencoder Fraud Detection (Anomaly Detection)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

A professional-grade anomaly detection system using Autoencoders to identify fraudulent credit card transactions. This project implements an unsupervised learning approach where the model learns to reconstruct "normal" transactions and flags high reconstruction errors as anomalies.

---

## 📋 Project Overview

### 1. The Problem
Credit card fraud causes billions in losses annually. Traditional supervised methods struggle with highly imbalanced datasets (often < 0.1% fraud). This project uses an Autoencoder that learns the pattern of legitimate transactions and flags deviations (anomalies) based on reconstruction error.

### 2. The Dataset
* **Source**: Credit Card Fraud Detection (Kaggle / IEEE-CIS)
* **Features**: 29 numerical features (V1-V28 anonymized + Amount)
* **Target**: Binary classification (Legitimate/Fraud)
* **Challenge**: Extreme class imbalance (~0.17% fraud)

### 3. The Solution (Autoencoder Architecture)

#### Network Structure
```
Encoder: 29 → 64 → 32 → 16 (Bottleneck)
Decoder: 16 → 32 → 64 → 29
```

#### Key Components
* **Bottleneck Dimension**: 16 (learns compressed representation)
* **Activation**: ReLU for hidden layers
* **Dropout**: 0.1 for regularization
* **Loss**: Mean Squared Error (MSE)
* **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)

#### Anomaly Detection Strategy
1. Train only on **normal transactions** (non-fraud)
2. Set threshold using reconstruction error distribution
3. Flag transactions with error > threshold as anomalies

---

## 📂 Project Structure

```text
Autoencoder Anomaly Detection/
├── app.py                    # Streamlit Dashboard (UI)
├── dashboard_core.py         # Autoencoder model & inference
├── path_utils.py             # Dynamic path management
├── Dockerfile                # Containerization
├── requirements.txt          # Dependencies
├── notebooks/                # Training pipelines
│   ├── s01_eda.py            # Exploratory Data Analysis
│   ├── s02_preprocessing.py  # Data cleaning & scaling
│   ├── s03_train_evaluate.py # Model training & evaluation
│   └── s04_pipeline.py      # Master pipeline
├── models/                   # Trained weights
│   ├── autoencoder.pkl       # Model state dict
│   └── threshold.json        # Anomaly threshold
├── charts/                   # Visualizations
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── reconstruction_dist.png
└── data/                     # Credit card dataset
    └── raw/
```

---

## 📈 Numerical Results & Performance

The Autoencoder successfully identifies fraudulent transactions with high recall:

| Metric | Value | Interpretation |
|:-------|:------|:---------------|
| **Recall** | ~85-90% | High fraud detection rate |
| **Precision** | ~70-80% | Moderate false positive rate |
| **F1-Score** | ~75-82% | Balanced detection |
| **ROC-AUC** | ~90-95% | Excellent separation |
| **Threshold** | Dynamic | Based on reconstruction error distribution |

### Anomaly Detection Approach
- **Training**: Unsupervised, only on normal (non-fraud) samples
- **Inference**: Reconstruction error > threshold → Fraud
- **Threshold**: Set at 95-99th percentile of normal reconstruction errors

---

## 🚀 Live Deployment

### Dockerized Hosting (HF Spaces / Cloud)
This project is fully containerized for deployment:

1. **Build the Image**:
   ```bash
   docker build -t autoencoder-fraud-detector .
   ```
2. **Run the Dashboard**:
   ```bash
   docker run -p 8501:7860 autoencoder-fraud-detector
   ```

### Standard Local Execution
1. **Activate Environment**:
   ```bash
   conda activate ml-env
   ```
2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ GitHub Configuration & Workflow

```bash
# 1. Initialize & track with LFS
git init
git lfs install
git lfs track "models/*.pkl" "charts/**/*.png"

# 2. Deploy to GitHub
git add .
git commit -m "feat: Professional Autoencoder implementation for fraud detection"
git branch -M main
git remote add origin git@github.com:Ashutosh-AIBOT/autoencoder-fraud-detection.git
git push -u origin main
```

---

## 🎯 Dashboard Features

The Streamlit dashboard provides:
- **📊 Reconstruction Analysis**: View error distributions
- **🔍 Live Anomaly Detection**: Test transactions in real-time
- **📈 Performance Metrics**: ROC curves, Precision-Recall analysis
- **📉 Threshold Tuning**: Adjust sensitivity dynamically

---

**Developed by [Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)**
