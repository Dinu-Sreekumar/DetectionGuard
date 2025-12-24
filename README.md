# DetectionGuard 🛡️

A machine learning-based network intrusion detection system featuring a real-time GUI application for traffic monitoring and classification.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## Features

- **Real-time Traffic Monitoring** - Live packet capture and analysis using Scapy
- **ML-based Detection** - Neural network classifier trained on CICIDS2017 dataset
- **Modern GUI** - CustomTkinter interface with dark mode support
- **Simulation Mode** - Test the system with simulated attack traffic
- **Multi-class Detection** - Identifies DoS, DDoS, Port Scanning, and other attack types

## Project Structure

```
DetectionGuard/
├── src/
│   ├── DetectionGuard.py    # Main GUI application
│   └── the_intruder.py      # Attack simulation tool
├── training/
│   ├── train_for_demo.py    # Demo training script
│   └── train_for_research.py # Research training with metrics
├── assets/                   # Pre-trained model & encoders
├── research_paper/           # Research documentation
├── data/                     # Dataset folder (not included)
└── requirements.txt
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DetectionGuard.git
   cd DetectionGuard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses the **CICIDS2017 Cleaned and Preprocessed** dataset.

📥 **Download:** [CICIDS2017 Cleaned Dataset on Kaggle](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)

After downloading, place the `cicids2017_cleaned.csv` file in the `data/` directory:
```
data/
└── cicids2017_cleaned.csv
```

## Usage

### Run the GUI Application

```bash
cd src
python DetectionGuard.py
```

### Train a New Model

```bash
cd training
python train_for_demo.py       # Quick demo training
python train_for_research.py   # Full research training with metrics
```

### Simulate Attack Traffic

```bash
cd src
python the_intruder.py
```

## Model Architecture

- **Input:** 52 network flow features
- **Hidden Layers:** Dense(64) → Dropout(0.5) → Dense(32)
- **Output:** Binary classification (Normal/Attack)
- **Optimizer:** Adam with Binary Crossentropy loss

## Research

This project is part of research on federated learning-based intrusion detection systems. See the `research_paper/` directory for:
- Full research paper (PDF)
- Supplementary materials with detailed methodology

## Acknowledgments

- CICIDS2017 Dataset by the Canadian Institute for Cybersecurity
- Cleaned dataset preprocessed by [Eric Anacleto Ribeiro](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)
