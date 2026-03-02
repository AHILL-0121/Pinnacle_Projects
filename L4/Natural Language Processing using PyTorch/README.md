# 🐦 Comparative Sentiment Analysis — BERT vs LSTM vs GRU vs RNN (Project 12)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

*End-to-end NLP notebook that trains and rigorously compares four neural architectures — Vanilla RNN, LSTM, GRU, and fine-tuned BERT — on the Twitter Sentiment140 dataset, measuring accuracy, F1-score, ROC-AUC, training time, and peak memory consumption side-by-side.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture Comparison](#-architecture-comparison)
- [Model Definitions](#-model-definitions)
- [Workflow](#-workflow)
- [Hyperparameters](#-hyperparameters)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Deployment Recommendations](#-deployment-recommendations)
- [Quick Start](#-quick-start)
- [Dependencies](#-dependencies)
- [File Structure](#-file-structure)

---

## 🎯 Overview

This notebook addresses **binary sentiment classification** (Negative / Positive) on real-world Twitter data. Rather than training a single model, it trains all four architectures on identical data splits and compares them across seven dimensions — performance, speed, and resource cost.

| Question Answered | Answer Source |
|-------------------|---------------|
| How much does BERT improve over recurrent models? | Accuracy / F1 comparison table |
| What is the speed-accuracy trade-off for each model? | Training time vs. F1 scatter |
| Which model fits a resource-constrained edge device? | Peak memory + speed chart |
| Where do each model fail? | Normalised confusion matrices |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | Sentiment140 |
| **Source** | Stanford / Kaggle |
| **Total Records** | 1,600,000 tweets |
| **Sampled for Training** | 6,000 (balanced — 3,000 per class) |
| **Classes** | Negative (0), Positive (1) |
| **Split** | 70% Train / 15% Validation / 15% Test |
| **File** | `training.1600000.processed.noemoticon.csv` |

### Text Preprocessing

```
Raw tweet
    → Lowercase
    → Remove URLs  (http/www)
    → Remove @mentions
    → Remove #hashtags
    → Remove non-alphabetic characters & digits
    → Collapse whitespace
    → Filter empty strings after cleaning
```

> **Dataset is NOT bundled** (1.6 GB CSV). Download from  
> [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)  
> and place `training.1600000.processed.noemoticon.csv` in the same folder as the notebook.

---

## 🧠 Architecture Comparison

| Attribute | Vanilla RNN | LSTM | GRU | BERT |
|-----------|:-----------:|:----:|:---:|:----:|
| **Architecture** | Recurrent | Recurrent + gates | Recurrent + gates | Transformer |
| **Parameters** | ~2M | ~2M | ~2M | ~110M |
| **Context memory** | Short | Long | Long | Full bidirectional |
| **Vanishing gradient** | Susceptible | Gated (resistant) | Gated (resistant) | Attention (immune) |
| **Pre-training** | ✗ | ✗ | ✗ | ✓ (`bert-base-uncased`) |
| **GPU speed-up** | Moderate | Moderate | Moderate | Large |
| **Best use case** | Baseline | CPU production | Edge / mobile | High-accuracy server |

---

## 🏗️ Model Definitions

### RNN / LSTM / GRU (from scratch)

All three recurrent models share the same skeleton, differing only in the core cell type:

```
Embedding(vocab_size=10000, dim=64, padding_idx=0)
        │
        ▼
RNN / LSTM / GRU (hidden=128, layers=2, dropout=0.3)
        │  [uses last hidden state]
        ▼
Dropout(0.3)
        │
        ▼
Linear(128 → 2)
        │
        ▼
CrossEntropyLoss
```

| Hyperparameter | Value |
|----------------|-------|
| Vocab Size | 10,000 (top-K words) |
| Max Sequence Length | 50 tokens |
| Embedding Dim | 64 |
| Hidden Dim | 128 |
| RNN Layers | 2 |
| Dropout | 0.3 |
| Optimizer | Adam (lr = 1e-3) |
| Batch Size | 64 |
| Epochs | 5 |

### BERT (fine-tuned)

```
bert-base-uncased (110M params, pre-trained on BooksCorpus + English Wikipedia)
        │  [CLS token representation]
        ▼
Linear(768 → 2)   [classification head]
        │
        ▼
CrossEntropyLoss
```

| Hyperparameter | Value |
|----------------|-------|
| Base model | `bert-base-uncased` |
| Max token length | 64 (32 recommended on CPU-only) |
| Batch Size | 32 |
| Optimizer | AdamW (lr = 2e-5, weight_decay = 0.01) |
| Scheduler | Linear warmup (10% of steps) → linear decay |
| Gradient clipping | 1.0 |
| Epochs | 5 |

---

## 🔄 Workflow

```
training.1600000.processed.noemoticon.csv
        │
        ▼
1. Environment Check (Python, PyTorch, CUDA)
        │
        ▼
2. Configuration (paths, hyperparameters, SEED=42)
        │
        ▼
3. Load & Preprocess
   ├── Map polarity {0→0, 4→1}
   ├── Clean tweets (regex pipeline)
   └── Balance + sample (6,000 total)
        │
        ▼
4. Train/Val/Test Split  (70 / 15 / 15, stratified)
        │
        ▼
5. Vocabulary Build (top-10K words, <PAD>=0, <UNK>=1)
        │
        ▼
6. DataLoaders
   ├── TextDataset  → integer sequences (RNN/LSTM/GRU)
   └── BertDataset  → subword token IDs + attention masks
        │
        ├──────────────────────────────┐
        ▼                              ▼
7a. Train RNN / LSTM / GRU        7b. Train BERT
    (5 epochs each, Adam)             (5 epochs, AdamW + scheduler)
        │                              │
        └──────────────┬───────────────┘
                       ▼
8. Comparative Metrics Table (7 columns × 4 models)
        │
        ▼
9. Visualizations (5 chart files saved to outputs/)
        │
        ▼
10. Per-Class Classification Reports
        │
        ▼
11. Final Summary + Deployment Recommendations
```

---

## ⚙️ Hyperparameters

| Parameter | RNN / LSTM / GRU | BERT |
|-----------|:----------------:|:----:|
| Epochs | 5 | 5 |
| Batch size | 64 | 32 |
| Learning rate | 1e-3 | 2e-5 |
| Optimizer | Adam | AdamW |
| LR scheduler | None | Linear warmup |
| Max seq len | 50 | 64 |
| Dropout | 0.3 | (internal BERT) |
| Gradient clip | — | 1.0 |

---

## 📈 Evaluation Metrics

All metrics are computed on the held-out **test set** (never seen during training):

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP+TN}{N}$ | Overall fraction correct |
| **Precision** | $\frac{TP}{TP+FP}$ | Weighted across classes |
| **Recall** | $\frac{TP}{TP+FN}$ | Weighted across classes |
| **F1-Score** | $2\cdot\frac{P \times R}{P+R}$ | Weighted harmonic mean |
| **ROC-AUC** | Area under ROC curve | Ranking quality (binary: P(pos) used) |
| **Train Time** | seconds | Wall-clock training duration |
| **Peak Memory** | MB | `tracemalloc` peak traced memory |

---

## 🖼️ Visualizations

All plots are saved to the `outputs/` folder:

| File | Content |
|------|---------|
| `f1_comparison.png` | Bar chart — F1-score per model |
| `metrics_comparison.png` | Grouped bar — Accuracy, Precision, Recall, F1, ROC-AUC |
| `training_curves.png` | 2×4 grid — loss & accuracy curves (train vs. val) per model |
| `confusion_matrices.png` | Normalised confusion matrices for all four models |
| `computational_cost.png` | Training time and peak memory per model |
| `metrics_summary.csv` | Full metrics table exported as CSV |

---

## 🚀 Deployment Recommendations

| Scenario | Recommended Model | Rationale |
|----------|:-----------------:|-----------|
| **Production (accuracy critical)** | BERT | Highest accuracy/F1 via contextual embeddings |
| **Edge / mobile / low-latency** | GRU | Best speed-accuracy trade-off among recurrent models |
| **Rapid prototyping / CPU-only** | LSTM or GRU | Fast training, no GPU requirement |
| **Academic baseline** | Vanilla RNN | Simplest architecture, useful lower-bound |

---

## 🚀 Quick Start

### 1. Get the dataset

Download from Kaggle and place the CSV in the notebook folder:

```
Natural Language Processing using PyTorch/
├── sentiment_analysis_comparative.ipynb
├── requirements.txt
└── training.1600000.processed.noemoticon.csv   ← place here
```

### 2. Set up environment

```powershell
cd "L4/Natural Language Processing using PyTorch"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. (Optional) GPU / CUDA

For significantly faster BERT training, install the CUDA build of PyTorch.  
Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and replace the `torch` line in `requirements.txt` with the matching CUDA wheel URL before running `pip install`.

### 4. Run the notebook

Open `sentiment_analysis_comparative.ipynb` in VS Code or JupyterLab, select the `venv` kernel, then run **Kernel → Restart & Run All**.

> CPU-only estimated runtimes (6,000 sample, 5 epochs):  
> RNN ≈ 30s · LSTM ≈ 45s · GRU ≈ 40s · BERT ≈ 10–20 min

---

## 📦 Dependencies

```
torch
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

Full pinned versions in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

> **Python version:** 3.10+ recommended (3.9+ minimum).

---

## 📁 File Structure

```
Natural Language Processing using PyTorch/
├── Natural Language Processing using PyTorch/
│   └── sentiment_analysis_comparative.ipynb   # Main notebook (13 sections, 33 cells)
├── requirements.txt                           # Pinned dependencies
└── README.md                                  # This file
```

### Notebook Sections

| # | Section | Key Output |
|---|---------|-----------|
| 1 | Imports & Environment Check | Python/PyTorch/CUDA versions |
| 2 | Configuration | Paths + hyperparameters |
| 3 | Load & Preprocess Dataset | Balanced 6,000-sample dataframe |
| 4 | Vocabulary (RNN-family) | 10,000-token vocabulary dict |
| 5 | Dataset & DataLoader Classes | `TextDataset`, `BertDataset` |
| 6 | Model Definitions | `SentimentRNN`, `SentimentLSTM`, `SentimentGRU`, `BertSentiment` |
| 7 | Training & Evaluation Helpers | `train_rnn`, `eval_rnn`, `train_bert`, `eval_bert`, `compute_metrics` |
| 8 | Train RNN, LSTM & GRU | Per-epoch loss/accuracy logs |
| 9 | Train BERT | Fine-tuning with ETA progress |
| 10 | Comparative Metrics Table | 7-column summary DataFrame |
| 11 | Visualizations | 5 saved chart files |
| 12 | Per-Class Classification Reports | Precision/Recall/F1 per class |
| 13 | Final Summary & Deployment Recommendations | Best model + use-case guide |

---

## 📄 License & Author

**License:** MIT — Use freely for educational and portfolio purposes.

**Author:** AHILL S
