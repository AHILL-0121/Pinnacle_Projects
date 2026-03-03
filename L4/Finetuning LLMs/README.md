# 🔧 Parameter-Efficient Fine-Tuning with QLoRA

### Memory-Optimized BERT Fine-Tuning for Local GPUs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow?style=flat)](https://huggingface.co/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.10+-blue?style=flat)](https://github.com/huggingface/peft)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)](https://wandb.ai)

*Production-grade implementation of QLoRA (Quantized Low-Rank Adaptation) for fine-tuning BERT on sentiment analysis using 4-bit quantization and LoRA adapters — optimized for local GPU training with comprehensive experiment tracking.*

---

## 🎯 Project Overview

This project demonstrates **parameter-efficient fine-tuning** using QLoRA, combining 4-bit quantization with Low-Rank Adaptation to fine-tune BERT for sentiment classification while using **90%+ less GPU memory** than traditional fine-tuning approaches.

### Key Innovations

- **QLoRA Architecture**: 4-bit NF4 quantization + LoRA adapters
- **Memory Optimization**: Trains on 4GB+ GPUs (adjustable batch sizes)
- **Parameter Efficiency**: Only ~0.16% of parameters are trainable
- **Production Ready**: Comprehensive logging and experiment tracking

---

## 📊 Technical Specifications

### Model Architecture
| Component | Configuration |
|-----------|---------------|
| **Base Model** | `bert-base-uncased` (110M parameters) |
| **Quantization** | 4-bit NF4 with double quantization |
| **LoRA Rank** | 16 (configurable) |
| **LoRA Alpha** | 32 |
| **Target Modules** | Query & Value attention layers |
| **Trainable Parameters** | ~177K (0.16% of total) |

### Dataset & Task
| Aspect | Detail |
|--------|--------|
| **Dataset** | IMDB Sentiment (20K samples) |
| **Task** | Binary sentiment classification |
| **Classes** | Positive, Negative |
| **Max Sequence Length** | 256 tokens |
| **Train/Test Split** | 90/10 |

### Memory Requirements
| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 4GB      | 8 |
| 6GB      | 16 |
| 8GB+     | 32 |

---

## 🚀 Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **CUDA Drivers** installed (`nvidia-smi` working)
3. **Python 3.8+**
4. **Weights & Biases account** (free at [wandb.ai](https://wandb.ai))

### Installation & Setup

```bash
# Clone and navigate
git clone <repository-url>
cd "L4/Finetuning LLMs"

# Install dependencies (automatic in notebook)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 datasets peft>=0.10.0 bitsandbytes>=0.43.0 
pip install accelerate>=0.29.0 scikit-learn wandb matplotlib
```

### Configuration

1. **Get W&B API Key**: Sign up at [wandb.ai](https://wandb.ai) → Settings → API Keys
2. **Update Notebook**: Replace `"PASTE_YOUR_KEY_HERE"` with your actual key
3. **Adjust Settings**: Modify batch size based on your GPU memory

### Training Pipeline

The notebook is structured for step-by-step execution:

```python
# 1. Install dependencies and restart kernel
# 2. GPU check and imports
# 3. Configure hyperparameters
# 4. Setup W&B logging
# 5. Load and preprocess IMDB dataset
# 6. Setup quantized model with LoRA
# 7. Train with automatic evaluation
# 8. Evaluate and export results
```

---

## 💡 Architecture Deep Dive

### QLoRA Implementation

```
┌─────────────────────────────────────┐
│           Original BERT             │
│        (110M parameters)            │
│                                     │
│  ┌─────────────────────────────┐   │
│  │     4-bit NF4 Quantization  │   │  ← Base model frozen
│  │     (Reduces memory 4x)     │   │
│  └─────────────────────────────┘   │
│                 │                   │
│                 ▼                   │
│  ┌─────────────────────────────┐   │
│  │       LoRA Adapters         │   │  ← Only these train
│  │     (177K parameters)       │   │     (~0.16% of total)
│  │                             │   │
│  │  Query LoRA: W_q + α·B·A    │   │
│  │  Value LoRA: W_v + α·B·A    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Memory Efficiency Comparison

| Method | GPU Memory | Trainable Params | Performance |
|--------|------------|------------------|-------------|
| **Full Fine-tuning** | ~16GB | 110M (100%) | Baseline |
| **LoRA** | ~8GB | 177K (0.16%) | ~98% of baseline |
| **QLoRA** | ~4GB | 177K (0.16%) | ~97% of baseline |

### Training Configuration

```python
# Optimized for local GPU training
BATCH_SIZE = 16     # Adjust based on VRAM
GRAD_ACCUM = 2      # Effective batch = 32
LEARNING_RATE = 2e-4
LORA_R = 16         # Rank (higher = more capacity)
LORA_ALPHA = 32     # Scaling factor
```

---

## 📈 Expected Results

After 3 epochs of training, you should expect:

| Metric | Value |
|--------|-------|
| **Training Loss** | ~0.15-0.25 |
| **Validation Accuracy** | ~92-95% |
| **F1 Score** | ~0.92-0.95 |
| **Training Time** | ~10-15 minutes (RTX 3080) |

### W&B Experiment Tracking

The notebook automatically logs:
- Training/validation loss curves
- Accuracy and F1 score progression
- GPU utilization metrics
- Hyperparameter configuration
- Model parameter counts

Access your results at [wandb.ai](https://wandb.ai/your-username/qlora-bert-sentiment)

---

## 🛠️ Customization Options

### Different Models
```python
MODEL_NAME = "bert-large-uncased"    # For better accuracy
MODEL_NAME = "roberta-base"          # Alternative architecture
MODEL_NAME = "distilbert-base-uncased"  # Faster training
```

### Different Datasets
```python
# Replace with any HuggingFace sentiment dataset
dataset = load_dataset("your-dataset-name")
TEXT_COL = "your_text_column"
LABEL_COL = "your_label_column"
```

### LoRA Configuration
```python
LORA_R = 8          # Lower rank = fewer parameters
LORA_R = 32         # Higher rank = more capacity
LORA_ALPHA = 16     # Lower scaling
LORA_ALPHA = 64     # Higher scaling
```

---

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 4
GRAD_ACCUM = 4  # Keep effective batch size
```

**Slow Training**
```python
# Enable gradient checkpointing
training_args.gradient_checkpointing = True
```

**W&B Login Issues**
```python
# Manual login
import wandb
wandb.login()  # Follow prompts
```

### GPU Requirements

| Issue | Solution |
|-------|----------|
| No CUDA detected | Install NVIDIA drivers + CUDA toolkit |
| Out of memory | Reduce `BATCH_SIZE` to 4 or 8 |
| Slow training | Use mixed precision: `fp16=True` |

---

## 🎓 Learning Objectives

Upon completion, you will understand:

1. **Parameter-Efficient Fine-Tuning** concepts and benefits
2. **Quantization techniques** for memory optimization
3. **LoRA methodology** and implementation details
4. **Experiment tracking** with Weights & Biases
5. **Production deployment** considerations for fine-tuned models

---

## 📚 References & Further Reading

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - "QLoRA: Efficient Finetuning of Quantized LLMs"
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - "LoRA: Low-Rank Adaptation of Large Language Models"
- [PEFT Documentation](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization library

---

## 📄 License & Usage

This project is part of the Pinnacle Projects educational portfolio. Feel free to adapt and extend for your own experiments and commercial applications.

**Next Steps**: After mastering QLoRA, explore training language models from scratch in the neighboring "Training LLMs from Scratch" project.