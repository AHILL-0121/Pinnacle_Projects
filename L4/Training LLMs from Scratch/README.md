# 🔤 Building Custom Tokenizers from Scratch

### BPE Tokenizer Implementation for Language Model Training

[![Tokenizers](https://img.shields.io/badge/🤗%20Tokenizers-Latest-orange?style=flat)](https://github.com/huggingface/tokenizers)
[![Datasets](https://img.shields.io/badge/🤗%20Datasets-Latest-blue?style=flat)](https://huggingface.co/datasets)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![WikiText-2](https://img.shields.io/badge/Dataset-WikiText--2-green?style=flat)](https://huggingface.co/datasets/Salesforce/wikitext)

*Production-grade implementation of a Byte Pair Encoding (BPE) tokenizer built from scratch using the WikiText-2 dataset — the foundational step for training language models with proper text preprocessing, normalization, and vocabulary construction.*

---

## 🎯 Project Overview

This project teaches the **fundamental building blocks** of language model training by implementing a custom BPE tokenizer from the ground up. Understanding tokenization is crucial for anyone working with LLMs, as it directly impacts model performance, efficiency, and behavior.

### Key Learning Outcomes

- **BPE Algorithm**: Understand how Byte Pair Encoding compresses text into subwords
- **Tokenizer Architecture**: Build production-ready tokenizers with proper normalization
- **Vocabulary Construction**: Learn optimal vocab size and special token handling
- **Text Preprocessing**: Master dataset cleaning and preparation techniques
- **Subword Tokenization**: Bridge character-level and word-level representations

---

## 📊 Technical Specifications

### Tokenizer Configuration
| Component | Configuration |
|-----------|---------------|
| **Algorithm** | Byte Pair Encoding (BPE) |
| **Vocabulary Size** | 30,000 tokens |
| **Training Corpus** | WikiText-2 (~4.5M tokens) |
| **Normalization** | NFD → Lowercase → Strip Accents |
| **Pre-tokenization** | Whitespace splitting |
| **Special Tokens** | `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` |

### Dataset Statistics
| Metric | Value |
|--------|-------|
| **Source** | WikiText-2-v1 (Salesforce) |
| **Training Samples** | ~36,718 articles |
| **Average Length** | ~145 characters |
| **Total Characters** | ~11.8M |
| **Vocabulary Coverage** | 99.5%+ after BPE |

### Processing Pipeline
| Stage | Operations |
|-------|-----------|
| **Loading** | HuggingFace datasets integration |
| **Cleaning** | Remove `<unk>`, normalize headers, deduplicate |
| **Training** | BPE with minimum frequency filtering |
| **Post-processing** | Add `[CLS]`/`[SEP]` templates |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install datasets tokenizers transformers pandas
```

### Usage

1. **Open the Jupyter Notebook**: `bpe_tokenizer_wikitext2.ipynb`
2. **Run all cells sequentially** - each section builds on the previous
3. **Examine outputs** - vocabulary, merge operations, token distributions
4. **Export tokenizer** - save for use in downstream models

### Key Sections

```python
# 1. Dataset Loading & Exploration
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

# 2. Text Cleaning & Preprocessing  
cleaned_data = preprocess_corpus(dataset)

# 3. BPE Tokenizer Training
tokenizer = train_bpe_tokenizer(cleaned_data, vocab_size=30000)

# 4. Evaluation & Analysis
analyze_tokenizer_performance(tokenizer)

# 5. Export & Save
tokenizer.save("custom_bpe_tokenizer.json")
```

---

## 💡 BPE Algorithm Deep Dive

### Byte Pair Encoding Process

```
Initial Vocabulary: Individual characters
└── "h", "e", "l", "o", " ", "w", "r", "d"

Iteration 1: Most frequent pair = ("l", "o") 
└── Merge → "lo"
└── New vocab: "h", "e", "lo", " ", "w", "r", "d"

Iteration 2: Most frequent pair = ("h", "e")
└── Merge → "he" 
└── New vocab: "he", "lo", " ", "w", "r", "d"

...continue until target vocabulary size...

Final: Subword units optimized for frequency
└── "hello", "world", "the", "ing", etc.
```

### Subword Benefits

| Challenge | BPE Solution |
|-----------|--------------|
| **OOV Words** | Breaks unknown words into known subwords |
| **Morphology** | Captures prefixes/suffixes naturally |
| **Rare Words** | Represents efficiently without explosion |
| **Multiple Languages** | Single vocab handles multilingual text |
| **Model Size** | Compact representation vs. word-level |

---

## 📈 Tokenizer Architecture

### Full Pipeline Implementation

```
Raw Text: "Hello world! This is tokenization."
    │
    ▼
┌─────────────────────────────────────┐
│         Normalization               │  NFD → Lowercase → Strip Accents
│  "Hello world! This is tokenization." │
│           ↓                         │
│  "hello world! this is tokenization." │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       Pre-tokenization              │  Whitespace splitting
│  ["hello", "world!", "this", "is", "tokenization."] │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         BPE Encoding                │  Subword segmentation
│  ["hel", "lo", "world", "!", "th", "is", "tok", "en", "ization", "."] │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Post-processing                │  Add special tokens
│  ["[CLS]", "hel", "lo", "world", "!", "th", "is", "tok", "en", "ization", ".", "[SEP]"] │
└─────────────────────────────────────┘
    │
    ▼
Token IDs: [2, 1547, 2615, 8291, 15, 4832, 374, 10616, 7559, 2065, 18, 3]
```

### Special Token Functions

| Token | Purpose | Position |
|-------|---------|----------|
| `[PAD]` | Sequence padding | Variable |
| `[UNK]` | Out-of-vocabulary | Any |
| `[CLS]` | Classification/start | Beginning |
| `[SEP]` | Separator/end | End/between |
| `[MASK]` | Masked language modeling | Any |

---

## 🔬 Analysis & Evaluation

### Vocabulary Distribution

The notebook provides comprehensive analysis:

```python
# Token frequency analysis
plot_token_frequency_distribution()

# Subword length statistics  
analyze_subword_lengths()

# Coverage on validation set
calculate_vocabulary_coverage()

# Compression ratio
measure_text_compression()
```

### Key Metrics

| Metric | Expected Value | Significance |
|--------|----------------|--------------|
| **Vocab Size** | 30,000 | Balance between efficiency and coverage |
| **Avg Subword Length** | 3.2-4.1 chars | Optimal subword granularity |
| **UNK Rate** | <0.5% | Out-of-vocabulary handling |
| **Compression Ratio** | 3.2:1 | Space efficiency vs. characters |

### Performance Evaluation

```python
# Test tokenization on sample texts
sample_texts = [
    "Artificial intelligence and machine learning",
    "Transformers revolutionized natural language processing", 
    "Subword tokenization enables efficient vocabulary",
]

for text in sample_texts:
    tokens = tokenizer.encode(text).tokens
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)}")
```

---

## 🎛️ Customization Options

### Vocabulary Size Tuning

```python
# Small vocab (faster training, less precision)
vocab_size = 10000

# Large vocab (slower training, more precision)  
vocab_size = 50000

# Production standard
vocab_size = 30000  # Sweet spot for most applications
```

### Alternative Datasets

```python
# Other text corpora
dataset = load_dataset("openwebtext")        # Web text
dataset = load_dataset("bookcorpus")         # Books
dataset = load_dataset("cc_news")            # News articles
dataset = load_dataset("oscar", "en")        # Common Crawl
```

### Normalization Strategies

```python
# Minimal normalization (preserve casing)
tokenizer.normalizer = None

# Aggressive normalization
tokenizer.normalizer = NormSequence([
    NFD(), Lowercase(), StripAccents(), 
    Replace(Regex(r'[^\w\s]'), ' ')  # Remove punctuation
])
```

---

## 🚨 Troubleshooting

### Common Issues

**Slow Training**
```python
# Reduce dataset size for experimentation
small_dataset = dataset['train'].select(range(10000))
```

**Memory Issues**
```python
# Process in batches
def batch_iterator(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
```

**Poor Tokenization Quality**
```python
# Increase minimum frequency threshold
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=5,  # Higher threshold
    special_tokens=special_tokens
)
```

### Dataset Quality Issues

| Problem | Solution |
|---------|----------|
| Too many duplicates | Increase deduplication aggressiveness |
| Poor text quality | Add more cleaning rules |
| Imbalanced domains | Sample from multiple datasets |
| Encoding issues | Normalize Unicode properly |

---

## 🎓 Learning Path Integration

### Prerequisites
- Basic understanding of **text processing** concepts
- Familiarity with **Python** and **Jupyter notebooks**
- Knowledge of **machine learning** fundamentals

### Next Steps
1. **Use your tokenizer** in the "Finetuning LLMs" project
2. **Experiment** with different vocabulary sizes and datasets  
3. **Compare** with existing tokenizers (GPT-2, T5, etc.)
4. **Build** a simple language model using your tokenizer
5. **Explore** SentencePiece and other tokenization algorithms

### Advanced Extensions

```python
# Train domain-specific tokenizers
medical_tokenizer = train_on_medical_texts()
code_tokenizer = train_on_programming_languages()

# Multi-lingual tokenization
multilingual_tokenizer = train_on_multiple_languages()

# Byte-level BPE (like GPT-2)
byte_level_tokenizer = ByteLevelBPETokenizer()
```

---

## 📚 References & Further Reading

### Core Papers
- [BPE Paper](https://arxiv.org/abs/1508.07909) - "Neural Machine Translation of Rare Words with Subword Units"
- [WordPiece](https://arxiv.org/abs/1609.08144) - "Google's Neural Machine Translation System"
- [SentencePiece](https://arxiv.org/abs/1808.06226) - "Unsupervised Text Tokenizer for Neural Text Processing"

### Implementation Guides
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) - Fast tokenizer library
- [OpenAI BPE](https://github.com/openai/gpt-2) - GPT-2 tokenizer implementation
- [Google SentencePiece](https://github.com/google/sentencepiece) - Subword tokenization

### Dataset Resources
- [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) - Original dataset
- [OpenWebText](https://huggingface.co/datasets/openwebtext) - Larger corpus
- [The Pile](https://pile.eleuther.ai/) - Diverse text collection

---

## 💼 Production Considerations

### Deployment Checklist

- [ ] **Save tokenizer** with `tokenizer.save("path/to/tokenizer.json")`
- [ ] **Validate coverage** on target domain texts
- [ ] **Benchmark speed** for your expected throughput
- [ ] **Test edge cases** (empty strings, very long texts)
- [ ] **Version control** tokenizer artifacts
- [ ] **Document** preprocessing assumptions

### Integration Examples

```python
# Load saved tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("custom_bpe_tokenizer.json")

# Use with Transformers
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

---

## 📄 License & Usage

This project is part of the Pinnacle Projects educational portfolio. The custom tokenizer implementation can be freely used in both research and commercial applications.

**Recommended Next**: After building your tokenizer, proceed to the "Finetuning LLMs" project to see how tokenization integrates with model training pipelines.