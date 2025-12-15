# Scaling Laws for Language Models on Symbolic Music Data

**Course**: CS-GY 6923 Machine Learning  
**Author**: Pranjal Mishra (pm4084)  
**Institution**: New York University  
**Date**: December 2025

---

## Abstract

This project investigates whether neural scaling laws, originally discovered for natural language models, extend to symbolic music generation. We trained 9 models (5 GPT transformers and 4 LSTMs) ranging from 1.7M to 92.9M parameters on ABC notation music data. Through power law fitting, we determined a scaling exponent of α ≈ 0.19, indicating that model performance scales predictably with size. Our best model achieved a validation loss of 0.2886 (perplexity 1.33) on 100M tokens of training data.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Models Trained | 9 (5 GPT + 4 LSTM) |
| Parameter Range | 1.7M – 92.9M |
| Scaling Exponent (α) | 0.19 |
| Best Model | gpt_large (92.9M params) |
| Best Validation Loss | 0.2886 |
| Best Perplexity | 1.33 |
| Training Time (A100) | 30.5 minutes |

---

## Repository Structure

```
COMPREHENSIVE_SUBMISSION/
├── NOTEBOOKS/
│   └── A100_Gpt_large.ipynb           # Main training notebook (with outputs)
│
├── REPORT/
│   ├── Final_Report.pdf               # Compiled LaTeX report
│   └── main.tex                       # LaTeX source
│
├── RESULTS/
│   ├── scaling_plot_final.png         # Log-log scaling plot
│   ├── training_curves.png            # Loss vs. iterations
│   └── A100_TRAINING_RESULTS.md       # Detailed training metrics
│
├── SAMPLES/
│   ├── generated_samples/             # 10 ABC notation files
│   └── midi_files/                    # 10 MIDI conversions
│
├── DOCUMENTATION/
│   └── EXECUTION_PROOF.md             # Training verification details
│
└── README.md                          # This file
```

---

## Methodology

### Data Pipeline

1. **Source**: Lakh MIDI Dataset (178,561 MIDI files)
2. **Conversion**: MIDI → ABC notation using `midi2abc`
3. **Deduplication**: SHA-256 hashing removed 3,572 duplicates (2%)
4. **Tokenization**: Byte-Pair Encoding (BPE), vocabulary size 5,000
5. **Data Split**: 98% train / 1% validation / 1% test
6. **Training Subset**: 100M tokens (for fair 1-epoch comparison)

### Model Architectures

**Transformer (GPT-style)**:
- Decoder-only architecture with causal self-attention
- Pre-layer normalization
- GELU activation in feed-forward layers
- Learned positional embeddings

**LSTM (Baseline)**:
- Multi-layer LSTM with embedding layer
- Dropout regularization
- Linear output projection

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Learning Rate | 3×10⁻⁴ → 3×10⁻⁵ (cosine decay) |
| Warmup Steps | 500 |
| Weight Decay | 0.1 |
| Batch Size | 96 (A100) |
| Context Length | 256 tokens |
| Gradient Clipping | 1.0 |
| Precision | Mixed (FP16) |

---

## Development Process

### Phase 1: Initial Implementation

**Approach**: Character-level tokenization with fixed learning rate (3×10⁻⁴).

**Issues Identified**:
- Context window too limited (~256 characters insufficient for musical phrases)
- All models constrained by inability to see enough context
- Scaling behavior noisy and inconsistent
- No meaningful performance separation between model sizes

### Phase 2: Tokenization Revision

**Solution**: Switched to Byte-Pair Encoding (BPE) with vocabulary size 5,000.

**Impact**:
- Achieved 5x sequence compression
- Effective context expanded to cover phrase-level structure
- Common musical patterns (chords, scales) encoded as single tokens
- Clean scaling behavior emerged for the first time

### Phase 3: Architecture Correction

**Issue Discovered**: Initial transformer implementation used bidirectional attention (standard `nn.TransformerEncoder`), creating train-test mismatch.

**Fix**: Implemented custom `CausalSelfAttention` module with lower-triangular mask:

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

**Result**: Valid autoregressive generation; training metrics directly transfer to generation performance.

### Phase 4: Final Training on A100

**Configuration**:
- Model: gpt_large (92.9M parameters)
- GPU: NVIDIA A100-SXM4-40GB
- Batch Size: 96 (optimized for A100 memory)
- Mixed Precision: Enabled

**Training Progress**:
```
iter 0/6103     | loss 8.6335 | lr 0.00e+00
iter 500/6103   | loss 0.5555 | val_loss 0.7128
iter 1000/6103  | loss 0.4293 | val_loss 0.5297
iter 2000/6103  | loss 0.2778 | val_loss 0.3932
iter 4000/6103  | loss 0.2196 | val_loss 0.3391
iter 6103/6103  | loss 0.1808 | val_loss 0.2886
```

**Final Metrics**:
- Validation Loss: 0.2886
- Perplexity: 1.33
- Training Time: 30.5 minutes

---

## Complete Model Results

### Transformer Models

| Model | Layers | Embed Dim | Heads | Parameters | Val Loss | Time |
|-------|--------|-----------|-------|------------|----------|------|
| gpt_micro | 2 | 128 | 4 | 1,713,800 | 0.5878 | 3.6 min |
| gpt_tiny | 3 | 192 | 4 | 3,307,400 | 0.4297 | 7.2 min |
| gpt_small | 6 | 288 | 6 | 8,948,552 | 0.3856 | 21.0 min |
| gpt_medium | 8 | 512 | 8 | 30,463,880 | 0.3195 | 56.4 min |
| gpt_large | 12 | 768 | 12 | 92,909,960 | 0.2886 | 30.5 min* |

*Trained on A100; others on T4.

### LSTM Models

| Model | Layers | Hidden Size | Parameters | Val Loss | Time |
|-------|--------|-------------|------------|----------|------|
| rnn_tiny | 1 | 512 | 7,226,248 | 0.4331 | 6.0 min |
| rnn_small | 2 | 896 | 21,824,392 | 0.3132 | 14.4 min |
| rnn_medium | 2 | 1536 | 53,138,312 | 0.3655 | 39.6 min |
| rnn_large | 3 | 1536 | 72,024,968 | 0.3223 | 55.2 min |

---

## Key Findings

### 1. Power Law Scaling

Validation loss follows the relationship:

**L(N) = a · N^(-α) + c**

Where α ≈ 0.19 for transformers on symbolic music.

**Comparison to Natural Language**:
- Music: α ≈ 0.19 (this work)
- Language: α ≈ 0.076 (Kaplan et al., 2020)

The steeper exponent suggests that music structure may be more amenable to scaling-based improvements than natural language.

### 2. Transformer vs. LSTM Efficiency

| Scale | Comparison |
|-------|------------|
| ~20M params | Comparable performance |
| ~50M params | Transformers begin to dominate |
| ~90M params | Transformers 2.4x more parameter-efficient |

LSTMs plateau around 50-70M parameters, while transformers continue to improve.

### 3. Sample Quality Limitation

Generated samples achieved low perplexity (1.33) but consisted primarily of rest tokens rather than melodic content. This reveals an important insight: **perplexity measures token prediction accuracy, not musical creativity**.

Future work should incorporate domain-specific metrics:
- Pitch class diversity
- Rhythmic complexity
- Note density measures

---

## Reproducibility

### Model Checkpoints

All model weights and training logs are available at:

**Google Drive (Essential Files)**: [https://drive.google.com/drive/folders/1iORAP1jo7TST4TvToLfKtvl9nHV3Fzo3](https://drive.google.com/drive/folders/1iORAP1jo7TST4TvToLfKtvl9nHV3Fzo3?usp=sharing)

**Google Drive (Complete Project)**: [https://drive.google.com/drive/folders/19CmXEs7RvoGhT1I53YnjEBQgOGdv1sBc](https://drive.google.com/drive/folders/19CmXEs7RvoGhT1I53YnjEBQgOGdv1sBc?usp=sharing)

Contents:
- `gpt_large_final.pt` (360MB) - Final model weights
- `gpt_large_log.pkl` - Training metrics (6,103 iterations)
- `training_curves.png` - Loss visualization

### Running the Code

1. Upload `A100_Gpt_large.ipynb` to Google Colab
2. Select A100 GPU runtime (Runtime → Change runtime type)
3. Run all cells (~30 minutes)

---

## References

1. Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361.

2. Raffel, C. (2016). "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching." PhD Thesis, Columbia University.

3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems.

---

## Contact

Pranjal Mishra  
pm4084@nyu.edu  
New York University
