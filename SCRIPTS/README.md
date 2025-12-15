# Scaling Laws for Music Language Models - Scripts

This folder contains modular, clean implementations of all project components.

## Scripts

| Script | Description |
|--------|-------------|
| `01_preprocess_data.py` | Data preprocessing (MIDI→ABC, BPE tokenization) |
| `02_models.py` | GPT and LSTM model definitions |
| `03_train.py` | Training with LR scheduling and checkpointing |
| `04_scaling_analysis.py` | Power law fitting and scaling plots |
| `05_generate_samples.py` | Sample generation with nucleus sampling |

## Usage

### 1. Preprocess Data
```bash
python 01_preprocess_data.py \
    --input_dir /path/to/abc_files \
    --output_dir /path/to/processed \
    --vocab_size 5000
```

### 2. Train Model
```bash
python 03_train.py \
    --model gpt_large \
    --data_dir /path/to/processed \
    --output_dir /path/to/checkpoints \
    --batch_size 64 \
    --learning_rate 3e-4
```

### 3. Analyze Scaling
```bash
python 04_scaling_analysis.py \
    --results_dir /path/to/checkpoints \
    --output_dir /path/to/analysis
```

### 4. Generate Samples
```bash
python 05_generate_samples.py \
    --model_path /path/to/gpt_large_final.pt \
    --tokenizer_path /path/to/music_bpe.json \
    --output_dir /path/to/samples \
    --num_samples 10
```

## Model Configurations

### Transformers (GPT)
| Model | Layers | Heads | Embed | ~Params |
|-------|--------|-------|-------|---------|
| gpt_micro | 2 | 4 | 128 | 1.7M |
| gpt_tiny | 3 | 4 | 192 | 3.3M |
| gpt_small | 6 | 6 | 288 | 8.9M |
| gpt_medium | 8 | 8 | 512 | 30.5M |
| gpt_large | 12 | 12 | 768 | 92.9M |

### LSTMs
| Model | Layers | Hidden | ~Params |
|-------|--------|--------|---------|
| rnn_tiny | 1 | 512 | 7.2M |
| rnn_small | 2 | 896 | 21.8M |
| rnn_medium | 2 | 1536 | 53.1M |
| rnn_large | 3 | 1536 | 72.0M |
