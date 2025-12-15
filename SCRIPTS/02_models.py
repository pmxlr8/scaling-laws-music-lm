"""
Model Definitions for Music Language Modeling
=============================================
GPT (Transformer) and LSTM architectures for symbolic music generation.

This module defines:
    - GPTConfig: Configuration dataclass for GPT models
    - CausalSelfAttention: Self-attention with causal masking
    - GPTModel: Full GPT architecture
    - LSTMModel: LSTM baseline architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Causal mask: lower triangular
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block with pre-layer normalization."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT Language Model for symbolic music generation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class LSTMModel(nn.Module):
    """LSTM Language Model for symbolic music generation."""
    
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# Model configurations used in experiments
MODEL_CONFIGS = {
    'gpt_micro':  {'n_layer': 2,  'n_head': 4,  'n_embd': 128},
    'gpt_tiny':   {'n_layer': 3,  'n_head': 4,  'n_embd': 192},
    'gpt_small':  {'n_layer': 6,  'n_head': 6,  'n_embd': 288},
    'gpt_medium': {'n_layer': 8,  'n_head': 8,  'n_embd': 512},
    'gpt_large':  {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
}

LSTM_CONFIGS = {
    'rnn_tiny':   {'num_layers': 1, 'hidden_size': 512},
    'rnn_small':  {'num_layers': 2, 'hidden_size': 896},
    'rnn_medium': {'num_layers': 2, 'hidden_size': 1536},
    'rnn_large':  {'num_layers': 3, 'hidden_size': 1536},
}
