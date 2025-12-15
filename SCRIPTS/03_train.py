"""
Training Script for Music Language Models
==========================================
Train GPT or LSTM models on tokenized music data.

Usage:
    python 03_train.py --model gpt_large --data_dir /path/to/data --output_dir /path/to/output

Features:
    - Cosine learning rate schedule with warmup
    - Gradient clipping
    - Checkpointing
    - Training curve logging
"""

import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

from models import GPTModel, GPTConfig, LSTMModel, MODEL_CONFIGS, LSTM_CONFIGS


class DataLoader:
    """Simple data loader for binary token files."""
    
    def __init__(self, data_path, block_size, batch_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
    
    def get_batch(self, device='cuda'):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


def get_lr(step, max_steps, base_lr, min_lr, warmup_steps=500):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, eval_iters=50):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = val_loader.get_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train(args):
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load metadata
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    vocab_size = meta['vocab_size']
    train_tokens = meta['train_tokens']
    
    # Create model
    if args.model.startswith('gpt'):
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            **MODEL_CONFIGS[args.model]
        )
        model = GPTModel(config)
    else:
        model = LSTMModel(
            vocab_size=vocab_size,
            **LSTM_CONFIGS[args.model]
        )
    
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model: {args.model} ({n_params:,} parameters)")
    
    # Data loaders
    train_loader = DataLoader(
        os.path.join(args.data_dir, 'train.bin'),
        args.block_size, args.batch_size
    )
    val_loader = DataLoader(
        os.path.join(args.data_dir, 'val.bin'),
        args.block_size, args.batch_size
    )
    
    # Calculate steps
    max_steps = train_tokens // (args.batch_size * args.block_size)
    print(f"Max steps (1 epoch): {max_steps:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Update learning rate
        lr = get_lr(step, max_steps, args.learning_rate, args.learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        x, y = train_loader.get_batch(device)
        _, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
        
        # Evaluation
        if step % args.eval_interval == 0 or step == max_steps - 1:
            val_loss = evaluate(model, val_loader)
            val_losses.append((step, val_loss))
            print(f"  Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'val_loss': val_loss,
                'config': args
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'{args.model}_checkpoint.pt'))
    
    # Save final model
    final_val_loss = evaluate(model, val_loader)
    elapsed = time.time() - start_time
    
    torch.save({
        'model': model.state_dict(),
        'val_loss': final_val_loss,
        'train_time': elapsed,
        'n_params': n_params
    }, os.path.join(args.output_dir, f'{args.model}_final.pt'))
    
    # Save training log
    log = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val_loss': final_val_loss,
        'train_time': elapsed,
        'n_params': n_params
    }
    with open(os.path.join(args.output_dir, f'{args.model}_log.pkl'), 'wb') as f:
        pickle.dump(log, f)
    
    print(f"\nTraining complete!")
    print(f"  Final validation loss: {final_val_loss:.4f}")
    print(f"  Perplexity: {np.exp(final_val_loss):.2f}")
    print(f"  Training time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Train music language model')
    parser.add_argument('--model', type=str, required=True, 
                        choices=list(MODEL_CONFIGS.keys()) + list(LSTM_CONFIGS.keys()))
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=500)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
