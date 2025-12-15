"""
Music Sample Generation
=======================
Generate ABC notation samples from trained models.

Usage:
    python 05_generate_samples.py --model_path /path/to/model.pt --tokenizer_path /path/to/tokenizer.json --output_dir /path/to/output
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from models import GPTModel, GPTConfig, MODEL_CONFIGS


def nucleus_sample(logits, top_p=0.9, temperature=0.8):
    """Nucleus (top-p) sampling."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
    return next_token


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=400, top_p=0.9, temperature=0.8, device='cuda'):
    """Generate ABC notation from prompt."""
    model.eval()
    
    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    block_size = model.config.block_size
    
    for _ in range(max_tokens):
        # Crop to block size
        tokens_cond = tokens[:, -block_size:]
        
        # Get predictions
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]  # Last position
        
        # Sample next token
        next_token = nucleus_sample(logits, top_p, temperature)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())


def fix_abc_headers(text, sample_num, key='C', meter='4/4'):
    """Ensure ABC notation has proper headers."""
    lines = text.split('\n')
    
    has_x = any(line.startswith('X:') for line in lines[:5])
    has_m = any(line.startswith('M:') for line in lines[:5])
    has_k = any(line.startswith('K:') for line in lines[:5])
    
    if has_x and has_m and has_k:
        return text
    
    header = f"X:{sample_num}\nT:Generated Sample {sample_num}\nM:{meter}\nL:1/8\nK:{key}\n"
    
    # Find where music content starts
    music_start = 0
    for i, line in enumerate(lines):
        if line and not line.startswith(('X:', 'M:', 'K:', 'T:', 'L:', 'Q:', '%')):
            music_start = i
            break
    
    return header + '\n'.join(lines[music_start:])


def main():
    parser = argparse.ArgumentParser(description='Generate music samples')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt_large')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--max_tokens', type=int, default=400)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Load model
    config = GPTConfig(vocab_size=vocab_size, **MODEL_CONFIGS[args.model_name])
    model = GPTModel(config).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model from {args.model_path}")
    
    # Generate samples with different prompts
    prompts = [
        ("X:1\nM:4/4\nK:C\n", "C", "4/4"),
        ("X:1\nM:3/4\nK:G\n", "G", "3/4"),
        ("X:1\nM:6/8\nK:D\n", "D", "6/8"),
        ("X:1\nM:4/4\nK:Am\n", "Am", "4/4"),
        ("X:1\nM:2/4\nK:F\n", "F", "2/4"),
        ("X:1\nM:4/4\nK:Em\n", "Em", "4/4"),
        ("X:1\nM:6/8\nK:A\n", "A", "6/8"),
        ("X:1\nM:3/4\nK:Bm\n", "Bm", "3/4"),
        ("X:1\nM:4/4\nK:E\n", "E", "4/4"),
        ("X:1\nM:2/2\nK:Bb\n", "Bb", "2/2"),
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(min(args.num_samples, len(prompts))):
        prompt, key, meter = prompts[i]
        print(f"Generating sample {i+1}/{args.num_samples}...")
        
        text = generate(model, tokenizer, prompt, args.max_tokens, args.top_p, args.temperature, device)
        text = fix_abc_headers(text, i+1, key, meter)
        
        output_path = os.path.join(args.output_dir, f'sample_{i+1:02d}.abc')
        with open(output_path, 'w') as f:
            f.write(text)
        
        print(f"  Saved to {output_path}")
    
    print(f"\nGenerated {args.num_samples} samples to {args.output_dir}")
    print(f"\nTo convert to MIDI: abc2midi sample_01.abc -o sample_01.mid")
    print(f"To play online: paste ABC into https://abcjs.net/abcjs-editor.html")


if __name__ == '__main__':
    main()
