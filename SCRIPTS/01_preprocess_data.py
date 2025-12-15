"""
Data Preprocessing Pipeline for Symbolic Music
==============================================
Converts MIDI files to ABC notation and creates BPE tokenized datasets.

Usage:
    python 01_preprocess_data.py --input_dir /path/to/midi --output_dir /path/to/output

Requirements:
    - midi2abc (install: apt-get install abcmidi)
    - tokenizers
"""

import os
import hashlib
import pickle
import argparse
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def find_abc_files(root_dir, max_count=None):
    """Discover all .abc files in directory tree."""
    abc_files = []
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith('.abc'):
                abc_files.append(os.path.join(root, fn))
                if max_count and len(abc_files) >= max_count:
                    return abc_files
    return abc_files


def deduplicate_files(file_paths, min_length=50):
    """Remove duplicate files based on content hash."""
    seen_hashes = set()
    unique_files = []
    
    for fp in file_paths:
        try:
            text = Path(fp).read_text(errors='ignore')
            if len(text) < min_length:
                continue
            content_hash = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_files.append(fp)
        except Exception:
            continue
    
    return unique_files


def train_bpe_tokenizer(file_paths, vocab_size=5000, output_path='music_bpe.json'):
    """Train BPE tokenizer on ABC notation files."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=True
    )
    
    def text_iterator():
        for fp in file_paths:
            try:
                yield Path(fp).read_text(errors='ignore')
            except Exception:
                continue
    
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(output_path)
    
    return tokenizer


def encode_to_binary(file_paths, tokenizer, output_path):
    """Encode files to binary token format."""
    total_tokens = 0
    
    with open(output_path, 'wb') as f:
        for fp in file_paths:
            try:
                text = Path(fp).read_text(errors='ignore')
                encoded = tokenizer.encode(text)
                if encoded.ids:
                    arr = np.array(encoded.ids, dtype=np.uint16)
                    arr.tofile(f)
                    total_tokens += len(arr)
            except Exception:
                continue
    
    return total_tokens


def split_data(all_bin_path, output_dir, train_ratio=0.98, val_ratio=0.01):
    """Split binary data into train/val/test sets."""
    data = np.memmap(all_bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    
    n_train = int(total_tokens * train_ratio)
    n_val = int(total_tokens * val_ratio)
    n_test = total_tokens - n_train - n_val
    
    # Write splits
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    test_path = os.path.join(output_dir, 'test.bin')
    
    np.asarray(data[:n_train]).tofile(train_path)
    np.asarray(data[n_train:n_train+n_val]).tofile(val_path)
    np.asarray(data[n_train+n_val:]).tofile(test_path)
    
    del data
    
    return n_train, n_val, n_test


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIDI/ABC data for music LM')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing ABC files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--vocab_size', type=int, default=5000, help='BPE vocabulary size')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Find ABC files
    print("Step 1/4: Finding ABC files...")
    abc_files = find_abc_files(args.input_dir)
    print(f"  Found {len(abc_files):,} ABC files")
    
    # Step 2: Deduplicate
    print("Step 2/4: Deduplicating...")
    unique_files = deduplicate_files(abc_files)
    print(f"  Kept {len(unique_files):,} unique files")
    
    # Step 3: Train tokenizer
    print("Step 3/4: Training BPE tokenizer...")
    tokenizer_path = os.path.join(args.output_dir, 'music_bpe.json')
    tokenizer = train_bpe_tokenizer(unique_files, args.vocab_size, tokenizer_path)
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Step 4: Encode and split
    print("Step 4/4: Encoding and splitting data...")
    all_bin_path = os.path.join(args.output_dir, 'all.bin')
    total_tokens = encode_to_binary(unique_files, tokenizer, all_bin_path)
    n_train, n_val, n_test = split_data(all_bin_path, args.output_dir)
    os.remove(all_bin_path)
    
    # Save metadata
    meta = {
        'vocab_size': tokenizer.get_vocab_size(),
        'train_tokens': n_train,
        'val_tokens': n_val,
        'test_tokens': n_test,
        'total_files': len(unique_files)
    }
    with open(os.path.join(args.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"\nDone!")
    print(f"  Train tokens: {n_train:,}")
    print(f"  Val tokens: {n_val:,}")
    print(f"  Test tokens: {n_test:,}")


if __name__ == '__main__':
    main()
