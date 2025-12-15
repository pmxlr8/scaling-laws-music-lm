# GPT-Large A100 Training Results

## Final Metrics
- **Validation Loss:** 0.2886
- **Training Loss:** 0.1948  
- **Perplexity:** 1.33
- **Training Time:** 30.5 minutes
- **Hardware:** NVIDIA A100-SXM4-40GB

## Model Config
- Parameters: 92,932,608 (92.9M)
- Layers: 12, Heads: 12, Embed: 768
- Vocabulary: 5,000 (BPE)
- Context: 256 tokens

## Training Checkpoints
| Iter | Train Loss | Val Loss | Time |
|------|------------|----------|------|
| 0 | 8.6183 | 8.6646 | 1m |
| 500 | 0.5396 | 0.7128 | 8m |
| 1000 | 0.3627 | 0.5297 | 10m |
| 2000 | 0.2990 | 0.3932 | 14m |
| 3000 | 0.2476 | 0.3441 | 18m |
| 4000 | 0.2677 | 0.3391 | 22m |
| 5000 | 0.2270 | 0.3168 | 26m |
| 6000 | 0.2246 | 0.2929 | 30m |
| 6102 | 0.1948 | 0.2886 | 30.5m |

## Files in Google Drive
- Model: `checkpoints_final/gpt_large_final.pt`
- Curves: `outputs_final/training_curves.png`
- Log: `outputs_final/training_logs/gpt_large_log.pkl`
- Samples: `outputs_final/generated_samples/` (needs header fix)

## Next Steps
1. Run sample regeneration code in Colab
2. Download files from Drive to local project
3. Update LaTeX report with real results
4. Add training curves figure to report
