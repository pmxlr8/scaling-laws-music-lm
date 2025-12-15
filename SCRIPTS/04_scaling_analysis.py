"""
Scaling Law Analysis
====================
Fit power law to model results and generate scaling plots.

Usage:
    python 04_scaling_analysis.py --results_dir /path/to/results --output_dir /path/to/output
"""

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(N, a, alpha, c):
    """Power law function: L(N) = a * N^(-alpha) + c"""
    return a * N**(-alpha) + c


def load_results(results_dir):
    """Load training results from all model logs."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_log.pkl'):
            model_name = filename.replace('_log.pkl', '')
            with open(os.path.join(results_dir, filename), 'rb') as f:
                log = pickle.load(f)
            
            results[model_name] = {
                'n_params': log['n_params'],
                'val_loss': log['final_val_loss'],
                'train_time': log['train_time'],
                'type': 'gpt' if model_name.startswith('gpt') else 'lstm'
            }
    
    return results


def fit_scaling_law(results, model_type='gpt'):
    """Fit power law to results of specified model type."""
    filtered = {k: v for k, v in results.items() if v['type'] == model_type}
    
    if len(filtered) < 3:
        print(f"Warning: Not enough {model_type} models for reliable fit")
        return None
    
    params = np.array([v['n_params'] for v in filtered.values()])
    losses = np.array([v['val_loss'] for v in filtered.values()])
    
    try:
        popt, _ = curve_fit(power_law, params, losses, p0=[1.0, 0.1, 0.1], maxfev=10000)
        a, alpha, c = popt
        return {'a': a, 'alpha': alpha, 'c': c}
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def plot_scaling_laws(results, gpt_fit, lstm_fit, output_path):
    """Generate scaling law plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Separate GPT and LSTM results
    gpt_params = [v['n_params'] for k, v in results.items() if v['type'] == 'gpt']
    gpt_losses = [v['val_loss'] for k, v in results.items() if v['type'] == 'gpt']
    lstm_params = [v['n_params'] for k, v in results.items() if v['type'] == 'lstm']
    lstm_losses = [v['val_loss'] for k, v in results.items() if v['type'] == 'lstm']
    
    # Plot data points
    ax.scatter(gpt_params, gpt_losses, s=100, c='blue', alpha=0.7, label='GPT (Transformer)', zorder=5)
    ax.scatter(lstm_params, lstm_losses, s=100, c='red', marker='x', alpha=0.7, label='LSTM (RNN)', zorder=5)
    
    # Plot fitted lines
    if gpt_fit:
        N_fit = np.logspace(np.log10(min(gpt_params)), np.log10(max(gpt_params)), 100)
        L_fit = power_law(N_fit, gpt_fit['a'], gpt_fit['alpha'], gpt_fit['c'])
        ax.plot(N_fit, L_fit, 'b--', linewidth=2, label=f'GPT fit (α={gpt_fit["alpha"]:.3f})')
    
    if lstm_fit:
        N_fit = np.logspace(np.log10(min(lstm_params)), np.log10(max(lstm_params)), 100)
        L_fit = power_law(N_fit, lstm_fit['a'], lstm_fit['alpha'], lstm_fit['c'])
        ax.plot(N_fit, L_fit, 'r--', linewidth=2, label=f'LSTM fit (α={lstm_fit["alpha"]:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Scaling Laws for Music Language Models', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scaling plot to {output_path}")


def plot_training_curves(results_dir, model_name, output_path):
    """Plot training curves for a specific model."""
    log_path = os.path.join(results_dir, f'{model_name}_log.pkl')
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    with open(log_path, 'rb') as f:
        log = pickle.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    train_losses = log['train_losses']
    ax1.plot(train_losses, alpha=0.5, color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Validation loss
    val_steps, val_losses = zip(*log['val_losses'])
    ax2.plot(val_steps, val_losses, 'o-', color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves: {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze scaling laws')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    print(f"Loaded results for {len(results)} models")
    
    # Fit scaling laws
    gpt_fit = fit_scaling_law(results, 'gpt')
    lstm_fit = fit_scaling_law(results, 'lstm')
    
    if gpt_fit:
        print(f"GPT scaling: L = {gpt_fit['a']:.4f} * N^(-{gpt_fit['alpha']:.4f}) + {gpt_fit['c']:.4f}")
    if lstm_fit:
        print(f"LSTM scaling: L = {lstm_fit['a']:.4f} * N^(-{lstm_fit['alpha']:.4f}) + {lstm_fit['c']:.4f}")
    
    # Generate plots
    plot_scaling_laws(results, gpt_fit, lstm_fit, os.path.join(args.output_dir, 'scaling_plot.png'))
    
    # Save results summary
    summary = {
        'models': results,
        'gpt_fit': gpt_fit,
        'lstm_fit': lstm_fit
    }
    with open(os.path.join(args.output_dir, 'scaling_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
