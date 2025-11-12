"""
Main experiment runner for systematic evaluation of RNN architectures
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os
import json
from datetime import datetime

from preprocess import load_and_preprocess_data
from models import create_model
from train import train_model
from evaluate import evaluate_model
from utils import set_seed, get_device, get_hardware_info


def run_single_experiment(config, X_train, y_train, X_test, y_test, vocab_size, device):
    """
    Run a single experiment with given configuration

    Args:
        config: Experiment configuration dictionary
        X_train, y_train: Training data
        X_test, y_test: Test data
        vocab_size: Vocabulary size
        device: Device to train on

    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print(f"Running experiment: {config['name']}")
    print("="*80)
    print(f"Configuration: {config}")

    # Set seed for reproducibility
    set_seed(42)

    # Create model
    model = create_model(
        config['architecture'],
        vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        activation=config['activation']
    )

    # Training configuration
    train_config = {
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'optimizer': config['optimizer'],
        'learning_rate': config['learning_rate'],
        'gradient_clipping': config['gradient_clipping'],
        'clip_value': config.get('clip_value', 1.0)
    }

    # Train model
    history, avg_epoch_time = train_model(
        model, X_train, y_train, X_test, y_test, train_config, device
    )

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, device, batch_size=config['batch_size'])

    # Compile results
    results = {
        'config': config,
        'history': history,
        'avg_epoch_time': avg_epoch_time,
        'test_accuracy': metrics['accuracy'],
        'test_f1_macro': metrics['f1_macro'],
        'test_f1_binary': metrics['f1_binary'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'model': model
    }

    print(f"\nResults:")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")

    return results


def run_all_experiments():
    """
    Run all experiments systematically
    """
    # Get device
    device = get_device()
    hardware_info = get_hardware_info()

    # Base configuration
    base_config = {
        'embedding_dim': 100,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.4,
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001,
        'clip_value': 1.0
    }

    # All experiments to run
    experiments = []

    # Experiment 1: Different architectures (fixing other parameters)
    for arch in ['rnn', 'lstm', 'bilstm']:
        exp_config = base_config.copy()
        exp_config.update({
            'name': f'architecture_{arch}',
            'architecture': arch,
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 50,
            'gradient_clipping': False,
            'experiment_type': 'architecture'
        })
        experiments.append(exp_config)

    # Experiment 2: Different activation functions (LSTM as baseline)
    for activation in ['sigmoid', 'relu', 'tanh']:
        exp_config = base_config.copy()
        exp_config.update({
            'name': f'activation_{activation}',
            'architecture': 'lstm',
            'activation': activation,
            'optimizer': 'adam',
            'seq_length': 50,
            'gradient_clipping': False,
            'experiment_type': 'activation'
        })
        experiments.append(exp_config)

    # Experiment 3: Different optimizers (LSTM as baseline)
    for optimizer in ['adam', 'sgd', 'rmsprop']:
        exp_config = base_config.copy()
        exp_config.update({
            'name': f'optimizer_{optimizer}',
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': optimizer,
            'seq_length': 50,
            'gradient_clipping': False,
            'experiment_type': 'optimizer'
        })
        experiments.append(exp_config)

    # Experiment 4: Different sequence lengths (LSTM as baseline)
    for seq_len in [25, 50, 100]:
        exp_config = base_config.copy()
        exp_config.update({
            'name': f'seqlen_{seq_len}',
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': seq_len,
            'gradient_clipping': False,
            'experiment_type': 'sequence_length'
        })
        experiments.append(exp_config)

    # Experiment 5: Gradient clipping (LSTM as baseline)
    for grad_clip in [False, True]:
        exp_config = base_config.copy()
        exp_config.update({
            'name': f'gradclip_{grad_clip}',
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 50,
            'gradient_clipping': grad_clip,
            'experiment_type': 'gradient_clipping'
        })
        experiments.append(exp_config)

    # Store all results
    all_results = []

    # Data cache for different sequence lengths
    data_cache = {}

    # Run experiments
    for i, exp_config in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiments)}")
        print(f"{'='*80}")

        seq_len = exp_config['seq_length']

        # Load data with appropriate sequence length (cache to avoid reloading)
        if seq_len not in data_cache:
            print(f"Loading data with sequence length {seq_len}...")
            X_train, X_test, y_train, y_test, preprocessor, stats = load_and_preprocess_data(
                "../data/IMDB Dataset.csv",
                max_words=10000,
                seq_length=seq_len,
                test_size=0.5
            )
            data_cache[seq_len] = (X_train, X_test, y_train, y_test, preprocessor, stats)
        else:
            X_train, X_test, y_train, y_test, preprocessor, stats = data_cache[seq_len]

        # Run experiment
        results = run_single_experiment(
            exp_config, X_train, y_train, X_test, y_test,
            preprocessor.vocab_size, device
        )

        all_results.append(results)

        # Save intermediate results
        save_results(all_results, hardware_info)

    return all_results, hardware_info


def save_results(all_results, hardware_info):
    """
    Save experiment results to CSV and JSON

    Args:
        all_results: List of result dictionaries
        hardware_info: Hardware information dictionary
    """
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)

    # Prepare data for CSV
    csv_data = []
    for result in all_results:
        config = result['config']
        row = {
            'Model': config['architecture'].upper(),
            'Activation': config['activation'],
            'Optimizer': config['optimizer'],
            'Seq_Length': config['seq_length'],
            'Grad_Clipping': 'Yes' if config['gradient_clipping'] else 'No',
            'Accuracy': result['test_accuracy'],
            'F1': result['test_f1_macro'],
            'Epoch_Time_s': result['avg_epoch_time'],
            'Experiment_Type': config['experiment_type']
        }
        csv_data.append(row)

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('../results/metrics.csv', index=False)
    print(f"\nResults saved to ../results/metrics.csv")

    # Save detailed results to JSON (without model objects)
    json_results = []
    for result in all_results:
        json_result = {
            'config': result['config'],
            'avg_epoch_time': result['avg_epoch_time'],
            'test_accuracy': result['test_accuracy'],
            'test_f1_macro': result['test_f1_macro'],
            'test_f1_binary': result['test_f1_binary'],
            'test_precision': result['test_precision'],
            'test_recall': result['test_recall'],
            'train_loss': result['history']['train_loss'],
            'val_loss': result['history']['val_loss'],
            'train_acc': result['history']['train_acc'],
            'val_acc': result['history']['val_acc']
        }
        json_results.append(json_result)

    output_data = {
        'hardware_info': hardware_info,
        'timestamp': datetime.now().isoformat(),
        'results': json_results
    }

    with open('../results/detailed_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results saved to ../results/detailed_results.json")

    return df


def generate_plots(all_results):
    """
    Generate required plots

    Args:
        all_results: List of result dictionaries
    """
    os.makedirs('../results/plots', exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # Plot 1: Accuracy and F1 vs. Sequence Length
    seq_length_results = [r for r in all_results if r['config']['experiment_type'] == 'sequence_length']

    if seq_length_results:
        seq_lengths = [r['config']['seq_length'] for r in seq_length_results]
        accuracies = [r['test_accuracy'] for r in seq_length_results]
        f1_scores = [r['test_f1_macro'] for r in seq_length_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        ax1.plot(seq_lengths, accuracies, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs. Sequence Length', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # F1 plot
        ax2.plot(seq_lengths, f1_scores, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
        ax2.set_title('F1-Score vs. Sequence Length', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/plots/accuracy_f1_vs_seqlen.png', dpi=300, bbox_inches='tight')
        print("Saved: ../results/plots/accuracy_f1_vs_seqlen.png")
        plt.close()

    # Plot 2: Training Loss vs. Epochs for best and worst models
    # Find best and worst models based on test accuracy
    sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'])
    worst_model = sorted_results[0]
    best_model = sorted_results[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Best model
    epochs = range(1, len(best_model['history']['train_loss']) + 1)
    ax1.plot(epochs, best_model['history']['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, best_model['history']['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Best Model: {best_model["config"]["name"]}\n(Acc: {best_model["test_accuracy"]:.4f})',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Worst model
    epochs = range(1, len(worst_model['history']['train_loss']) + 1)
    ax2.plot(epochs, worst_model['history']['train_loss'], label='Train Loss', linewidth=2)
    ax2.plot(epochs, worst_model['history']['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Worst Model: {worst_model["config"]["name"]}\n(Acc: {worst_model["test_accuracy"]:.4f})',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/plots/loss_best_worst_models.png', dpi=300, bbox_inches='tight')
    print("Saved: ../results/plots/loss_best_worst_models.png")
    plt.close()

    # Additional plot: Comparison by experiment type
    experiment_types = ['architecture', 'activation', 'optimizer', 'gradient_clipping']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, exp_type in enumerate(experiment_types):
        exp_results = [r for r in all_results if r['config']['experiment_type'] == exp_type]

        if exp_results:
            names = [r['config'][exp_type] if exp_type != 'gradient_clipping'
                    else ('Yes' if r['config']['gradient_clipping'] else 'No')
                    for r in exp_results]
            accuracies = [r['test_accuracy'] for r in exp_results]
            f1_scores = [r['test_f1_macro'] for r in exp_results]

            x = np.arange(len(names))
            width = 0.35

            axes[idx].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            axes[idx].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            axes[idx].set_xlabel(exp_type.replace('_', ' ').title(), fontsize=11)
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].set_title(f'Performance by {exp_type.replace("_", " ").title()}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(names, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/plots/comparison_by_experiment_type.png', dpi=300, bbox_inches='tight')
    print("Saved: ../results/plots/comparison_by_experiment_type.png")
    plt.close()

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    print("Starting comprehensive RNN sentiment analysis experiments...")
    print("This will run all required experiments systematically.\n")

    # Run all experiments
    all_results, hardware_info = run_all_experiments()

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    df = pd.read_csv('../results/metrics.csv')
    print("\nResults Table:")
    print(df.to_string(index=False))

    print(f"\nHardware Used:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")

    print("\nBest performing model:")
    best_idx = df['Accuracy'].idxmax()
    print(df.iloc[best_idx].to_string())

    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)
