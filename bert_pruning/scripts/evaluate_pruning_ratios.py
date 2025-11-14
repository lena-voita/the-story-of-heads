"""
Evaluate BERT with different pruning ratios (10%, 30%, 50%) on GLUE tasks.

This script:
1. Trains a baseline model
2. Trains models with L0 regularization for different pruning ratios
3. Evaluates all models and compares performance

Usage:
    python evaluate_pruning_ratios.py --task sst2 --output_dir ./output/sst2_pruning
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BertForSequenceClassificationWithGates
from utils.glue import get_task_info
from utils.pruning import compute_pruning_threshold, get_pruning_statistics
import train_glue

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--task', type=str, required=True,
                        choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb'],
                        help='GLUE task name')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained model name')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Pruning arguments
    parser.add_argument('--pruning_ratios', type=float, nargs='+',
                        default=[0.1, 0.3, 0.5],
                        help='Pruning ratios to evaluate')
    parser.add_argument('--l0_coefficients', type=float, nargs='+',
                        default=[0.01, 0.1, 0.5],
                        help='L0 coefficients for each pruning ratio')

    # Other arguments
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip training baseline model')

    return parser.parse_args()


def train_model_with_pruning(
    task: str,
    output_dir: str,
    l0_coefficient: float,
    args
) -> Dict:
    """Train a single model with specified L0 coefficient."""

    logger.info(f'\n{"="*60}')
    logger.info(f'Training with L0 coefficient: {l0_coefficient}')
    logger.info(f'{"="*60}\n')

    # Create training args
    train_args = argparse.Namespace(
        task=task,
        max_length=args.max_length,
        model_name=args.model_name,
        l0_coefficient=l0_coefficient,
        output_dir=output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        seed=args.seed,
        eval_steps=500,
        save_steps=500,
        visualize_gates=(l0_coefficient > 0)
    )

    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = ['train_glue.py']
    for k, v in vars(train_args).items():
        sys.argv.extend([f'--{k}', str(v)])

    try:
        # Run training
        train_glue.main()
    finally:
        sys.argv = old_argv

    # Load best metrics
    metrics_path = os.path.join(output_dir, 'best_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load sparsity info if available
    sparsity_path = os.path.join(output_dir, 'sparsity_info.json')
    if os.path.exists(sparsity_path):
        with open(sparsity_path, 'r') as f:
            sparsity_info = json.load(f)
        metrics['sparsity_info'] = sparsity_info

    return metrics


def evaluate_model_at_threshold(
    model_path: str,
    threshold: float,
    task: str,
    tokenizer,
    eval_dataloader,
    device
) -> Dict:
    """Evaluate model with specific pruning threshold."""

    logger.info(f'Evaluating with threshold: {threshold:.4f}')

    # Load model
    model = BertForSequenceClassificationWithGates.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Get pruning statistics
    pruning_stats = get_pruning_statistics(model, threshold)

    # Evaluate
    eval_metrics = train_glue.evaluate(model, eval_dataloader, device, task)

    # Combine results
    results = {
        'threshold': threshold,
        'eval_metrics': eval_metrics,
        'pruning_stats': pruning_stats
    }

    return results


def find_best_threshold_for_ratio(
    model_path: str,
    target_ratio: float,
    task: str,
    tokenizer,
    eval_dataloader,
    device,
    tolerance: float = 0.02
) -> Dict:
    """
    Find threshold that achieves target pruning ratio.

    Args:
        model_path: Path to model checkpoint
        target_ratio: Target pruning ratio (e.g., 0.1 for 10%)
        task: GLUE task name
        tokenizer: Tokenizer
        eval_dataloader: Evaluation data loader
        device: Device to use
        tolerance: Acceptable deviation from target ratio

    Returns:
        Dictionary with results
    """
    logger.info(f'\nFinding threshold for {target_ratio*100:.0f}% pruning...')

    # Load model
    model = BertForSequenceClassificationWithGates.from_pretrained(model_path)
    model.to(device)

    # Compute threshold for target ratio
    threshold = compute_pruning_threshold(model, target_ratio)

    # Evaluate at this threshold
    results = evaluate_model_at_threshold(
        model_path, threshold, task, tokenizer, eval_dataloader, device
    )

    actual_ratio = results['pruning_stats']['pruning_ratio']
    logger.info(f'Target ratio: {target_ratio:.2%}, Actual ratio: {actual_ratio:.2%}')

    return results


def create_comparison_table(results: Dict, task_info: Dict) -> pd.DataFrame:
    """Create comparison table of results."""

    rows = []
    main_metric = task_info['metric']

    for config_name, config_results in results.items():
        eval_metrics = config_results.get('eval_metrics', {})
        pruning_stats = config_results.get('pruning_stats', {})

        row = {
            'Configuration': config_name,
            main_metric: eval_metrics.get(main_metric, 0),
            'Eval Loss': eval_metrics.get('eval_loss', 0),
            'Pruning Ratio': pruning_stats.get('pruning_ratio', 0),
            'Active Heads': pruning_stats.get('active_heads', '-'),
            'Total Heads': pruning_stats.get('total_heads', '-')
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    args = parse_args()
    train_glue.set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'experiment_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Get task info
    task_info = get_task_info(args.task)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Dictionary to store all results
    all_results = {}

    # 1. Train baseline model (no pruning)
    if not args.skip_baseline:
        logger.info('\n' + '='*60)
        logger.info('Training Baseline Model (No Pruning)')
        logger.info('='*60)

        baseline_dir = os.path.join(args.output_dir, 'baseline')
        baseline_metrics = train_model_with_pruning(
            args.task, baseline_dir, l0_coefficient=0.0, args=args
        )
        all_results['Baseline (0% pruning)'] = baseline_metrics

    # 2. Train models with different L0 coefficients
    for i, (ratio, l0_coef) in enumerate(zip(args.pruning_ratios, args.l0_coefficients)):
        logger.info(f'\n{"="*60}')
        logger.info(f'Training Model for {ratio*100:.0f}% Pruning Target')
        logger.info(f'L0 Coefficient: {l0_coef}')
        logger.info(f'{"="*60}')

        pruning_dir = os.path.join(args.output_dir, f'pruning_{int(ratio*100)}pct')
        pruning_metrics = train_model_with_pruning(
            args.task, pruning_dir, l0_coefficient=l0_coef, args=args
        )

        # Find actual pruning ratio achieved
        if 'sparsity_info' in pruning_metrics:
            sparsity_info = pruning_metrics['sparsity_info']
            total_heads = len(sparsity_info) * (len(sparsity_info[0]['gate_values']) if sparsity_info else 0)
            active_heads = sum(info['num_active_heads'] for info in sparsity_info)
            actual_ratio = 1 - (active_heads / total_heads) if total_heads > 0 else 0

            logger.info(f'Achieved pruning ratio: {actual_ratio:.2%}')

            all_results[f'{ratio*100:.0f}% Pruning (L0={l0_coef})'] = pruning_metrics

    # 3. Create comparison table
    logger.info('\n' + '='*60)
    logger.info('Results Summary')
    logger.info('='*60 + '\n')

    # Prepare results for table
    table_results = {}
    for config_name, metrics in all_results.items():
        eval_metrics = metrics.get('eval_metrics', {})

        # Calculate pruning stats from sparsity info
        if 'sparsity_info' in metrics:
            sparsity_info = metrics['sparsity_info']
            total_heads = len(sparsity_info) * (len(sparsity_info[0]['gate_values']) if sparsity_info else 0)
            active_heads = sum(info['num_active_heads'] for info in sparsity_info)
            pruning_ratio = 1 - (active_heads / total_heads) if total_heads > 0 else 0

            pruning_stats = {
                'pruning_ratio': pruning_ratio,
                'active_heads': active_heads,
                'total_heads': total_heads
            }
        else:
            pruning_stats = {
                'pruning_ratio': 0.0,
                'active_heads': '-',
                'total_heads': '-'
            }

        table_results[config_name] = {
            'eval_metrics': eval_metrics,
            'pruning_stats': pruning_stats
        }

    # Create and display table
    comparison_df = create_comparison_table(table_results, task_info)
    print(comparison_df.to_string(index=False))

    # Save results
    results_path = os.path.join(args.output_dir, 'comparison_results.csv')
    comparison_df.to_csv(results_path, index=False)
    logger.info(f'\nResults saved to {results_path}')

    # Save detailed results
    detailed_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(detailed_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        json.dump(convert_to_native(all_results), f, indent=2)

    logger.info(f'Detailed results saved to {detailed_path}')
    logger.info('\nExperiment completed!')


if __name__ == '__main__':
    main()
