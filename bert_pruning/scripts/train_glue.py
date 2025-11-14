"""
Train BERT with Hard Concrete Gates on GLUE tasks.

Usage:
    python train_glue.py --task sst2 --output_dir ./output/sst2 --l0_coefficient 0.1
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BertForSequenceClassificationWithGates
from utils.glue import (
    load_glue_dataset,
    prepare_glue_data,
    compute_glue_metrics,
    get_task_info
)
from utils.pruning import (
    get_pruning_statistics,
    visualize_gate_values,
    compute_pruning_threshold
)

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
    parser.add_argument('--l0_coefficient', type=float, default=0.0,
                        help='L0 regularization coefficient (0 = no pruning)')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Evaluation arguments
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')

    # Pruning arguments
    parser.add_argument('--visualize_gates', action='store_true',
                        help='Visualize gate values after training')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, l0_coefficient, task_name):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_l0_penalty = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc='Training')

    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids'),
            labels=batch['labels'],
            l0_coefficient=l0_coefficient
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        if l0_coefficient > 0:
            l0_penalty = model.bert.get_total_l0_penalty().item()
            total_l0_penalty += l0_penalty
        num_batches += 1

        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    metrics = {
        'loss': total_loss / num_batches,
    }
    if l0_coefficient > 0:
        metrics['l0_penalty'] = total_l0_penalty / num_batches

    return metrics


def evaluate(model, dataloader, device, task_name):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    task_info = get_task_info(task_name)
    is_regression = task_info.get('is_regression', False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                labels=batch['labels'],
                l0_coefficient=0.0  # No L0 penalty during evaluation
            )

            logits = outputs.logits.cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            if is_regression:
                predictions = logits.squeeze()
            else:
                predictions = logits

            all_predictions.extend(predictions)
            all_labels.extend(labels)

            total_loss += outputs.loss.item()
            num_batches += 1

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_glue_metrics(task_name, all_predictions, all_labels)
    metrics['eval_loss'] = total_loss / num_batches

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load task info
    task_info = get_task_info(args.task)
    logger.info(f'Task: {args.task}, Num labels: {task_info["num_labels"]}, Metric: {task_info["metric"]}')

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Load dataset
    logger.info('Loading dataset...')
    dataset = load_glue_dataset(args.task)

    # Prepare datasets
    def tokenize_function(examples):
        return prepare_glue_data(examples, args.task, tokenizer, args.max_length)

    train_dataset = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    if args.task == 'mnli':
        eval_dataset = dataset['validation_matched'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['validation_matched'].column_names
        )
    else:
        eval_dataset = dataset['validation'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['validation'].column_names
        )

    # Set format for PyTorch
    train_dataset.set_format(type='torch')
    eval_dataset.set_format(type='torch')

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # Load model
    logger.info('Loading model...')
    config = BertConfig.from_pretrained(args.model_name)
    config.num_labels = task_info['num_labels']

    model = BertForSequenceClassificationWithGates.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Setup optimizer and scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    logger.info('Starting training...')
    best_metric = -float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):
        logger.info(f'\nEpoch {epoch + 1}/{args.num_epochs}')

        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            device, args.l0_coefficient, args.task
        )
        logger.info(f'Train metrics: {train_metrics}')

        # Evaluate
        eval_metrics = evaluate(model, eval_dataloader, device, args.task)
        logger.info(f'Eval metrics: {eval_metrics}')

        # Get sparsity info if using L0 regularization
        if args.l0_coefficient > 0:
            sparsity_stats = get_pruning_statistics(model)
            logger.info(f'Pruning ratio: {sparsity_stats["pruning_ratio"]:.2%}')
            logger.info(f'Active heads: {sparsity_stats["active_heads"]}/{sparsity_stats["total_heads"]}')

        # Save checkpoint if best
        main_metric = eval_metrics[task_info['metric']]
        if main_metric > best_metric:
            best_metric = main_metric
            best_epoch = epoch

            checkpoint_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f'Saved best model to {checkpoint_path}')

            # Save metrics
            with open(os.path.join(args.output_dir, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics
                }, f, indent=2)

    logger.info(f'\nBest epoch: {best_epoch + 1}')
    logger.info(f'Best {task_info["metric"]}: {best_metric:.4f}')

    # Visualize gates if requested
    if args.visualize_gates and args.l0_coefficient > 0:
        logger.info('Visualizing gate values...')
        vis_path = os.path.join(args.output_dir, 'gate_values.png')
        visualize_gate_values(model, vis_path)

        # Save detailed sparsity info
        sparsity_info = model.get_sparsity_info()
        with open(os.path.join(args.output_dir, 'sparsity_info.json'), 'w') as f:
            json.dump(sparsity_info, f, indent=2)

    logger.info('Training completed!')


if __name__ == '__main__':
    main()
