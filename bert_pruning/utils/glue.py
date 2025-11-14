"""
GLUE task utilities.
"""

from datasets import load_dataset
from typing import Dict, List


GLUE_TASKS = {
    'cola': {
        'num_labels': 2,
        'metric': 'matthews_correlation',
        'text_keys': ['sentence']
    },
    'mnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis']
    },
    'mrpc': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2']
    },
    'qnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['question', 'sentence']
    },
    'qqp': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['question1', 'question2']
    },
    'rte': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2']
    },
    'sst2': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence']
    },
    'stsb': {
        'num_labels': 1,
        'metric': 'pearson',
        'text_keys': ['sentence1', 'sentence2'],
        'is_regression': True
    },
    'wnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2']
    }
}


def get_glue_task_names() -> List[str]:
    """Return list of all GLUE task names."""
    return list(GLUE_TASKS.keys())


def get_task_info(task_name: str) -> Dict:
    """Get information about a GLUE task."""
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(GLUE_TASKS.keys())}")
    return GLUE_TASKS[task_name]


def load_glue_dataset(task_name: str):
    """Load GLUE dataset."""
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    if task_name == 'mnli':
        dataset = load_dataset('glue', task_name)
        # MNLI has two validation sets
        return dataset
    else:
        dataset = load_dataset('glue', task_name)
        return dataset


def prepare_glue_data(examples, task_name: str, tokenizer, max_length: int = 128):
    """
    Prepare GLUE data for training.

    Args:
        examples: Batch of examples from dataset
        task_name: Name of GLUE task
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenized inputs
    """
    task_info = get_task_info(task_name)
    text_keys = task_info['text_keys']

    if len(text_keys) == 1:
        # Single sentence tasks
        texts = examples[text_keys[0]]
        inputs = tokenizer(
            texts,
            padding='max_length',
            max_length=max_length,
            truncation=True
        )
    else:
        # Sentence pair tasks
        texts1 = examples[text_keys[0]]
        texts2 = examples[text_keys[1]]
        inputs = tokenizer(
            texts1,
            texts2,
            padding='max_length',
            max_length=max_length,
            truncation=True
        )

    return inputs


def compute_glue_metrics(task_name: str, predictions, labels):
    """
    Compute metrics for GLUE task.

    Args:
        task_name: Name of GLUE task
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        matthews_corrcoef,
        f1_score,
        accuracy_score
    )
    from scipy.stats import pearsonr, spearmanr
    import numpy as np

    task_info = get_task_info(task_name)

    if task_info.get('is_regression', False):
        # Regression task (STS-B)
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            'pearson': pearson_corr,
            'spearmanr': spearman_corr,
            'combined': (pearson_corr + spearman_corr) / 2
        }
    else:
        # Classification tasks
        preds = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        acc = accuracy_score(labels, preds)

        if task_info['metric'] == 'matthews_correlation':
            mcc = matthews_corrcoef(labels, preds)
            return {'matthews_correlation': mcc, 'accuracy': acc}
        elif task_info['metric'] == 'f1':
            f1 = f1_score(labels, preds)
            return {'f1': f1, 'accuracy': acc}
        else:
            return {'accuracy': acc}
