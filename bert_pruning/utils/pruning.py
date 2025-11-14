"""
Utility functions for pruning BERT attention heads.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_pruning_threshold(
    model,
    target_pruning_ratio: float
) -> float:
    """
    Compute gate threshold to achieve target pruning ratio.

    Args:
        model: BertForSequenceClassificationWithGates model
        target_pruning_ratio: Desired fraction of heads to prune (0.0 to 1.0)

    Returns:
        Threshold value for gate-based pruning
    """
    all_gate_values = []

    for gate in model.bert.get_all_gates():
        gate_values = gate.get_gate_values()
        all_gate_values.extend(gate_values.cpu().numpy().flatten().tolist())

    all_gate_values = np.array(all_gate_values)
    threshold = np.percentile(all_gate_values, target_pruning_ratio * 100)

    return float(threshold)


def get_pruning_statistics(
    model,
    threshold: float = 0.01
) -> Dict:
    """
    Get detailed pruning statistics.

    Args:
        model: BertForSequenceClassificationWithGates model
        threshold: Threshold for determining if a head is pruned

    Returns:
        Dictionary with pruning statistics
    """
    stats = {
        'total_heads': 0,
        'active_heads': 0,
        'pruned_heads': 0,
        'layer_stats': []
    }

    for layer_idx, gate in enumerate(model.bert.get_all_gates()):
        gate_values = gate.get_gate_values().cpu().numpy()
        active_mask = gate_values > threshold

        layer_stat = {
            'layer': layer_idx,
            'total': len(gate_values),
            'active': int(active_mask.sum()),
            'pruned': int((~active_mask).sum()),
            'gate_values': gate_values.tolist(),
            'sparsity_rate': float(active_mask.mean())
        }

        stats['total_heads'] += layer_stat['total']
        stats['active_heads'] += layer_stat['active']
        stats['pruned_heads'] += layer_stat['pruned']
        stats['layer_stats'].append(layer_stat)

    stats['overall_sparsity'] = stats['active_heads'] / stats['total_heads']
    stats['pruning_ratio'] = stats['pruned_heads'] / stats['total_heads']

    return stats


def visualize_gate_values(
    model,
    save_path: str = None
) -> None:
    """
    Visualize gate values across all layers.

    Args:
        model: BertForSequenceClassificationWithGates model
        save_path: Path to save the visualization (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sparsity_info = model.get_sparsity_info()
    num_layers = len(sparsity_info)

    gate_matrix = []
    for info in sparsity_info:
        gate_matrix.append(info['gate_values'])

    gate_matrix = np.array(gate_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        gate_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=[f'Head {i}' for i in range(gate_matrix.shape[1])],
        yticklabels=[f'Layer {i}' for i in range(num_layers)],
        cbar_kws={'label': 'Gate Value'},
        vmin=0.0,
        vmax=1.0
    )
    plt.title('BERT Attention Head Gate Values')
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def find_threshold_for_ratio(
    gate_values: List[float],
    target_ratio: float
) -> float:
    """
    Find threshold that achieves target pruning ratio.

    Args:
        gate_values: List of all gate values
        target_ratio: Target pruning ratio (0.0 to 1.0)

    Returns:
        Threshold value
    """
    sorted_values = sorted(gate_values)
    idx = int(len(sorted_values) * target_ratio)
    return sorted_values[idx] if idx < len(sorted_values) else sorted_values[-1]


def apply_pruning_mask(
    model,
    pruning_mask: Dict[int, List[int]]
) -> None:
    """
    Apply pruning mask to model by setting gate values to 0.

    Args:
        model: BertForSequenceClassificationWithGates model
        pruning_mask: Dict mapping layer_idx -> list of head indices to prune
    """
    gates = model.bert.get_all_gates()

    for layer_idx, head_indices in pruning_mask.items():
        if layer_idx < len(gates):
            gate = gates[layer_idx]
            with torch.no_grad():
                for head_idx in head_indices:
                    # Set log_a to very negative value to close the gate
                    gate.log_a[0, head_idx, 0, 0] = -10.0


def compute_flops_reduction(
    pruning_ratio: float,
    model_config
) -> Dict[str, float]:
    """
    Estimate FLOPs reduction from pruning.

    Args:
        pruning_ratio: Fraction of heads pruned
        model_config: BERT model configuration

    Returns:
        Dictionary with FLOPs estimates
    """
    num_heads = model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers

    # Approximate attention FLOPs per layer
    head_dim = hidden_size // num_heads
    attn_flops_per_head = 4 * head_dim  # QKV projections + output

    original_flops = num_heads * attn_flops_per_head * num_layers
    remaining_heads = num_heads * (1 - pruning_ratio)
    pruned_flops = remaining_heads * attn_flops_per_head * num_layers

    reduction = (original_flops - pruned_flops) / original_flops

    return {
        'original_relative_flops': 1.0,
        'pruned_relative_flops': 1 - reduction,
        'flops_reduction': reduction,
        'remaining_heads_per_layer': remaining_heads
    }
