"""
Hard Concrete Gate for BERT Attention Head Pruning
Based on "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting,
the Rest Can Be Pruned" (Voita et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcreteGate(nn.Module):
    """
    A gate made of Hard Concrete distribution for attention head pruning.

    Args:
        num_heads: Number of attention heads
        temperature: Concrete distribution temperature (lower = more discrete)
        stretch_limits: Min and max values before clipping to [0, 1]
        l0_penalty: Coefficient for L0 regularization
        hard: If True, use hard gate values at test time
    """

    def __init__(
        self,
        num_heads: int,
        temperature: float = 0.33,
        stretch_limits: tuple = (-0.1, 1.1),
        l0_penalty: float = 1.0,
        hard: bool = False,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.stretch_limits = stretch_limits
        self.l0_penalty = l0_penalty
        self.hard = hard
        self.eps = eps

        # Learnable gate parameters (log_alpha in the paper)
        self.log_a = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        """
        Apply gates to values.

        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim]
            is_train: Whether in training mode

        Returns:
            Gated tensor of the same shape as input
        """
        gates = self.get_gates(is_train, x.shape)
        return x * gates

    def get_gates(self, is_train: bool, shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Sample gate activations in [0, 1] interval."""
        low, high = self.stretch_limits

        if is_train:
            # Sample from Hard Concrete distribution during training
            if shape is None:
                shape = (1, self.num_heads, 1, 1)

            noise = torch.rand(shape, device=self.log_a.device)
            noise = torch.clamp(noise, self.eps, 1.0 - self.eps)

            concrete = torch.sigmoid(
                (torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature
            )
        else:
            # Deterministic gates at inference
            concrete = torch.sigmoid(self.log_a)

        # Stretch and clip
        stretched = concrete * (high - low) + low
        clipped = torch.clamp(stretched, 0, 1)

        if self.hard and not is_train:
            # Binarize at inference if hard=True
            hard_concrete = (clipped > 0.5).float()
            # Straight-through estimator
            clipped = clipped + (hard_concrete - clipped).detach()

        return clipped

    def get_penalty(self) -> torch.Tensor:
        """
        Compute L0 penalty (expected number of non-zero gates).

        Returns:
            Scalar tensor with the L0 regularization term
        """
        if self.l0_penalty == 0:
            return torch.tensor(0.0, device=self.log_a.device)

        low, high = self.stretch_limits
        assert low < 0.0, "Lower stretch limit must be negative for L0 penalty"

        # Compute p(gate > 0) = P(stretched_concrete > 0)
        p_open = torch.sigmoid(self.log_a - self.temperature * torch.log(torch.tensor(-low / high)))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        # L0 regularization: expected number of open gates
        l0_reg = self.l0_penalty * torch.sum(p_open)

        return l0_reg

    def get_sparsity_rate(self, is_train: bool = False) -> torch.Tensor:
        """Compute fraction of gates that are active (non-zero)."""
        gates = self.get_gates(is_train)
        is_nonzero = (gates > 0).float()
        return torch.mean(is_nonzero)

    def get_gate_values(self) -> torch.Tensor:
        """Get current gate values (deterministic)."""
        return self.get_gates(is_train=False).squeeze()

    def prune_heads(self, threshold: float = 0.01) -> torch.Tensor:
        """
        Return mask indicating which heads to keep.

        Args:
            threshold: Heads with gate values below this are pruned

        Returns:
            Binary mask [num_heads] where 1 = keep, 0 = prune
        """
        gate_values = self.get_gate_values()
        return (gate_values > threshold).long()
