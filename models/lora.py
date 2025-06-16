"""
Implementation of Low-Rank Adaptation (LoRA) for transformer models
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        rank: Rank of the low-rank matrices
        alpha: Scaling factor
        activation: Whether to apply GELU activation (Modified non-linear LoRA)
    """

    def __init__(self, in_dim, out_dim, rank=8, alpha=16, activation=True):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank))  # Low-rank matrix A
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # Low-rank matrix B
        self.scale = alpha / rank  # Scaling factor
        self.non_linearity = nn.GELU()  # Activation function
        self.activation = activation  # Whether to use activation (Modified non-linear LoRA)

    def forward(self, x):
        """
        Forward pass: BAx*scale with optional activation
        B and A are reversed
        """
        if self.activation:
            return self.non_linearity((x @ (self.B.transpose(0,1) @ self.A.transpose(0,1)))) * self.scale
        else:
            return (x @ self.A @ self.B) * self.scale


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a linear layer with a LoRA layer

    Args:
        linear_layer: Original linear layer to augment
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        activation: Whether LoRA uses activation (LoRA or Modified NL LoRA)
    """

    def __init__(self, linear_layer, rank=8, alpha=16, activation=True):
        super().__init__()
        self.linear = linear_layer  # Original linear layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank, alpha,
            activation=activation
        )

    def forward(self, x):
        """Forward pass: original output + LoRA output"""
        return self.linear(x) + self.lora(x)