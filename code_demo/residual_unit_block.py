# =============================================================================
# Add & Norm — Residual Connection + Layer Normalization (Vaswani et al., 2017)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of the Add & Norm unit.
#
# This module implements the "Add & Norm" unit that wraps every sub-layer
# (Attention, Feed-Forward) inside a Transformer encoder or decoder block.
#
# Formula:  AddNorm(x, SubLayer) = LayerNorm(x + SubLayer(x))
#
# Two critical ideas work together here:
#   1. Residual connection (He et al., 2016) — the skip path "x +" ensures
#      gradients can flow directly through the network, enabling training
#      of very deep stacks (6+ layers).
#   2. Layer Normalization (Ba et al., 2016) — normalizes across the
#      feature dimension (d_model) to stabilize training dynamics.
#
# Reference: https://arxiv.org/abs/1706.03762  (Section 5.4 & Figure 1)
# =============================================================================

import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Add & Norm — Residual Connection followed by Layer Normalization.

    Every sub-layer in the Transformer (both Attention and FFN) is wrapped
    by this unit.  The data flow for a single Transformer sub-layer is::

        x ──────────────────────┐
        │                       │  (residual / skip connection)
        ▼                       │
      SubLayer(x)               │
        │                       │
        ▼                       ▼
        +  ◄────────────────── x   →  x + SubLayer(x)
        │
        ▼
      LayerNorm  →  LayerNorm(x + SubLayer(x))
        │
        ▼
       output

    Why residual connections?
        Without skip connections, gradients must pass through every
        transformation layer during backpropagation.  As depth increases,
        gradients tend to vanish or explode.  The residual path provides a
        "gradient highway" — even if a sub-layer's gradient is small, the
        identity shortcut carries the gradient through undiminished.

    Why Layer Normalization (not Batch Normalization)?
        • Batch Norm normalizes across the batch dimension — it needs large
          batch sizes and struggles with variable-length sequences.
        • Layer Norm normalizes across the *feature* (d_model) dimension
          for each individual sample, making it batch-size independent and
          naturally suited for sequence models.

    Layer Normalization formula::

        μ  = mean(x, axis=-1)
        σ² = var(x,  axis=-1)
        x̂  = (x − μ) / √(σ² + ε)
        y  = γ ⊙ x̂ + β          (element-wise scale and shift)

    Attributes:
        d_model (int):
            Feature dimensionality — must match the model's embedding size.
        layer_norm (nn.LayerNorm):
            PyTorch LayerNorm module that normalizes over the last dimension.

    Example::

        add_norm  = AddNorm(d_model=512)
        x         = torch.randn(2, 10, 512)   # [batch, seq_len, d_model]
        sublayer_out = some_sublayer(x)        # [batch, seq_len, d_model]
        out       = add_norm(x, sublayer_out)  # [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        """
        Initialize the Add & Norm layer.

        Args:
            d_model: Feature dimensionality of the model embeddings.
                     Must match the d_model used in attention / FFN so that
                     the residual addition (x + sublayer_output) is valid.
                     Default is 512 as in the paper.
            eps:     Small constant for numerical stability in the
                     normalization denominator. Default is 1e-6.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # PyTorch's LayerNorm implements the full LayerNorm operation including
        # the learnable affine parameters (weight/gamma and bias/beta)
        # It normalizes over the last dimension by default
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Apply residual connection then layer normalization.

        Implements the post-norm formulation from the original paper::

            output = LayerNorm(x + SubLayer(x))

        Where ``x`` is the sub-layer's input (for the skip connection) and
        ``sublayer_output`` is the sub-layer's output (Attention or FFN).

        Where this sits in a Transformer encoder block::

            x ──→ [Attention] ──→ Add & Norm ──→ [FFN] ──→ Add & Norm ──→ out
                      │              ▲                │          ▲
                      └── residual ──┘                └── residual ──┘

        Note:
            Some modern architectures (GPT-2, LLaMA) use *pre-norm* instead:
                output = x + SubLayer(LayerNorm(x))
            Pre-norm often improves training stability for very deep models,
            but the original Transformer uses post-norm as implemented here.

        Args:
            x: Original input *before* the sub-layer, shape
               [batch_size, seq_len, d_model]. This is the residual path.
            sublayer_output: Output of the sub-layer (Attention or FFN),
                             shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor of shape [batch_size, seq_len, d_model] — the
            normalized result, ready for the next sub-layer.
        """
        # Residual connection: add the original input to the sub-layer output
        residual = x + sublayer_output

        # Layer normalization over the combined result
        return self.layer_norm(residual)

    def __repr__(self) -> str:
        return f"AddNorm(d_model={self.d_model}, eps={self.eps})"
