# =============================================================================
# Position-wise Feed-Forward Network (Vaswani et al., 2017, Section 3.3)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of the Position-wise Feed-Forward Network.
#
# This module implements the two-layer fully-connected network that is applied
# to each position separately and identically. It is the second sub-layer
# inside every Transformer encoder and decoder block.
#
# Formula:  FFN(x) = ReLU(x W₁ + b₁) W₂ + b₂
#
# "Position-wise" means the same weights are shared across all sequence
# positions, but each position is transformed independently — there is NO
# information exchange between positions in this layer. Cross-position
# communication is the sole responsibility of the attention sub-layer.
#
# This is mathematically equivalent to two stacked 1×1 convolutions.
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Vaswani et al., 2017, Section 3.3).

    Inside every Transformer block the data flows through two sub-layers:

        1. Multi-Head Attention  — gathers context across positions.
        2. **This FFN**          — processes each position independently.

    Architecture::

        x ──→ Linear(d_model → d_ff) ──→ ReLU ──→ Linear(d_ff → d_model) ──→ out
              (expand to 2048)                     (compress back to 512)

    Formula:
        FFN(x) = ReLU(x W₁ + b₁) W₂ + b₂

    Why is this layer needed?
        Attention is powerful at *mixing* information between tokens, but its
        core operation (softmax-weighted average of Values) is essentially
        linear with respect to V.  The FFN introduces the **non-linearity**
        (ReLU) that gives the Transformer its universal-approximation power.

        Intuition: Attention = "gather relevant context."
                   FFN       = "think about what was gathered."

    Why the 4× expansion (d_ff = 4 × d_model)?
        The hidden layer expands the representation to a higher-dimensional
        space (512 → 2048) so the ReLU can carve out richer, more complex
        feature interactions before compressing back to d_model.  This
        expand → activate → compress pattern appears throughout deep learning
        (inverted bottleneck in MobileNetV2, SwiGLU in LLaMA, etc.).

    Attributes:
        d_model (int):
            Input and output dimensionality. The paper uses 512.
        d_ff (int):
            Inner (hidden) dimensionality. The paper uses 2048 (= 4 × d_model).
        linear1 (nn.Linear):
            First linear layer, expands d_model → d_ff.
        linear2 (nn.Linear):
            Second linear layer, compresses d_ff → d_model.

    Example::

        ffn = PositionWiseFeedForward(d_model=512, d_ff=2048)
        x   = torch.randn(2, 10, 512)   # [batch=2, seq_len=10, d_model=512]
        out = ffn(x)                     # [2, 10, 512]  — same shape as input
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        """
        Initialize the Position-wise Feed-Forward Network.

        Both linear layers use Kaiming/He initialization by default (PyTorch's
        default for nn.Linear), which is well-suited for networks with ReLU
        activations.

        Args:
            d_model: Dimensionality of model embeddings (input & output).
                     Must match the d_model used in the attention layer so
                     that residual connections (x + FFN(x)) are valid.
                     Default is 512 as in the paper.
            d_ff:    Inner (hidden) dimensionality (the "expansion" factor).
                     Default is 2048 (= 4 × 512) as in the paper.
            dropout: Dropout rate applied after the first linear layer and ReLU.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Layer 1: expand  d_model → d_ff   (512 → 2048)
        self.linear1 = nn.Linear(d_model, d_ff)

        # Layer 2: compress  d_ff → d_model  (2048 → 512)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout layer (internal to FFN, applied after ReLU)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward network to each position.

        Data flow::

            x   ─── [batch, seq_len, 512]
            │
            ├── Linear1 + ReLU + Dropout ──→ [batch, seq_len, 2048]   (expand + activate + dropout)
            │
            ├── Linear2 ──────────────────→ [batch, seq_len, 512]    (compress)
            │
            out ── [batch, seq_len, 512]

        Key insight — "position-wise":
            The linear layers broadcast over the batch and seq_len dimensions.
            This means every token at every position goes through the *exact
            same* learned linear transformation, but completely independently
            of other tokens.

            Mathematically, this is identical to running two 1×1
            convolutions (kernel_size=1) along the sequence — a pattern
            also seen in convolutional feed-forward blocks (e.g., NiN).

        Where this sits in the Transformer::

            Input ──→ [Attention] ──→ Add & Norm ──→ [THIS FFN] ──→ Add & Norm ──→ Output
                          ▲                               ▲
                          └─── residual ──┘               └─── residual ──┘

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor of shape [batch_size, seq_len, d_model] — same shape
            as input, ready for the downstream residual connection and
            layer normalization.
        """
        # Layer 1: Expand to higher-dimensional space + apply ReLU non-linearity + Dropout
        hidden = self.dropout(F.relu(self.linear1(x)))  # [batch, seq_len, d_ff]

        # Layer 2: Compress back to model dimension
        output = self.linear2(hidden)  # [batch, seq_len, d_model]

        return output

    def __repr__(self) -> str:
        return f"PositionWiseFeedForward(d_model={self.d_model}, d_ff={self.d_ff})"
