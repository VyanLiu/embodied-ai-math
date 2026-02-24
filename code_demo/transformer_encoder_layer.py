# =============================================================================
# Transformer Encoder Layer (Vaswani et al., 2017, Section 3.1)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of the Transformer Encoder Layer.
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from positionwise_feed_forward import PositionWiseFeedForward
from residual_unit_block import AddNorm


class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder layer (Vaswani et al., 2017, Section 3.1).

    The Transformer encoder is built by stacking N identical layers. Each
    layer applies two sub-layers in sequence, and each sub-layer is wrapped
    by a residual connection followed by layer normalization (Add & Norm):

        1. **Multi-Head Self-Attention** — Every token attends to every
           other token in the sequence, building a context-aware
           representation. This is the mechanism by which long-range
           dependencies are captured in a single step (unlike RNNs, which
           must propagate information step-by-step).

        2. **Position-wise Feed-Forward Network** — A two-layer MLP applied
           independently and identically at each position. This provides the
           non-linearity and additional representational capacity that
           attention alone cannot supply.

    Architecture of one encoder layer::

        x ──────────────────────────┐
        │                           │  (residual 1)
        ▼                           │
        Multi-Head Self-Attention   │
        │   Q, K, V all from x      │
        ▼                           ▼
        Add & Norm₁ ◄──── x + attn_out
        │
        ├──────────────────────────┐
        │                          │  (residual 2)
        ▼                          │
        Position-wise FFN          │
        │                          │
        ▼                          ▼
        Add & Norm₂ ◄──── norm₁_out + ffn_out
        │
        ▼
        output  [batch, seq_len, d_model]

    Why two sub-layers with different roles?
        • **Self-Attention** = "gather relevant context."
          It mixes information *across* positions so each token can see
          the entire sequence. However, attention is essentially a weighted
          average — it is linear with respect to the Value vectors.

        • **FFN** = "think about what was gathered."
          It introduces non-linearity (ReLU) that gives the Transformer
          its universal-approximation power, processing each position's
          enriched representation independently.

    Why Add & Norm after each sub-layer?
        The residual connection (``x + SubLayer(x)``) provides a gradient
        highway for training deep stacks. Layer Normalization stabilizes
        the activation magnitudes, preventing internal covariate shift.
        Together they allow the paper's 6-layer encoder to train smoothly.

    Attributes:
        self_attention (MultiHeadAttention):
            Multi-Head Self-Attention sub-layer. ``masked=False`` because
            the encoder has no causal constraint — every position is
            allowed to attend to every other position.
        feed_forward (PositionWiseFeedForward):
            Two-layer FFN with ReLU activation (expand → activate → compress).
        add_norm1 (AddNorm):
            Residual connection + Layer Normalization wrapping the
            self-attention sub-layer.
        add_norm2 (AddNorm):
            Residual connection + Layer Normalization wrapping the
            feed-forward sub-layer.

    Example::

        layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        x = torch.randn(2, 10, 512)   # [batch=2, seq_len=10, d_model=512]
        out = layer(x)                # [2, 10, 512] — same shape, richer repr
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        """
        Initialize one Transformer Encoder layer.

        Creates independently parameterized sub-layers. When multiple
        encoder layers are stacked, each layer gets its own instance of
        this class with its own weights — they are NOT weight-shared.

        Args:
            d_model:   Dimensionality of model embeddings. Must be divisible
                       by num_heads. Default is 512 as in the paper.
            num_heads: Number of parallel attention heads. Each head operates
                       on a d_k = d_model // num_heads dimensional subspace.
                       Default is 8 as in the paper (d_k = 64).
            d_ff:      Inner dimensionality of the feed-forward network.
                       Default is 2048 (= 4 × d_model) as in the paper.
            dropout:   Dropout rate to be applied to the output of each sub-layer
                       before the residual connection and normalization.
                       Default is 0.1 as in the paper.
        """
        super().__init__()
        
        # Sub-layer 1: Multi-Head Self-Attention (unmasked for encoder)
        self.self_attention = MultiHeadAttention(d_model, num_heads, masked=False)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

        # Dropout layers applied before residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Add & Norm wrappers — one for each sub-layer
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one encoder layer.

        Applies self-attention and feed-forward sub-layers in sequence,
        each followed by dropout, a residual connection, and layer normalization.

        Data flow::

            x  [batch, seq_len, d_model]
            │
            ├──── Self-Attention(x, x)
            │         Q, K, V are ALL derived from x (self-attention)
            │         → attn_output  [batch, seq_len, d_model]
            │
            ├──── Dropout(attn_output)
            │
            ├──── Add & Norm₁:  LayerNorm(x + dropout(attn_output))
            │         → x  [batch, seq_len, d_model]
            │
            ├──── FFN(x)
            │         Expand 512→2048 → ReLU → Compress 2048→512
            │         → ffn_output  [batch, seq_len, d_model]
            │
            ├──── Dropout(ffn_output)
            │
            ├──── Add & Norm₂:  LayerNorm(x + dropout(ffn_output))
            │         → x  [batch, seq_len, d_model]
            │
            ▼
            output  [batch, seq_len, d_model]

        After this layer, every token's representation has been:
            1. Enriched with contextual information from all other tokens
               (via self-attention).
            2. Non-linearly transformed to increase representational
               capacity (via FFN).

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
               For the first encoder layer this comes from the positional
               encoding stage; for subsequent layers it is the output of
               the previous encoder layer.

        Returns:
            torch.Tensor of shape [batch_size, seq_len, d_model] — the
            context-enriched representation, ready for the next encoder
            layer or for consumption by the decoder / task head.
        """
        # Sub-layer 1: Self-Attention → Dropout → Add & Norm
        attention_output = self.self_attention(x, x)
        x = self.add_norm1(x, self.dropout1(attention_output))

        # Sub-layer 2: Feed-Forward → Dropout → Add & Norm
        ffn_output = self.feed_forward(x)
        x = self.add_norm2(x, self.dropout2(ffn_output))
        return x

    def __repr__(self) -> str:
        return (f"EncoderLayer(\n"
                f"  self_attention={self.self_attention},\n"
                f"  feed_forward={self.feed_forward},\n"
                f"  add_norm_1={self.add_norm1},\n"
                f"  add_norm_2={self.add_norm2}\n"
                f")")
