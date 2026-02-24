# =============================================================================
# Transformer Decoder Layer (Vaswani et al., 2017, Section 3.1)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of the Transformer Decoder Layer.
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from positionwise_feed_forward import PositionWiseFeedForward
from residual_unit_block import AddNorm


class TransformerDecoderLayer(nn.Module):
    """
    A single Transformer Decoder layer (Vaswani et al., 2017, Section 3.1).

    The decoder layer extends the encoder layer with an additional
    cross-attention sub-layer that allows the decoder to "look at" the
    encoder's output. This is the bridge between the source sequence
    (e.g., an English sentence) and the target sequence (e.g., a French
    translation).

    The three sub-layers, each wrapped by Add & Norm, are:

        1. **Masked Multi-Head Self-Attention** — The decoder attends to
           its own previous outputs, but with a causal mask that prevents
           position t from attending to any future position t+1, t+2, ….
           This preserves the auto-regressive property: during generation,
           the model can only use tokens it has already produced.

        2. **Multi-Head Cross-Attention** — Queries come from the decoder
           (sub-layer 1's output); Keys and Values come from the encoder
           output. This is how the decoder reads the source sequence.
           It answers the question: "Given what I've generated so far,
           which parts of the source are most relevant for predicting the
           next token?"

        3. **Position-wise Feed-Forward Network** — Same as in the encoder:
           a two-layer MLP with ReLU that transforms each position
           independently to add non-linearity and capacity.

    Architecture of one decoder layer::

        y ──────────────────────────┐
        │                           │  (residual 1)
        ▼                           │
        Masked Self-Attention       │
        │   Q, K, V all from y      │
        │   (causal mask applied)   │
        ▼                           ▼
        Add & Norm₁ ◄──── y + masked_attn_out
        │
        ├──────────────────────────┐
        │                          │  (residual 2)
        ▼                          │
        Cross-Attention            │
        │   Q from decoder (y)     │
        │   K, V from encoder (x)  │
        ▼                          ▼
        Add & Norm₂ ◄──── norm₁_out + cross_attn_out
        │
        ├──────────────────────────┐
        │                          │  (residual 3)
        ▼                          │
        Position-wise FFN          │
        │                          │
        ▼                          ▼
        Add & Norm₃ ◄──── norm₂_out + ffn_out
        │
        ▼
        output  [batch, tgt_seq_len, d_model]

    Encoder layer vs. Decoder layer — what's different?

        ┌──────────────────────┬──────────────────┬───────────────────┐
        │                      │  Encoder Layer   │  Decoder Layer    │
        ├──────────────────────┼──────────────────┼───────────────────┤
        │ Number of sub-layers │  2               │  3                │
        │ Self-Attention       │  Unmasked        │  Causal-masked    │
        │ Cross-Attention      │  ✗ (not needed)  │  ✓ (reads encoder)│
        │ FFN                  │  ✓               │  ✓                │
        │ Add & Norm units     │  2               │  3                │
        └──────────────────────┴──────────────────┴───────────────────┘

    Why is the self-attention masked in the decoder?
        During training, the entire target sequence is available, but the
        model must learn to predict each token using only the tokens that
        precede it (auto-regressive constraint). The causal mask enforces
        this by setting future-position attention scores to −∞ before the
        softmax, resulting in zero attention weight for future tokens.

    Why does cross-attention use encoder output for K and V?
        The encoder has already built a rich, context-aware representation
        of the source sequence. By using it as Keys and Values, the decoder
        can "search" through the source for the information most relevant
        to producing the next target token. The decoder's Queries determine
        *what* to look for; the encoder's Keys determine *where* to look;
        and the encoder's Values provide *what* to retrieve.

    Attributes:
        self_attention (MultiHeadAttention):
            Masked Multi-Head Self-Attention sub-layer. ``masked=True``
            enforces the causal constraint via a lower-triangular mask.
        cross_attention (MultiHeadAttention):
            Multi-Head Cross-Attention sub-layer. ``masked=False`` because
            the decoder is allowed to attend to all encoder positions.
        feed_forward (PositionWiseFeedForward):
            Two-layer FFN with ReLU activation (expand → activate → compress).
        add_norm1 (AddNorm):
            Residual + Layer Norm wrapping the masked self-attention.
        add_norm2 (AddNorm):
            Residual + Layer Norm wrapping the cross-attention.
        add_norm3 (AddNorm):
            Residual + Layer Norm wrapping the feed-forward network.

    Example::

        dec_layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)
        enc_out = torch.randn(2, 12, 512)  # encoder output [batch, src_len, d_model]
        y       = torch.randn(2, 10, 512)  # decoder input  [batch, tgt_len, d_model]
        out     = dec_layer(enc_out, y)    # [2, 10, 512]
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        """
        Initialize one Transformer Decoder layer.

        Creates three independently parameterized sub-layers. When multiple
        decoder layers are stacked, each gets its own instance with its own
        weights — they are NOT weight-shared.

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
        
        # Sub-layer 1: Masked Multi-Head Self-Attention (causal for decoder)
        self.self_attention = MultiHeadAttention(d_model, num_heads, masked=True)

        # Sub-layer 2: Multi-Head Cross-Attention (unmasked — full encoder visible)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, masked=False)

        # Sub-layer 3: Position-wise Feed-Forward Network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

        # Dropout layers applied before residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Add & Norm wrappers — one for each sub-layer
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one decoder layer.

        Applies masked self-attention, cross-attention, and feed-forward
        sub-layers in sequence, each followed by dropout, a residual connection
        and layer normalization.

        Data flow::

            y  [batch, tgt_seq_len, d_model]   (decoder input)
            x  [batch, src_seq_len, d_model]   (encoder output)
            │
            ├──── Masked Self-Attention(y, y)
            │         Q, K, V all from y
            │         Causal mask: position t sees only {0, …, t}
            │         → attn_output  [batch, tgt_seq_len, d_model]
            │
            ├──── Dropout(attn_output)
            │
            ├──── Add & Norm₁:  LayerNorm(y + dropout(attn_output))
            │         → y  [batch, tgt_seq_len, d_model]
            │
            ├──── Cross-Attention(x, y)
            │         Q from decoder (y)  — "what am I looking for?"
            │         K from encoder (x)  — "where should I look?"
            │         V from encoder (x)  — "what do I retrieve?"
            │         → cross_attn_output  [batch, tgt_seq_len, d_model]
            │
            ├──── Dropout(cross_attn_output)
            │
            ├──── Add & Norm₂:  LayerNorm(y + dropout(cross_attn_output))
            │         → y  [batch, tgt_seq_len, d_model]
            │
            ├──── FFN(y)
            │         Expand 512→2048 → ReLU → Compress 2048→512
            │         → ffn_output  [batch, tgt_seq_len, d_model]
            │
            ├──── Dropout(ffn_output)
            │
            ├──── Add & Norm₃:  LayerNorm(y + dropout(ffn_output))
            │         → y  [batch, tgt_seq_len, d_model]
            │
            ▼
            output  [batch, tgt_seq_len, d_model]

        After this layer, every target token's representation has been:
            1. Enriched with context from previously generated tokens
               (via masked self-attention — no future leakage).
            2. Enriched with relevant information from the source sequence
               (via cross-attention to the encoder output).
            3. Non-linearly transformed to increase representational
               capacity (via FFN).

        Args:
            x: Encoder output of shape [batch_size, src_seq_len, d_model].
               This provides the Keys and Values for the cross-attention
               sub-layer. It is the same tensor for every decoder layer
               in the stack — the encoder is run once and its output is
               reused by all N decoder layers.
            y: Decoder input of shape [batch_size, tgt_seq_len, d_model].
               For the first decoder layer this comes from the target
               embedding + positional encoding stage; for subsequent
               layers it is the output of the previous decoder layer.

        Returns:
            torch.Tensor of shape [batch_size, tgt_seq_len, d_model] — the
            decoder representation, ready for the next decoder layer or
            for the final linear + softmax output projection.
        """
        # Sub-layer 1: Masked Self-Attention → Dropout → Add & Norm
        attention_output = self.self_attention(y, y)
        y = self.add_norm1(y, self.dropout1(attention_output))

        # Sub-layer 2: Cross-Attention → Dropout → Add & Norm
        cross_attention_output = self.cross_attention(x, y)
        y = self.add_norm2(y, self.dropout2(cross_attention_output))

        # Sub-layer 3: Feed-Forward → Dropout → Add & Norm
        ffn_output = self.feed_forward(y)
        y = self.add_norm3(y, self.dropout3(ffn_output))

        return y

    def __repr__(self) -> str:
        return (f"DecoderLayer(\n"
                f"  self_attention={self.self_attention},\n"
                f"  cross_attention={self.cross_attention},\n"
                f"  feed_forward={self.feed_forward},\n"
                f"  add_norm_1={self.add_norm1},\n"
                f"  add_norm_2={self.add_norm2},\n"
                f"  add_norm_3={self.add_norm3}\n"
                f")")
