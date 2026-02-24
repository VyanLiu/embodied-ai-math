# =============================================================================
# Re-implementation of "Attention Is All You Need" (Vaswani et al., 2017)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of Multi-Head Attention.
#
# This module provides the multi-head attention mechanism described in
# Section 3.2 of the original Transformer paper, implemented using PyTorch
# for GPU acceleration.
#
# It supports three Transformer attention patterns via its forward(x, y) API:
#   • Self-Attention   — forward(x, x)      (encoder self-attention)
#   • Masked Self-Attn — forward(x, x)      (decoder, with masked=True)
#   • Cross-Attention  — forward(enc_out, dec)  (decoder cross-attention)
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer (Vaswani et al., 2017, Section 3.2).

    This class implements the general multi-head attention mechanism that
    underpins every attention sub-layer in the Transformer architecture.
    By accepting two inputs (x for Keys/Values, y for Queries), it
    naturally supports all three attention configurations used in the paper:

    ┌───────────────────┬──────────────┬──────────────┬──────────────────────┐
    │ Configuration     │ x (K, V src) │ y (Q src)    │ Usage                │
    ├───────────────────┼──────────────┼──────────────┼──────────────────────┤
    │ Encoder Self-Attn │ encoder in   │ encoder in   │ forward(x, x)        │
    │ Masked Self-Attn  │ decoder in   │ decoder in   │ forward(x, x) masked │
    │ Cross-Attention   │ encoder out  │ decoder state│ forward(enc, dec)    │
    └───────────────────┴──────────────┴──────────────────────┘

    Core formulas (paper Section 3.2):
        head_i       = Attention(y W_Q_i,  x W_K_i,  x W_V_i)
        Attention(Q, K, V) = softmax(Q K^T / √d_k) V
        MultiHead    = Concat(head_1, …, head_h) W_O

    Attributes:
        d_model (int):
            Dimensionality of the model's embedding space. The paper uses 512.
        num_heads (int):
            Number of parallel attention heads. The paper uses 8.
        d_k (int):
            Dimensionality per head (d_model // num_heads). The paper uses 64.
        masked (bool):
            When True, a causal (lower-triangular) mask is applied so that
            position t can only attend to positions ≤ t.  Required for the
            decoder's auto-regressive self-attention.
        w_q, w_k, w_v (nn.Linear):
            Learned projection matrices for Query, Key, Value.
        w_o (nn.Linear):
            Output projection matrix that recombines the concatenated head outputs.

    Example::

        # Encoder self-attention (unmasked)
        enc_attn = MultiHeadAttention(d_model=512, num_heads=8, masked=False)
        out = enc_attn(x, x)          # Q, K, V all from x

        # Decoder masked self-attention
        dec_attn = MultiHeadAttention(d_model=512, num_heads=8, masked=True)
        out = dec_attn(dec_input, dec_input)

        # Decoder cross-attention (Q from decoder, K/V from encoder)
        cross_attn = MultiHeadAttention(d_model=512, num_heads=8, masked=False)
        out = cross_attn(encoder_output, decoder_state)
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, masked: bool = False):
        """
        Initialize the Multi-Head Attention layer.

        Args:
            d_model: Total dimensionality of the model embedding (must be
                     divisible by num_heads). Default is 512 as in the paper.
            num_heads: Number of parallel attention heads. Default is 8 as
                       in the paper.
            masked: Whether to apply a causal mask (for decoder self-attention).
                    Default is False (encoder self-attention).

        Raises:
            AssertionError: If d_model is not evenly divisible by num_heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.masked = masked

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention_weight = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a lower-triangular causal mask.

        Args:
            seq_len: Length of the target sequence.
            device: Device to place the mask on.

        Returns:
            torch.Tensor of shape [seq_len, seq_len] — boolean mask with
            True on and below the main diagonal, False above.
        """
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the d_model dimension into num_heads independent subspaces.

        Shape transformation:
            [batch, seq_len, d_model]
              ──reshape──▶ [batch, seq_len, num_heads, d_k]
              ──transpose──▶ [batch, num_heads, seq_len, d_k]

        Args:
            x: Projected tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor of shape [batch_size, num_heads, seq_len, d_k].
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_head — reassemble all heads into a single d_model vector.

        Shape transformation:
            [batch, num_heads, seq_len, d_k]
              ──transpose──▶ [batch, seq_len, num_heads, d_k]
              ──reshape──▶   [batch, seq_len, d_model]

        Args:
            x: Per-head attention output.

        Returns:
            torch.Tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Scaled Dot-Product Attention (Vaswani et al., Section 3.2.1).

        Formula:
            Attention(Q, K, V) = softmax( Q K^T / √d_k ) V

        Args:
            q: Query tensor,  shape [batch, num_heads, seq_len_q, d_k].
            k: Key tensor,    shape [batch, num_heads, seq_len_k, d_k].
            v: Value tensor,  shape [batch, num_heads, seq_len_k, d_k].
            mask: Optional boolean mask, broadcastable to
                  [batch, num_heads, seq_len_q, seq_len_k].

        Returns:
            tuple of (output, attention_weights):
                output:           shape [batch, heads, seq_len_q, d_k]
                attention_weights: shape [batch, heads, seq_len_q, seq_len_k]
        """
        # Compute raw attention scores: Q @ K^T
        # Shape: [batch, heads, seq_len_q, seq_len_k]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale by 1 / sqrt(d_k) to stabilize gradients
        attention_scores = attention_scores / (self.d_k ** 0.5)

        # Apply causal mask: set future positions to -inf so softmax yields 0
        if mask is not None:
            # Expand mask to match attention_scores shape
            # mask: [seq_len_q, seq_len_k] -> [1, 1, seq_len_q, seq_len_k]
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(0).unsqueeze(0), 
                float('-inf')
            )

        # Convert scores to probabilities (attention weights)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted combination of value vectors
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of Multi-Head Attention.

        Queries are derived from ``y``; Keys and Values are derived from ``x``.

        Args:
            x: Key / Value source tensor, shape [batch_size, seq_len_k, d_model].
            y: Query source tensor, shape [batch_size, seq_len_q, d_model].

        Returns:
            torch.Tensor of shape [batch_size, seq_len_q, d_model].
        """
        batch_size = y.shape[0]
        device = y.device

        # Step 1: Linear projections
        q = self.w_q(y)  # [batch, seq_len_q, d_model]
        k = self.w_k(x)  # [batch, seq_len_k, d_model]
        v = self.w_v(x)  # [batch, seq_len_k, d_model]

        # Step 2: Split into heads
        q = self._split_heads(q)  # [batch, num_heads, seq_len_q, d_k]
        k = self._split_heads(k)  # [batch, num_heads, seq_len_k, d_k]
        v = self._split_heads(v)  # [batch, num_heads, seq_len_k, d_k]

        # Step 3: Build causal mask if needed
        mask = None
        if self.masked:
            seq_len_q = q.shape[2]
            mask = self._causal_mask(seq_len_q, device)

        # Step 4: Scaled dot-product attention
        attn_output, self.attention_weight = self._scaled_dot_product_attention(
            q, k, v, mask=mask
        )

        # Step 5: Concatenate heads and apply output projection
        concat = self._merge_heads(attn_output)  # [batch, seq_len_q, d_model]
        output = self.w_o(concat)  # [batch, seq_len_q, d_model]
        return output
