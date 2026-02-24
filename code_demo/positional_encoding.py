# =============================================================================
# Positional Encoding (Vaswani et al., 2017, Section 3.5)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of Sinusoidal Positional Encoding.
#
# The Transformer architecture contains no recurrence and no convolution,
# so it has absolutely no built-in notion of token order. Without positional
# encoding, the sentence "the cat sat on the mat" and "mat the on sat cat the"
# would produce identical representations.
#
# This module injects position information into the token embeddings by adding
# fixed sinusoidal signals of varying frequencies. The result is that each
# position in the sequence gets a unique, deterministic "fingerprint" that
# the model can learn to interpret.
#
# Formulas (paper Section 3.5):
#     PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017, Section 3.5).

    Because the Transformer's self-attention is permutation-invariant
    (swapping two tokens in the input and swapping the corresponding
    rows in Q, K, V produces the same output), it cannot distinguish
    "the cat chased the dog" from "the dog chased the cat."

    This class generates a fixed (non-learned) encoding matrix where
    each row is a unique positional signature, using sine and cosine
    functions at geometrically increasing wavelengths.

    Encoding formulas::

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    where:
        pos  = position index in the sequence (0, 1, 2, …)
        i    = dimension index (0, 1, 2, …, d_model/2 - 1)

    Why sin/cos instead of learned embeddings?
        1. **Generalization to unseen lengths:** Since the functions are
           continuous, the model can extrapolate to sequence lengths longer
           than those seen during training.
        2. **Relative position via linear transformation:** For any fixed
           offset k, PE(pos+k) can be expressed as a linear function of
           PE(pos), which helps the model learn relative position relationships.
        3. **No extra parameters:** The encoding is deterministic and
           adds zero learnable parameters to the model.

    How it's used in the Transformer::

        token_embeddings = Embedding(tokens) × √d_model
        model_input      = token_embeddings + PositionalEncoding(positions)

    Attributes:
        d_model (int):
            Dimensionality of the model's embedding space. Must be even
            so that dimensions can be paired into (sin, cos). The paper
            uses 512.
        pe (torch.Tensor):
            Pre-computed positional encoding buffer of shape [max_len, d_model].
            Registered as a non-parameter buffer so it moves with the model
            to GPU/CPU automatically.

    Example::

        pe = PositionalEncoding(d_model=512)
        embeddings = torch.randn(2, 10, 512)   # [batch=2, seq_len=10, 512]
        encoded = pe(embeddings)                # [2, 10, 512] — with position info
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000):
        """
        Initialize the Positional Encoding module.

        Args:
            d_model: Dimensionality of the model embedding. Must be even
                     (each dimension pair uses one sin and one cos).
                     Default is 512 as in the paper.
            max_len: Maximum sequence length to pre-compute encodings for.
                     Default is 5000, which is sufficient for most applications.

        Raises:
            AssertionError: If d_model is not even.
        """
        super().__init__()
        assert d_model % 2 == 0, \
            f"d_model ({d_model}) must be even for sin/cos positional encoding"
        self.d_model = d_model

        # Create positional encoding buffer [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len - 1] as a column vector
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Dimension indices: [0, 2, 4, ..., d_model - 2]
        dim_indices = torch.arange(0, d_model, 2).float()  # [d_model/2]

        # Denominator: 10000^(2i / d_model)
        # Computed as exp(2i * ln(10000) / d_model) for numerical stability
        div_term = torch.exp(dim_indices * (-math.log(10000.0) / d_model))  # [d_model/2]

        # Angle matrix: each entry is  pos / 10000^(2i / d_model)
        angles = pos * div_term  # [max_len, d_model/2]

        # Even dimensions (0, 2, 4, ...) get sine
        pe[:, 0::2] = torch.sin(angles)

        # Odd dimensions (1, 3, 5, ...) get cosine
        pe[:, 1::2] = torch.cos(angles)

        # Register as buffer (not a parameter, so it doesn't get gradients)
        # This ensures it moves with the model to GPU/CPU automatically
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Data flow::

            x (token embeddings)    [batch, seq_len, d_model]
            +                       element-wise addition
            PE (positional encoding) [1, seq_len, d_model]  (broadcast over batch)
            =
            output                  [batch, seq_len, d_model]

        Where this sits in the Transformer::

            tokens → Embedding × √d_model → [+ THIS PE] → Encoder / Decoder
                                                ▲
                                        position info injected here

        Note:
            The paper adds (not concatenates!) the positional encoding to
            the token embeddings. This works because both live in the same
            d_model-dimensional space, and the model learns to disentangle
            positional information from semantic information through training.

        Args:
            x: Token embeddings of shape [batch_size, seq_len, d_model].
               These should already be scaled by √d_model (as the paper
               specifies) before being passed to this method.

        Returns:
            torch.Tensor of shape [batch_size, seq_len, d_model] — the input
            embeddings with positional information added.

        Raises:
            RuntimeError: If seq_len exceeds the pre-computed max_len.
        """
        seq_len = x.shape[1]

        # Ensure we have pre-computed encodings for this sequence length
        if seq_len > self.pe.shape[1]:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds maximum positional encoding "
                f"length {self.pe.shape[1]}. Increase max_len when initializing."
            )

        # Add positional encoding (broadcast over batch dimension)
        # self.pe shape: [1, max_len, d_model]
        # We slice to get: [1, seq_len, d_model]
        return x + self.pe[:, :seq_len, :]

    def __repr__(self) -> str:
        return f"PositionalEncoding(d_model={self.d_model})"
