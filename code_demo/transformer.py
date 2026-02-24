# =============================================================================
# Transformer (Vaswani et al., 2017)
# Author: Chang Liu
#
# PyTorch CUDA-accelerated implementation of the full Transformer model.
#
# This script assembles the full Transformer (Encoder + Decoder) from the
# building-block modules in this project.
#
# Architecture (paper Figure 1):
#
#   SOURCE tokens                          TARGET tokens (shifted right)
#       │                                       │
#   Embedding × √d_model                  Embedding × √d_model
#       │                                       │
#   + Positional Encoding                  + Positional Encoding
#       │                                       │
#   ┌───▼──────────────┐                  ┌───▼──────────────┐
#   │ Encoder Layer ×N │                  │ Decoder Layer ×N │
#   │  (Self-Attn+FFN) │──── enc_out ───▶│  (Masked Self,   │
#   └──────────────────┘                  │   Cross, FFN)    │
#                                         └───┬──────────────┘
#                                             │
#                                         Linear → Softmax
#                                             │
#                                         next-token probabilities
#
# Reference: https://arxiv.org/abs/1706.03762
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import PositionalEncoding
from transformer_encoder_layer import TransformerEncoderLayer
from transformer_decoder_layer import TransformerDecoderLayer


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks (Vaswani et al., 2017).

    This class wires together every component of the original Transformer
    architecture into a single end-to-end model:

        • Source & target token embeddings (with √d_model scaling)
        • Sinusoidal positional encoding
        • N encoder layers (self-attention + FFN)
        • N decoder layers (masked self-attention + cross-attention + FFN)
        • Final linear projection to produce logits over target vocabulary

    Full data flow::

        src_tokens [batch, src_len]          tgt_tokens [batch, tgt_len]
            │                                     │
        Embedding Lookup                     Embedding Lookup
            │                                     │
        × √d_model                            × √d_model
            │                                     │
        + Positional Encoding                + Positional Encoding
            │                                     │
        ┌───▼──────────────────┐            ┌───▼──────────────────┐
        │  EncoderLayer 1      │            │  DecoderLayer 1      │
        │  EncoderLayer 2      │            │  DecoderLayer 2      │
        │       ...            │── enc ──▶  │       ...            │
        │  EncoderLayer N      │   out      │  DecoderLayer N      │
        └──────────────────────┘            └───┬──────────────────┘
                                                │
                                            Linear(d_model → tgt_vocab)
                                                │
                                            logits [batch, tgt_len, tgt_vocab]

    Training objective:
        The model is trained with **teacher forcing** — at each decoder
        position t, the ground-truth token at position t-1 is fed as input
        (rather than the model's own prediction). The loss is the standard
        **cross-entropy** between the predicted logits and the target tokens.

    Attributes:
        d_model (int):       Model dimensionality (paper: 512).
        num_heads (int):     Attention heads per layer (paper: 8).
        d_ff (int):          FFN inner dimensionality (paper: 2048).
        num_layers (int):    Number of encoder/decoder layers (paper: 6).
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        src_embedding (nn.Embedding): Source token embeddings.
        tgt_embedding (nn.Embedding): Target token embeddings.
        pos_encoding (PositionalEncoding): Shared positional encoding module.
        encoder_layers (nn.ModuleList): Encoder stack.
        decoder_layers (nn.ModuleList): Decoder stack.
        output_projection (nn.Linear): Final projection layer [d_model, tgt_vocab].

    Example::

        model = Transformer(src_vocab_size=8000, tgt_vocab_size=8000)
        src = torch.tensor([[1, 45, 302, 7, 2]])       # [batch=1, src_len=5]
        tgt = torch.tensor([[1, 88, 156, 2]])           # [batch=1, tgt_len=4]
        logits = model(src, tgt)                        # [1, 4, 8000]
    """

    def __init__(
        self,
        src_vocab_size: int = 8000,
        tgt_vocab_size: int = 8000,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """
        Initialize the full Transformer model.

        Args:
            src_vocab_size: Number of tokens in the source vocabulary.
            tgt_vocab_size: Number of tokens in the target vocabulary.
            d_model:    Model embedding dimensionality. Default 512.
            num_heads:  Number of attention heads. Default 8.
            d_ff:       FFN hidden dimensionality. Default 2048.
            num_layers: Number of encoder/decoder layers (N). Default 6.
            dropout:    Dropout rate. Default 0.1 as in the paper.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # ── Token Embeddings ─────────────────────────────────────────────
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # ── Positional Encoding (shared between encoder and decoder) ─────
        self.pos_encoding = PositionalEncoding(d_model=d_model)

        # ── Dropout ──────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Encoder Stack (N layers) ─────────────────────────────────────
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Decoder Stack (N layers) ─────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Output Projection (d_model → tgt_vocab_size) ────────────────
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize weights (Xavier/Glorot initialization)
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _scale_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Scale embeddings by √d_model as in the paper."""
        return x * math.sqrt(self.d_model)

    def encode(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder: embedding → positional encoding → N encoder layers.

        Data flow::

            src_tokens [batch, src_len]
                │
            Embedding Lookup  →  [batch, src_len, d_model]
                │
            × √d_model
                │
            + Positional Encoding
                │
            EncoderLayer 1 → … → EncoderLayer N
                │
            encoder_output  [batch, src_len, d_model]

        Args:
            src_tokens: Source token IDs, shape [batch_size, src_len].

        Returns:
            torch.Tensor of shape [batch_size, src_len, d_model].
        """
        # Embed + scale
        x = self.src_embedding(src_tokens)
        x = self._scale_embeddings(x)
        
        # Add positional encoding and dropout
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder stack
        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def decode(
        self, 
        encoder_output: torch.Tensor, 
        tgt_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the decoder: embedding → positional encoding → N decoder layers.

        Data flow::

            tgt_tokens [batch, tgt_len]
                │
            Embedding Lookup  →  [batch, tgt_len, d_model]
                │
            × √d_model
                │
            + Positional Encoding
                │
            DecoderLayer 1(enc_out, y) → … → DecoderLayer N(enc_out, y)
                │
            decoder_output  [batch, tgt_len, d_model]

        Each decoder layer receives the *same* encoder output — the encoder
        is run once and its result is reused by all N decoder layers.

        Args:
            encoder_output: Output of the encoder stack,
                            shape [batch_size, src_len, d_model].
            tgt_tokens:     Target token IDs, shape [batch_size, tgt_len].

        Returns:
            torch.Tensor of shape [batch_size, tgt_len, d_model].
        """
        # Embed + scale
        y = self.tgt_embedding(tgt_tokens)
        y = self._scale_embeddings(y)
        
        # Add positional encoding and dropout
        y = self.pos_encoding(y)
        y = self.dropout(y)

        # Pass through decoder stack (each layer reads from encoder_output)
        for layer in self.decoder_layers:
            y = layer(encoder_output, y)

        return y

    def forward(
        self, 
        src_tokens: torch.Tensor, 
        tgt_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Complete Transformer forward pass: encode → decode → project.

        This implements teacher-forcing: the full target sequence is provided
        to the decoder (shifted right by one position), and the causal mask
        inside the decoder prevents future-token leakage.

        Data flow::

            src_tokens ──→ encode() ──→ encoder_output
                                              │
            tgt_tokens ──→ decode(enc_out) ──→ decoder_output
                                              │
                                         Linear(d_model → vocab)
                                              │
                                         logits [batch, tgt_len, vocab]

        Args:
            src_tokens: Source token IDs, shape [batch_size, src_len].
            tgt_tokens: Target token IDs (teacher forcing input),
                        shape [batch_size, tgt_len].

        Returns:
            torch.Tensor of shape [batch_size, tgt_len, tgt_vocab_size] —
            logits over the target vocabulary at each decoder position.
        """
        # Step 1: Encode the source sequence
        encoder_output = self.encode(src_tokens)

        # Step 2: Decode the target sequence (with cross-attention to encoder)
        decoder_output = self.decode(encoder_output, tgt_tokens)

        # Step 3: Project to vocabulary size (logits, no softmax for numerical stability)
        logits = self.output_projection(decoder_output)

        return logits

    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        pad_id: int = 0
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing option.

        Args:
            logits: Model output logits, shape [batch_size, tgt_len, tgt_vocab_size].
            targets: Ground-truth token IDs, shape [batch_size, tgt_len].
            pad_id: Padding token ID to ignore in loss computation.

        Returns:
            torch.Tensor — scalar loss value.
        """
        # Reshape for cross_entropy: [batch * tgt_len, vocab] and [batch * tgt_len]
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # FIX: Added label_smoothing=0.1 to match paper's regularization strategy.
        # DOCUMENTATION: Paper Section 5.4 - "During training, we employed label smoothing
        # of value εls = 0.1. This hurts perplexity, as the model learns to be more unsure,
        # but improves accuracy and BLEU score."
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=pad_id,
            label_smoothing=0.1
        )

        return loss

    def generate(
        self,
        src_tokens: torch.Tensor,
        max_len: int = 50,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate target sequence using greedy decoding (auto-regressive).

        Args:
            src_tokens: Source token IDs, shape [batch_size, src_len].
            max_len: Maximum generation length.
            bos_id: Beginning-of-sequence token ID.
            eos_id: End-of-sequence token ID.
            pad_id: Padding token ID.

        Returns:
            torch.Tensor of shape [batch_size, generated_len] — generated token IDs.
        """
        self.eval()
        device = src_tokens.device
        batch_size = src_tokens.shape[0]

        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        # Encode source once (cached for all generation steps)
        encoder_output = self.encode(src_tokens)

        with torch.no_grad():
            for _ in range(max_len - 1):
                # Get model predictions
                logits = self.decode(encoder_output, generated)
                logits = self.output_projection(logits)
                
                # Get probability distribution over last position
                probs = F.softmax(logits[:, -1, :], dim=-1)
                
                # Greedy: select most likely token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch, 1]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check if all sequences have generated EOS
                if (next_token == eos_id).all():
                    break

        return generated

    def __repr__(self) -> str:
        return (
            f"Transformer(\n"
            f"  src_vocab={self.src_vocab_size}, tgt_vocab={self.tgt_vocab_size},\n"
            f"  d_model={self.d_model}, heads={self.num_heads}, "
            f"d_ff={self.d_ff}, layers={self.num_layers}\n"
            f")"
        )
