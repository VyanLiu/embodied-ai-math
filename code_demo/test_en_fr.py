# =============================================================================
# Test Demo — English→French Translation (Greedy Decoding)
# Loads the trained Transformer model and vocabularies, then translates
# test sentences from English to French using greedy (argmax) decoding.
#
# Usage:
#   python test_en_fr.py
#
# Prerequisites:
#   Run train_en_fr.py first to generate model_saved/ artifacts.
#
# Author: Chang Liu
# =============================================================================

import os
import sys
import csv
import json
import pickle

import numpy as np
import torch

# Ensure code_demo modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code_demo"))

from transformer import Transformer

# Subword Tokenization (BPE)
from tokenizers import Tokenizer

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


# ═══════════════════════════════════════════════════════════════════════════
#  Vocabulary (same as training)
# ═══════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Subword-level vocabulary using Byte-Pair Encoding (BPE) loaded from file."""

    def __init__(self):
        self.tokenizer = None

    def load(self, path: str):
        """Load a saved BPE tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, sentence: str) -> list[int]:
        """Convert a sentence string to a list of subword IDs."""
        return self.tokenizer.encode(sentence).ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def size(self) -> int:
        return self.tokenizer.get_vocab_size()


# ═══════════════════════════════════════════════════════════════════════════
#  Greedy Decoding
# ═══════════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src_tokens: np.ndarray,
    max_len: int = 20,
) -> np.ndarray:
    """
    Greedy auto-regressive decoding.

    At each step, feed the generated tokens so far into the decoder,
    take the argmax of the output distribution, and append it.

    Args:
        model:      Trained Transformer model.
        src_tokens: Source token IDs, shape [1, src_len].
        max_len:    Maximum number of tokens to generate.

    Returns:
        np.ndarray of generated token IDs, shape [max_len].
    """
    # Encode the source once
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(torch.from_numpy(src_tokens).to(next(model.parameters()).device))

    # Start with <BOS>
    generated = [BOS_ID]

    for _ in range(max_len):
        tgt_input = torch.tensor([generated], dtype=torch.long, device=next(model.parameters()).device)

        # Decode
        decoder_output = model.decode(encoder_output, tgt_input)

        # Project to vocabulary size
        logits = model.output_projection(decoder_output[:, -1, :])  # [1, vocab]
        probs = torch.softmax(logits, dim=-1)

        # Greedy: pick the token with the highest probability
        next_token = int(torch.argmax(probs, dim=-1).item())
        generated.append(next_token)

        # Stop if <EOS> is generated
        if next_token == EOS_ID:
            break

    return np.array(generated)


# ═══════════════════════════════════════════════════════════════════════════
#  Test Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def test(
    save_dir: str = "model_saved",
    csv_path: str = "en-fr.csv",
    num_test_samples: int = 20,
    max_seq_len: int = 12,
):
    """
    Load the saved model and test it on held-out sentence pairs.
    """
    # ── Load config ──────────────────────────────────────────────────────
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    src_max_len = config["src_max_len"]

    # ── Load vocabularies ────────────────────────────────────────────────
    en_vocab = Vocabulary()
    fr_vocab = Vocabulary()
    en_vocab.load(os.path.join(save_dir, "en_vocab.json"))
    fr_vocab.load(os.path.join(save_dir, "fr_vocab.json"))

    print(f"English vocab: {en_vocab.size} tokens")
    print(f"French  vocab: {fr_vocab.size} tokens")

    # ── Load model ───────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, "transformer_en_fr.pt")
    
    # Initialize model with config
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1)
    )
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"\nLoaded model from {model_path} on {device}")
    print(model)

    # ── Load test sentences from the TAIL of the dataset ─────────────────
    # (Training used the first N samples, so we grab from further in the file)
    print(f"\nLoading test sentences from {csv_path} ...")
    test_en = []
    test_fr = []
    skip_count = 0
    target_skip = 5000  # skip past training data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row["en"].strip()
            fr = row["fr"].strip()

            if not (1 <= len(en.split()) <= max_seq_len
                    and 1 <= len(fr.split()) <= max_seq_len):
                continue

            skip_count += 1
            if skip_count <= target_skip:
                continue

            test_en.append(en)
            test_fr.append(fr)

            if len(test_en) >= num_test_samples:
                break

    print(f"  Loaded {len(test_en)} test sentence pairs\n")

    # ── Run Translation ──────────────────────────────────────────────────
    print("=" * 70)
    print("  Translation Results (Greedy Decoding)")
    print("=" * 70)

    total_loss = 0.0

    for i, (en_sent, fr_sent) in enumerate(zip(test_en, test_fr)):
        # Tokenize source
        en_ids = en_vocab.encode(en_sent) + [EOS_ID]
        en_ids = en_ids[:src_max_len]
        en_ids = en_ids + [PAD_ID] * (src_max_len - len(en_ids))
        src = torch.tensor([en_ids], dtype=torch.long, device=device)

        # Greedy decode
        output_ids = greedy_decode(model, src.cpu().numpy(), max_len=config["tgt_max_len"])
        translation = fr_vocab.decode(output_ids.tolist())

        # Also compute teacher-forced loss for this sample
        fr_ids = [BOS_ID] + fr_vocab.encode(fr_sent) + [EOS_ID]
        tgt_max = config["tgt_max_len"]
        fr_ids = fr_ids[:tgt_max]
        fr_ids = fr_ids + [PAD_ID] * (tgt_max - len(fr_ids))
        tgt = torch.tensor([fr_ids], dtype=torch.long, device=device)

        dec_in = tgt[:, :-1]
        dec_lbl = tgt[:, 1:]
        
        with torch.no_grad():
            logits = model(src, dec_in)
            loss = model.compute_loss(logits, dec_lbl, pad_id=PAD_ID)
            total_loss += loss.item()

        print(f"\n  [{i+1}]")
        print(f"    English (input):     {en_sent}")
        print(f"    French  (reference): {fr_sent}")
        print(f"    French  (model):     {translation}")
        print(f"    Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(1, len(test_en))
    print(f"\n{'=' * 70}")
    print(f"  Average test loss: {avg_loss:.4f}")
    print(f"{'=' * 70}")

    print("\n  📝 Note: This is a small-scale NumPy demo. The translations")
    print("  will NOT be fluent — only the output-layer weights are trained.")
    print("  For real translation, use a full framework (PyTorch/TensorFlow)")
    print("  with backpropagation through all layers.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    CSV_PATH = "../../../DataSet/kaggle/en-fr.csv"

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset file '{CSV_PATH}' not found!")
        print(f"Please download from:")
        print(f"  https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset")
        sys.exit(1)

    if not os.path.exists("../model_saved/transformer_translator_en_fr"):
        print("ERROR: No trained model found! Run train_en_fr.py first.")
        sys.exit(1)

    test(
        save_dir="../model_saved/transformer_translator_en_fr",
        csv_path=CSV_PATH,
        num_test_samples=30000,
        max_seq_len=64,
    )