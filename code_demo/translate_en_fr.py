"""
Simple English→French translation script using the saved Transformer model.

This script loads the trained checkpoint and vocabularies from:
  ../model_saved/transformer_translator_en_fr/

Usage examples:
  # One-off translation
  python code_demo/translate_en_fr.py --text "how are you?"

  # Interactive mode
  python code_demo/translate_en_fr.py --interactive

Notes:
- This relies on the same tokenization/vocabulary used during training.
- Translations are produced via greedy decoding for simplicity.
"""

import os
import json
import argparse
from typing import List

import numpy as np
import torch
import sys

# Import the Transformer model implementation
from transformer import Transformer


# Special token IDs (must match training)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class Vocabulary:
    """Minimal vocabulary to load/save token<->id maps compatible with training."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support multiple vocab serialization formats
        # 1) Our simple format: {"token_to_id": {...}}
        # 2) HuggingFace Tokenizers JSON: {"model": {"vocab": {...}}, ...}
        # 3) torchtext-like: {"stoi": {...}}
        if isinstance(data, dict) and "token_to_id" in data:
            token_to_id = data["token_to_id"]
        elif isinstance(data, dict) and isinstance(data.get("model"), dict) and "vocab" in data["model"]:
            token_to_id = data["model"]["vocab"]
        elif isinstance(data, dict) and "stoi" in data:
            token_to_id = data["stoi"]
        else:
            raise KeyError(
                "Unrecognized vocabulary JSON schema. Expected keys 'token_to_id' or 'model.vocab' or 'stoi'."
            )

        # Ensure IDs are ints (JSON may already store them as ints)
        self.token_to_id = {str(tok): int(idx) for tok, idx in token_to_id.items()}

        # Build reverse map id -> token
        id_to_token = {int(idx): tok for tok, idx in self.token_to_id.items()}
        # Ensure special tokens have their canonical IDs
        for tok, idx in [("<PAD>", PAD_ID), ("<BOS>", BOS_ID), ("<EOS>", EOS_ID)]:
            # If mapping differs, respect stored mapping (training time ground truth)
            if tok in self.token_to_id:
                id_to_token[self.token_to_id[tok]] = tok
        self.id_to_token = id_to_token

    def encode(self, text: str) -> List[int]:
        # Very simple whitespace tokenization to match training's vocab build
        ids = []
        for tok in text.strip().split():
            tid = self.token_to_id.get(tok, self.token_to_id.get("<UNK>", PAD_ID))
            ids.append(tid)
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i in (PAD_ID, BOS_ID):
                continue
            if i == EOS_ID:
                break
            toks.append(self.id_to_token.get(int(i), "<UNK>"))
        return " ".join(toks)


def greedy_decode(model: Transformer, src_ids: np.ndarray, max_len: int) -> np.ndarray:
    """Greedy decode a single example.

    Args:
        model: loaded Transformer
        src_ids: numpy array shape [1, S]
        max_len: maximum target length (including EOS)
    Returns:
        numpy array of generated token IDs (without initial BOS)
    """
    model_device = next(model.parameters()).device
    src = torch.tensor(src_ids, dtype=torch.long, device=model_device)

    generated = [BOS_ID]
    for _ in range(max_len - 1):
        dec_in = torch.tensor([generated], dtype=torch.long, device=model_device)
        with torch.no_grad():
            logits = model(src, dec_in)  # [1, T, V]
            next_token = torch.argmax(logits[0, -1], dim=-1).item()
        generated.append(next_token)
        if next_token == EOS_ID:
            break

    # drop leading BOS
    return np.array(generated[1:], dtype=np.int64)


def load_assets(save_dir: str):
    # Load config
    with open(os.path.join(save_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load vocabs
    en_vocab = Vocabulary()
    fr_vocab = Vocabulary()
    en_vocab.load(os.path.join(save_dir, "en_vocab.json"))
    fr_vocab.load(os.path.join(save_dir, "fr_vocab.json"))

    # Build model
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1),
    )

    # Load weights
    ckpt_path = os.path.join(save_dir, "transformer_en_fr.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    return model, en_vocab, fr_vocab, config


def translate(text_en: str, save_dir: str, device: str = None) -> str:
    model, en_vocab, fr_vocab, config = load_assets(save_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Encode source with EOS and pad
    src_ids = en_vocab.encode(text_en) + [EOS_ID]
    src_ids = src_ids[: config["src_max_len"]]
    src_ids = src_ids + [PAD_ID] * (config["src_max_len"] - len(src_ids))
    src = np.array([src_ids], dtype=np.int64)

    out_ids = greedy_decode(model, src, max_len=config["tgt_max_len"])  # np array
    return fr_vocab.decode(out_ids.tolist())


def main():
    parser = argparse.ArgumentParser(description="English→French translation (greedy)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "model_saved", "transformer_translator_en_fr"),
        help="Directory containing config.json, vocab jsons, and transformer_en_fr.pt",
    )
    parser.add_argument("--text", type=str, default=None, help="English input to translate")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        raise FileNotFoundError(f"Save dir not found: {args.save_dir}")

    if args.interactive:
        print("Enter English sentences (Ctrl+C to quit):")
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    continue
                fr = translate(line, args.save_dir, device=args.device)
                print(fr)
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
    else:
        # Fallback behavior improvements:
        # - If --text is provided: translate the single sentence
        # - Else if stdin is not a TTY (piped input): translate each non-empty line from stdin
        # - Else: start interactive mode automatically
        if args.text:
            print(translate(args.text, args.save_dir, device=args.device))
        else:
            if not sys.stdin.isatty():
                # Read lines from stdin, translate each
                for line in sys.stdin:
                    line = line.strip()
                    if not line:
                        continue
                    print(translate(line, args.save_dir, device=args.device))
            else:
                # Default to interactive if nothing provided
                print("No --text given; entering interactive mode. (Ctrl+C to quit)")
                try:
                    while True:
                        line = input("> ").strip()
                        if not line:
                            continue
                        fr = translate(line, args.save_dir, device=args.device)
                        print(fr)
                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")


if __name__ == "__main__":
    main()
