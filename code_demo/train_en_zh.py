# =============================================================================
# Training Demo — English→Chinese Translation (PyTorch CUDA)
# Uses the EN-ZH dataset
#
# This script:
#   1. Downloads / loads the en-zh.csv dataset
#   2. Builds word-level vocabularies for English (source) and Chinese (target)
#   3. Tokenizes and pads sentence pairs
#   4. Trains the Transformer using PyTorch with GPU acceleration
#   5. Saves the model + vocabularies to model_saved/
#
# Prerequisites:
#   pip install torch tokenizers tqdm sacrebleu
#   Place en-zh.csv in the project root, or let the script tell you where to put it.
#
# Author: Chang Liu
# =============================================================================

import os
import sys
import csv
import json
import time
from typing import Tuple, List, Dict
from datetime import datetime
from collections import Counter
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
# Use the unified AMP API (torch.amp) to avoid deprecation warnings
from torch.amp import autocast, GradScaler

# Subword Tokenization (BPE)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Progress bar library
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

# Optional: TensorBoard for real-time visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Note: TensorBoard not installed. Run: pip install tensorboard for real-time graphs")
    SummaryWriter = None

# BLEU metric (sacrebleu)
try:
    import sacrebleu  # type: ignore
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    try:
        print("Installing sacrebleu for BLEU computation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu", "-q"])
        import sacrebleu  # type: ignore
        SACREBLEU_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not install sacrebleu automatically: {e}")
        SACREBLEU_AVAILABLE = False

# Ensure code_demo modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code_demo"))

from transformer import Transformer


# ═══════════════════════════════════════════════════════════════════════════
#  Constants / Special Tokens
# ═══════════════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════
#  Vocabulary Builder
# ═══════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Subword-level vocabulary using Byte-Pair Encoding (BPE)."""

    def __init__(self, max_vocab_size: int = 32000):
        self.max_vocab_size = max_vocab_size
        # Initialize BPE model with UNK token
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        # Use Whitespace pre-tokenizer for English
        # For Chinese, BPE will handle characters correctly if trained on them.
        self.tokenizer.pre_tokenizer = Whitespace()

    def build(self, sentences: List[str]):
        """Train the BPE tokenizer on the provided sentences."""
        trainer = BpeTrainer(
            vocab_size=self.max_vocab_size, 
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        )
        # Train from the list of sentences
        self.tokenizer.train_from_iterator(sentences, trainer)

    def encode(self, sentence: str) -> List[int]:
        """Convert a sentence string to a list of subword IDs."""
        return self.tokenizer.encode(sentence).ids

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        # skip_special_tokens=True results in cleaner output for human reading
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, path: str):
        """Save the BPE tokenizer configuration."""
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load a saved BPE tokenizer from a tokenizer.json file."""
        vocab = cls()
        vocab.tokenizer = Tokenizer.from_file(path)
        return vocab


# ═══════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs with vocabulary tracking."""

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Pre-tokenize all sentences
        self.src_tokens = []
        self.tgt_tokens = []
        
        # Track vocabulary usage
        self.src_vocab_usage = Counter()
        self.tgt_vocab_usage = Counter()
        self.unk_count = 0
        self.total_tokens = 0

        for src, tgt in zip(src_sentences, tgt_sentences):
            # Source: add EOS only
            src_ids = src_vocab.encode(src) + [EOS_ID]
            src_ids = src_ids[:max_src_len]
            src_ids = src_ids + [PAD_ID] * (max_src_len - len(src_ids))
            self.src_tokens.append(src_ids)
            
            # Track source vocabulary usage
            for token_id in src_ids:
                if token_id != PAD_ID:
                    self.src_vocab_usage[token_id] += 1
                    self.total_tokens += 1
                    if token_id == UNK_ID:
                        self.unk_count += 1

            # Target: add BOS and EOS
            tgt_ids = [BOS_ID] + tgt_vocab.encode(tgt) + [EOS_ID]
            tgt_ids = tgt_ids[:max_tgt_len]
            tgt_ids = tgt_ids + [PAD_ID] * (max_tgt_len - len(tgt_ids))
            self.tgt_tokens.append(tgt_ids)
            
            # Track target vocabulary usage
            for token_id in tgt_ids:
                if token_id != PAD_ID:
                    self.tgt_vocab_usage[token_id] += 1

    def get_vocab_stats(self) -> dict:
        """Get vocabulary usage statistics."""
        src_vocab_size = self.src_vocab.size
        tgt_vocab_size = self.tgt_vocab.size
        
        src_used = len(self.src_vocab_usage)
        tgt_used = len(self.tgt_vocab_usage)
        
        unk_rate = self.unk_count / max(1, self.total_tokens)
        
        # Top 10 most used tokens
        src_top_10 = self.src_vocab_usage.most_common(10)
        tgt_top_10 = self.tgt_vocab_usage.most_common(10)
        
        return {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'src_vocab_used': src_used,
            'tgt_vocab_used': tgt_used,
            'src_vocab_coverage': src_used / src_vocab_size * 100,
            'tgt_vocab_coverage': tgt_used / tgt_vocab_size * 100,
            'unk_rate': unk_rate * 100,
            'total_tokens': self.total_tokens,
            'src_top_10': src_top_10,
            'tgt_top_10': tgt_top_10,
        }

    def __len__(self) -> int:
        return len(self.src_tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_tokens[idx], dtype=torch.long),
            torch.tensor(self.tgt_tokens[idx], dtype=torch.long),
        )


def collate_fn(batch, pad_id: int = PAD_ID):
    """Collate function for DataLoader - handles variable length sequences."""
    src_batch, tgt_batch = zip(*batch)

    # Get actual lengths (for potential future use with padding masks)
    src_lengths = [(src != pad_id).sum().item() for src in src_batch]
    tgt_lengths = [(tgt != pad_id).sum().item() for tgt in tgt_batch]

    # Stack into batches (already padded in dataset)
    src_batch = torch.stack(src_batch, dim=0)
    tgt_batch = torch.stack(tgt_batch, dim=0)

    return src_batch, tgt_batch


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(
    data_path: str,
    max_samples: int = 5000,
    max_seq_len: int = 20
) -> Tuple[List[str], List[str]]:
    """
    Load and filter EN-ZH sentence pairs from CSV or Arrow files.

    Args:
        data_path:   Path to en-zh.csv or directory containing .arrow files.
        max_samples: Maximum number of sentence pairs to load.
        max_seq_len: Maximum sentence length (in words/characters) to keep.

    Returns:
        Tuple of (en_sentences, zh_sentences) — lists of strings.
    """
    en_sentences = []
    zh_sentences = []

    # URL masking regex: replace any URL with a special token to reduce noise
    url_re = re.compile(r"(https?://\S+|www\.[^\s]+)", re.IGNORECASE)

    def _mask_urls(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return url_re.sub("<URL>", text)

    print(f"Loading dataset from {data_path} ...")
    
    # Check for Arrow files first (WMT18 format)
    arrow_files = []
    if os.path.isdir(data_path):
        arrow_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.arrow')]
    
    if arrow_files:
        print(f"  Detected {len(arrow_files)} Arrow files. Using 'datasets' library to load...")
        try:
            from datasets import load_dataset as load_arrow_dataset
            # Filter for training files to avoid using test/validation for training if not specified
            train_files = [f for f in arrow_files if 'train' in os.path.basename(f)]
            if not train_files:
                train_files = arrow_files # Fallback to all arrow files
            
            ds = load_arrow_dataset('arrow', data_files=train_files, split='train')
            
            # Use a generator to iterate through the dataset to avoid loading everything into RAM at once
            # although we still collect into lists for the current Vocabulary builder.
            count = 0
            for item in tqdm(ds, desc="Processing Arrow data", total=min(len(ds), max_samples * 2)): # Estimate total
                trans = item.get('translation')
                if not trans:
                    continue
                
                en = _mask_urls(trans.get('en', '').strip())
                zh = _mask_urls(trans.get('zh', '').strip())
                
                if not en or not zh:
                    continue
                
                if (1 <= len(en.split()) <= max_seq_len and 1 <= len(zh) <= max_seq_len * 2):
                    en_sentences.append(en)
                    zh_sentences.append(zh)
                    count += 1
                
                if count >= max_samples:
                    break
            
            print(f"  Loaded {len(en_sentences)} sentence pairs from Arrow files.")
            return en_sentences, zh_sentences
            
        except ImportError:
            print("  Warning: 'datasets' library not found. Please install it with 'pip install datasets'.")
            print("  Attempting to fallback to CSV loading...")
        except Exception as e:
            print(f"  Error loading Arrow files: {e}")
            print("  Attempting to fallback to CSV loading...")

    # Fallback to CSV loading
    if os.path.isdir(data_path):
        actual_path = os.path.join(data_path, "en-zh.csv")
    else:
        actual_path = data_path

    if not os.path.exists(actual_path):
        print(f"Warning: {actual_path} not found. Searching for any .csv file in {data_path}...")
        if os.path.isdir(data_path):
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if csv_files:
                actual_path = os.path.join(data_path, csv_files[0])
                print(f"Using {actual_path}")
            else:
                raise FileNotFoundError(f"No CSV or Arrow files found in {data_path}")
        else:
            raise FileNotFoundError(f"File {actual_path} not found")

    with open(actual_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = _mask_urls((row.get("en") or row.get("english") or row.get("src") or "").strip())
            zh = _mask_urls((row.get("zh") or row.get("chinese") or row.get("tgt") or "").strip())

            if not en or not zh:
                continue

            if (
                1 <= len(en.split()) <= max_seq_len
                and 1 <= len(zh) <= max_seq_len * 2
            ):
                en_sentences.append(en)
                zh_sentences.append(zh)

            if len(en_sentences) >= max_samples:
                break

    print(f"  Loaded {len(en_sentences)} sentence pairs from CSV.")
    return en_sentences, zh_sentences


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    """Cross-entropy with label smoothing, ignoring PAD tokens.

    Args:
        logits: [B, T, V]
        targets: [B, T]
        epsilon: smoothing factor in [0, 1)
        pad_id: PAD token id to ignore

    Returns:
        Scalar loss tensor.
    """
    B, T, V = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1)
    # Gather NLL for true targets
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Mean over classes for smoothing loss
    smooth = -log_probs.mean(dim=-1)
    # Combine
    loss = (1.0 - epsilon) * nll + epsilon * smooth
    # Mask out PAD positions
    mask = (targets != pad_id).float()
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss

def setup_device(device: str = None) -> str:
    """Sets up and returns the computation device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}")
    print(f"  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 70}\n")
    return device


def prepare_data(
    data_path: str,
    max_samples: int,
    max_seq_len: int,
    vocab_size: int,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary, Dict]:
    """Loads dataset, builds vocabularies, and creates DataLoaders."""
    # Load Data
    en_sentences, zh_sentences = load_dataset(data_path, max_samples, max_seq_len)

    # Build Vocabularies
    print("\nBuilding vocabularies ...")
    en_vocab = Vocabulary(max_vocab_size=vocab_size)
    zh_vocab = Vocabulary(max_vocab_size=vocab_size)
    en_vocab.build(en_sentences)
    zh_vocab.build(zh_sentences)
    print(f"  English vocab size: {en_vocab.size}")
    print(f"  Chinese vocab size: {zh_vocab.size}")

    # Create Datasets
    src_max_len = max_seq_len + 1  # +1 for EOS
    tgt_max_len = max_seq_len + 2  # +2 for BOS and EOS

    dataset = TranslationDataset(
        src_sentences=en_sentences,
        tgt_sentences=zh_sentences,
        src_vocab=en_vocab,
        tgt_vocab=zh_vocab,
        max_src_len=src_max_len,
        max_tgt_len=tgt_max_len,
    )

    # Train / validation split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"\n  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Vocabulary Statistics
    vocab_stats = dataset.get_vocab_stats()
    print(f"\n  📚 Vocabulary Statistics:")
    print(f"     English: {vocab_stats['src_vocab_used']:,} / {vocab_stats['src_vocab_size']:,} tokens used ({vocab_stats['src_vocab_coverage']:.1f}% coverage)")
    print(f"     Chinese: {vocab_stats['tgt_vocab_used']:,} / {vocab_stats['tgt_vocab_size']:,} tokens used ({vocab_stats['tgt_vocab_coverage']:.1f}% coverage)")
    print(f"     UNK rate: {vocab_stats['unk_rate']:.2f}%")

    return train_loader, val_loader, en_vocab, zh_vocab, vocab_stats


def initialize_training(
    en_vocab: Vocabulary,
    zh_vocab: Vocabulary,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    device: str,
    min_lr: float = 1e-06,
) -> Tuple[nn.Module, AdamW, LambdaLR]:
    """Initializes the model, optimizer, and learning rate scheduler."""
    print("\nInitializing Transformer model ...")
    model = Transformer(
        src_vocab_size=en_vocab.size,
        tgt_vocab_size=zh_vocab.size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Learning rate scheduler with warmup + inverse sqrt decay and a floor (min_lr)
    base_lr = learning_rate
    min_factor = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            factor = (step + 1) / warmup_steps
        else:
            # Standard inverse sqrt decay after warmup
            factor = (step / warmup_steps) ** -0.5
        # Apply LR floor
        return max(min_factor, factor)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return model, optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: str,
    epoch: int,
    num_epochs: int,
    global_step: int,
    writer: SummaryWriter = None,
    *,
    use_amp: bool = False,
    scaler: GradScaler = None,
    label_smoothing: float = 0.0,
    pad_id: int = PAD_ID,
) -> Tuple[float, int, float]:
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_train_batches = 0

    train_pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
        unit="batch",
        ncols=100,
        colour='green'
    )

    current_lr = scheduler.get_last_lr()[0]

    for batch_idx, (src_batch, tgt_batch) in enumerate(train_pbar):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # Decoder input = tgt[:, :-1], labels = tgt[:, 1:]
        decoder_input = tgt_batch[:, :-1]
        decoder_labels = tgt_batch[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # Use new torch.amp.autocast API (specify device type explicitly)
            with autocast(device_type='cuda', enabled=True):
                logits = model(src_batch, decoder_input)
                if label_smoothing and label_smoothing > 0.0:
                    loss = label_smoothed_ce(logits, decoder_labels, label_smoothing, pad_id)
                else:
                    loss = model.compute_loss(logits, decoder_labels, pad_id=pad_id)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            logits = model(src_batch, decoder_input)
            if label_smoothing and label_smoothing > 0.0:
                loss = label_smoothed_ce(logits, decoder_labels, label_smoothing, pad_id)
            else:
                loss = model.compute_loss(logits, decoder_labels, pad_id=pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        epoch_loss += loss.item()
        num_train_batches += 1
        current_lr = scheduler.get_last_lr()[0]

        if writer is not None:
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('LearningRate', current_lr, global_step)
            writer.add_scalar('GPU/Memory_GB', torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0, global_step)

        global_step += 1

        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
            'lr': f'{current_lr:.2e}',
            'GPU': f'{torch.cuda.memory_allocated() / 1e9:.2f}GB' if torch.cuda.is_available() else 'N/A'
        })

    return epoch_loss / num_train_batches, global_step, current_lr


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    num_epochs: int,
    *,
    label_smoothing: float = 0.0,
    pad_id: int = PAD_ID,
) -> float:
    """Validates the model for one epoch."""
    model.eval()
    val_loss = 0.0
    num_val_batches = 0

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} [Val] ",
        unit="batch",
        ncols=100,
        colour='blue'
    )

    with torch.no_grad():
        for batch_idx, (src_batch, tgt_batch) in enumerate(val_pbar):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            decoder_input = tgt_batch[:, :-1]
            decoder_labels = tgt_batch[:, 1:]

            logits = model(src_batch, decoder_input)
            if label_smoothing and label_smoothing > 0.0:
                loss = label_smoothed_ce(logits, decoder_labels, label_smoothing, pad_id)
            else:
                loss = model.compute_loss(logits, decoder_labels, pad_id=pad_id)

            val_loss += loss.item()
            num_val_batches += 1

            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{val_loss / (batch_idx + 1):.4f}'
            })

    return val_loss / num_val_batches


def _greedy_decode_single(model: nn.Module, src_ids: List[int], max_len: int, device: str) -> List[int]:
    """Greedy decode for a single example; returns generated IDs excluding the initial BOS."""
    model_device = next(model.parameters()).device
    if device is not None and str(model_device) != str(device):
        model_device = device
    src = torch.tensor([src_ids], dtype=torch.long, device=model_device)
    generated = [BOS_ID]
    with torch.no_grad():
        for _ in range(max_len - 1):
            dec_in = torch.tensor([generated], dtype=torch.long, device=model_device)
            logits = model(src, dec_in)  # [1, T, V]
            next_token = torch.argmax(logits[0, -1], dim=-1).item()
            generated.append(next_token)
            if next_token == EOS_ID:
                break
    return generated[1:]


def compute_bleu_on_loader(
    model: nn.Module,
    val_loader: DataLoader,
    zh_vocab: Vocabulary,
    device: str,
    max_tgt_len: int,
) -> float:
    """Compute corpus BLEU on the validation DataLoader using greedy decoding.

    Returns BLEU score (sacrebleu.corpus_bleu .score) or -1.0 if sacrebleu unavailable.
    """
    if not SACREBLEU_AVAILABLE:
        return -1.0

    model.eval()
    sys_refs: List[str] = []
    sys_hyps: List[str] = []

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            batch_size = src_batch.size(0)
            for i in range(batch_size):
                # Prepare single example
                src_ids = src_batch[i].tolist()
                # Remove padding beyond EOS for cleaner decoding
                if EOS_ID in src_ids:
                    eos_pos = src_ids.index(EOS_ID)
                    src_ids = src_ids[: eos_pos + 1]
                # Hypothesis via greedy decode
                hyp_ids = _greedy_decode_single(model, src_ids, max_tgt_len, device)
                hyp_text = zh_vocab.decode(hyp_ids)

                # Reference: strip BOS and everything after EOS, then decode
                ref_ids = tgt_batch[i].tolist()
                # drop leading BOS
                if len(ref_ids) > 0 and ref_ids[0] == BOS_ID:
                    ref_ids = ref_ids[1:]
                # cut at EOS
                if EOS_ID in ref_ids:
                    ref_ids = ref_ids[: ref_ids.index(EOS_ID)]
                ref_text = zh_vocab.decode(ref_ids)

                sys_hyps.append(hyp_text)
                sys_refs.append(ref_text)

    # sacrebleu handles Chinese characters well when zh is specified
    # but we might need to handle tokenization for BLEU calculation on Chinese
    # sacrebleu's corpus_bleu has a 'tokenize' parameter.
    bleu = sacrebleu.corpus_bleu(sys_hyps, [sys_refs], tokenize='zh')
    return float(bleu.score)


def save_artifacts(
    save_dir: str,
    model: nn.Module,
    optimizer: AdamW,
    en_vocab: Vocabulary,
    zh_vocab: Vocabulary,
    training_history: List[Dict],
    config_params: Dict,
    epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    avg_bleu: float = None,
):
    """Saves model state, history, vocabularies, and configuration."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model state dict
    model_path = os.path.join(save_dir, "transformer_en_zh.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'bleu': avg_bleu,
        'training_history': training_history,
        'config': config_params
    }, model_path)
    print(f"  💾 Model saved to {model_path}")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"  📊 Training history saved to {history_path}")

    # Save vocabularies
    en_vocab_path = os.path.join(save_dir, "en_vocab.json")
    zh_vocab_path = os.path.join(save_dir, "zh_vocab.json")
    en_vocab.save(en_vocab_path)
    zh_vocab.save(zh_vocab_path)
    print(f"  📚 BPE Tokenizers saved to {en_vocab_path}, {zh_vocab_path}")

    # Save config for test script
    config = {
        **config_params,
        "src_max_len": config_params.get("max_seq_len", 0) + 1,
        "tgt_max_len": config_params.get("max_seq_len", 0) + 2,
        "src_vocab_size": en_vocab.size,
        "tgt_vocab_size": zh_vocab.size,
    }
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ⚙️ Config saved to {config_path}")


def train(
    data_path: str,
    save_dir: str = "../model_saved/transformer_translator_en_zh",
    max_samples: int = 200000,
    max_seq_len: int = 50,
    vocab_size: int = 32000,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    num_layers: int = 6,
    num_epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.01,
    warmup_steps: int = 6000,
    dropout: float = 0.3,
    device: str = None,
    # Training strategies
    patience: int = 6,
    min_delta: float = 1e-3,
    min_epochs: int = 5,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
    min_lr: float = 1e-6,
):
    """
    Train the Transformer on the EN-ZH dataset using PyTorch with GPU support.
    Uses Adam optimizer with learning rate scheduling as in the original paper.
    """
    device = setup_device(device)

    train_loader, val_loader, en_vocab, zh_vocab, vocab_stats = prepare_data(
        data_path, max_samples, max_seq_len, vocab_size, batch_size
    )

    model, optimizer, scheduler = initialize_training(
        en_vocab, zh_vocab, d_model, num_heads, d_ff, num_layers,
        dropout, learning_rate, weight_decay, warmup_steps, device, min_lr
    )

    # AMP scaler (optional)
    scaler = GradScaler(enabled=bool(use_amp) and torch.cuda.is_available())

    # ── Training Loop Setup ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  🚀 Starting Training")
    print(f"{'=' * 80}")
    print(f"  Model: Transformer (d_model={d_model}, heads={num_heads}, layers={num_layers})")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Initial LR: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Dropout: {dropout}")
    print(f"  Vocabulary Coverage:")
    print(f"     English: {vocab_stats['src_vocab_coverage']:.1f}% ({vocab_stats['src_vocab_used']:,}/{vocab_stats['src_vocab_size']:,})")
    print(f"     Chinese: {vocab_stats['tgt_vocab_coverage']:.1f}% ({vocab_stats['tgt_vocab_used']:,}/{vocab_stats['tgt_vocab_size']:,})")
    print(f"  UNK Rate: {vocab_stats['unk_rate']:.2f}%")
    print(f"{'=' * 80}\n")

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_epochs = 0
    training_history = []
    start_time = time.time()
    global_step = 0
    writer = None

    if TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"  📊 TensorBoard logs: {log_dir}")
        print(f"  Run 'tensorboard --logdir {log_dir}' to view real-time graphs\n")
        
        # Log vocabulary statistics to TensorBoard
        writer.add_scalar('Vocabulary/en_vocab_size', vocab_stats['src_vocab_size'], 0)
        writer.add_scalar('Vocabulary/zh_vocab_size', vocab_stats['tgt_vocab_size'], 0)
        writer.add_scalar('Vocabulary/en_vocab_coverage', vocab_stats['src_vocab_coverage'], 0)
        writer.add_scalar('Vocabulary/zh_vocab_coverage', vocab_stats['tgt_vocab_coverage'], 0)
        writer.add_scalar('Vocabulary/unk_rate', vocab_stats['unk_rate'], 0)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        avg_train_loss, global_step, current_lr = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, num_epochs, global_step, writer,
            use_amp=use_amp and torch.cuda.is_available(), scaler=scaler, label_smoothing=label_smoothing, pad_id=PAD_ID
        )
        
        train_elapsed = time.time() - epoch_start
        
        avg_val_loss = validate_one_epoch(
            model, val_loader, device, epoch, num_epochs, label_smoothing=label_smoothing, pad_id=PAD_ID
        )
        # Compute BLEU on the validation set (greedy decoding)
        bleu_score = compute_bleu_on_loader(
            model, val_loader, zh_vocab, device, max_tgt_len=max_seq_len + 2
        )
        
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        
        # Estimate remaining time
        epochs_per_hour = (epoch + 1) / (total_elapsed / 3600) if total_elapsed > 0 else 0
        remaining_epochs = num_epochs - (epoch + 1)
        eta_hours = remaining_epochs / epochs_per_hour if epochs_per_hour > 0 else 0
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'bleu': bleu_score if bleu_score is not None else None,
            'train_time': train_elapsed,
            'epoch_time': epoch_elapsed,
            'lr': current_lr,
        })
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch + 1)
            if bleu_score is not None and bleu_score >= 0:
                writer.add_scalar('Metrics/BLEU', bleu_score, epoch + 1)
            writer.add_scalar('Metrics/learning_rate', current_lr, epoch + 1)
            writer.add_scalar('Metrics/epochs_per_hour', epochs_per_hour, epoch + 1)
            # Early stopping related logs
            writer.add_scalar('EarlyStop/no_improve_epochs', no_improve_epochs, epoch + 1)
            writer.add_scalar('EarlyStop/patience', patience, epoch + 1)
            writer.flush()
        
        # Save training history after each epoch
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        # Print epoch summary
        print(f"\n{'─' * 80}")
        print(f"  ✅ Epoch {epoch+1}/{num_epochs} Complete")
        print(f"{'─' * 80}")
        print(f"  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")
        print(f"  Train Time: {train_elapsed:.1f}s  |  Total Time: {total_elapsed/60:.1f}min")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "")
        print(f"  ETA: {eta_hours:.1f} hours remaining ({epochs_per_hour:.1f} epochs/hour)")
        
        # Save best model + Early stopping bookkeeping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improve_epochs = 0
            print(f"\n  🏆 New best model! (Epoch {best_epoch}) Val Loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"\n  No improvement for {no_improve_epochs} epoch(s). Best {best_val_loss:.4f} @ epoch {best_epoch}.")

        # Early stopping check
        if (epoch + 1) >= min_epochs and no_improve_epochs >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}. Best val: {best_val_loss:.4f} (epoch {best_epoch}).")
            # Persist latest history and break
            history_path = os.path.join(save_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)
            print(f"{ '─' * 80 }\n")
            break
        print(f"{'─' * 80}\n")

    # ── Save Model & Vocabularies ────────────────────────────────────────
    config_params = {
        'd_model': d_model,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'max_seq_len': max_seq_len,
        'patience': patience,
        'min_delta': min_delta,
        'min_epochs': min_epochs,
        'label_smoothing': label_smoothing,
        'use_amp': bool(use_amp and torch.cuda.is_available()),
        'min_lr': min_lr,
    }
    
    save_artifacts(
        save_dir, model, optimizer, en_vocab, zh_vocab, 
        training_history, config_params, num_epochs - 1, 
        avg_train_loss, avg_val_loss, 
        avg_bleu=training_history[-1].get('bleu') if len(training_history) > 0 else None
    )

    # Print final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"  🎉 Training Complete!")
    print(f"{'=' * 80}")
    print(f"  Total Time: {total_time / 3600:.2f} hours")
    print(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  Final Train Loss: {avg_train_loss:.4f}")
    print(f"  Final Val Loss: {avg_val_loss:.4f}")
    
    if writer is not None:
        writer.close()
        print(f"\n  📊 TensorBoard logs saved to: {log_dir}")
    
    print(f"\n  To test the model, run:")
    print(f"    python test_en_zh.py")
    print(f"{'=' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Path to the dataset
    DATA_PATH = "/home/changliu/DataSet/kaggle/wmt18-en-zh"

    # ═══════════════════════════════════════════════════════════════════════
    #  High-Precision Configuration for RTX 5060 Ti
    # ═══════════════════════════════════════════════════════════════════════
    
    train(
        data_path=DATA_PATH,
        
        # ── Model Architecture (Transformer Base) ─────────────────────────
        d_model=512,           # Model dimension - standard for good quality
        num_heads=8,           # Attention heads (d_k = 64 per head)
        d_ff=2048,             # FFN hidden dimension (4x expansion)
        num_layers=6,          # Number of encoder/decoder layers
        dropout=0.2,           # Regularization
        
        # ── Training Configuration ────────────────────────────────────────
        num_epochs=40,         # More epochs for better convergence
        batch_size=64,         # Batch size (adjust based on VRAM)
        learning_rate=0.0004,  # AdamW learning rate
        weight_decay=0.01,     # Weight decay (L2 regularization)
        warmup_steps=4000,     # LR warmup
        # Early stopping & regularization strategies
        patience=6,
        min_delta=1e-3,
        min_epochs=8,
        label_smoothing=0.1,
        use_amp=True,
        min_lr=4e-6,
        
        # ── Data Configuration ────────────────────────────────────────────
        max_samples=1500000,    # Use more data for better quality
        max_seq_len=64,        # Handle longer sentences
        vocab_size=37000,
    )
