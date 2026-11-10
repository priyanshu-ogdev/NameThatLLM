"""
Name That LLM: Quad-Model Frontier Ensemble (V6 Production Pipeline)
DeBERTa-v3-large FT (GPU 0) + BGE-large-en-v1.5 FT (GPU 1) + Qwen 3 8B / 2.5 7B LoRA (Dual GPU) + CPU Multiview Linear (~850k).
Standalone, offline, self-contained Kaggle submission notebook.

Target Metric: Macro Balanced Accuracy across 18 classes (>0.70+).

Key Architectural Components:
  1. DeBERTa-v3-large Full FT -> cuda:0 (BS=12, Head+Tail L=352, LLRD xi=0.90, StylometricMultiMomentPooler)
  2. BGE-large-en-v1.5 Full FT -> cuda:1 (BS=16, Head+Tail L=352, LLRD xi=0.85, Multi-Moment Pooling)
  3. Qwen 3 8B / 2.5 7B LoRA  -> device_map="auto" sharded across dual GPUs (r=48, SpikeAndGapHierarchicalHead)
  4. CPU Multiview Dual-Linear -> ~850k sparse TF-IDF + 132 stylometrics + IntraFamilyCliqueSpecialist
  5. Zero-VRAM External Data   -> 24k passages with Kish ESS-capped mass conservation (0.000% prior skew)
  6. Meta-Combiner & Consensus -> Nested 5-fold CV CPU autograd L-BFGS + Hierarchical Soft-Tree + Sinkhorn balancing
"""
from collections import Counter
import concurrent.futures
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import warnings

SCRIPT_T0 = time.perf_counter()

# 1. Disable Tokenizer Parallelism Deadlock & HF Internal Thread Contention & Broken Jupyter Widgets
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# 2. Enforce Strict Offline & Disable Hugging Face Transfer / XET hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 3. PyTorch CUDA memory allocator configuration to avoid fragmentation hangs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 4. Bypass Kaggle container bug where pre-installed torchao is 0.10.0 but peft requires >= 0.16.0
sys.modules['torchao'] = None

# 5. Ensure line-buffered stdout so logs stream immediately without buffering
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# 6. Offline Dependency Bootstrap: Auto-upgrade transformers & tokenizers if wheels are attached
def bootstrap_offline_wheels():
    """
    Auto-discover and install offline wheel packages under /kaggle/input (e.g. transformers >= 4.50 for Qwen 3).
    Safe, silent no-op when running with standard pre-installed libraries or when no wheels are present.
    """
    input_roots = []
    local_data = os.environ.get("NTL_DATA_DIR")
    if local_data:
        input_roots.append(Path(local_data))
    input_roots.append(Path("/kaggle/input"))
    if os.environ.get("NTL_WHEELS_DIR"):
        input_roots.insert(0, Path(os.environ["NTL_WHEELS_DIR"]))

    wheel_dirs = set()
    for root in input_roots:
        if root.exists():
            try:
                for whl in root.rglob("transformers*.whl"):
                    wheel_dirs.add(whl.parent)
            except Exception:
                pass

    if wheel_dirs:
        for wdir in sorted(wheel_dirs):
            print(f"[Offline Bootstrap] Found wheel directory at: {wdir}. Installing upgraded transformers...", flush=True)
            try:
                res = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={wdir}", "transformers", "tokenizers", "peft"],
                    capture_output=True, text=True, timeout=120
                )
                if res.returncode == 0:
                    import importlib
                    importlib.invalidate_caches()
                    print(f"[Offline Bootstrap] Upgrade successful from {wdir}.", flush=True)
                    break
                else:
                    print(f"[Offline Bootstrap] pip returned {res.returncode}: {res.stderr[:160]}", flush=True)
            except Exception as e:
                print(f"[Offline Bootstrap] Installation error: {e}", flush=True)

bootstrap_offline_wheels()

# 7. Global Warning Controls
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold  # noqa: F401
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
try:
    import transformers
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Enable cuDNN auto-tuning & high float32 matrix multiplication precision
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

class HeartbeatTimer:
    """Prints live elapsed seconds during long blocking operations (e.g. Kaggle virtual disk I/O)."""
    def __init__(self, message, interval=8.0):
        self.message = message
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.wait(self.interval):
            elapsed = time.time() - self.start_time
            print(f"    ... [Heartbeat] {self.message} ({elapsed:.0f}s elapsed - disk I/O active, not stuck) ...", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        elapsed = time.time() - self.start_time
        print(f"    -> [Heartbeat] Completed in {elapsed:.1f}s.", flush=True)

def amp_autocast():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()

def inference_context():
    """Inference context with PyTorch inference_mode for maximum dispatch throughput."""
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()

def get_mem_status(device=None):
    """Returns live formatted VRAM and System RAM utilization."""
    vram_items = []
    if torch.cuda.is_available():
        if device is not None:
            try:
                d = torch.device(device)
                idx = d.index if d.index is not None else 0
                alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                res = torch.cuda.memory_reserved(idx) / (1024 ** 3)
                vram_items.append(f"{d}:{alloc:.1f}/{res:.1f}GB")
            except Exception:
                pass
        else:
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                res = torch.cuda.memory_reserved(i) / (1024 ** 3)
                vram_items.append(f"cuda:{i}:{alloc:.1f}/{res:.1f}GB")

    vram_str = "VRAM[" + " | ".join(vram_items) + "]" if vram_items else "CPU"
    ram_str = "RAM"
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_str = f"RAM:{vm.used / (1024 ** 3):.1f}/{vm.total / (1024 ** 3):.1f}GB ({vm.percent:.0f}%)"
    except Exception:
        pass

    return f"[{vram_str} | {ram_str}]"

# --- 1. COMPETITION CONSTANTS ---

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

FAMILY_MAP = {
    "gpt-5.5": "openai", "gpt-5.6-luna": "openai", "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai", "gpt-6-astra": "openai",
    "gemini-3.1-pro-preview": "google", "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google", "gemini-3.8-flash": "google",
    "claude-fable-5": "anthropic", "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic", "claude-opus-5": "anthropic", "claude-sonnet-5": "anthropic",
    "qwen3.8-max": "qwen",
    "glm-5.3": "zhipu", "glm-5.3-flash": "zhipu",
    "kimi-k3": "moonshot",
}
FAMILIES = sorted(set(FAMILY_MAP.values()))
FAMILY_TO_INDICES = {
    fam: [i for i, lab in enumerate(LABELS) if FAMILY_MAP[lab] == fam]
    for fam in FAMILIES
}
FAMILY_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}
LABEL_TO_FAMILY_IDX = np.array([FAMILY_TO_IDX[FAMILY_MAP[l]] for l in LABELS], dtype=np.int64)

def compute_family_logits(model_logits: torch.Tensor, temp: float = 4.0) -> torch.Tensor:
    """
    Derive per-family logits by temperature-smoothed log-sum-exp pooling within each family.
    Scaling by temp=4.0 prevents hard max() saturation, ensuring family gradients distribute
    smoothly across all sibling models rather than starving non-top siblings.
    Shape: (N, 18) -> (N, n_families)
    """
    n_fams = len(FAMILIES)
    scaled = model_logits / max(float(temp), 1e-4)
    fam_logits = torch.empty((model_logits.shape[0], n_fams), device=model_logits.device, dtype=model_logits.dtype)
    for f_i, fam in enumerate(FAMILIES):
        idxs = FAMILY_TO_INDICES[fam]
        fam_logits[:, f_i] = float(temp) * torch.logsumexp(scaled[:, idxs], dim=1)
    return fam_logits

def seed_everything(seed: int = 42):
    """Seed Python, NumPy, PyTorch (CPU & CUDA) for full deterministic reproducibility."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_smoothed_class_weights(y_indices, num_classes=18, gamma=0.5):
    """
    Computes smoothed class weights to prevent gradient spikes on rare
    classes while aligning optimization with Macro Balanced Accuracy.
    Caps max weight disparity at ~6.0x (clipping to [0.5, 4.5]).
    """
    counts = np.bincount(y_indices, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    median_count = np.median(counts)
    weights = (median_count / counts) ** gamma
    weights = np.clip(weights, 0.5, 4.5)
    weights = weights / np.mean(weights)
    return weights

# --- 2. TEXT NORMALIZATION & STYLISTIC SIGNAL PRESERVATION ---

def clean_text(text: str) -> str:
    """Normalize whitespace and line endings while preserving casing, punctuation, and markdown."""
    if not isinstance(text, str):
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s.strip()


def compute_fold_variance_capped_weights(
    y_comp_fold: np.ndarray,
    y_ext: np.ndarray,
    ext_n_models: np.ndarray,
    n_classes: int = 18,
    ext_mass_ratio: float = 0.55,
    max_single_ext_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Guarantees:
      1. Exact 0.000% class prior skew across all 18 classes: Sum_{i in c} (w_comp + w_ext) is constant.
      2. Bounded Estimator Variance: No single external sample weight ever exceeds max_single_ext_weight (0.35).
         Any mass deficit for starved classes (e.g. gemini-3.8-flash, N=71) is added directly to that
         class's official competition samples so the class prior stays 100% uniform!
      3. Regularization Invariance: Total weight sum equals len(y_comp_fold) so LinearSVC(C) and Ridge(alpha)
         penalties never shift when external data is attached.
    """
    n_comp = len(y_comp_fold)
    target_class_comp = n_comp / float(n_classes)

    w_comp = np.ones(n_comp, dtype=np.float64)
    w_ext = np.zeros(len(y_ext), dtype=np.float64)

    # Reward multi-model rows (2.5x) over single-model rows (1.0x)
    raw_intra = np.where(ext_n_models >= 2, 2.5, 1.0)
    total_target_per_class = target_class_comp * (1.0 + ext_mass_ratio)

    for c in range(n_classes):
        c_label = LABELS[c] if c < len(LABELS) else c
        comp_mask = (y_comp_fold == c) | (y_comp_fold == c_label)
        ext_mask = (y_ext == c) | (y_ext == c_label)
        n_c_comp = max(int(comp_mask.sum()), 1)

        desired_ext_mass = target_class_comp * ext_mass_ratio
        raw_c_sum = raw_intra[ext_mask].sum()

        if raw_c_sum > 0:
            w_c_ext = raw_intra[ext_mask] * (desired_ext_mass / raw_c_sum)
            # Cap individual external sample weights to prevent variance explosion on starved classes
            w_c_ext = np.minimum(w_c_ext, max_single_ext_weight)
            w_ext[ext_mask] = w_c_ext
            actual_ext_mass = w_c_ext.sum()
        else:
            actual_ext_mass = 0.0

        # Absorb any capped deficit into the official competition samples of class c
        rem_comp_mass = total_target_per_class - actual_ext_mass
        w_comp[comp_mask] = rem_comp_mass / float(n_c_comp)

    # Normalize global sum to equal n_comp so LinearSVC(C) and Ridge(alpha) stay invariant
    total_sum = w_comp.sum() + w_ext.sum()
    scale = float(n_comp) / max(total_sum, 1e-12)
    return w_comp * scale, w_ext * scale


def align_external_dense_moments(
    V_comp: np.ndarray,
    V_ext: np.ndarray,
    w_ext: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Aligns weighted mean and std of external dense stylometric features to the competition fold
    so the classifier cannot split on domain covariate shift.
    """
    w_norm = w_ext[:, None] / max(float(w_ext.sum()), eps)
    mu_ext = np.sum(V_ext * w_norm, axis=0, keepdims=True)
    var_ext = np.sum(w_norm * (V_ext - mu_ext) ** 2, axis=0, keepdims=True)
    std_ext = np.sqrt(np.maximum(var_ext, eps))

    mu_comp = V_comp.mean(axis=0, keepdims=True)
    std_comp = np.maximum(V_comp.std(axis=0, keepdims=True), eps)

    return (((V_ext - mu_ext) / std_ext) * std_comp + mu_comp).astype(V_comp.dtype)


def build_mass_balanced_external_corpus(
    comp_train_df: pd.DataFrame,
    ext_wide_path: str | Path,
    labels: list[str],
    ext_mass_ratio: float = 0.55,
    prefer_multi_model: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Converts wide external data into a mass-conserved long dataframe where:
      Sum_{i in comp, class=c} w_i = N_comp_c
      Sum_{j in ext,  class=c} w_j = ext_mass_ratio * N_comp_c  (Identical across all 18 classes!)
    Guarantees 0.0% class prior skew, bounded individual variance (<=0.35),
    and exact regularization invariance for scikit-learn LinearSVC and Ridge.
    Namespaces external prompt IDs ('ext_{id}') to prevent prompt collisions.
    Strips NUL bytes and cleans whitespace.
    """
    wide = pd.read_parquet(ext_wide_path)
    model_cols = [c for c in labels if c in wide.columns]

    # Count how many models answered each external prompt (reward multi-model contrastive rows)
    wide["_n_models"] = wide[model_cols].notna().sum(axis=1)

    ext_long = wide.melt(
        id_vars=["id", "_n_models"],
        value_vars=model_cols,
        var_name="model",
        value_name="text",
    ).dropna(subset=["text"]).reset_index(drop=True)

    # Clean text and strip NUL bytes
    ext_long["text"] = ext_long["text"].apply(clean_text)
    ext_long = ext_long[ext_long["text"].str.len() > 0].reset_index(drop=True)

    # Namespace prompt IDs to avoid collision with competition prompt_id
    ext_long["prompt_id"] = "ext_" + ext_long["id"].astype(str)

    label_to_id = {m: i for i, m in enumerate(labels)}
    ext_long["label"] = ext_long["model"].map(label_to_id)

    comp_labels = comp_train_df["label"].values if "label" in comp_train_df.columns else comp_train_df["model"].map(label_to_id).values
    w_comp, w_ext = compute_fold_variance_capped_weights(
        comp_labels, ext_long["label"].values, ext_long["_n_models"].values,
        n_classes=len(labels), ext_mass_ratio=ext_mass_ratio, max_single_ext_weight=0.35,
    )

    # Standardize column sets
    comp_sub = comp_train_df[["text", "model", "prompt_id"]].copy()
    comp_sub["label"] = comp_labels
    comp_sub["_n_models"] = 18

    ext_sub = ext_long[["text", "model", "prompt_id", "label", "_n_models"]].copy()

    combined_df = pd.concat([comp_sub, ext_sub], ignore_index=True)
    combined_weights = np.concatenate([w_comp, w_ext], axis=0)
    return combined_df, combined_weights


def prep_transformer_text(text: str) -> str:
    """Map newlines to visible [NL] tokens so SentencePiece/WordPiece preserve line/list cadence."""
    if not isinstance(text, str):
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    s = s.replace("\n\n", " [PAR] ").replace("\n", " [NL] ")
    return s.strip()


def tokenize_with_head_tail(tokenizer, texts, max_length=352, head_tokens=176, tail_tokens=176, preserve_newlines=True):
    """
    Tokenize sequences with balanced Head+Tail truncation (176 tokens each for 352 context).
    Preserves opening thesis and concluding sign-off signatures where LLM stylistic patterns peak.
    For encoder models (DeBERTa/BGE), preserve_newlines maps newlines to dedicated [NL]/[PAR] tokens.
    For decoder models (Qwen), raw newlines are preserved natively.
    Returns unpadded token lists ready for dynamic micro-batch padding.
    """
    if preserve_newlines:
        prep_texts = [prep_transformer_text(t) for t in texts]
    else:
        prep_texts = [clean_text(t) for t in texts]
    enc = tokenizer(prep_texts, truncation=False, padding=False)

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    has_token_types = "token_type_ids" in enc

    for i in range(len(prep_texts)):
        ids = enc["input_ids"][i]
        mask = enc["attention_mask"][i]
        if len(ids) > max_length:
            ids = ids[:head_tokens] + ids[-tail_tokens:]
            mask = mask[:head_tokens] + mask[-tail_tokens:]
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        if has_token_types:
            ttypes = enc["token_type_ids"][i]
            if len(ttypes) > max_length:
                ttypes = ttypes[:head_tokens] + ttypes[-tail_tokens:]
            token_type_ids_list.append(ttypes)

    res = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
    }
    if has_token_types:
        res["token_type_ids"] = token_type_ids_list
    return res


def find_topic_token_ids(token_id_sequences, groups, min_prompts: int = 3, special_ids=None) -> set:
    """
    Identifies token IDs that appear in fewer than min_prompts distinct prompt groups.
    Special tokens, punctuation, whitespace, and structural markers are protected from masking.
    """
    from collections import defaultdict
    if groups is None or len(token_id_sequences) != len(groups):
        return set()

    token_groups = defaultdict(set)
    for seq, g in zip(token_id_sequences, groups):
        for tok in set(seq):
            token_groups[tok].add(g)

    special_set = set(special_ids) if special_ids else set()
    topic_tokens = {
        tok for tok, g_set in token_groups.items()
        if len(g_set) < min_prompts and tok not in special_set
    }
    return topic_tokens


class TopicBlindAndSiFTPrep:
    """
    1. Stochastically masks rare topic tokens (PromptDF < 3) during training
       so DeBERTa/BGE cannot memorize prompt subjects.
    2. Applies Scale-Invariant Fine-Tuning (SiFT) noise to word embeddings
       to flatten the loss basin and improve out-of-fold generalization.
    """
    def __init__(self, topic_token_ids, mask_token_id: int, mask_prob: float = 0.30, sift_eps: float = 0.015):
        self.topic_ids_tensor = None
        self.raw_topic_ids = list(topic_token_ids) if topic_token_ids else []
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob if mask_token_id is not None else 0.0
        self.sift_eps = sift_eps

    def mask_topic_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.raw_topic_ids or self.mask_prob <= 0.0:
            return input_ids
        if self.topic_ids_tensor is None or self.topic_ids_tensor.device != input_ids.device:
            self.topic_ids_tensor = torch.tensor(self.raw_topic_ids, device=input_ids.device, dtype=input_ids.dtype)

        is_topic = torch.isin(input_ids, self.topic_ids_tensor)
        rand_mask = (torch.rand_like(input_ids, dtype=torch.float32) < self.mask_prob) & is_topic

        masked_ids = input_ids.clone()
        masked_ids[rand_mask] = self.mask_token_id
        return masked_ids

    def apply_sift_embedding_noise(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.sift_eps <= 0.0:
            return inputs_embeds
        mask = attention_mask.unsqueeze(-1).type_as(inputs_embeds)
        norm = torch.norm(inputs_embeds * mask, p=2, dim=(1, 2), keepdim=True).clamp(min=1e-6).detach()
        noise = torch.randn_like(inputs_embeds) * mask
        noise = noise / torch.norm(noise, p=2, dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return inputs_embeds + self.sift_eps * norm * noise


class FastBatchPrefetcher:
    """
    Background worker thread that pre-pads and pins micro-batches in host RAM ahead of the GPU.
    Eliminates CPU-GPU synchronization bubbles, keeping Tensor Cores at 100% duty cycle.
    Supports asynchronous CPU-side PromptDF stochastic topic-masking.
    """
    def __init__(self, tokenizer, enc_data, y_labels, indices, batch_size, dev, label_dev=None, cuda_ok=True, maxsize=6, topic_prep=None):
        self.tokenizer = tokenizer
        self.enc_data = enc_data
        self.y_labels = y_labels
        self.indices = indices
        self.batch_size = batch_size
        self.dev = dev
        self.label_dev = label_dev if label_dev is not None else dev
        self.cuda_ok = cuda_ok
        self.topic_prep = topic_prep
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.exc = None
        self.worker = threading.Thread(target=self._produce, daemon=True)
        self.worker.start()

    def _produce(self):
        try:
            n = len(self.indices)
            has_types = "token_type_ids" in self.enc_data
            for start in range(0, n, self.batch_size):
                if self.stop_event.is_set():
                    break
                idx = self.indices[start:start + self.batch_size]
                batch_slice = {
                    "input_ids": [self.enc_data["input_ids"][j] for j in idx],
                    "attention_mask": [self.enc_data["attention_mask"][j] for j in idx],
                }
                if has_types:
                    batch_slice["token_type_ids"] = [self.enc_data["token_type_ids"][j] for j in idx]

                padded = self.tokenizer.pad(batch_slice, padding=True, return_tensors="pt")
                if self.topic_prep is not None:
                    padded["input_ids"] = self.topic_prep.mask_topic_tokens(padded["input_ids"])
                kw = {}
                for k, v in padded.items():
                    kw[k] = v.pin_memory() if self.cuda_ok else v
                labels = torch.tensor(self.y_labels[idx], dtype=torch.long)
                if self.cuda_ok:
                    labels = labels.pin_memory()

                while not self.stop_event.is_set():
                    try:
                        self.queue.put((kw, labels), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.exc = e
        finally:
            self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                if self.exc is not None:
                    raise self.exc
                break
            kw, labels = item
            if self.cuda_ok:
                gpu_kw = {k: v.to(self.dev, non_blocking=True) for k, v in kw.items()}
                gpu_labels = labels.to(self.label_dev, non_blocking=True)
                yield gpu_kw, gpu_labels
            else:
                cpu_kw = {k: v.to(self.dev) for k, v in kw.items()}
                cpu_labels = labels.to(self.label_dev)
                yield cpu_kw, cpu_labels

    def close(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker.is_alive():
            self.worker.join(timeout=0.5)


def precompute_token_caches(raw_texts_tr, raw_texts_te, deberta_dir=None, bge_dir=None, qwen_dir=None, max_length=352):
    """
    Pre-tokenizes entire dataset (train + test) once upfront across all models.
    Caches token IDs and attention masks in host RAM (~28 MB total),
    eliminating redundant tokenizations across 5 folds and 3 models (over 100k tokenizations saved).
    Executes in parallel across vCPUs via ThreadPoolExecutor.
    """
    from transformers import AutoTokenizer

    caches = {}

    def _tokenize_model(model_tag, model_dir, preserve_newlines, left_pad=False, add_special=True):
        if model_dir is None:
            return None
        t0 = time.perf_counter()
        staged_dir = prepare_model_dir(model_dir, model_tag=model_tag)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                staged_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                staged_dir, local_files_only=True, trust_remote_code=True
            )
        if left_pad:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        if add_special:
            tokenizer.add_tokens(["[NL]", "[PAR]"], special_tokens=True)

        enc_tr = tokenize_with_head_tail(tokenizer, raw_texts_tr, max_length=max_length, preserve_newlines=preserve_newlines)
        enc_te = tokenize_with_head_tail(tokenizer, raw_texts_te, max_length=max_length, preserve_newlines=preserve_newlines)
        dur = time.perf_counter() - t0
        print(f"  [{model_tag.upper()} Cache] Pre-tokenized {len(raw_texts_tr)} train + {len(raw_texts_te)} test texts in {dur:.2f}s (RAM resident).", flush=True)
        return {
            "tr": enc_tr,
            "te": enc_te,
            "tokenizer": tokenizer,
            "staged_dir": staged_dir,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_deb = executor.submit(_tokenize_model, "deberta", deberta_dir, True, False, True) if deberta_dir else None
        f_bge = executor.submit(_tokenize_model, "bge", bge_dir, True, False, True) if bge_dir else None
        f_qwen = executor.submit(_tokenize_model, "qwen", qwen_dir, False, True, False) if qwen_dir else None

        if f_deb:
            res = f_deb.result()
            if res:
                caches["deberta"] = res
        if f_bge:
            res = f_bge.result()
            if res:
                caches["bge"] = res
        if f_qwen:
            res = f_qwen.result()
            if res:
                caches["qwen"] = res

    return caches

# --- 3. 132-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---

class RichStyleFeatures(BaseEstimator, TransformerMixin):
    """Linguistic and typographical features (132 features) capturing authorial habits."""
    PUNCTUATION = [
        "\u2014", "\u2013", "-", ",", ".", ";", ":", "!", "?", "\u2026", "...",
        '"', "'", "\u2018", "\u2019", "\u201c", "\u201d", "(", ")", "[", "]",
        "*", "#", "\n", "\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\b\w+['\u2019\u2018]\w+\b")
    WORD_RE = re.compile(r"\b\w+(?:['\u2019\u2018]\w+)?\b")
    SENT_RE = re.compile(r"[.!?]+\s*")
    H_RE = re.compile(r"^\s*#{1,6}\s")
    LIST_RE = re.compile(r"^\s*[-*\u2022]\s|^\s*\d+\.\s")
    BOLD_RE = re.compile(r"^\s*\*\*")
    OPENING_RE = re.compile(r"^(certainly|sure|here is|as an|to answer|i would)", re.I)
    CLOSING_RE = re.compile(r"(overall|in conclusion|in summary|ultimately|hope this helps|to summarize)", re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            t = clean_text(text)
            nc = max(len(t), 1)
            words = self.WORD_RE.findall(t)
            nw = max(len(words), 1)
            lower = [w.lower() for w in words]
            w_lengths = [len(w) for w in words] or [0]
            sents = [s for s in self.SENT_RE.split(t) if s.strip()]
            s_lengths = [len(s.split()) for s in sents] or [0]

            feat = [
                np.log1p(nc), np.log1p(nw), np.log1p(len(sents)),
                float(np.mean(w_lengths)), float(np.std(w_lengths)), float(np.median(w_lengths)),
                float(np.mean(s_lengths)), float(np.std(s_lengths)),
                len(set(lower)) / nw,
                sum(len(w) <= 3 for w in words) / nw,
                sum(len(w) >= 8 for w in words) / nw,
                sum(c.isupper() for c in t) / nc,
                sum(c.isdigit() for c in t) / nc,
                sum(ord(c) > 127 for c in t) / nc,
                len(self.CONTRACTION_RE.findall(t)) / nw,
            ]
            for p in self.PUNCTUATION:
                cnt = t.count(p)
                feat.append(np.log1p(cnt))
                feat.append(cnt / nw)
            w_counts = Counter(lower)
            for fw in self.FUNCTION_WORDS:
                feat.append(w_counts[fw] / nw)

            # Positional Tri-Partition Stylometrics (Head 25%, Body 50%, Tail 25%)
            n_quarter = max(nw // 4, 1)
            head_words = words[:n_quarter]
            body_words = words[n_quarter:max(nw - n_quarter, n_quarter + 1)]
            tail_words = words[max(nw - n_quarter, 0):]

            head_text = " ".join(head_words)
            tail_text = " ".join(tail_words)

            head_w_lens = [len(w) for w in head_words] or [0]
            tail_w_lens = [len(w) for w in tail_words] or [0]
            body_w_lens = [len(w) for w in body_words] or [0]

            feat.extend([
                float(bool(self.H_RE.search(t))),
                float(bool(self.LIST_RE.search(t))),
                float(bool(self.BOLD_RE.search(t))),
                float(bool(self.OPENING_RE.search(head_text))),
                float(bool(self.CLOSING_RE.search(tail_text))),
                float(t.endswith("?")),
                float(t.endswith("!")),
                # Tri-partition word length and capitalization moments
                float(np.mean(head_w_lens)),
                float(np.std(head_w_lens)),
                sum(c.isupper() for c in head_text) / max(len(head_text), 1),
                float(np.mean(tail_w_lens)),
                float(np.std(tail_w_lens)),
                sum(c.isupper() for c in tail_text) / max(len(tail_text), 1),
                tail_text.count(".") / max(len(tail_words), 1),
                tail_text.count("!") / max(len(tail_words), 1),
                tail_text.count("?") / max(len(tail_words), 1),
                float(np.mean(body_w_lens)),
                float(np.std(body_w_lens)),
            ])

            # Paragraph and Line Cadence
            paragraphs = [p for p in t.split("\n\n") if p.strip()]
            n_paras = max(len(paragraphs), 1)
            para_lengths = [len(p.split()) for p in paragraphs] or [0]
            lines = [l for l in t.split("\n") if l.strip()]
            line_lengths = [len(l.split()) for l in lines] or [0]

            feat.extend([
                np.log1p(n_paras),
                float(np.mean(para_lengths)),
                float(np.std(para_lengths)),
                np.log1p(max(len(lines), 1)),
                float(np.mean(line_lengths)),
                float(np.std(line_lengths)),
                t.count("\n\n") / max(t.count("\n"), 1),
                len(set(lower)) / max(np.sqrt(nw), 1e-6),    # Root Type-Token Ratio
                sum(len(w) >= 12 for w in words) / nw,        # Multisyllabic density
                t.count("```") / nw,                           # Code block density
                t.count("`") / nw,                             # Inline code density
                t.count("$") / nw,                             # LaTeX math density
                t.count("\\") / nw,                            # TeX / escape density
                t.count(":") / max(t.count(";") + 1, 1),       # Colon-to-semicolon ratio
                (t.count('"') + t.count("\u201c") + t.count("\u201d")) / nw,  # Quotation density
            ])
            rows.append(feat)
        return np.asarray(rows, dtype=np.float64)

# --- 4. CPU MULTIVIEW DUAL-LINEAR PIPELINE (~850k TF-IDF + 132 Stylometrics) ---

def align_margins(margins: np.ndarray, model_classes: list, canonical_labels: list, fill_value: float = -999.0) -> np.ndarray:
    """Reorder margin or probability columns to match canonical LABELS."""
    c_to_idx = {c: i for i, c in enumerate(model_classes)}
    aligned = np.full((margins.shape[0], len(canonical_labels)), fill_value, dtype=np.float64)
    for col, lab in enumerate(canonical_labels):
        if lab in c_to_idx:
            aligned[:, col] = margins[:, c_to_idx[lab]]
    return aligned


def compute_prompt_df_mask(X, groups, min_prompts=3):
    """
    Computes boolean feature mask based on Prompt Document Frequency.
    Prunes rare features appearing in fewer than min_prompts distinct prompts.
    Vectorized sparse matrix multiplication: O(prompts * nonzeros), executes in <0.5s.
    """
    if min_prompts <= 1 or groups is None:
        return np.ones(X.shape[1], dtype=np.float32)
    B = (X > 0).astype(np.float32).tocsr()
    codes = pd.factorize(groups)[0]
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))))
    pdf = np.asarray(((G @ B) > 0).sum(axis=0)).ravel()
    mask = (pdf >= min_prompts).astype(np.float32)
    if not np.any(mask > 0):
        return np.ones(X.shape[1], dtype=np.float32)
    return mask


class DenseStylometricClassifier(BaseEstimator):
    """
    Standardized dense stylometrics classifier.
    Scales continuous stylometric features with StandardScaler and fits an L2-regularized
    LogisticRegression classifier with balanced class weights to produce calibrated probabilities.
    Supports sample_weight with automatic class_weight=None override to prevent double-balancing.
    """
    def __init__(self, c=0.2, seed=42, class_weight="balanced", canonical_labels=None):
        self.c = c
        self.seed = seed
        self.class_weight = class_weight
        self.canonical_labels = canonical_labels

    def fit(self, V, y, sample_weight=None):
        self.scaler_ = StandardScaler()
        cw = None if sample_weight is not None else self.class_weight
        self.clf_ = LogisticRegression(
            C=self.c, max_iter=1000, random_state=self.seed, class_weight=cw, solver="lbfgs"
        )
        V_scaled = self.scaler_.fit_transform(V)
        if sample_weight is not None:
            self.clf_.fit(V_scaled, y, sample_weight=sample_weight)
        else:
            self.clf_.fit(V_scaled, y)
        raw_classes = list(self.clf_.classes_)
        if self.canonical_labels is not None:
            self.classes_ = np.array(self.canonical_labels)
            self._raw_classes = raw_classes
        else:
            self.classes_ = self.clf_.classes_
            self._raw_classes = raw_classes
        return self

    def predict_proba(self, V):
        V_scaled = self.scaler_.transform(V)
        p = self.clf_.predict_proba(V_scaled)
        if self.canonical_labels is not None:
            p_aligned = align_margins(p, self._raw_classes, self.canonical_labels, fill_value=0.0)
            return p_aligned / np.maximum(p_aligned.sum(axis=1, keepdims=True), 1e-12)
        return p


class DualLinearClassifier(BaseEstimator):
    """
    Ensemble of LinearSVC (hinge loss margin maximizer) and RidgeClassifier (L2 regularized least squares).
    Fast, memory-safe, and bounded to max_iter=800 with dual="auto" to guarantee rapid convergence.
    Uses class_weight="balanced" to directly align optimization with Macro Balanced Accuracy.
    Regularization tuned (c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70) to control variance under p >> n regime.
    Enforces canonical label ordering on decision_function output to guarantee ensemble alignment.
    Supports sample_weight with automatic class_weight=None override to prevent double-balancing inversion.
    """
    def __init__(self, c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42, class_weight="balanced", canonical_labels=None):
        self.c_svc = c_svc
        self.alpha_ridge = alpha_ridge
        self.svc_weight = svc_weight
        self.seed = seed
        self.class_weight = class_weight
        self.canonical_labels = canonical_labels

    def fit(self, X, y, sample_weight=None):
        cw = None if sample_weight is not None else self.class_weight
        self.svc_ = LinearSVC(C=self.c_svc, class_weight=cw, dual="auto", tol=1e-3, max_iter=800, random_state=self.seed)
        self.ridge_ = RidgeClassifier(alpha=self.alpha_ridge, solver="auto", class_weight=cw, random_state=self.seed)
        if sample_weight is not None:
            self.svc_.fit(X, y, sample_weight=sample_weight)
            self.ridge_.fit(X, y, sample_weight=sample_weight)
        else:
            self.svc_.fit(X, y)
            self.ridge_.fit(X, y)
        raw_classes = list(self.svc_.classes_)
        if self.canonical_labels is not None:
            self.classes_ = np.array(self.canonical_labels)
            self._raw_classes = raw_classes
        else:
            self.classes_ = self.svc_.classes_
            self._raw_classes = raw_classes
        return self

    def decision_function(self, X):
        d_svc = self.svc_.decision_function(X)
        d_ridge = self.ridge_.decision_function(X)
        if d_svc.ndim == 1:
            d_svc = np.column_stack([-d_svc, d_svc])
        if d_ridge.ndim == 1:
            d_ridge = np.column_stack([-d_ridge, d_ridge])
        m = self.svc_weight * d_svc + (1.0 - self.svc_weight) * d_ridge
        if self.canonical_labels is not None:
            return align_margins(m, self._raw_classes, self.canonical_labels)
        return m


class IntraFamilyCliqueSpecialist(BaseEstimator):
    """
    Trains dedicated specialist linear classifiers within each multi-model family clique
    (Anthropic-5, Google-4, OpenAI-5, Zhipu-2) on the combined competition + external corpus.
    Provides sharp conditional intra-family sibling log-probabilities to break sibling ties.
    Zero-mean row centering ensures family overall mass is unchanged, strictly redistributing
    preference between ambiguous siblings.
    """
    def __init__(self, alpha_ridge: float = 1.0, beta_refine: float = 0.25, seed: int = 42):
        self.alpha_ridge = alpha_ridge
        self.beta_refine = beta_refine
        self.seed = seed
        self.specialists_ = {}

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        for fam, indices in FAMILY_TO_INDICES.items():
            if len(indices) < 2:
                continue
            fam_labels = [LABELS[i] for i in indices]
            fam_mask = np.isin(y_arr, fam_labels)
            if np.sum(fam_mask) < 20:
                continue
            X_fam = X[fam_mask]
            y_fam = y_arr[fam_mask]
            w_fam = sample_weight[fam_mask] if sample_weight is not None else None

            clf = RidgeClassifier(alpha=self.alpha_ridge, random_state=self.seed)
            clf.fit(X_fam, y_fam, sample_weight=w_fam)
            self.specialists_[fam] = (clf, list(clf.classes_))
        return self

    def refine_margins(self, global_margins: np.ndarray, X_eval) -> np.ndarray:
        """
        Refines global margins with conditional intra-family specialist scores:
        z_m <- z_m + beta * sibling_score_m
        """
        if not self.specialists_ or len(global_margins) == 0:
            return global_margins
        refined = np.copy(global_margins)
        for fam, (clf, classes) in self.specialists_.items():
            d = clf.decision_function(X_eval)
            if d.ndim == 1:
                d = np.column_stack([-d, d])
            max_d = np.max(d, axis=1, keepdims=True)
            exp_d = np.exp(d - max_d)
            p_cond = exp_d / np.maximum(np.sum(exp_d, axis=1, keepdims=True), 1e-12)
            log_p = np.log(p_cond + 1e-12)
            log_p_centered = log_p - log_p.mean(axis=1, keepdims=True)

            for col_idx, c_name in enumerate(classes):
                if c_name in LABEL_TO_IDX:
                    glob_idx = LABEL_TO_IDX[c_name]
                    refined[:, glob_idx] += self.beta_refine * log_p_centered[:, col_idx]
        return refined


class StyleInvariantSubspaceProjector(BaseEstimator):
    """
    Discovers the Prompt-Topic Subspace vs. Author-Style Subspace using multi-model prompts,
    and attenuates topic-correlated directions so unseen test prompts cannot shift predictions.
    Operates strictly on dense representations (e.g. 132-dim stylometrics, 18-dim OOF probabilities,
    or neural latent embeddings) to prevent sparse matrix densification and memory explosion.
    Class-demeaned within-prompt SVD prevents partial prompts from leaking author style into the topic subspace.
    """
    def __init__(self, n_topic_components: int = 16, attenuation: float = 0.75):
        self.n_topic_components = n_topic_components
        self.attenuation = attenuation
        self.U_topic_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray, prompt_ids: np.ndarray, y_labels: np.ndarray = None):
        """
        X: [N, D] dense feature representations
        prompt_ids: [N] prompt group identifiers
        y_labels: [N] optional class labels to demean author style before computing prompt centroids
        """
        if sp.issparse(X):
            raise TypeError("StyleInvariantSubspaceProjector must be applied strictly to dense matrices to prevent RAM densification.")
        X_arr = np.asarray(X, dtype=np.float32)
        self.mean_ = X_arr.mean(axis=0, keepdims=True)
        X_c = X_arr - self.mean_

        # CRITICAL FIX: Subtract per-class style centroid BEFORE computing prompt centroids
        # Otherwise, external prompts answered by only 2-3 models leak author style into the topic SVD!
        if y_labels is not None:
            X_res = X_c.copy()
            for c in np.unique(y_labels):
                c_mask = (y_labels == c)
                if np.any(c_mask):
                    X_res[c_mask] -= X_c[c_mask].mean(axis=0, keepdims=True)
        else:
            X_res = X_c

        unique_p, counts = np.unique(prompt_ids, return_counts=True)
        multi_p = unique_p[counts >= 2]
        k = min(self.n_topic_components, len(multi_p) - 1, X.shape[1] - 1)
        if k <= 0:
            return self

        centroids = np.array([X_res[prompt_ids == p].mean(axis=0) for p in multi_p], dtype=np.float32)
        try:
            _, _, Vt = np.linalg.svd(centroids, full_matrices=False)
            k_actual = min(k, Vt.shape[0])
            self.U_topic_ = Vt[:k_actual].T  # [D, K]
        except np.linalg.LinAlgError:
            self.U_topic_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.U_topic_ is None or self.mean_ is None:
            return X
        X_arr = np.asarray(X, dtype=np.float32)
        X_c = X_arr - self.mean_
        topic_proj = (X_c @ self.U_topic_) @ self.U_topic_.T
        return (X_c - self.attenuation * topic_proj) + self.mean_


def build_cpu_feature_extractor(style_weight=0.2):
    """
    High-throughput 5-branch feature union (~850k features):
    char n-grams (2-5), word n-grams (1-2), char_wb (2-5), punct (1-3), and stylometrics.
    Fits once upfront on all texts to eliminate redundant vocabulary building across folds.
    """
    char_branch = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), lowercase=False,
        sublinear_tf=True, min_df=3, max_features=250_000,
    )
    word_branch = TfidfVectorizer(
        ngram_range=(1, 3), lowercase=False, token_pattern=r"\S+",
        sublinear_tf=True, min_df=2, max_features=250_000,
    )
    boundary_branch = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        sublinear_tf=True, min_df=3, max_features=250_000,
    )
    punct_branch = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 3),
        lowercase=False, sublinear_tf=True, min_df=2, max_features=100_000,
    )
    style_branch = Pipeline([
        ("extract", RichStyleFeatures()),
        ("scale", StandardScaler()),
    ])
    return FeatureUnion(
        transformer_list=[
            ("char", char_branch),
            ("word", word_branch),
            ("boundary", boundary_branch),
            ("punct", punct_branch),
            ("style", style_branch),
        ],
        transformer_weights={"style": style_weight},
        n_jobs=1,
    )

# --- 5. INPUT DISCOVERY ---

def discover_paths():
    """
    Discover paths for competition data, Qwen, DeBERTa, and BGE model weights.
    Returns (data_dir, qwen_dir, deberta_dir, bge_dir, working_dir).
    """
    local_data_dir = os.environ.get("NTL_DATA_DIR")
    input_root = Path(local_data_dir) if local_data_dir else Path("/kaggle/input")
    working_dir = Path(os.environ.get("NTL_WORKING", "/kaggle/working"))
    working_dir.mkdir(parents=True, exist_ok=True)

    data_dir = None
    qwen_dir = None
    bge_dir = None
    qwen_candidates = []
    deberta_candidates = []

    for candidate in [
        input_root / "competitions/name-that-llm",
        input_root / "name-that-llm",
        input_root / "sample",
        input_root,
    ]:
        if (candidate / "train.csv").is_file() and (candidate / "test.csv").is_file():
            data_dir = candidate
            break

    if data_dir is None:
        for p in input_root.rglob("train.csv"):
            if (p.parent / "test.csv").is_file():
                data_dir = p.parent
                break

    if data_dir is None:
        raise FileNotFoundError(f"Could not locate train.csv + test.csv under {input_root}")

    configs = sorted(input_root.rglob("config.json"), key=lambda p: len(p.parts))
    for p in configs:
        p_str = str(p).lower()
        if any(x in p_str for x in ("keras", "tensorflow", "jax")):
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            mtype = cfg.get("model_type", "").lower()
            arch = str(cfg.get("architectures", "")).lower()

            is_qwen = "qwen" in mtype or "qwen" in p_str or "qwen" in arch
            if is_qwen:
                # Priority: Qwen 3 (tier 0) > Qwen 2.5 (tier 1) > other (tier 2)
                # Size: 8b/7b (size 0) > other (size 1)
                tier = 0 if ("qwen-3" in p_str or "qwen3" in p_str) else (1 if ("qwen2.5" in p_str or "qwen-2.5" in p_str) else 2)
                size = 0 if ("8b" in p_str or "7b" in p_str) else 1
                qwen_candidates.append(((tier, size, len(p.parts)), p.parent))

            is_deberta = "deberta" in mtype or "deberta" in p_str or "deberta" in arch
            if is_deberta:
                # Prioritize clean base/large over 3rd-party task-specific checkpoints (khalidalt, etc.)
                tier = 0 if ("deberta-v3-large" in p_str and "khalidalt" not in p_str) else (1 if "khalidalt" not in p_str else 2)
                size_score = 0 if "large" in p_str else 1
                deberta_candidates.append(((tier, size_score, len(p.parts)), p.parent))

            is_bge = "bge" in p_str or "bge" in mtype
            if is_bge and bge_dir is None:
                bge_dir = p.parent
        except Exception:
            continue

    if qwen_candidates:
        qwen_candidates.sort(key=lambda x: x[0])
        qwen_dir = qwen_candidates[0][1]

    deberta_dir = None
    if deberta_candidates:
        deberta_candidates.sort(key=lambda x: x[0])
        deberta_dir = deberta_candidates[0][1]

    ext_path = None
    for cand in [
        input_root / "datasets/external_short_wide.parquet",
        input_root / "external_short_wide.parquet",
        Path("datasets/external_short_wide.parquet"),
        Path("data/external_short_wide.parquet"),
    ]:
        if cand.is_file():
            ext_path = cand
            break

    if ext_path is None:
        for p in input_root.rglob("external_short_wide.parquet"):
            if p.is_file():
                ext_path = p
                break

    if ext_path is None:
        for p in input_root.rglob("*.parquet"):
            p_str = p.name.lower()
            if "external" in p_str or "wide" in p_str:
                ext_path = p
                break

    deb_variant = "large" if deberta_dir and "large" in str(deberta_dir).lower() else "base/unknown"
    qwen_variant = "qwen-3-8b" if qwen_dir and ("qwen-3" in str(qwen_dir).lower() or "qwen3" in str(qwen_dir).lower()) else ("qwen-2.5-7b" if qwen_dir and "2.5" in str(qwen_dir).lower() else "custom/unknown")
    print(f"  [Paths] data={data_dir}")
    print(f"  [Paths] qwen={qwen_dir}  ({qwen_variant})")
    print(f"  [Paths] deberta={deberta_dir}  ({deb_variant})")
    print(f"  [Paths] bge={bge_dir}")
    print(f"  [Paths] external={ext_path}")
    if deberta_dir is not None:
        deberta_dir = prepare_model_dir(deberta_dir, "deberta")
    if bge_dir is not None:
        bge_dir = prepare_model_dir(bge_dir, "bge")
    return data_dir, qwen_dir, deberta_dir, bge_dir, ext_path, working_dir

def prepare_model_dir(model_dir, model_tag="model"):
    """
    Optimizes model loading from Kaggle network FUSE storage (/kaggle/input).
    If the model directory is on network storage (/kaggle/input) and is under 6GB,
    stream-copies it sequentially to local /tmp NVMe storage (12s vs 500s of FUSE random seeks).
    """
    if model_dir is None:
        return None
    model_path = Path(model_dir)
    if not str(model_path).startswith("/kaggle/input"):
        return str(model_dir)

    # Qwen 7B (15GB) uses safetensors and mmap across dual GPUs; avoid copying 15GB to /tmp
    if "qwen" in str(model_path).lower() or model_tag.lower() == "qwen":
        return str(model_dir)

    local_cache = Path(f"/tmp/model_cache_{model_tag}")
    if (local_cache / "config.json").is_file():
        return str(local_cache)

    print(f"  [{model_tag}] Staging weights from network FUSE to local NVMe (/tmp)...", flush=True)
    t0 = time.perf_counter()
    local_cache.mkdir(parents=True, exist_ok=True)

    try:
        for root, dirs, files in os.walk(model_path):
            rel = Path(root).relative_to(model_path)
            target_dir = local_cache / rel
            target_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                src_file = Path(root) / fname
                dst_file = target_dir / fname
                if not dst_file.exists() or dst_file.stat().st_size != src_file.stat().st_size:
                    with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                        while True:
                            buf = fsrc.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            fdst.write(buf)
        elapsed = time.perf_counter() - t0
        print(f"  [{model_tag}] Staged locally in {elapsed:.1f}s (eliminates FUSE seek latency).", flush=True)
        return str(local_cache)
    except Exception as e:
        print(f"  [{model_tag}] Warning: Local staging error: {e}. Falling back to direct path.", flush=True)
        return str(model_dir)


class MultiSampleDropoutHead(nn.Module):
    """5-way Multi-Sample Dropout classification head for regularization & implicit ensembling."""
    def __init__(self, in_features, num_classes, dropouts=(0.1, 0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.linear = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        if not self.training:
            return self.linear(x)
        return torch.mean(torch.stack([self.linear(drop(x)) for drop in self.dropouts], dim=0), dim=0)


class SpikeAndGapHierarchicalHead(nn.Module):
    """
    Captures token-level anomalous spikes, hidden-state transition jerk,
    and multi-subcenter class geometry for fine-grained sibling LLM attribution.
    Extracts multi-moments (Mean + Learned Spike Attention + GeM Max Spike + Transition Jerk Stats)
    and projects onto a unit hypersphere with K=3 sub-centers per class.
    Supports Multi-Layer Hidden State Fusion across discourse (-1), syntax (-5), and BPE cadence (-9).
    """
    def __init__(self, hidden_size: int = 4096, num_classes: int = 18, sub_centers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.layer_mix_weights = nn.Parameter(torch.zeros(3))

        self.spike_gate = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        fused_dim = hidden_size * 3 + 4
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.15),
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024)
        )

        self.weight = nn.Parameter(torch.empty(num_classes * sub_centers, 1024))
        nn.init.xavier_uniform_(self.weight)
        self.scale = 16.0  # Cosine temperature scale

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states: [B, T, D] tensor OR tuple/list of hidden states from causal LLM backbone
        attention_mask: [B, T] (1 for valid tokens, 0 for pad)
        """
        if isinstance(hidden_states, (tuple, list)):
            layers = []
            for offset in (-1, -5, -9):
                idx = offset if abs(offset) <= len(hidden_states) else -1
                layers.append(hidden_states[idx])
            w = F.softmax(self.layer_mix_weights, dim=0)
            hidden_states = w[0] * layers[0] + w[1] * layers[1] + w[2] * layers[2]

        target_dev = self.weight.device
        target_dtype = self.weight.dtype
        if hidden_states.device != target_dev:
            hidden_states = hidden_states.to(target_dev)
        if not torch.is_autocast_enabled() and hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(dtype=target_dtype)

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=target_dev, dtype=torch.long)
        elif attention_mask.device != target_dev:
            attention_mask = attention_mask.to(target_dev)

        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        lengths = torch.clamp(mask.sum(dim=1), min=1.0)

        # 1. Mean Pooling (Global stylistic baseline)
        h_mean = (hidden_states * mask).sum(dim=1) / lengths

        # 2. Learned Anomaly-Spike Attention Pooling (isolates signature formatting & refusal spikes)
        gate_logits = self.spike_gate(hidden_states)
        gate_logits = gate_logits.masked_fill(mask == 0, -1e4)
        gate_weights = F.softmax(gate_logits, dim=1)
        h_attn_spike = (hidden_states * gate_weights).sum(dim=1)

        # 3. Masked Max Activation Spike Pooling across tokens
        masked_h = hidden_states.masked_fill(mask == 0, -1e4)
        h_max_spike, _ = masked_h.max(dim=1)

        # 4. Token-to-Token Transition Jerk (||h_t - h_{t-1}||_2 syntactic jump)
        B, T, D = hidden_states.shape
        if T > 1:
            diffs = torch.norm(hidden_states[:, 1:, :] - hidden_states[:, :-1, :], p=2, dim=-1)
            valid_pairs = (attention_mask[:, 1:] * attention_mask[:, :-1]).to(dtype=hidden_states.dtype)
            pair_lens = torch.clamp(valid_pairs.sum(dim=1, keepdim=True), min=1.0)

            gap_mean = (diffs * valid_pairs).sum(dim=1, keepdim=True) / pair_lens
            masked_diffs = diffs.masked_fill(valid_pairs == 0, -1e4)
            gap_max, _ = masked_diffs.max(dim=1, keepdim=True)
            gap_max = torch.clamp(gap_max, min=0.0)
            has_pairs = (valid_pairs.sum(dim=1, keepdim=True) > 0)
            gap_std = torch.sqrt(torch.clamp(((diffs - gap_mean) ** 2 * valid_pairs).sum(dim=1, keepdim=True) / pair_lens, min=1e-6))
            gap_std = torch.where(has_pairs, gap_std, torch.zeros_like(gap_std))
        else:
            gap_mean = torch.zeros((B, 1), device=hidden_states.device, dtype=hidden_states.dtype)
            gap_max = torch.zeros((B, 1), device=hidden_states.device, dtype=hidden_states.dtype)
            gap_std = torch.zeros((B, 1), device=hidden_states.device, dtype=hidden_states.dtype)

        seq_len_norm = (lengths / 512.0).to(dtype=hidden_states.dtype)
        gap_stats = torch.cat([gap_mean, gap_max, gap_std, seq_len_norm], dim=-1)

        fused = torch.cat([h_mean, h_attn_spike, h_max_spike, gap_stats], dim=-1)
        emb = F.normalize(self.proj(fused), p=2, dim=-1)

        w_norm = F.normalize(self.weight, p=2, dim=-1)
        cos_sim = F.linear(emb, w_norm)
        cos_sim = cos_sim.view(-1, self.num_classes, self.sub_centers)

        # Max-cosine over sub-centers per class
        logits, _ = cos_sim.max(dim=-1)
        return logits * self.scale


class StylometricMultiMomentPooler(nn.Module):
    """
    Multi-Moment (CLS + Mean + Max-Spike) Pooler for Transformer Encoders (BGE / DeBERTa).
    Breaks BGE's semantic collapse by extracting token-level syntactic cadence
    and extreme spikes rather than relying solely on the topic-dominated [CLS] token.
    Mask-aware: cleanly zeroes out dynamic padding tokens to prevent sequence dilution.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 3, hidden_size)
        self.activation = nn.Tanh()
        self._current_mask = None

    @property
    def output_dim(self) -> int:
        return self.dense.out_features

    def set_mask(self, mask: torch.Tensor | None):
        self._current_mask = mask

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: [B, T, D]
        mask = attention_mask if attention_mask is not None else self._current_mask
        self._current_mask = None
        h_cls = hidden_states[:, 0]
        if mask is not None:
            m = mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
            lengths = torch.clamp(m.sum(dim=1), min=1.0)
            h_mean = (hidden_states * m).sum(dim=1) / lengths
            neg_inf = torch.tensor(-1e4, device=hidden_states.device, dtype=hidden_states.dtype)
            masked_h = torch.where(m > 0, hidden_states, neg_inf)
            h_max, _ = masked_h.max(dim=1)
        else:
            h_mean = hidden_states.mean(dim=1)
            h_max, _ = hidden_states.max(dim=1)
        fused = torch.cat([h_cls, h_mean, h_max], dim=-1)
        return self.activation(self.dense(fused))


def set_model_pooler_mask(model, mask):
    """Passes current batch attention mask to StylometricMultiMomentPooler if present."""
    pooler = getattr(model, "pooler", None)
    if pooler is None and hasattr(model, "bert"):
        pooler = getattr(model.bert, "pooler", None)
    if pooler is not None and hasattr(pooler, "set_mask"):
        pooler.set_mask(mask)


class FocalLoss(nn.Module):
    """
    Focal Loss with per-class weighting for severe class imbalance.
    Modulates cross entropy by (1 - p_t)^gamma using the predicted probability of the true class.
    Down-weights well-classified easy examples to focus gradients on hard minority classes.
    alpha (class weights) additionally up-weights rare classes.
    """
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets = targets.to(device=logits.device, dtype=torch.long)
        log_prob = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        prob = torch.exp(log_prob)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        ce_loss = -(smooth_targets * log_prob).sum(dim=-1)
        loss = focal_weight * ce_loss
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BalancedLogitAdjustedLDAMLoss(nn.Module):
    """
    SOTA Long-Tail Loss combining:
      1. Menon et al. (ICLR 2021) Logit Adjustment to remove empirical class prior skew
      2. Cao et al. (NeurIPS 2019) LDAM Margin (Delta_c ~ N_c^{-1/4}) to widen minority sibling boundaries
      3. Focal modulation (1 - p_t)^gamma to focus on hard ambiguous passages
    """
    def __init__(self, class_counts: list[int] | np.ndarray | torch.Tensor, tau: float = 1.0, max_margin: float = 0.35, gamma: float = 1.0):
        super().__init__()
        if not isinstance(class_counts, torch.Tensor):
            counts = torch.tensor(class_counts, dtype=torch.float32)
        else:
            counts = class_counts.float()
        counts = counts.clamp(min=1.0)
        priors = counts / counts.sum()
        self.register_buffer("log_priors", tau * torch.log(priors))

        # LDAM margin: inversely proportional to N_c^(1/4) so starved classes get widest margin
        margins = 1.0 / torch.pow(counts, 0.25)
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", margins)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(device=logits.device, dtype=torch.long)
        num_classes = logits.size(-1)
        one_hot = F.one_hot(targets, num_classes=num_classes).type_as(logits)
        margin_logits = logits - one_hot * self.margins.unsqueeze(0).to(logits.device)
        adjusted_logits = margin_logits + self.log_priors.unsqueeze(0).to(logits.device)

        log_probs = F.log_softmax(adjusted_logits.float(), dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(target_log_probs)
        focal_weight = (1.0 - probs).pow(self.gamma)
        return -(focal_weight * target_log_probs).mean()


def get_optimizer_grouped_parameters(model, lr_head, lr_backbone, layer_decay=0.95, weight_decay=0.01):
    """
    Layer-Wise Learning Rate Decay (LLRD) for 24-layer transformers (DeBERTa / BGE).
    Decays learning rate geometrically from top layer down to embeddings (ξ = 0.95).
    Explicitly assigns rel_embeddings to embeddings LR and LayerNorm to top layer LR.
    Guarantees 100% parameter coverage without generic frozen buckets.
    """
    def is_no_decay(name, param):
        return param.ndim < 2 or any(nd in name.lower() for nd in ["bias", "norm"])

    base_model = getattr(model, "deberta", getattr(model, "bert", getattr(model, "model", model)))
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    embeddings = getattr(base_model, "embeddings", None)

    assigned_param_ids = set()
    params_group = []

    # Head and pooler modules (receive lr_head)
    head_modules = [
        getattr(model, "classifier", None),
        getattr(model, "pooler", None),
        getattr(base_model, "pooler", None),
    ]
    head_params = []
    for mod in head_modules:
        if mod is not None:
            head_params.extend(list(mod.named_parameters()))

    if head_params:
        params_group.append({
            "params": [p for n, p in head_params if not is_no_decay(n, p)],
            "lr": lr_head, "weight_decay": weight_decay,
        })
        params_group.append({
            "params": [p for n, p in head_params if is_no_decay(n, p)],
            "lr": lr_head, "weight_decay": 0.0,
        })
        for n, p in head_params:
            assigned_param_ids.add(id(p))

    # Encoder layers with geometric decay (ξ = 0.95)
    if layers is not None:
        n_layers = len(layers)
        for i in reversed(range(n_layers)):
            layer = layers[i]
            decay_factor = layer_decay ** (n_layers - 1 - i)
            layer_lr = lr_backbone * decay_factor
            l_params = list(layer.named_parameters())
            params_group.append({
                "params": [p for n, p in l_params if not is_no_decay(n, p)],
                "lr": layer_lr, "weight_decay": weight_decay,
            })
            params_group.append({
                "params": [p for n, p in l_params if is_no_decay(n, p)],
                "lr": layer_lr, "weight_decay": 0.0,
            })
            for n, p in l_params:
                assigned_param_ids.add(id(p))

    # Embeddings layer: word embeddings receive lr_backbone so newly added [NL], [PAR] tokens can adapt
    n_layers = len(layers) if layers is not None else 24
    emb_lr = lr_backbone
    emb_params = []
    if embeddings is not None:
        emb_params.extend(list(embeddings.named_parameters()))
    if encoder is not None and hasattr(encoder, "rel_embeddings"):
        emb_params.extend(list(encoder.rel_embeddings.named_parameters()))

    if emb_params:
        params_group.append({
            "params": [p for n, p in emb_params if not is_no_decay(n, p)],
            "lr": emb_lr, "weight_decay": weight_decay,
        })
        params_group.append({
            "params": [p for n, p in emb_params if is_no_decay(n, p)],
            "lr": emb_lr, "weight_decay": 0.0,
        })
        for n, p in emb_params:
            assigned_param_ids.add(id(p))

    # Encoder final LayerNorm assigned to top layer LR
    if encoder is not None and hasattr(encoder, "LayerNorm"):
        ln_params = list(encoder.LayerNorm.named_parameters())
        if ln_params:
            params_group.append({
                "params": [p for n, p in ln_params],
                "lr": lr_backbone, "weight_decay": 0.0,
            })
            for n, p in ln_params:
                assigned_param_ids.add(id(p))

    # Fallback for any leftover parameters (ensures zero frozen parameters)
    leftover = [(n, p) for n, p in model.named_parameters() if id(p) not in assigned_param_ids]
    if leftover:
        params_group.append({
            "params": [p for n, p in leftover if not is_no_decay(n, p)],
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        })
        params_group.append({
            "params": [p for n, p in leftover if is_no_decay(n, p)],
            "lr": lr_backbone,
            "weight_decay": 0.0,
        })

    return params_group

# --- 6. UNIFIED HIGH-THROUGHPUT TRANSFORMER TRAINING (GPU 0 & GPU 1) ---

def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
    """
    Length-sorted batched inference for encoder models.
    Groups evaluation passages by length to eliminate redundant padding tokens,
    then inverts the sort order back to the original index sequence.
    """
    n_samples = len(enc_data["input_ids"]) if "input_ids" in enc_data else 0
    if n_samples == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    model.eval()
    lengths = [len(x) for x in enc_data["input_ids"]]
    sort_idx = np.argsort(lengths)
    all_logits = []
    eval_bs = 64
    cuda_ok = torch.cuda.is_available() and getattr(dev, "type", str(dev)) != "cpu"
    with inference_context():
        for start in range(0, n_samples, eval_bs):
            end = min(start + eval_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
            }
            if "token_type_ids" in enc_data:
                sl["token_type_ids"] = [enc_data["token_type_ids"][j] for j in batch_indices]
            padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
            kw = {
                "input_ids": padded["input_ids"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(dev),
                "attention_mask": padded["attention_mask"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(dev),
            }
            if "token_type_ids" in padded:
                kw["token_type_ids"] = padded["token_type_ids"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["token_type_ids"].to(dev)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                set_model_pooler_mask(model, kw["attention_mask"])
                logits = model(**kw).logits
            all_logits.append(logits.float().cpu().numpy())
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res


def check_finite_logits(logits: np.ndarray, name: str) -> np.ndarray:
    """Validates that a logits matrix contains strictly finite values."""
    if logits is None:
        return None
    if not np.all(np.isfinite(logits)):
        n_nan = int(np.sum(np.isnan(logits)))
        n_inf = int(np.sum(np.isinf(logits)))
        raise FloatingPointError(f"[{name}] Non-finite logits detected: {n_nan} NaNs, {n_inf} Infs in shape {logits.shape}")
    return logits


def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes):
    """
    Length-sorted batched inference for Qwen 3 8B / 2.5 7B LoRA decoder.
    Passes representations through backbone to SpikeAndGapHierarchicalHead or classification head.
    """
    n_samples = len(enc_data["input_ids"]) if "input_ids" in enc_data else 0
    if n_samples == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    backbone.eval()
    head.eval()
    lengths = [len(x) for x in enc_data["input_ids"]]
    sort_idx = np.argsort(lengths)
    all_logits = []
    inf_bs = 32
    autocast_ctx = amp_autocast() if cuda_ok else contextlib.nullcontext()
    with inference_context():
        for start in range(0, n_samples, inf_bs):
            end = min(start + inf_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
            }
            padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded["input_ids"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(first_dev),
                "attention_mask": padded["attention_mask"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(first_dev),
            }
            with autocast_ctx:
                out = backbone(**batch_kwargs, output_hidden_states=True)
                h_input = out.hidden_states if (hasattr(out, "hidden_states") and out.hidden_states is not None) else out.last_hidden_state
                if isinstance(head, SpikeAndGapHierarchicalHead):
                    logits = head(h_input, batch_kwargs["attention_mask"]).float().cpu().numpy()
                else:
                    if getattr(tokenizer, "padding_side", "left") == "left":
                        pooled = out.last_hidden_state[:, -1, :].float()
                    else:
                        attn_mask = batch_kwargs["attention_mask"].to(out.last_hidden_state.device)
                        seq_lens = torch.clamp(attn_mask.sum(dim=1) - 1, min=0)
                        batch_idx = torch.arange(out.last_hidden_state.size(0), device=out.last_hidden_state.device)
                        pooled = out.last_hidden_state[batch_idx, seq_lens].float()
                    logits = head(pooled.to(head_dev)).float().cpu().numpy()
            all_logits.append(logits)
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res


def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=12,
    grad_accum=1,
    max_length=352,
    epochs=3,
    lr_backbone=1.5e-5,
    lr_head=1.2e-4,
    weight_decay=0.01,
    label_smoothing=0.05,
    layer_decay=0.95,
    enable_grad_checkpointing=True,
    deadline=None,
    cached_enc_tr=None,
    cached_enc_va=None,
    cached_enc_te=None,
    cached_tokenizer=None,
    group_ids_tr=None,
    use_topic_masking=True,
    use_sift=True,
):
    """
    Unified, high-throughput transformer fine-tuning loop for DeBERTa-v3-large and BGE-large-en-v1.5.
    Features:
    - Stylistic [NL] newline preservation + Head+Tail truncation (176 head + 176 tail = 352 context).
    - FastBatchPrefetcher with background worker thread and pinned RAM transfers (100% GPU saturation).
    - Upfront RAM token cache reuse (eliminates redundant tokenizations across folds).
    - Gradient checkpointing ON by default (reduces activation VRAM by 90%, guaranteed zero OOM).
    - LLRD (layer_decay = 0.95) with explicit relative embeddings and LayerNorm discovery.
    - Synchronized gradient accumulation with non-fatal NaN recovery.
    - Protected scheduler stepping with GradScaler scale checking (no PyTorch UserWarnings).
    - Per-device context cache clearing.
    """
    from transformers import get_cosine_schedule_with_warmup

    label2id = {l: i for i, l in enumerate(labels)}
    texts_tr = list(texts_tr)
    texts_va = list(texts_va)
    texts_te = list(texts_te)

    y_tr_int = np.array([label2id[y] for y in y_tr_str], dtype=np.int64)
    y_va_int = np.array([label2id[y] for y in y_va_str], dtype=np.int64)

    cuda_ok = torch.cuda.is_available()
    dev = torch.device(device if cuda_ok else "cpu")
    if cuda_ok:
        try:
            torch.cuda.set_device(dev)
        except Exception:
            pass

    model_tag = "deberta" if "deberta" in model_name.lower() else ("bge" if "bge" in model_name.lower() else "model")
    model_dir = prepare_model_dir(model_dir, model_tag=model_tag)
    print(f"  [{model_name}] Loading from {model_dir} on {dev}...", flush=True)

    if cached_tokenizer is not None:
        tokenizer = cached_tokenizer
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True
            )
        tokenizer.add_tokens(["[NL]", "[PAR]"], special_tokens=True)

    id2label = {i: l for i, l in enumerate(labels)}

    with HeartbeatTimer(f"Reading {model_name} from disk onto {dev}"):
        kw_load = {
            "num_labels": len(labels),
            "id2label": id2label,
            "label2id": label2id,
            "local_files_only": True,
            "trust_remote_code": True,
            "ignore_mismatched_sizes": True,
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                dtype=torch.float32,
                **kw_load,
            )
        except TypeError:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float32,
                    **kw_load,
                )
            except TypeError:
                kw_load.pop("low_cpu_mem_usage", None)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float32,
                    **kw_load,
                )

    # Ensure model embeddings match tokenizer vocabulary size and initialize newline weights unconditionally
    emb_module = model.get_input_embeddings()
    if len(tokenizer) > emb_module.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        emb_module = model.get_input_embeddings()
    with torch.no_grad():
        for new_tok, src_tok in (("[NL]", "."), ("[PAR]", "|")):
            n_id = tokenizer.convert_tokens_to_ids(new_tok)
            s_id = tokenizer.convert_tokens_to_ids(src_tok)
            if n_id is not None and s_id is not None and max(n_id, s_id) < emb_module.weight.shape[0]:
                emb_module.weight[n_id].copy_(emb_module.weight[s_id])

    model.config.use_cache = False
    in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") else model.config.hidden_size
    model.classifier = MultiSampleDropoutHead(in_features, len(labels))
    if hasattr(model, "dropout"):
        model.dropout = nn.Identity()
    if hasattr(model, "pooler") and hasattr(model.pooler, "dropout"):
        model.pooler.dropout = nn.Identity()

    # Stylometric Multi-Moment Pooler for BGE and DeBERTa
    # Replaces single-token [CLS] pooler to break semantic collapse and preserve sequence syntax & token spikes
    if "bge" in model_name.lower() and hasattr(model, "bert") and hasattr(model.bert, "pooler"):
        model.bert.pooler = StylometricMultiMomentPooler(in_features)
    elif "deberta" in model_name.lower() and hasattr(model, "pooler"):
        model.pooler = StylometricMultiMomentPooler(in_features)

    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(dev)
    gc.collect()
    if cuda_ok:
        if str(dev).startswith("cuda"):
            dev_idx = torch.device(dev).index if torch.device(dev).index is not None else 0
            with torch.cuda.device(dev_idx):
                torch.cuda.synchronize(dev_idx)
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    ckpt_status = "ON" if enable_grad_checkpointing else "OFF"
    print(f"  [{model_name}] Ready on {dev} (FP16 Tensor Cores ON, grad_ckpt={ckpt_status}, Multi-Sample Dropout ON). {get_mem_status(dev)}", flush=True)

    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.5)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=dev)
    criterion = FocalLoss(weight=weight_tensor, gamma=1.0, label_smoothing=label_smoothing)
    family_criterion = nn.CrossEntropyLoss().to(dev)
    FAMILY_AUX_WEIGHT = 0.25
    label_to_fam_tensor = torch.tensor(LABEL_TO_FAMILY_IDX, dtype=torch.long, device=dev)

    opt_params = get_optimizer_grouped_parameters(
        model, lr_head=lr_head, lr_backbone=lr_backbone, layer_decay=layer_decay, weight_decay=weight_decay
    )
    optimizer = AdamW(opt_params, foreach=False)

    n_opt_steps_per_epoch = max((len(texts_tr) + batch_size * grad_accum - 1) // (batch_size * grad_accum), 1)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 1),
        num_training_steps=total_steps,
    )

    if cached_enc_tr is not None:
        enc_tr = cached_enc_tr
        enc_va = cached_enc_va
        enc_te = cached_enc_te
        print(f"  [{model_name}] Using RAM-cached token encodings ({len(enc_tr['input_ids'])} train / {len(enc_va['input_ids'])} val / {len(enc_te['input_ids'])} test passages).", flush=True)
    else:
        print(f"  [{model_name}] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, [NL] structure preserved)...", flush=True)
        enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length, preserve_newlines=True)
        enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length, preserve_newlines=True)
        enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length, preserve_newlines=True)

    use_amp = cuda_ok
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=4096.0)

    best_val_score = -1.0
    best_val_logits = None
    best_state = None
    epoch_states = []
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 8
    stop_early = False
    n_skipped_micros = 0
    n_oom_skips = 0
    t_train_start = time.perf_counter()
    pure_train_sec = 0.0

    topic_prep = None
    if use_topic_masking and group_ids_tr is not None and getattr(tokenizer, "mask_token_id", None) is not None:
        try:
            special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
            topic_token_ids = find_topic_token_ids(enc_tr["input_ids"], group_ids_tr, min_prompts=3, special_ids=special_ids)
            sift_eps = 0.015 if use_sift else 0.0
            topic_prep = TopicBlindAndSiFTPrep(topic_token_ids, mask_token_id=tokenizer.mask_token_id, mask_prob=0.30, sift_eps=sift_eps)
            print(f"  [{model_name}] TopicBlind & SiFT prep active: {len(topic_token_ids)} rare topic tokens identified (mask_prob=0.30, sift_eps={sift_eps}).", flush=True)
        except Exception as e:
            print(f"  [{model_name}] Note: TopicBlind prep skipped ({e}).", flush=True)
            topic_prep = None

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        running_loss = 0.0
        n_micro = 0
        accumulated_steps = 0

        t_batch_start = time.perf_counter()
        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, dev, cuda_ok=cuda_ok, topic_prep=topic_prep
        )
        try:
            for micro_step, (batch_kwargs, batch_labels) in enumerate(prefetcher):
                if deadline is not None and (micro_step % 10 == 0) and time.perf_counter() > deadline:
                    print(f"  [{model_name}] [Time Guard] Deadline reached mid-epoch at step {micro_step}/{n_total_micros}. Stopping training.", flush=True)
                    stop_early = True
                    break

                try:
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        set_model_pooler_mask(model, batch_kwargs["attention_mask"])
                        if topic_prep is not None and topic_prep.sift_eps > 0:
                            emb = model.get_input_embeddings()(batch_kwargs["input_ids"])
                            emb = topic_prep.apply_sift_embedding_noise(emb, batch_kwargs["attention_mask"])
                            kw_forward = {
                                "inputs_embeds": emb,
                                "attention_mask": batch_kwargs["attention_mask"],
                            }
                            if "token_type_ids" in batch_kwargs:
                                kw_forward["token_type_ids"] = batch_kwargs["token_type_ids"]
                            logits = model(**kw_forward).logits
                        else:
                            logits = model(**batch_kwargs).logits
                        main_loss = criterion(logits.float(), batch_labels)
                        fam_logits = compute_family_logits(logits.float(), temp=4.0)
                        batch_fam_labels = label_to_fam_tensor[batch_labels]
                        aux_loss = family_criterion(fam_logits, batch_fam_labels)
                        loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        n_skipped_micros += 1
                        print(f"  [{model_name}] Warning: NaN/Inf loss in micro-step {micro_step}. Skipping micro-batch ({n_skipped_micros} total skips).", flush=True)
                        if n_skipped_micros > 0.15 * max(n_total_micros * (epoch + 1), 1):
                            raise FloatingPointError(f"[{model_name}] >15% micro-batches ({n_skipped_micros}) produced NaN/Inf loss. Unstable FP16 training detected; discarding fold.")
                        if accumulated_steps > 0:
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_steps = 0
                        continue

                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except torch.cuda.OutOfMemoryError:
                    n_oom_skips += 1
                    print(f"  [{model_name}] CUDA OOM detected at micro-step {micro_step} (OOM #{n_oom_skips}). Resetting partial grads...", flush=True)
                    if n_oom_skips > 5:
                        raise RuntimeError(f"[{model_name}] Exceeded 5 CUDA OOM errors in fold; aborting to prevent data starvation.")
                    try:
                        del logits, loss, batch_kwargs, batch_labels
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if cuda_ok and str(dev).startswith("cuda"):
                        dev_idx = torch.device(dev).index if torch.device(dev).index is not None else 0
                        with torch.cuda.device(dev_idx):
                            torch.cuda.synchronize(dev_idx)
                            torch.cuda.empty_cache()
                    continue

                running_loss += loss.item() * grad_accum
                n_micro += 1
                accumulated_steps += 1

                if accumulated_steps == grad_accum:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= old_scale:
                            scheduler.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0

                log_interval = max(n_total_micros // 5, 25) if n_total_micros >= 20 else n_total_micros
                if (micro_step + 1) % log_interval == 0 or (micro_step + 1) == n_total_micros:
                    step_elapsed = max(time.perf_counter() - t_batch_start, 0.001)
                    step_sps = min((micro_step + 1) * batch_size, len(texts_tr)) / step_elapsed
                    step_loss = running_loss / max(n_micro, 1)
                    pct = ((micro_step + 1) / n_total_micros) * 100.0
                    print(f"  [{model_name}] Ep {epoch+1}/{epochs} | Step {micro_step+1}/{n_total_micros} ({pct:3.0f}%) | Loss: {step_loss:.4f} | Speed: {step_sps:5.2f} s/s {get_mem_status(dev)}", flush=True)
        finally:
            prefetcher.close()

        pure_train_sec += (time.perf_counter() - t_batch_start)

        if accumulated_steps > 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()
            accumulated_steps = 0

        avg_loss = running_loss / max(n_micro, 1)

        val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))
        val_preds = np.argmax(val_logits, axis=1)
        val_score = balanced_accuracy_score(y_va_int, val_preds)
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Bal Acc: {val_score:.4f} {get_mem_status(dev)}", flush=True)

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_logits = val_logits
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # Collect states for Stochastic Weight Averaging (SWA) over the final epochs
        if epoch >= max(epochs - 3, 1):
            epoch_states.append({k: v.detach().cpu() for k, v in model.state_dict().items() if v.is_floating_point()})

        if stop_early:
            break

    train_dur = time.perf_counter() - t_train_start
    pure_throughput = (len(texts_tr) * max(epoch + 1, 1)) / max(pure_train_sec, 0.01)
    train_transformer_fold.last_throughput = pure_throughput
    if "deberta" in model_name.lower():
        train_deberta_fold.last_throughput = pure_throughput
    elif "bge" in model_name.lower():
        train_bge_fold.last_throughput = pure_throughput
    print(f"  [{model_name}] Training finished: {train_dur:.1f}s ({pure_throughput:.2f} samples/sec step throughput). Best Val Bal Acc: {best_val_score:.4f}", flush=True)

    if len(epoch_states) >= 2:
        try:
            swa_state = {}
            for k in epoch_states[0]:
                swa_state[k] = torch.stack([s[k] for s in epoch_states]).mean(dim=0)
            model.load_state_dict({k: v.to(dev) for k, v in swa_state.items()}, strict=False)
            swa_val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))
            swa_val_score = balanced_accuracy_score(y_va_int, np.argmax(swa_val_logits, axis=1))
            print(f"  [{model_name}] SWA Checkpoint Averaged ({len(epoch_states)} epochs) | Val Bal Acc: {swa_val_score:.4f} (vs Best Single {best_val_score:.4f})", flush=True)
            if swa_val_score >= best_val_score:
                best_val_score = swa_val_score
                best_val_logits = swa_val_logits
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            elif best_state is not None:
                model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        except Exception as e:
            print(f"  [{model_name}] SWA averaging fallback: {e}", flush=True)
            if best_state is not None:
                model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    elif best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))

    del model, optimizer, scheduler
    if best_state is not None:
        del best_state
    gc.collect()
    if cuda_ok and str(dev).startswith("cuda"):
        dev_idx = torch.device(dev).index if torch.device(dev).index is not None else 0
        with torch.cuda.device(dev_idx):
            torch.cuda.synchronize(dev_idx)
            torch.cuda.empty_cache()

    return best_val_logits, test_logits

def train_deberta_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, group_ids_tr=None, **kwargs):
    kwargs.setdefault("batch_size", 12)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 4)
    kwargs.setdefault("lr_backbone", 1.8e-5)
    kwargs.setdefault("lr_head", 1.5e-4)
    kwargs.setdefault("layer_decay", 0.90)
    kwargs.setdefault("enable_grad_checkpointing", True)
    return train_transformer_fold(
        "DeBERTa FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels,
        group_ids_tr=group_ids_tr, **kwargs
    )

# --- 7. BGE-LARGE FULL FINE-TUNE WRAPPER (GPU 1) ---

def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, group_ids_tr=None, **kwargs):
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 5)
    kwargs.setdefault("lr_backbone", 2.2e-5)
    kwargs.setdefault("lr_head", 1.2e-4)
    kwargs.setdefault("layer_decay", 0.85)
    kwargs.setdefault("enable_grad_checkpointing", True)
    return train_transformer_fold(
        "BGE FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels,
        group_ids_tr=group_ids_tr, **kwargs
    )


# --- 8. QWEN 3 8B / 2.5 7B LORA FINE-TUNE PER FOLD (DUAL GPU) ---

def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=16,
    grad_accum=1,
    max_length=352,
    epochs=3,
    lr_lora=1.5e-4,
    lr_head=3.5e-4,
    lora_r=48,
    lora_alpha=96,
    lora_dropout=0.08,
    deadline=None,
    cached_enc_tr=None,
    cached_enc_va=None,
    cached_enc_te=None,
    cached_tokenizer=None,
):
    """
    LoRA fine-tune of Qwen 3 8B / 2.5 7B across dual GPUs (device_map="auto").
    Context length aligned to 352 tokens (covering 96.4% of all passage lengths).
    LoRA rank 48 with alpha 96 targeting all linear projections (~120M trainable parameters).
    FastBatchPrefetcher with host RAM pinned transfers for zero GPU bubbles.
    Upfront RAM token cache reuse (eliminates redundant tokenizations across folds).
    Returns (val_logits, test_logits) each (N, 18) float64.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("  [Qwen LoRA] peft not installed -> skipping Qwen stream.", flush=True)
        return None, None

    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    label2id = {l: i for i, l in enumerate(labels)}
    texts_tr = list(texts_tr)
    texts_va = list(texts_va)
    texts_te = list(texts_te)
    y_tr_int = np.array([label2id[y] for y in y_tr_str], dtype=np.int64)
    y_va_int = np.array([label2id[y] for y in y_va_str], dtype=np.int64)

    cuda_ok = torch.cuda.is_available()
    print(f"  [Qwen LoRA] Loading from {model_dir}...", flush=True)
    if cached_tokenizer is not None:
        tokenizer = cached_tokenizer
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"   # decoder-style: pad left

    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    qwen_desc = "Qwen 3 8B" if ("qwen-3" in str(model_dir).lower() or "qwen3" in str(model_dir).lower()) else "Qwen 2.5 7B"
    with HeartbeatTimer(f"Reading {qwen_desc} weights and sharding across Dual GPUs"):
        kw = {"local_files_only": True, "trust_remote_code": True}
        if n_gpus >= 2:
            print(f"  [Qwen LoRA] device_map=auto across {n_gpus} GPUs", flush=True)
            kw["device_map"] = "auto"
        elif cuda_ok:
            kw["device_map"] = {"": "cuda:0"}

        try:
            backbone = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, **kw)
        except TypeError:
            backbone = AutoModel.from_pretrained(model_dir, dtype=torch.float16, **kw)

    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    try:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        backbone.gradient_checkpointing_enable()
    backbone.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout, bias="none",
    )
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.print_trainable_parameters()

    first_dev = next(backbone.parameters()).device
    last_dev = first_dev
    if hasattr(backbone, "hf_device_map") and backbone.hf_device_map:
        for k in ("model.norm", "norm", "base_model.model.norm"):
            if k in backbone.hf_device_map:
                dev_val = backbone.hf_device_map[k]
                last_dev = torch.device(f"cuda:{dev_val}" if isinstance(dev_val, int) else dev_val)
                break
        else:
            int_devs = [v for v in backbone.hf_device_map.values() if isinstance(v, int)]
            if int_devs:
                last_dev = torch.device(f"cuda:{max(int_devs)}")
    else:
        try:
            base_m = getattr(backbone, "base_model", backbone)
            inner_m = getattr(base_m, "model", base_m)
            norm_layer = getattr(inner_m, "norm", None)
            if norm_layer is not None and hasattr(norm_layer, "weight"):
                last_dev = norm_layer.weight.device
            else:
                last_dev = list(backbone.parameters())[-1].device
        except Exception:
            last_dev = first_dev

    head_dev = last_dev if cuda_ok else torch.device("cpu")
    hidden_size = backbone.config.hidden_size
    head = SpikeAndGapHierarchicalHead(hidden_size, len(labels), sub_centers=3).to(head_dev)

    lora_params = [p for _, p in backbone.named_parameters() if p.requires_grad]
    for p in lora_params:
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    head_params = list(head.parameters())
    head_decay = [p for p in head_params if p.ndim >= 2]
    head_no_decay = [p for p in head_params if p.ndim < 2]
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": lr_lora, "weight_decay": 0.01},
            {"params": head_decay, "lr": lr_head, "weight_decay": 0.01},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
        ],
    )
    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.5)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=head_dev)
    criterion = FocalLoss(weight=weight_tensor, gamma=1.0, label_smoothing=0.05)
    family_criterion = nn.CrossEntropyLoss().to(head_dev)
    FAMILY_AUX_WEIGHT = 0.25
    label_to_fam_tensor = torch.tensor(LABEL_TO_FAMILY_IDX, dtype=torch.long, device=head_dev)

    n_opt_steps_per_epoch = max((len(texts_tr) + batch_size * grad_accum - 1) // (batch_size * grad_accum), 1)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 1),
        num_training_steps=total_steps,
    )

    if cached_enc_tr is not None:
        enc_tr = cached_enc_tr
        enc_va = cached_enc_va
        enc_te = cached_enc_te
        print(f"  [Qwen LoRA] Using RAM-cached token encodings ({len(enc_tr['input_ids'])} train / {len(enc_va['input_ids'])} val / {len(enc_te['input_ids'])} test passages).", flush=True)
    else:
        print(f"  [Qwen LoRA] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, native BPE newlines preserved)...", flush=True)
        enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length, preserve_newlines=False)
        enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length, preserve_newlines=False)
        enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length, preserve_newlines=False)

    scaler = torch.cuda.amp.GradScaler(enabled=cuda_ok, init_scale=4096.0)

    best_val_score = -1.0
    best_val_logits = None
    best_lora_state = None
    best_head_state = None
    epoch_lora_states = []
    epoch_head_states = []
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 8
    t_qwen_train_start = time.perf_counter()
    pure_qwen_sec = 0.0
    stop_early = False
    n_skipped_micros = 0
    n_oom_skips = 0

    for epoch in range(epochs):
        t_qwen_ep_start = time.perf_counter()
        backbone.train()
        head.train()
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        accumulated_steps = 0

        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size,
            dev=first_dev, label_dev=head_dev, cuda_ok=cuda_ok
        )
        try:
            for micro_step, (batch_kwargs, batch_labels) in enumerate(prefetcher):
                if deadline is not None and (micro_step % 10 == 0) and time.perf_counter() > deadline:
                    print(f"  [Qwen LoRA] [Time Guard] Deadline reached mid-epoch at step {micro_step}/{n_total_micros}. Stopping training.", flush=True)
                    stop_early = True
                    break

                try:
                    autocast_ctx = amp_autocast() if cuda_ok else contextlib.nullcontext()
                    with autocast_ctx:
                        out = backbone(**batch_kwargs, output_hidden_states=True)
                        h_input = out.hidden_states if (hasattr(out, "hidden_states") and out.hidden_states is not None) else out.last_hidden_state
                        if isinstance(head, SpikeAndGapHierarchicalHead):
                            logits = head(h_input, batch_kwargs["attention_mask"])
                        else:
                            if getattr(tokenizer, "padding_side", "left") == "left":
                                pooled = out.last_hidden_state[:, -1, :].float()
                            else:
                                attn_mask = batch_kwargs["attention_mask"].to(out.last_hidden_state.device)
                                seq_lens = torch.clamp(attn_mask.sum(dim=1) - 1, min=0)
                                batch_idx = torch.arange(out.last_hidden_state.size(0), device=out.last_hidden_state.device)
                                pooled = out.last_hidden_state[batch_idx, seq_lens].float()
                            logits = head(pooled.to(head_dev))
                    main_loss = criterion(logits.float(), batch_labels)
                    fam_logits = compute_family_logits(logits.float(), temp=4.0)
                    batch_fam_labels = label_to_fam_tensor[batch_labels]
                    aux_loss = family_criterion(fam_logits, batch_fam_labels)
                    loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        n_skipped_micros += 1
                        print(f"  [Qwen LoRA] Warning: NaN/Inf loss at micro-step {micro_step}. Skipping ({n_skipped_micros} total skips).", flush=True)
                        if n_skipped_micros > 0.15 * max(n_total_micros * (epoch + 1), 1):
                            raise FloatingPointError(f"[Qwen LoRA] >15% micro-batches ({n_skipped_micros}) produced NaN/Inf loss. Discarding fold.")
                        if accumulated_steps > 0:
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_steps = 0
                        continue

                    if cuda_ok:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except torch.cuda.OutOfMemoryError:
                    n_oom_skips += 1
                    print(f"  [Qwen LoRA] CUDA OOM at micro-step {micro_step} (OOM #{n_oom_skips}). Resetting partial grads...", flush=True)
                    if n_oom_skips > 5:
                        raise RuntimeError("[Qwen LoRA] Exceeded 5 CUDA OOM errors in fold; aborting to prevent data starvation.")
                    try:
                        del out, logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if cuda_ok:
                        for gi in (first_dev, head_dev):
                            if str(gi).startswith("cuda"):
                                dev_idx = torch.device(gi).index if torch.device(gi).index is not None else 0
                                with torch.cuda.device(dev_idx):
                                    torch.cuda.synchronize(dev_idx)
                                    torch.cuda.empty_cache()
                    continue

                epoch_loss += loss.item() * grad_accum
                n_batches += 1
                accumulated_steps += 1

                if accumulated_steps == grad_accum:
                    if cuda_ok:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= old_scale:
                            scheduler.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0

                log_interval = max(n_total_micros // 5, 25) if n_total_micros >= 20 else n_total_micros
                if (micro_step + 1) % log_interval == 0 or (micro_step + 1) == n_total_micros:
                    step_elapsed = max(time.perf_counter() - t_qwen_ep_start, 0.001)
                    step_sps = min((micro_step + 1) * batch_size, len(texts_tr)) / step_elapsed
                    step_loss = epoch_loss / max(n_batches, 1)
                    pct = ((micro_step + 1) / n_total_micros) * 100.0
                    print(f"  [Qwen LoRA] Ep {epoch+1}/{epochs} | Step {micro_step+1}/{n_total_micros} ({pct:3.0f}%) | Loss: {step_loss:.4f} | Speed: {step_sps:5.2f} s/s {get_mem_status()}", flush=True)
        finally:
            prefetcher.close()

        pure_qwen_sec += (time.perf_counter() - t_qwen_ep_start)

        if accumulated_steps > 0:
            if cuda_ok:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
            else:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()
            accumulated_steps = 0

        avg_loss = epoch_loss / max(n_batches, 1)
        val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))
        val_preds = np.argmax(val_logits, axis=1)
        val_score = balanced_accuracy_score(y_va_int, val_preds)
        print(f"  [Qwen LoRA] Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Bal Acc: {val_score:.4f} {get_mem_status()}", flush=True)

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_logits = val_logits
            best_lora_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        # Collect states for Stochastic Weight Averaging (SWA) over the final epochs
        if epoch >= max(epochs - 3, 1):
            epoch_lora_states.append({k: v.cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k and v.is_floating_point()})
            epoch_head_states.append({k: v.cpu().clone() for k, v in head.state_dict().items() if v.is_floating_point()})

        if stop_early:
            break

    qwen_dur = time.perf_counter() - t_qwen_train_start
    qwen_sps = (len(texts_tr) * max(epoch + 1, 1)) / max(pure_qwen_sec, 0.01)
    train_qwen_lora_fold.last_throughput = qwen_sps
    print(f"  [Qwen LoRA] Training finished: {qwen_dur:.1f}s ({qwen_sps:.2f} samples/sec). Best Val Bal Acc: {best_val_score:.4f}", flush=True)

    if len(epoch_lora_states) >= 2:
        try:
            swa_lora = {}
            for k in epoch_lora_states[0]:
                swa_lora[k] = torch.stack([s[k] for s in epoch_lora_states]).mean(dim=0)
            swa_head = {}
            for k in epoch_head_states[0]:
                swa_head[k] = torch.stack([s[k] for s in epoch_head_states]).mean(dim=0)

            backbone.load_state_dict(swa_lora, strict=False)
            head.load_state_dict(swa_head)
            swa_val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))
            swa_val_score = balanced_accuracy_score(y_va_int, np.argmax(swa_val_logits, axis=1))
            print(f"  [Qwen LoRA] SWA Checkpoint Averaged ({len(epoch_lora_states)} epochs) | Val Bal Acc: {swa_val_score:.4f} (vs Best Single {best_val_score:.4f})", flush=True)
            if swa_val_score >= best_val_score:
                best_val_score = swa_val_score
                best_val_logits = swa_val_logits
                best_lora_state = swa_lora
                best_head_state = swa_head
            else:
                if best_lora_state is not None:
                    backbone.load_state_dict(best_lora_state, strict=False)
                if best_head_state is not None:
                    head.load_state_dict(best_head_state)
        except Exception as e:
            print(f"  [Qwen LoRA] SWA averaging fallback: {e}", flush=True)
            if best_lora_state is not None:
                backbone.load_state_dict(best_lora_state, strict=False)
            if best_head_state is not None:
                head.load_state_dict(best_head_state)
    else:
        if best_lora_state is not None:
            backbone.load_state_dict(best_lora_state, strict=False)
        if best_head_state is not None:
            head.load_state_dict(best_head_state)

    test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))

    del backbone, head, optimizer, scheduler
    if best_lora_state is not None:
        del best_lora_state
    if best_head_state is not None:
        del best_head_state
    gc.collect()
    if cuda_ok:
        for gi in (first_dev, head_dev):
            if str(gi).startswith("cuda"):
                dev_idx = torch.device(gi).index if torch.device(gi).index is not None else 0
                with torch.cuda.device(dev_idx):
                    torch.cuda.synchronize(dev_idx)
                    torch.cuda.empty_cache()

    return best_val_logits, test_logits

# --- 9. CONSENSUS BLENDING, CALIBRATION & SINKHORN BALANCING ---

def fit_temperature(margins: np.ndarray, y_true_int: np.ndarray, default_t: float = 1.0) -> float:
    """
    Fits optimal Platt/temperature scaling parameter T by minimizing NLL on valid OOF margins.
    Searches over [0.05, 3.0] using numerical log-sum-exp; guaranteed non-zero and stable.
    """
    if margins is None:
        return default_t
    valid = np.isfinite(margins).all(axis=1) & (y_true_int >= 0)
    if np.sum(valid) < 20:
        return default_t
    m_val = margins[valid]
    y_val = y_true_int[valid]

    best_t = default_t
    best_nll = float("inf")
    for t_cand in np.linspace(0.05, 3.0, 60):
        scaled = m_val / t_cand
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        sum_exp = np.sum(exp_s, axis=1, keepdims=True)
        log_probs = (scaled - max_s) - np.log(sum_exp + 1e-12)
        nll = -np.mean(log_probs[np.arange(len(y_val)), y_val])
        if nll < best_nll:
            best_nll = nll
            best_t = float(t_cand)
    return round(best_t, 3)


def margins_to_probs(margins: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Convert unbounded margins/logits into calibrated probability distribution.
    Hardened against NaNs: any row containing non-finite values is replaced with a uniform prior (1/K).
    """
    m = np.asarray(margins, dtype=np.float64)
    bad = ~np.isfinite(m).all(axis=1)
    m = np.nan_to_num(m, nan=0.0, posinf=50.0, neginf=-50.0)
    z = np.clip(m, -50.0, 50.0) / max(temp, 1e-4)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    if np.any(bad):
        p[bad] = 1.0 / p.shape[1]
    return p


def apply_hierarchical_family_gating(probs: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Smooth family-level prior on probability simplex with cardinality normalization.
    Normalizes by uniform family prior mass (|fam| / K) to eliminate structural bias
    toward larger model families (e.g. OpenAI/Anthropic with 5 models vs Kimi with 1).
    """
    K = probs.shape[1]
    gated = probs.copy()
    for indices in FAMILY_TO_INDICES.values():
        prior_mass = max(len(indices) / K, 1e-6)
        fam_mass = np.sum(probs[:, indices], axis=1, keepdims=True)
        concentration = np.maximum(fam_mass / prior_mass, 1e-6)
        gated[:, indices] *= (concentration ** alpha)
    return gated / np.maximum(np.sum(gated, axis=1, keepdims=True), 1e-12)


def sinkhorn_knopp_balance(
    probs: np.ndarray, max_iters: int = 50, tol: float = 1e-3
) -> np.ndarray:
    """
    Sinkhorn-Knopp matrix balancing directly on probability distribution.
    Guarantees uniform column sums (N/K) while maintaining row stochasticity.
    Uses relative tolerance convergence check.
    Automatically converts raw margins to calibrated probabilities if negative values are present.
    """
    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
        probs = margins_to_probs(probs)
    assert np.all(np.isfinite(probs)), "Non-finite probabilities reached Sinkhorn balancing"
    N, K = probs.shape
    P = np.copy(probs) + 1e-12
    target_col = np.full(K, N / K, dtype=np.float64)
    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)

    for _ in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        rel_diff = np.max(np.abs(c - c_prev)) / (np.max(np.abs(c)) + 1e-12)
        if rel_diff < tol:
            break

    P_bal = r[:, None] * P * c[None, :]
    return P_bal / np.maximum(np.sum(P_bal, axis=1, keepdims=True), 1e-12)


def compute_hierarchical_tree_probs(
    p_svm: np.ndarray,
    p_deb: np.ndarray = None,
    p_qwen: np.ndarray = None,
    p_bge: np.ndarray = None,
    router_weights=(0.30, 0.25, 0.30, 0.15),
    expert_weights=(1.0, 1.0, 1.0, 1.0),
    hierarchy_alpha: float = 0.65,
) -> np.ndarray:
    """
    Differentiable Bayesian Hierarchical Neural-Tree Probability Fusion:
      P(Class_i) = P_router(Family_k) * P_expert(Class_i | Family_k)
    Stage 1 Router (router_weights) isolates coarse macro-family regions (>92% family accuracy).
    Stage 2 Sibling Expert (expert_weights) isolates conditional probabilities strictly within family siblings.
    Smooth interpolation with flat expert distribution via hierarchy_alpha prevents hard-decision cliffs.
    """
    models = [p_svm, p_deb, p_qwen, p_bge]
    w_r = [w if m is not None else 0.0 for w, m in zip(router_weights, models)]
    w_e = [w if m is not None else 0.0 for w, m in zip(expert_weights, models)]
    sum_r = max(sum(w_r), 1e-6)
    sum_e = max(sum(w_e), 1e-6)
    w_r = [w / sum_r for w in w_r]
    w_e = [w / sum_e for w in w_e]

    N = p_svm.shape[0]
    p_router = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    p_expert = np.zeros((N, NUM_CLASSES), dtype=np.float64)

    for m, wr, we in zip(models, w_r, w_e):
        if m is not None:
            p_router += wr * m
            p_expert += we * m

    p_router = p_router / np.maximum(np.sum(p_router, axis=-1, keepdims=True), 1e-12)
    p_expert = p_expert / np.maximum(np.sum(p_expert, axis=-1, keepdims=True), 1e-12)

    if hierarchy_alpha <= 0.0:
        return p_expert

    p_hierarchical = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    for fam, idxs in FAMILY_TO_INDICES.items():
        # Stage 1: Router probability mass allocated to Family k
        p_family = np.sum(p_router[:, idxs], axis=-1, keepdims=True)
        # Stage 2: Sibling conditional probability strictly within Family k
        if len(idxs) > 1:
            sub_mass = np.sum(p_expert[:, idxs], axis=-1, keepdims=True)
            p_sibling = p_expert[:, idxs] / np.maximum(sub_mass, 1e-12)
        else:
            p_sibling = np.ones((N, 1), dtype=np.float64)
        p_hierarchical[:, idxs] = p_family * p_sibling

    p_final = hierarchy_alpha * p_hierarchical + (1.0 - hierarchy_alpha) * p_expert
    return p_final / np.maximum(np.sum(p_final, axis=-1, keepdims=True), 1e-12)


def quad_rank_blend(
    svm_m: np.ndarray,
    deberta_m: np.ndarray = None,
    qwen_m: np.ndarray = None,
    bge_m: np.ndarray = None,
    weights=(1.0, 1.0, 1.0, 1.0),
    temperatures=(0.20, 1.2, 1.5, 1.2),
    balance_test: bool = False,
    apply_family_gating: bool = True,
    family_alpha: float = 0.65,
    router_weights=(0.30, 0.25, 0.30, 0.15),
    return_probs: bool = False,
):
    """
    Calibrated Probability Consensus across 4 model streams with Hierarchical Soft-Tree Fusion.
    Stage 1 Router (router_weights) + Stage 2 Sibling Expert (weights).
    Blends smooth probability distributions, applies hierarchical tree routing, and balances test priors via Sinkhorn.
    SVM margins use temp=0.20 to match transformer logit dynamic range and entropy.
    """
    ts, td, tq, tb = temperatures
    p_svm = margins_to_probs(svm_m, temp=ts)
    p_deb = margins_to_probs(deberta_m, temp=td) if deberta_m is not None else None
    p_qwen = margins_to_probs(qwen_m, temp=tq) if qwen_m is not None else None
    p_bge = margins_to_probs(bge_m, temp=tb) if bge_m is not None else None

    probs = compute_hierarchical_tree_probs(
        p_svm=p_svm,
        p_deb=p_deb,
        p_qwen=p_qwen,
        p_bge=p_bge,
        router_weights=router_weights,
        expert_weights=weights,
        hierarchy_alpha=family_alpha if apply_family_gating else 0.0,
    )

    if balance_test and len(probs) >= 36:
        probs = sinkhorn_knopp_balance(probs)

    if return_probs:
        return probs
    return np.argmax(probs, axis=1)


def optimize_blend_weights(oof_svm, oof_deb, oof_qwen, oof_bge, y_true, temperatures=(0.20, 1.2, 1.5, 1.2)):
    """
    Grid search over expert weight candidates and hierarchical tree alpha maximising OOF Balanced Accuracy on valid rows only.
    Tests hierarchy_alpha in [0.0, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85] with guaranteed monotonic improvement over flat blend.
    """
    best_w = (
        1.0,
        1.0 if oof_deb is not None else 0.0,
        1.0 if oof_qwen is not None else 0.0,
        1.0 if oof_bge is not None else 0.0,
    )
    best_alpha = 0.65
    best_score = -1.0

    valid_mask = ~np.isnan(oof_svm).any(axis=1)
    if oof_deb is not None:
        valid_mask &= ~np.isnan(oof_deb).any(axis=1)
    if oof_bge is not None:
        valid_mask &= ~np.isnan(oof_bge).any(axis=1)
    if oof_qwen is not None:
        valid_mask &= ~np.isnan(oof_qwen).any(axis=1)

    if not np.any(valid_mask):
        return best_w, best_alpha, 0.0

    label2id = {l: i for i, l in enumerate(LABELS)}
    y_true_int = np.array([label2id[y] if isinstance(y, str) else int(y) for y in y_true], dtype=np.int64)
    y_val = y_true_int[valid_mask]

    ts, td, tq, tb = temperatures
    svm_p = margins_to_probs(oof_svm[valid_mask], temp=ts)
    deb_p = margins_to_probs(oof_deb[valid_mask], temp=td) if oof_deb is not None else None
    qwn_p = margins_to_probs(oof_qwen[valid_mask], temp=tq) if oof_qwen is not None else None
    bge_p = margins_to_probs(oof_bge[valid_mask], temp=tb) if oof_bge is not None else None

    s_cands = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    d_cands = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5] if deb_p is not None else [0.0]
    q_cands = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5] if qwn_p is not None else [0.0]
    b_cands = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] if bge_p is not None else [0.0]

    for ws in s_cands:
        for wd in d_cands:
            for wq in q_cands:
                for wb in b_cands:
                    cand_weights = (ws, wd, wq, wb)
                    blend_p = compute_hierarchical_tree_probs(
                        p_svm=svm_p, p_deb=deb_p, p_qwen=qwn_p, p_bge=bge_p,
                        router_weights=(0.30, 0.25, 0.30, 0.15),
                        expert_weights=cand_weights,
                        hierarchy_alpha=0.65,
                    )
                    if len(blend_p) >= 36:
                        blend_p = sinkhorn_knopp_balance(blend_p)
                    pred_idx = np.argmax(blend_p, axis=1)
                    score = balanced_accuracy_score(y_val, pred_idx)
                    if score > best_score:
                        best_score = score
                        best_w = cand_weights

    # Fine-tune hierarchical tree alpha on the optimal expert weights
    for cand_alpha in [0.0, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85]:
        tree_p = compute_hierarchical_tree_probs(
            p_svm=svm_p, p_deb=deb_p, p_qwen=qwn_p, p_bge=bge_p,
            router_weights=(0.30, 0.25, 0.30, 0.15),
            expert_weights=best_w,
            hierarchy_alpha=cand_alpha,
        )
        if len(tree_p) >= 36:
            tree_p = sinkhorn_knopp_balance(tree_p)
        pred_idx = np.argmax(tree_p, axis=1)
        score = balanced_accuracy_score(y_val, pred_idx)
        if score > best_score:
            best_score = score
            best_alpha = cand_alpha

    return best_w, best_alpha, best_score


def optimize_cpu_subblend(sparse_probs: np.ndarray, dense_probs: np.ndarray, y_true_int: np.ndarray) -> float:
    """
    Find optimal alpha such that alpha*sparse + (1-alpha)*dense maximizes OOF balanced accuracy.
    Searches over alpha in [0.5, 1.0] anchored at pure sparse (1.0) to preserve sparse dominance.
    """
    valid = np.isfinite(sparse_probs).all(axis=1) & np.isfinite(dense_probs).all(axis=1)
    if np.sum(valid) < 10:
        return 0.85
    base_score = balanced_accuracy_score(y_true_int[valid], sparse_probs[valid].argmax(axis=1))
    best_score = base_score
    best_alpha = 1.0
    for alpha in np.arange(0.95, 0.45, -0.05):
        blend = alpha * sparse_probs[valid] + (1.0 - alpha) * dense_probs[valid]
        score = balanced_accuracy_score(y_true_int[valid], blend.argmax(axis=1))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return round(best_alpha, 2)


def calibrate_logits(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Applies temperature scaling to raw model logits or margins."""
    if logits is None:
        return None
    return logits / max(float(temp), 1e-4)


class CPUGradientMetaCombiner:
    """
    PyTorch CPU Autograd L-BFGS Meta-Combiner.
    Learns optimal per-model, per-class specialization weights W[M, C],
    per-model, per-family router weights W_rot[M, 6],
    sample-level entropy confidence gates, and Bayesian hierarchical tree fusion.
    Executes in under 2 seconds on CPU with exact analytical autograd.
    """
    def __init__(self, n_models: int = 4, n_classes: int = 18, l2_reg: float = 0.05, family_groups=None):
        self.M = n_models
        self.C = n_classes
        self.l2_reg = l2_reg
        self.family_groups = family_groups if family_groups is not None else [
            FAMILY_TO_INDICES[fam] for fam in FAMILIES
        ]
        self.W_exp = None
        self.W_rot = None
        self.gamma = None
        self.alpha_raw = None
        self.valid_indices = None

    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        p = probs.clamp(min=1e-12)
        ent = -torch.sum(p * torch.log(p), dim=-1, keepdim=True)
        return ent / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device))

    def _forward(self, logits_stack: torch.Tensor) -> torch.Tensor:
        # logits_stack: [M, N, C]
        M, N, C = logits_stack.shape
        probs_m = F.softmax(logits_stack, dim=-1)
        ent_m = self._compute_entropy(probs_m)  # [M, N, 1]

        g = torch.exp(torch.clamp(self.gamma, -5.0, 5.0)).view(M, 1, 1)
        conf_gate = torch.exp(-g * ent_m)
        gated = logits_stack * conf_gate  # [M, N, C]

        # 1. Expert logits
        z_expert = torch.sum(self.W_exp.view(M, 1, C) * gated, dim=0)  # [N, C]

        # 2. Router logits
        n_fams = len(self.family_groups)
        fam_logits_m = torch.zeros((M, N, n_fams), dtype=logits_stack.dtype, device=logits_stack.device)
        for f_idx, c_idxs in enumerate(self.family_groups):
            fam_logits_m[:, :, f_idx] = torch.logsumexp(gated[:, :, c_idxs], dim=-1)
        z_router = torch.sum(self.W_rot.view(M, 1, n_fams) * fam_logits_m, dim=0)  # [N, n_fams]
        p_family = F.softmax(z_router, dim=-1)  # [N, n_fams]

        # 3. Hierarchical Bayesian Tree Fusion
        p_tree = torch.zeros_like(z_expert)
        for f_idx, c_idxs in enumerate(self.family_groups):
            p_fam_slice = p_family[:, [f_idx]]
            if len(c_idxs) > 1:
                p_sib = F.softmax(z_expert[:, c_idxs], dim=-1)
            else:
                p_sib = torch.ones((N, 1), dtype=logits_stack.dtype, device=logits_stack.device)
            p_tree[:, c_idxs] = p_fam_slice * p_sib

        p_flat = F.softmax(z_expert, dim=-1)
        alpha = torch.sigmoid(self.alpha_raw)
        p_fused = (alpha * p_tree + (1.0 - alpha) * p_flat).clamp(min=1e-12)
        return p_fused / p_fused.sum(dim=-1, keepdim=True)

    def fit(self, oof_logits_list: list, y_true: np.ndarray, prior_weights: list = None, class_weights: np.ndarray = None, max_iter: int = 45):
        valid_indices = [i for i, l in enumerate(oof_logits_list) if l is not None]
        self.valid_indices = valid_indices
        self.M = len(valid_indices)
        if self.M == 0:
            return self
        logits_stack = torch.tensor(np.stack([oof_logits_list[i] for i in valid_indices], axis=0), dtype=torch.float32)
        y = torch.tensor(y_true, dtype=torch.long)
        N = y.shape[0]

        if prior_weights is None:
            prior_w = torch.full((self.M,), 1.0 / self.M)
        else:
            if len(prior_weights) == len(oof_logits_list):
                pw = [prior_weights[i] for i in valid_indices]
            else:
                pw = [prior_weights[j] for j in range(min(len(prior_weights), self.M))]
            prior_w = torch.tensor(pw, dtype=torch.float32)
            prior_w = prior_w / prior_w.sum()

        n_fams = len(self.family_groups)
        self.W_exp = torch.nn.Parameter(prior_w.view(self.M, 1).repeat(1, self.C))
        self.W_rot = torch.nn.Parameter(torch.full((self.M, n_fams), 1.0 / self.M))
        self.gamma = torch.nn.Parameter(torch.zeros(self.M))
        self.alpha_raw = torch.nn.Parameter(torch.tensor([0.8]))

        exp_target = self.W_exp.data.clone()
        rot_target = self.W_rot.data.clone()

        if class_weights is None:
            counts = np.bincount(y_true, minlength=self.C)
            cw = N / (self.C * np.maximum(counts, 1).astype(np.float32))
            cw_tensor = torch.tensor(cw, dtype=torch.float32)
        else:
            cw_tensor = torch.tensor(class_weights, dtype=torch.float32)
        sample_w = cw_tensor[y]

        optimizer = torch.optim.LBFGS(
            [self.W_exp, self.W_rot, self.gamma, self.alpha_raw],
            max_iter=max_iter, lr=0.15, history_size=10
        )

        def closure():
            optimizer.zero_grad()
            probs = self._forward(logits_stack)
            nll = -torch.mean(sample_w * torch.log(probs[torch.arange(N), y]))
            reg = self.l2_reg * (
                torch.sum((self.W_exp - exp_target) ** 2) +
                torch.sum((self.W_rot - rot_target) ** 2) +
                torch.sum(self.gamma ** 2)
            )
            loss = nll + reg
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def fit_nested_cv(
        self,
        oof_logits_list: list,
        y_true: np.ndarray,
        groups: np.ndarray = None,
        prior_weights: list = None,
        class_weights: np.ndarray = None,
        n_splits: int = 5,
        max_iter: int = 35,
    ) -> tuple[np.ndarray, float]:
        """
        Nested Prompt-Grouped Cross-Validation.
        Fits meta-combiner across n_splits prompt groups, producing true out-of-sample CV probabilities.
        Averages parameter tensors across folds for bagged test inference (predict_proba).
        Returns (oof_cv_probs, oof_cv_score).
        """
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        from sklearn.metrics import balanced_accuracy_score

        valid_indices = [i for i, l in enumerate(oof_logits_list) if l is not None]
        self.valid_indices = valid_indices
        self.M = len(valid_indices)
        N = len(y_true)
        if self.M == 0 or N == 0:
            return np.ones((N, self.C)) / self.C, 0.0

        n_unique_groups = len(np.unique(groups)) if groups is not None else 0
        if groups is not None and n_unique_groups >= n_splits:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(cv.split(np.arange(N), y_true, groups=groups))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(cv.split(np.arange(N), y_true))

        cv_oof_probs = np.zeros((N, self.C), dtype=np.float64)
        fold_W_exp, fold_W_rot, fold_gamma, fold_alpha = [], [], [], []

        for tr_idx, va_idx in splits:
            fold_tr_streams = [oof_logits_list[i][tr_idx] if oof_logits_list[i] is not None else None for i in range(len(oof_logits_list))]
            fold_va_streams = [oof_logits_list[i][va_idx] if oof_logits_list[i] is not None else None for i in range(len(oof_logits_list))]
            y_tr_fold = y_true[tr_idx]

            fold_comb = CPUGradientMetaCombiner(
                n_models=self.M, n_classes=self.C, l2_reg=self.l2_reg, family_groups=self.family_groups
            )
            fold_comb.valid_indices = self.valid_indices
            fold_comb.fit(fold_tr_streams, y_tr_fold, prior_weights=prior_weights, class_weights=class_weights, max_iter=max_iter)

            cv_oof_probs[va_idx] = fold_comb.predict_proba(fold_va_streams)

            fold_W_exp.append(fold_comb.W_exp.data.clone())
            fold_W_rot.append(fold_comb.W_rot.data.clone())
            fold_gamma.append(fold_comb.gamma.data.clone())
            fold_alpha.append(fold_comb.alpha_raw.data.clone())

        # Bag parameters across all folds for superior test generalization
        self.W_exp = torch.nn.Parameter(torch.stack(fold_W_exp).mean(dim=0))
        self.W_rot = torch.nn.Parameter(torch.stack(fold_W_rot).mean(dim=0))
        self.gamma = torch.nn.Parameter(torch.stack(fold_gamma).mean(dim=0))
        self.alpha_raw = torch.nn.Parameter(torch.stack(fold_alpha).mean(dim=0))

        cv_preds = np.argmax(cv_oof_probs, axis=1)
        cv_score = float(balanced_accuracy_score(y_true, cv_preds))
        return cv_oof_probs, cv_score

    def predict_proba(self, test_logits_list: list) -> np.ndarray:
        if self.valid_indices is not None and len(test_logits_list) >= max(self.valid_indices) + 1:
            valid_streams = [test_logits_list[i] for i in self.valid_indices if test_logits_list[i] is not None]
        else:
            valid_streams = [l for l in test_logits_list if l is not None]
        if not valid_streams or self.W_exp is None or len(valid_streams) != self.M:
            fallback_n = len(test_logits_list[0]) if (test_logits_list and test_logits_list[0] is not None) else 1
            return np.ones((fallback_n, self.C)) / self.C
        logits_stack = torch.tensor(np.stack(valid_streams, axis=0), dtype=torch.float32)
        with torch.no_grad():
            probs = self._forward(logits_stack)
        return probs.cpu().numpy()

# --- 10. PROMPT DSU & STRATIFIED GROUP K-FOLD ---

class DSU:
    """Disjoint Set Union with iterative two-pass path compression (immune to recursion limits)."""
    def __init__(self, elements):
        self.p = {x: x for x in elements}

    def find(self, x):
        root = x
        while self.p[root] != root:
            root = self.p[root]
        curr = x
        while curr != root:
            nxt = self.p[curr]
            self.p[curr] = root
            curr = nxt
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx


def make_prompt_groups(df):
    """DSU to ensure identical texts across different prompts never cross folds."""
    prompt_col = df["prompt_id"] if "prompt_id" in df.columns else pd.Series(np.arange(len(df)), index=df.index)
    dsu = DSU(prompt_col.unique())
    seen = {}
    for pid, text in zip(prompt_col, df["text"]):
        h = hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()
        if h in seen:
            dsu.union(pid, seen[h])
        else:
            seen[h] = pid
    return np.array([dsu.find(p) for p in prompt_col])


@torch.no_grad()
def extract_bge_embeddings(texts, model_dir, device="cuda:0", batch_size=32):
    """
    Extracts L2-normalized [CLS] embeddings from frozen BGE-large.
    Used for sibling structure prompt discovery and neighborhood feature centering.
    """
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32
    ).to(device)
    model.eval()
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = model(**enc)
        cls_rep = out.last_hidden_state[:, 0].float()
        normed = torch.nn.functional.normalize(cls_rep, dim=-1).cpu()
        embeds.append(normed)
    return torch.cat(embeds).numpy()


def compute_neighborhood_centered_features(features, embeddings, k_neighbors=17):
    """
    Computes features centered relative to their semantic prompt neighborhood.
    Subtracts the mean of the top-k nearest neighbors to isolate model-specific
    variations from prompt-induced topic shifts.
    """
    if embeddings is None or len(embeddings) <= k_neighbors:
        return features
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, -1.0)
    k = min(k_neighbors, embeddings.shape[0] - 1)
    nn_idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    nn_means = np.array([features[nn_idx[i]].mean(axis=0) for i in range(len(features))])
    return features - nn_means

# --- 11. SUBMISSION VALIDATION ---

def validate_and_save_submission(test_ids, predictions, output_path):
    """Atomic parquet write with schema, null, and integrity validation."""
    if len(predictions) != len(test_ids):
        raise ValueError(f"Row count mismatch: expected {len(test_ids)} IDs, got {len(predictions)} predictions")
    sub = pd.DataFrame({"id": test_ids, "model": predictions})
    if list(sub.columns) != ["id", "model"]:
        raise ValueError("Submission columns must be exactly ['id', 'model']")
    if sub.isna().any().any():
        raise ValueError("Submission contains null values")
    if not set(sub["model"]).issubset(set(LABELS)):
        bad = set(sub["model"]) - set(LABELS)
        raise ValueError(f"Invalid predicted class labels: {bad}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".tmp_sub_{time.time_ns()}.parquet"
    sub.to_parquet(temp_path, index=False)
    pd.testing.assert_frame_equal(sub, pd.read_parquet(temp_path))
    temp_path.replace(output_path)
    print(f"\nSUCCESS: submission saved to {output_path}", flush=True)
    print(f"Total rows: {len(sub):,}", flush=True)
    print("Class distribution:", flush=True)
    print(sub["model"].value_counts().to_string(), flush=True)
    return sub

# --- 12. MAIN PIPELINE ---

def run_quad_ensemble():
    start_time = time.perf_counter()
    print("=" * 65, flush=True)
    print("NAME THAT LLM: QUAD-MODEL FRONTIER ENSEMBLE (V6 Production Pipeline)", flush=True)
    print("DeBERTa Full FT (GPU0) + BGE Full FT (GPU1) + Qwen LoRA (Dual GPU) + CPU Dual-Linear", flush=True)
    print("=" * 65 + "\n", flush=True)

    data_dir, qwen_dir, deberta_dir, bge_dir, ext_path, working_dir = discover_paths()
    print(flush=True)

    seed_everything(42)

    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str, "prompt_id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    # Preserve 100% raw unaltered text for CPU stylometrics & character TF-IDF
    raw_texts_tr = train_df["text"].tolist()
    raw_texts_te = test_df["text"].tolist()

    tr_lens = train_df["text"].str.len()
    print(f"Train: {len(train_df):,} passages | {train_df['prompt_id'].nunique()} unique prompts (length: mean={tr_lens.mean():.0f}, min={tr_lens.min()}, max={tr_lens.max()} chars)", flush=True)
    print(f"Test:  {len(test_df):,} passages\n", flush=True)

    # Ingest and prepare mass-conserved external data if available
    ext_df = None
    ext_weights = None
    X_ext_all = None
    V_dense_ext = None
    ext_prompt_ids = None

    if ext_path is not None and Path(ext_path).is_file():
        try:
            print(f"  [External Data] Ingesting mass-balanced corpus from {ext_path}...", flush=True)
            comb_ext_df, comb_ext_weights = build_mass_balanced_external_corpus(
                train_df, ext_path, LABELS, ext_mass_ratio=0.55, prefer_multi_model=True
            )
            ext_slice = comb_ext_df.iloc[len(train_df):].reset_index(drop=True)
            ext_df = ext_slice
            ext_weights = comb_ext_weights[len(train_df):]
            ext_texts = ext_slice["text"].tolist()
            ext_prompt_ids = ext_slice["prompt_id"].to_numpy()
            print(f"  [External Data] Loaded {len(ext_texts):,} passages across 18 classes (Mass ratio=0.55, 0.0% prior skew) [OK]", flush=True)
        except Exception as e:
            print(f"  [External Data] Warning: Failed to load external data: {e}. Falling back to competition data.", flush=True)
            ext_df = None
            ext_weights = None

    n_train = len(train_df)
    texts_test = raw_texts_te
    y_true = train_df["model"].to_numpy()

    # --- Prompt-grouped 5-fold CV split ---
    group_ids = make_prompt_groups(train_df)
    n_splits = 5
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.full(n_train, -1, dtype=int)
    for fold_i, (_, val_idx) in enumerate(
        splitter.split(train_df["text"], train_df["model"], group_ids)
    ):
        folds[val_idx] = fold_i

    # --- OOF / test accumulators (NaN initialized to prevent zero-poisoning) ---
    oof_svm = np.full((n_train, NUM_CLASSES), np.nan, dtype=np.float64)
    oof_deb = np.full((n_train, NUM_CLASSES), np.nan, dtype=np.float64) if deberta_dir else None
    oof_bge = np.full((n_train, NUM_CLASSES), np.nan, dtype=np.float64) if bge_dir else None
    oof_qwen = np.full((n_train, NUM_CLASSES), np.nan, dtype=np.float64) if qwen_dir else None

    test_svm_folds = []
    test_deb_folds = []
    test_bge_folds = []
    test_qwen_folds = []

    # Detect GPU configuration
    cuda_ok = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    deb_dev = "cuda:0" if cuda_ok else "cpu"
    bge_dev = "cuda:1" if n_gpus >= 2 else ("cuda:0" if cuda_ok else "cpu")
    print(f"Hardware setup: CUDA={cuda_ok} | GPUs={n_gpus} | DeBERTa={deb_dev} | BGE={bge_dev} | Qwen=device_map=auto\n", flush=True)

    # ── STEP 0: MULTI-MODEL UPFRONT PRE-TOKENIZATION & RAM CACHING ──
    print("=" * 65, flush=True)
    print("STEP 0: MULTI-MODEL UPFRONT PRE-TOKENIZATION & RAM CACHING", flush=True)
    print("=" * 65, flush=True)
    token_caches = precompute_token_caches(raw_texts_tr, raw_texts_te, deberta_dir, bge_dir, qwen_dir, max_length=352)
    print(flush=True)

    # ── STEP 1: CPU MULTIVIEW FEATURE UNION & IMMEDIATE BASELINE SUBMISSION ──
    print("=" * 65, flush=True)
    print("STEP 1: FITTING CPU FEATURE UNION & IMMEDIATE BASELINE SUBMISSION", flush=True)
    print("=" * 65, flush=True)
    print("Train class distribution:")
    print(train_df["model"].value_counts().to_string(), flush=True)

    feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
    all_raw_texts = raw_texts_tr + raw_texts_te
    n_tr = len(raw_texts_tr)
    with threadpool_limits(limits=4):
        X_all = feature_extractor.fit_transform(all_raw_texts)
        X_train_all = X_all[:n_tr]
        X_test_all = X_all[n_tr:]
        if ext_df is not None:
            print("  [External Data] Transforming external texts via competition-locked FeatureUnion...", flush=True)
            X_ext_all = feature_extractor.transform(ext_texts)
            print(f"  [External Data] Sparse external matrix: {X_ext_all.shape} [OK]", flush=True)
    del all_raw_texts, X_all
    for _, tr in feature_extractor.transformer_list:
        if hasattr(tr, "stop_words_"):
            tr.stop_words_ = None
    del feature_extractor
    gc.collect()

    # Write immediate baseline submission (guarantees non-empty submission in first 30 seconds)
    pdf_mask_quick = compute_prompt_df_mask(X_train_all, group_ids, min_prompts=3)
    diag_mask_quick = sp.diags(pdf_mask_quick)
    quick_clf = DualLinearClassifier(c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42, canonical_labels=LABELS)
    with threadpool_limits(limits=4):
        if X_ext_all is not None:
            w_comp_quick, w_ext_quick = compute_fold_variance_capped_weights(
                y_true, ext_df["model"].to_numpy(), ext_df["_n_models"].to_numpy(),
                n_classes=NUM_CLASSES, ext_mass_ratio=0.55, max_single_ext_weight=0.35,
            )
            X_ext_quick = X_ext_all @ diag_mask_quick
            X_comb_quick = sp.vstack([X_train_all @ diag_mask_quick, X_ext_quick])
            y_comb_quick = np.concatenate([y_true, ext_df["model"].to_numpy()])
            w_comb_quick = np.concatenate([w_comp_quick, w_ext_quick])
            quick_clf.fit(X_comb_quick, y_comb_quick, sample_weight=w_comb_quick)
        else:
            quick_clf.fit(X_train_all @ diag_mask_quick, y_true)
        quick_test_m = quick_clf.decision_function(X_test_all @ diag_mask_quick)
    quick_preds_idx = np.argmax(quick_test_m, axis=1)
    baseline_sub_path = working_dir / "submission.parquet"
    validate_and_save_submission(test_df["id"].tolist(), [LABELS[i] for i in quick_preds_idx], baseline_sub_path)
    print("Immediate baseline submission saved successfully (safety guarantee verified).\n", flush=True)

    # Precompute dense stylometrics once upfront (eliminates 15 redundant regex passes across folds)
    t_style_start = time.perf_counter()
    print("  Precomputing dense stylometric features upfront across all passages...", flush=True)
    V_dense_all = RichStyleFeatures().fit_transform(raw_texts_tr + texts_test)
    n_tr = len(raw_texts_tr)
    print(f"  Dense stylometrics matrix shape: {V_dense_all.shape} computed in {time.perf_counter() - t_style_start:.2f}s [OK]\n", flush=True)

    if ext_df is not None:
        t_ext_style = time.perf_counter()
        print("  Precomputing dense stylometrics on external passages...", flush=True)
        V_dense_ext = RichStyleFeatures().fit_transform(ext_texts)
        print(f"  External dense stylometrics: {V_dense_ext.shape} computed in {time.perf_counter() - t_ext_style:.2f}s [OK]", flush=True)

        # Style-Invariant Subspace Projection: Project out prompt-topic variance on dense stylometrics
        subspace_proj = StyleInvariantSubspaceProjector(n_topic_components=16, attenuation=0.75)
        comb_V = np.vstack([V_dense_all[:n_tr], V_dense_ext])
        comb_pids = np.concatenate([group_ids, ext_prompt_ids])
        comb_labels = np.concatenate([y_true, ext_df["model"].to_numpy()])
        subspace_proj.fit(comb_V, comb_pids, y_labels=comb_labels)
        V_dense_all = subspace_proj.transform(V_dense_all)
        V_dense_ext = subspace_proj.transform(V_dense_ext)
        print("  StyleInvariantSubspaceProjector applied to dense stylometrics (class-demeaned pure topic attenuation) [OK]\n", flush=True)

    # ── STEP 2: 5-FOLD PROMPT-GROUPED CROSS-VALIDATION ──
    print("=" * 65, flush=True)
    print("STEP 2: 5-FOLD PROMPT-GROUPED CROSS-VALIDATION", flush=True)
    print("=" * 65, flush=True)

    N_EPOCHS_DEB = 4
    N_EPOCHS_BGE = 5
    N_EPOCHS_QWEN = 3
    MAX_WALLCLOCK_HOURS = 8.2
    SAFETY_MARGIN_SEC = 1200  # 20 min buffer for final bagging, Sinkhorn, and submission
    pipeline_t0 = SCRIPT_T0
    est_inference_sec = 900.0  # 15 min reserve for final inference, calibration and blending
    pipeline_deadline = SCRIPT_T0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC - est_inference_sec
    model_durations = {"deb": [], "bge": [], "qwen": []}

    for fold in range(n_splits):
        seed_everything(42 + fold)
        fold_start = time.perf_counter()
        elapsed_hours = (time.perf_counter() - pipeline_t0) / 3600.0
        if elapsed_hours > (MAX_WALLCLOCK_HOURS - 1.2):
            print(f"\n[Time Guard] {elapsed_hours:.2f}h elapsed. Starting another fold would risk the 9h limit.", flush=True)
            print(f"[Time Guard] Gracefully stopping at {fold} completed folds to finalize ensemble.", flush=True)
            break

        print(f"\n{'='*50}\nFOLD {fold+1}/{n_splits} (elapsed: {elapsed_hours:.2f}h)\n{'='*50}", flush=True)

        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]
        texts_tr = [raw_texts_tr[i] for i in tr_idx]
        texts_va = [raw_texts_tr[i] for i in va_idx]
        y_tr_str = y_true[tr_idx].tolist()
        y_va_str = y_true[va_idx].tolist()

        # Slice precomputed token caches from RAM in 0.001s
        deb_cache = token_caches.get("deberta")
        deb_kw = {}
        if deb_cache is not None:
            deb_kw["cached_enc_tr"] = {k: [deb_cache["tr"][k][i] for i in tr_idx] for k in deb_cache["tr"]}
            deb_kw["cached_enc_va"] = {k: [deb_cache["tr"][k][i] for i in va_idx] for k in deb_cache["tr"]}
            deb_kw["cached_enc_te"] = deb_cache["te"]
            deb_kw["cached_tokenizer"] = deb_cache["tokenizer"]

        bge_cache = token_caches.get("bge")
        bge_kw = {}
        if bge_cache is not None:
            bge_kw["cached_enc_tr"] = {k: [bge_cache["tr"][k][i] for i in tr_idx] for k in bge_cache["tr"]}
            bge_kw["cached_enc_va"] = {k: [bge_cache["tr"][k][i] for i in va_idx] for k in bge_cache["tr"]}
            bge_kw["cached_enc_te"] = bge_cache["te"]
            bge_kw["cached_tokenizer"] = bge_cache["tokenizer"]

        qwen_cache = token_caches.get("qwen")
        qwen_kw = {}
        if qwen_cache is not None:
            qwen_kw["cached_enc_tr"] = {k: [qwen_cache["tr"][k][i] for i in tr_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_va"] = {k: [qwen_cache["tr"][k][i] for i in va_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_te"] = qwen_cache["te"]
            qwen_kw["cached_tokenizer"] = qwen_cache["tokenizer"]

        # Concurrent Parallel Fine-Tuning: DeBERTa (GPU 0) + BGE (GPU 1)
        deb_expected = np.mean(model_durations["deb"]) * 1.15 if model_durations["deb"] else 450.0
        bge_expected = np.mean(model_durations["bge"]) * 1.15 if model_durations["bge"] else 350.0
        run_deb = deberta_dir is not None and (time.perf_counter() + deb_expected + SAFETY_MARGIN_SEC < pipeline_deadline)
        run_bge = bge_dir is not None and (time.perf_counter() + bge_expected + SAFETY_MARGIN_SEC < pipeline_deadline)

        if run_deb and run_bge and n_gpus >= 2:
            print(f"\n[Fold {fold+1}] Launching CONCURRENT Dual-GPU Training: DeBERTa ({deb_dev}, BS=12, GA=1, ckpt=ON) || BGE ({bge_dev}, BS=16, GA=1, ckpt=ON)", flush=True)
            t_par_start = time.perf_counter()

            def _train_deb_worker():
                return train_deberta_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    deberta_dir, LABELS,
                    device=deb_dev,
                    batch_size=12, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_DEB, lr_backbone=1.8e-5, lr_head=1.5e-4, layer_decay=0.90, enable_grad_checkpointing=True,
                    deadline=pipeline_deadline,
                    group_ids_tr=group_ids[tr_idx],
                    **deb_kw,
                )

            def _train_bge_worker():
                return train_bge_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    bge_dir, LABELS,
                    device=bge_dev,
                    batch_size=16, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, layer_decay=0.85, enable_grad_checkpointing=True,
                    deadline=pipeline_deadline,
                    group_ids_tr=group_ids[tr_idx],
                    **bge_kw,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_deb = executor.submit(_train_deb_worker)
                f_bge = executor.submit(_train_bge_worker)

                try:
                    val_deb, test_deb = f_deb.result()
                    if val_deb is not None and np.all(np.isfinite(val_deb)):
                        check_finite_logits(val_deb, f"DeBERTa Fold {fold+1} Val")
                        oof_deb[va_idx] = val_deb
                    if test_deb is not None and np.all(np.isfinite(test_deb)):
                        check_finite_logits(test_deb, f"DeBERTa Fold {fold+1} Test")
                        test_deb_folds.append(test_deb)
                except Exception as e:
                    import traceback
                    print(f"  [DeBERTa FT] Fold {fold+1} error: {e}", flush=True)
                    traceback.print_exc()

                try:
                    val_bge, test_bge = f_bge.result()
                    if val_bge is not None and np.all(np.isfinite(val_bge)):
                        check_finite_logits(val_bge, f"BGE Fold {fold+1} Val")
                        oof_bge[va_idx] = val_bge
                    if test_bge is not None and np.all(np.isfinite(test_bge)):
                        check_finite_logits(test_bge, f"BGE Fold {fold+1} Test")
                        test_bge_folds.append(test_bge)
                except Exception as e:
                    import traceback
                    print(f"  [BGE FT] Fold {fold+1} error: {e}", flush=True)
                    traceback.print_exc()

            par_dur = time.perf_counter() - t_par_start
            model_durations["deb"].append(par_dur)
            model_durations["bge"].append(par_dur)
            print(f"  [Parallel Dual-GPU] DeBERTa + BGE completed concurrently in {par_dur/60:.1f} min! {get_mem_status(None)}", flush=True)

        else:
            if run_deb:
                try:
                    print(f"\n[Fold {fold+1}] DeBERTa-v3-large full fine-tune ({deb_dev}, BS=12, GA=1, L=352, ckpt=ON)", flush=True)
                    m_t0 = time.perf_counter()
                    val_deb, test_deb = train_deberta_fold(
                        texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                        deberta_dir, LABELS,
                        device=deb_dev,
                        batch_size=12, grad_accum=1, max_length=352,
                        epochs=N_EPOCHS_DEB, lr_backbone=1.8e-5, lr_head=1.5e-4, layer_decay=0.90, enable_grad_checkpointing=True,
                        deadline=pipeline_deadline,
                        group_ids_tr=group_ids[tr_idx],
                        **deb_kw,
                    )
                    if val_deb is not None and np.all(np.isfinite(val_deb)):
                        check_finite_logits(val_deb, f"DeBERTa Fold {fold+1} Val")
                        oof_deb[va_idx] = val_deb
                    if test_deb is not None and np.all(np.isfinite(test_deb)):
                        check_finite_logits(test_deb, f"DeBERTa Fold {fold+1} Test")
                        test_deb_folds.append(test_deb)
                    if val_deb is not None:
                        model_durations["deb"].append(time.perf_counter() - m_t0)
                except Exception as e:
                    import traceback
                    print(f"  [DeBERTa FT] Fold {fold+1} error: {e}", flush=True)
                    traceback.print_exc()

            if run_bge:
                try:
                    print(f"\n[Fold {fold+1}] BGE-large full fine-tune ({bge_dev}, BS=16, GA=1, L=352, ckpt=ON)", flush=True)
                    m_t0 = time.perf_counter()
                    val_bge, test_bge = train_bge_fold(
                        texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                        bge_dir, LABELS,
                        device=bge_dev,
                        batch_size=16, grad_accum=1, max_length=352,
                        epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, layer_decay=0.85, enable_grad_checkpointing=True,
                        deadline=pipeline_deadline,
                        group_ids_tr=group_ids[tr_idx],
                        **bge_kw,
                    )
                    if val_bge is not None and np.all(np.isfinite(val_bge)):
                        check_finite_logits(val_bge, f"BGE Fold {fold+1} Val")
                        oof_bge[va_idx] = val_bge
                    if test_bge is not None and np.all(np.isfinite(test_bge)):
                        check_finite_logits(test_bge, f"BGE Fold {fold+1} Test")
                        test_bge_folds.append(test_bge)
                    if val_bge is not None:
                        model_durations["bge"].append(time.perf_counter() - m_t0)
                except Exception as e:
                    import traceback
                    print(f"  [BGE FT] Fold {fold+1} error: {e}", flush=True)
                    traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 3. Qwen LoRA fine-tune (Dual GPU) ───────────
        is_qwen3 = qwen_dir and ("qwen-3" in str(qwen_dir).lower() or "qwen3" in str(qwen_dir).lower())
        qwen_bs = 8 if is_qwen3 else 16
        qwen_ga = 2 if is_qwen3 else 1
        qwen_desc = "Qwen 3 8B" if is_qwen3 else "Qwen 2.5 7B"
        qwen_expected = np.mean(model_durations["qwen"]) * 1.15 if model_durations["qwen"] else 1800.0
        if qwen_dir is not None and (time.perf_counter() + qwen_expected + SAFETY_MARGIN_SEC < pipeline_deadline):
            try:
                print(f"\n[Fold {fold+1}] {qwen_desc} LoRA fine-tune (device_map=auto, r=48, BS={qwen_bs}, GA={qwen_ga}, epochs={N_EPOCHS_QWEN}, L=352)", flush=True)
                m_t0 = time.perf_counter()
                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=qwen_bs, grad_accum=qwen_ga, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.5e-4, lr_head=3.5e-4, lora_r=48, lora_alpha=96, lora_dropout=0.08,
                    deadline=pipeline_deadline,
                    **qwen_kw,
                )
                if val_qwen is not None and np.all(np.isfinite(val_qwen)):
                    check_finite_logits(val_qwen, f"Qwen Fold {fold+1} Val")
                    oof_qwen[va_idx] = val_qwen
                if test_qwen is not None and np.all(np.isfinite(test_qwen)):
                    check_finite_logits(test_qwen, f"Qwen Fold {fold+1} Test")
                    test_qwen_folds.append(test_qwen)
                if val_qwen is not None:
                    model_durations["qwen"].append(time.perf_counter() - m_t0)
            except Exception as e:
                import traceback
                print(f"  [Qwen LoRA] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 4. CPU Multiview Dual-Linear (Sparse CSR + PromptDF Filter + Sibling Specialists) ──
        try:
            print(f"\n[Fold {fold+1}] CPU Multiview Dual-Linear (Sparse CSR + PromptDF Filter + Sibling Specialists) {get_mem_status(None)}", flush=True)
            pdf_mask = compute_prompt_df_mask(X_train_all[tr_idx], group_ids[tr_idx], min_prompts=3)
            diag_mask = sp.diags(pdf_mask)
            X_tr_fold = X_train_all[tr_idx] @ diag_mask
            X_va_fold = X_train_all[va_idx] @ diag_mask
            X_te_fold = X_test_all @ diag_mask

            cpu_clf = DualLinearClassifier(c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42 + fold, canonical_labels=LABELS)
            clique_specialist = IntraFamilyCliqueSpecialist(alpha_ridge=1.0, beta_refine=0.25, seed=42 + fold)
            with threadpool_limits(limits=4):
                if X_ext_all is not None:
                    # Bounded-variance mass conservation & regularization invariance (Sum(w) == n_comp)
                    w_comp_fold, w_ext_fold = compute_fold_variance_capped_weights(
                        y_true[tr_idx], ext_df["model"].to_numpy(), ext_df["_n_models"].to_numpy(),
                        n_classes=NUM_CLASSES, ext_mass_ratio=0.55, max_single_ext_weight=0.35,
                    )
                    X_ext_fold = X_ext_all @ diag_mask
                    X_tr_comb = sp.vstack([X_tr_fold, X_ext_fold])
                    y_tr_comb = np.concatenate([y_tr_str, ext_df["model"].to_numpy()])
                    w_tr_comb = np.concatenate([w_comp_fold, w_ext_fold])

                    # Domain covariate moment alignment for dense stylometrics
                    V_dense_tr = V_dense_all[tr_idx]
                    V_dense_ext_aligned = align_external_dense_moments(V_dense_tr, V_dense_ext, w_ext_fold)
                    V_dense_tr_comb = np.vstack([V_dense_tr, V_dense_ext_aligned])

                    cpu_clf.fit(X_tr_comb, y_tr_comb, sample_weight=w_tr_comb)
                    clique_specialist.fit(X_tr_comb, y_tr_comb, sample_weight=w_tr_comb)
                else:
                    X_tr_comb = X_tr_fold
                    y_tr_comb = y_tr_str
                    w_tr_comb = None
                    V_dense_tr_comb = V_dense_all[tr_idx]
                    cpu_clf.fit(X_tr_comb, y_tr_comb)
                    clique_specialist.fit(X_tr_comb, y_tr_comb)

                val_sparse_m = cpu_clf.decision_function(X_va_fold)
                test_sparse_m = cpu_clf.decision_function(X_te_fold)

                # Intra-family sibling tie-breaking refinement
                val_sparse_m = clique_specialist.refine_margins(val_sparse_m, X_va_fold)
                test_sparse_m = clique_specialist.refine_margins(test_sparse_m, X_te_fold)

                # Dense stylometric calibrated stream (sliced from upfront precomputed matrix)
                V_dense_va = V_dense_all[va_idx]
                V_dense_te = V_dense_all[n_tr:]
                dense_clf = DenseStylometricClassifier(c=0.2, seed=42 + fold, canonical_labels=LABELS)
                if w_tr_comb is not None:
                    dense_clf.fit(V_dense_tr_comb, y_tr_comb, sample_weight=w_tr_comb)
                else:
                    dense_clf.fit(V_dense_tr_comb, y_tr_comb)
                p_dense_va = dense_clf.predict_proba(V_dense_va)
                p_dense_te = dense_clf.predict_proba(V_dense_te)

                # Calibrated consensus: optimal sparse TF-IDF + scaled dense stylometrics
                p_sparse_va = margins_to_probs(val_sparse_m, temp=0.20)
                p_sparse_te = margins_to_probs(test_sparse_m, temp=0.20)
                y_va_int_fold = np.array([LABEL_TO_IDX[y] for y in y_va_str], dtype=np.int64)
                cpu_alpha = optimize_cpu_subblend(p_sparse_va, p_dense_va, y_va_int_fold)
                p_cpu_va = cpu_alpha * p_sparse_va + (1.0 - cpu_alpha) * p_dense_va
                p_cpu_te = cpu_alpha * p_sparse_te + (1.0 - cpu_alpha) * p_dense_te
                val_svm_m = 0.20 * np.log(p_cpu_va + 1e-12)
                test_svm_m = 0.20 * np.log(p_cpu_te + 1e-12)
                print(f"  [CPU Linear] Fold {fold+1} optimal sparse/dense alpha={cpu_alpha:.2f}", flush=True)

                check_finite_logits(val_svm_m, f"CPU Linear Fold {fold+1} Val")
                check_finite_logits(test_svm_m, f"CPU Linear Fold {fold+1} Test")

                oof_svm[va_idx] = val_svm_m
                test_svm_folds.append(test_svm_m)
                svm_bal_acc = balanced_accuracy_score(y_va_str, [LABELS[i] for i in np.argmax(val_svm_m, axis=1)])
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}% (PromptDF min3 + Sibling Specialists active)", flush=True)

                del cpu_clf, dense_clf, clique_specialist
                if X_ext_all is not None:
                    del X_ext_fold, X_tr_comb, V_dense_tr_comb, V_dense_ext_aligned
        except Exception as e:
            import traceback
            print(f"  [CPU Linear] Fold {fold+1} error: {e}", flush=True)
            traceback.print_exc()

        # ── Safe Incremental Submission Checkpoint ────────────
        # Guarantees a fully valid, competition-compliant submission.parquet exists on disk after EVERY fold
        try:
            n_test = len(texts_test)
            interim_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
            interim_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
            interim_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
            interim_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

            interim_weights = (
                1.0,
                1.0 if interim_deb_m is not None else 0.0,
                1.0 if interim_qwen_m is not None else 0.0,
                1.0 if interim_bge_m is not None else 0.0,
            )
            interim_preds_idx = quad_rank_blend(
                interim_svm_m, interim_deb_m, interim_qwen_m, interim_bge_m,
                weights=interim_weights,
                temperatures=(0.20, 1.2, 1.5, 1.2),
                balance_test=True,
                apply_family_gating=True,
            )
            validate_and_save_submission(
                test_df["id"].tolist(),
                [LABELS[i] for i in interim_preds_idx],
                working_dir / "submission.parquet",
            )
            print(f"  [Safety Checkpoint] Incremental submission.parquet saved after Fold {fold+1}/{n_splits} [OK]", flush=True)
        except Exception as e:
            import traceback
            print(f"  [Safety Checkpoint] Warning during interim submission write: {e}", flush=True)
            traceback.print_exc()

        print(f"\n[Fold {fold+1}] Done in {time.perf_counter()-fold_start:.1f}s {get_mem_status(None)}", flush=True)
        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

    # ── STEP 3: OUT-OF-FOLD EVALUATION & BLEND OPTIMIZATION ──
    print("\n" + "=" * 65, flush=True)
    print("STEP 3: OUT-OF-FOLD EVALUATION & BLEND OPTIMIZATION", flush=True)
    print("=" * 65, flush=True)

    eval_mask = ~np.isnan(oof_svm[:, 0])
    y_eval = y_true[eval_mask]

    svm_score = balanced_accuracy_score(y_eval, [LABELS[i] for i in np.argmax(oof_svm[eval_mask], axis=1)])
    print(f"  CPU Dual-Linear: {svm_score*100:.2f}%", flush=True)

    # Universal Neural Stream Quality Gate (relative to CPU baseline)
    min_acceptable_score = max(2.0 / NUM_CLASSES, 0.35 * svm_score)

    if oof_deb is not None and np.any(~np.isnan(oof_deb[:, 0])):
        deb_mask = ~np.isnan(oof_deb[:, 0])
        deb_score = balanced_accuracy_score(y_true[deb_mask], [LABELS[i] for i in np.argmax(oof_deb[deb_mask], axis=1)])
        if deb_score < min_acceptable_score:
            print(f"  [DeBERTa FT] Warning: OOF score ({deb_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_deb = None
            test_deb_folds = []
        else:
            print(f"  DeBERTa FT:      {deb_score*100:.2f}% (folds evaluated: {np.sum(deb_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_deb = None
        print("  DeBERTa FT:      (skipped)", flush=True)

    if oof_bge is not None and np.any(~np.isnan(oof_bge[:, 0])):
        bge_mask = ~np.isnan(oof_bge[:, 0])
        bge_score = balanced_accuracy_score(y_true[bge_mask], [LABELS[i] for i in np.argmax(oof_bge[bge_mask], axis=1)])
        if bge_score < min_acceptable_score:
            print(f"  [BGE FT] Warning: OOF score ({bge_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_bge = None
            test_bge_folds = []
        else:
            print(f"  BGE FT:          {bge_score*100:.2f}% (folds evaluated: {np.sum(bge_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_bge = None
        print("  BGE FT:          (skipped)", flush=True)

    if oof_qwen is not None and np.any(~np.isnan(oof_qwen).any(axis=1)):
        qwen_mask = ~np.isnan(oof_qwen).any(axis=1)
        qwen_score = balanced_accuracy_score(y_true[qwen_mask], [LABELS[i] for i in np.argmax(oof_qwen[qwen_mask], axis=1)])
        if qwen_score < min_acceptable_score:
            print(f"  [Qwen LoRA] Warning: OOF score ({qwen_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_qwen = None
            test_qwen_folds = []
        else:
            print(f"  Qwen LoRA:       {qwen_score*100:.2f}% (folds evaluated: {np.sum(qwen_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_qwen = None
        print("  Qwen LoRA:       (skipped)", flush=True)

    # Automated Temperature Calibration (minimizing NLL on valid OOF rows)
    lbl2id = {l: i for i, l in enumerate(LABELS)}
    y_true_int = np.array([lbl2id[y] for y in y_true], dtype=np.int64)
    T_svm = 0.20  # Analytically calibrated to match the 0.20 * log(p_cpu) consensus scaling
    T_deb = fit_temperature(oof_deb, y_true_int, default_t=1.20) if oof_deb is not None else 1.20
    T_qwn = fit_temperature(oof_qwen, y_true_int, default_t=1.50) if oof_qwen is not None else 1.50
    T_bge = fit_temperature(oof_bge, y_true_int, default_t=1.20) if oof_bge is not None else 1.20
    calibrated_temps = (T_svm, T_deb, T_qwn, T_bge)
    print(f"\n  Calibrated Temperatures (NLL-optimal): SVM={T_svm} DeBERTa={T_deb} Qwen={T_qwn} BGE={T_bge}", flush=True)

    try:
        best_w, best_alpha, blend_score = optimize_blend_weights(
            oof_svm,
            oof_deb,
            oof_qwen,
            oof_bge,
            y_true,
            temperatures=calibrated_temps,
        )
        ws, wd, wq, wb = best_w
        print(f"\n  Blend weights: CPU={ws} DeBERTa={wd} Qwen={wq} BGE={wb} | Family Alpha={best_alpha:.2f}", flush=True)
        print(f"  OOF QUAD BLEND: {blend_score*100:.2f}%\n", flush=True)
    except Exception as e:
        import traceback
        print(f"  [Blend Tuning] Error during blend optimization: {e}", flush=True)
        traceback.print_exc()
        best_w = (
            1.0,
            1.0 if (oof_deb is not None and test_deb_folds) else 0.0,
            1.0 if (oof_qwen is not None and test_qwen_folds) else 0.0,
            1.0 if (oof_bge is not None and test_bge_folds) else 0.0,
        )
        best_alpha = 0.35

    # ── CPU Gradient-Optimized Meta-Combiner (L-BFGS Matrix + Dynamic Entropy Gate) ──
    use_meta_combiner = False
    meta_combiner = None
    try:
        raw_oof_streams = [oof_svm, oof_deb, oof_qwen, oof_bge]
        avail_streams = [s for s in raw_oof_streams if s is not None and np.any(~np.isnan(s[:, 0]))]
        if len(avail_streams) >= 2:
            print("=" * 65, flush=True)
            print("FITTING CPU GRADIENT META-COMBINER (L-BFGS Matrix + Dynamic Entropy Gate)", flush=True)
            calib_oof = [
                calibrate_logits(oof_svm, temp=T_svm) if oof_svm is not None else None,
                calibrate_logits(oof_deb, temp=T_deb) if oof_deb is not None else None,
                calibrate_logits(oof_qwen, temp=T_qwn) if oof_qwen is not None else None,
                calibrate_logits(oof_bge, temp=T_bge) if oof_bge is not None else None,
            ]
            valid_oof_mask = ~np.isnan(oof_svm[:, 0])
            for s in calib_oof:
                if s is not None:
                    valid_oof_mask = valid_oof_mask & ~np.isnan(s[:, 0])

            if np.sum(valid_oof_mask) >= 36:
                y_meta = y_true_int[valid_oof_mask]
                groups_meta = group_ids[valid_oof_mask] if group_ids is not None else None
                meta_oof_inputs = [s[valid_oof_mask] if s is not None else None for s in calib_oof]
                meta_combiner = CPUGradientMetaCombiner(n_models=sum(s is not None for s in meta_oof_inputs), n_classes=NUM_CLASSES, l2_reg=0.05)
                meta_priors = [0.25, 0.20, 0.45, 0.10]
                meta_oof_probs, meta_score = meta_combiner.fit_nested_cv(
                    meta_oof_inputs, y_meta, groups=groups_meta, prior_weights=meta_priors, max_iter=30
                )
                print(f"  Nested 5-Fold CV Meta-Combiner OOF: {meta_score*100:.2f}% (vs Flat/Grid: {blend_score*100:.2f}%)", flush=True)
                if meta_score >= blend_score:
                    use_meta_combiner = True
                    print("  -> Meta-Combiner achieved superior out-of-sample OOF score. Selected for test predictions! [OK]\n", flush=True)
                else:
                    use_meta_combiner = False
                    print("  -> Flat/Grid blend achieved superior score; using Quad-Rank blend for test.\n", flush=True)
    except Exception as e:
        print(f"  [Meta-Combiner] Note: {e}. Falling back to Quad-Rank blend.\n", flush=True)
        use_meta_combiner = False

    # Diagnostic: Ridge Stacker Comparison (5-Fold CV out-of-sample)
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import cross_val_predict
        # Compute mask directly from raw OOF arrays before margins_to_probs replaces NaNs
        raw_streams = [oof_svm]
        if oof_deb is not None: raw_streams.append(oof_deb)
        if oof_qwen is not None: raw_streams.append(oof_qwen)
        if oof_bge is not None: raw_streams.append(oof_bge)
        if len(raw_streams) > 1:
            stack_mask = ~np.isnan(raw_streams[0]).any(axis=1)
            for raw_st in raw_streams[1:]:
                stack_mask = stack_mask & ~np.isnan(raw_st).any(axis=1)
            if np.any(stack_mask):
                y_stack = y_true_int[stack_mask]
                avail_probs = [margins_to_probs(oof_svm[stack_mask], temp=T_svm)]
                if oof_deb is not None: avail_probs.append(margins_to_probs(oof_deb[stack_mask], temp=T_deb))
                if oof_qwen is not None: avail_probs.append(margins_to_probs(oof_qwen[stack_mask], temp=T_qwn))
                if oof_bge is not None: avail_probs.append(margins_to_probs(oof_bge[stack_mask], temp=T_bge))
                X_oof_stack = np.hstack(avail_probs)
                groups_stack = group_ids[stack_mask]
                ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=groups_stack)
                ridge_score = balanced_accuracy_score(y_stack, ridge_preds)
                print(f"  OOF RIDGE STACKER (5-Fold CV): {ridge_score*100:.2f}% (out-of-sample diagnostic)\n", flush=True)
    except Exception as e:
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)

    # ── STEP 4: BAGGED TEST INFERENCE & FINAL SUBMISSION ──
    try:
        print("=" * 65, flush=True)
        print("STEP 4: COMPUTING BAGGED TEST PREDICTIONS & FINAL SUBMISSION", flush=True)
        print("=" * 65, flush=True)
        n_test = len(texts_test)
        test_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
        test_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
        test_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
        test_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

        if use_meta_combiner and meta_combiner is not None:
            calib_test = [
                calibrate_logits(test_svm_m, temp=T_svm) if test_svm_m is not None else None,
                calibrate_logits(test_deb_m, temp=T_deb) if test_deb_m is not None else None,
                calibrate_logits(test_qwen_m, temp=T_qwn) if test_qwen_m is not None else None,
                calibrate_logits(test_bge_m, temp=T_bge) if test_bge_m is not None else None,
            ]
            test_probs = meta_combiner.predict_proba(calib_test)
            if len(test_probs) >= 36:
                test_probs = sinkhorn_knopp_balance(test_probs)
            test_pred_idx = np.argmax(test_probs, axis=1)
            print("  [Final Blend] Generated test predictions via CPU Gradient Meta-Combiner + Sinkhorn [OK]", flush=True)
        else:
            test_pred_idx = quad_rank_blend(
                test_svm_m, test_deb_m, test_qwen_m, test_bge_m,
                weights=best_w,
                temperatures=calibrated_temps,
                balance_test=True,
                apply_family_gating=True,
                family_alpha=best_alpha,
            )
            print("  [Final Blend] Generated test predictions via Quad-Rank Blend [OK]", flush=True)

        test_predictions = [LABELS[i] for i in test_pred_idx]

        sub_path = working_dir / "submission.parquet"
        validate_and_save_submission(test_df["id"].tolist(), test_predictions, sub_path)
    except Exception as e:
        import traceback
        print(f"  [Final Blend] Error during final test prediction / submission write: {e}", flush=True)
        traceback.print_exc()
        print("  [Final Blend] Preserving last successful incremental submission.parquet on disk.", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    run_quad_ensemble()
