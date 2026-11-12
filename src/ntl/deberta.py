"""DeBERTa-v3 bidirectional contextual feature extraction for Name That LLM."""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ntl.preprocessing import clean_text


def discover_deberta_path(input_root: Path) -> Optional[Path]:
    """Search for DeBERTa weights directory under input_root."""
    for p in Path(input_root).rglob("config.json"):
        try:
            cfg = json.loads(p.read_text())
            mtype = cfg.get("model_type", "").lower()
            if "deberta" in mtype or "deberta" in str(p).lower():
                return p.parent
        except Exception:
            continue
    return None


def load_deberta_model(
    model_dir: Path,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[AutoModel, AutoTokenizer]:
    """Loads DeBERTa-v3 model and tokenizer in evaluation mode."""
    if device is None:
        # Default to GPU 1 if dual GPUs exist, else GPU 0, else CPU
        device = "cuda:1" if torch.cuda.device_count() >= 2 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def extract_deberta_embeddings(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[str] = None
) -> np.ndarray:
    """Extracts mean-pooled token contextual representations from DeBERTa-v3."""
    if device is None:
        device = next(model.parameters()).device

    extracted = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [clean_text(t) for t in texts[i:i + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            mask = inputs["attention_mask"].unsqueeze(-1)
            last_hidden = outputs.last_hidden_state
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            extracted.append(pooled.float().cpu().numpy())

    return np.concatenate(extracted, axis=0)

