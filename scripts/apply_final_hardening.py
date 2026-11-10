import re
from pathlib import Path

def apply_hardening():
    path = Path("notebooks/kaggle_tri_ensemble.py")
    text = path.read_text(encoding="utf-8")

    # 1. DualLinearClassifier: L1 and L2 diversity
    old_svc = '''        self.svc_1_ = LinearSVC(C=self.c_svc * 0.55, class_weight=cw, dual="auto", tol=1e-3, max_iter=800, random_state=self.seed)
        self.svc_2_ = LinearSVC(C=self.c_svc * 1.15, class_weight=cw, dual="auto", tol=1e-3, max_iter=800, random_state=self.seed + 7)'''
    new_svc = '''        self.svc_1_ = LinearSVC(penalty="l2", loss="squared_hinge", C=self.c_svc * 0.75, class_weight=cw, dual="auto", tol=1e-3, max_iter=800, random_state=self.seed)
        self.svc_2_ = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=self.c_svc * 1.0, class_weight=cw, tol=1e-2, max_iter=500, random_state=self.seed + 7)'''
    assert old_svc in text, "Could not find DualLinearClassifier svc definitions"
    text = text.replace(old_svc, new_svc, 1)

    # 2. StylometricMultiMomentPooler: Multi-Layer fusion support
    old_pooler_init = '''class StylometricMultiMomentPooler(nn.Module):
    """
    Multi-Moment (CLS + Mean + Max-Spike + Std-Dispersion) Pooler for Transformer Encoders (BGE / DeBERTa).
    Normalizes each statistical moment independently before projection so extreme h_max spikes
    never drown out h_mean or syntactic dispersion h_std.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_cls = nn.LayerNorm(hidden_size)
        self.norm_mean = nn.LayerNorm(hidden_size)
        self.norm_max = nn.LayerNorm(hidden_size)
        self.norm_std = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        self.norm_out = nn.LayerNorm(hidden_size)
        self._current_mask = None
        nn.init.xavier_uniform_(self.dense.weight, gain=0.5)
        nn.init.zeros_(self.dense.bias)'''

    new_pooler_init = '''class StylometricMultiMomentPooler(nn.Module):
    """
    Multi-Moment (CLS + Mean + Max-Spike + Std-Dispersion) Pooler for Transformer Encoders (BGE / DeBERTa).
    Normalizes each statistical moment independently before projection so extreme h_max spikes
    never drown out h_mean or syntactic dispersion h_std.
    Supports Multi-Layer Hidden State Fusion across discourse (-1), syntax (-3), and surface structure (-6).
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_mix_weights = nn.Parameter(torch.zeros(3))
        self.norm_cls = nn.LayerNorm(hidden_size)
        self.norm_mean = nn.LayerNorm(hidden_size)
        self.norm_max = nn.LayerNorm(hidden_size)
        self.norm_std = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        self.norm_out = nn.LayerNorm(hidden_size)
        self._current_mask = None
        self._extra_layers = None
        nn.init.xavier_uniform_(self.dense.weight, gain=0.5)
        nn.init.zeros_(self.dense.bias)'''
    assert old_pooler_init in text, "Could not find StylometricMultiMomentPooler init"
    text = text.replace(old_pooler_init, new_pooler_init, 1)

    old_pooler_fwd = '''    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = attention_mask if attention_mask is not None else self._current_mask
        h_cls = self.norm_cls(hidden_states[:, 0])'''
    new_pooler_fwd = '''    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = attention_mask if attention_mask is not None else self._current_mask
        if self._extra_layers is not None and len(self._extra_layers) == 3:
            w = F.softmax(self.layer_mix_weights, dim=0)
            h_fused = None
            for idx, h_l in enumerate(self._extra_layers):
                h_l_dev = h_l.to(device=hidden_states.device, dtype=hidden_states.dtype)
                h_fused = (w[idx] * h_l_dev) if h_fused is None else (h_fused + w[idx] * h_l_dev)
            hidden_states = h_fused
            self._extra_layers = None
        h_cls = self.norm_cls(hidden_states[:, 0])'''
    assert old_pooler_fwd in text, "Could not find StylometricMultiMomentPooler forward"
    text = text.replace(old_pooler_fwd, new_pooler_fwd, 1)

    # 3. Add get_model_pooler helper
    old_set_mask = '''def set_model_pooler_mask(model, mask):
    """Passes current batch attention mask to StylometricMultiMomentPooler if present."""
    candidates = [
        getattr(model, "pooler", None),
        getattr(getattr(model, "bert", None), "pooler", None),
        getattr(getattr(model, "deberta", None), "pooler", None),
        getattr(getattr(model, "model", None), "pooler", None),
    ]
    for pooler in candidates:
        if pooler is not None and hasattr(pooler, "set_mask"):
            pooler.set_mask(mask)'''

    new_set_mask = '''def get_model_pooler(model):
    """Retrieves StylometricMultiMomentPooler from DeBERTa, BERT/BGE, or generic model."""
    candidates = [
        getattr(model, "pooler", None),
        getattr(getattr(model, "bert", None), "pooler", None),
        getattr(getattr(model, "deberta", None), "pooler", None),
        getattr(getattr(model, "model", None), "pooler", None),
    ]
    for p in candidates:
        if p is not None and hasattr(p, "layer_mix_weights"):
            return p
    for p in candidates:
        if p is not None and hasattr(p, "dense"):
            return p
    return None


def set_model_pooler_mask(model, mask):
    """Passes current batch attention mask to StylometricMultiMomentPooler if present."""
    pooler = get_model_pooler(model)
    if pooler is not None and hasattr(pooler, "set_mask"):
        pooler.set_mask(mask)'''
    assert old_set_mask in text, "Could not find set_model_pooler_mask"
    text = text.replace(old_set_mask, new_set_mask, 1)

    # 4. FocalLoss label_smoothing reduction to 0.01
    old_focal_init = '''def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05, reduction="mean", use_sibling_margins: bool = True, sibling_margins=None):'''
    new_focal_init = '''def __init__(self, weight=None, gamma=1.0, label_smoothing=0.01, reduction="mean", use_sibling_margins: bool = True, sibling_margins=None):'''
    assert old_focal_init in text, "Could not find FocalLoss init"
    text = text.replace(old_focal_init, new_focal_init, 1)

    # 5. capture_intermediate_layers: add target_pooler parameter & encoder layer discovery
    old_cap_fn = '''@contextlib.contextmanager
def capture_intermediate_layers(model, layer_indices=(-1, -5, -9)):
    """
    Lightweight forward-hook capture of intermediate decoder layers (L1, L5, L9).
    Captures ONLY the requested layers without materializing all 37 hidden state tuples,
    saving ~3.5 GB of peak VRAM during forward passes.
    """
    base_m = getattr(model, "base_model", model)
    inner_m = getattr(base_m, "model", base_m)
    dec_layers = getattr(inner_m, "layers", getattr(getattr(inner_m, "model", None), "layers", None))
    captured = {}
    handles = []
    if dec_layers is not None:
        n_dec = len(dec_layers)
        for idx in layer_indices:
            actual_idx = idx if idx >= 0 else n_dec + idx
            if 0 <= actual_idx < n_dec:
                def _make_hook(i):
                    def _hook(m, inp, out):
                        captured[i] = out[0] if isinstance(out, tuple) else out
                    return _hook
                handles.append(dec_layers[actual_idx].register_forward_hook(_make_hook(idx)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()'''

    new_cap_fn = '''@contextlib.contextmanager
def capture_intermediate_layers(model, layer_indices=(-1, -5, -9), target_pooler=None):
    """
    Lightweight forward-hook capture of intermediate decoder/encoder layers.
    If target_pooler is provided, injects the captured layers directly into the pooler
    before it executes.
    """
    base_m = getattr(model, "base_model", model)
    inner_m = getattr(base_m, "encoder", getattr(base_m, "deberta", getattr(base_m, "bert", getattr(base_m, "model", base_m))))
    dec_layers = getattr(inner_m, "layer", getattr(inner_m, "layers", getattr(getattr(inner_m, "encoder", None), "layer", None)))
    if dec_layers is None and hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        dec_layers = model.encoder.layer
    captured = {}
    handles = []
    if dec_layers is not None:
        n_dec = len(dec_layers)
        for idx in layer_indices:
            actual_idx = idx if idx >= 0 else n_dec + idx
            if 0 <= actual_idx < n_dec:
                def _make_hook(i):
                    def _hook(m, inp, out):
                        captured[i] = out[0] if isinstance(out, tuple) else out
                        if target_pooler is not None and len(captured) == len(layer_indices):
                            target_pooler._extra_layers = [captured[x] for x in layer_indices]
                    return _hook
                handles.append(dec_layers[actual_idx].register_forward_hook(_make_hook(idx)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
        if target_pooler is not None:
            target_pooler._extra_layers = None'''
    assert old_cap_fn in text, "Could not find capture_intermediate_layers"
    text = text.replace(old_cap_fn, new_cap_fn, 1)

    # 6. evaluate_transformer_split: eval_bs=32, multi-layer fusion & TTA
    old_eval_split = '''def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
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
    set_model_pooler_mask(model, None)
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    new_eval_split = '''def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes, enc_data_tta=None):
    """
    Length-sorted batched inference for encoder models with eval_bs=32 and Multi-Layer Fusion.
    Groups evaluation passages by length to eliminate redundant padding tokens,
    then inverts the sort order back to the original index sequence.
    Optionally evaluates enc_data_tta and averages test predictions (Test-Time Augmentation).
    """
    def _eval_single(enc):
        n_samples = len(enc["input_ids"]) if "input_ids" in enc else 0
        if n_samples == 0:
            return np.zeros((0, num_classes), dtype=np.float64)
        lengths = [len(x) for x in enc["input_ids"]]
        sort_idx = np.argsort(lengths)
        all_logits = []
        eval_bs = 32  # Memory-safe for Tesla T4 under L=448
        cuda_ok = torch.cuda.is_available() and getattr(dev, "type", str(dev)) != "cpu"
        pooler = get_model_pooler(model)
        with inference_context():
            for start in range(0, n_samples, eval_bs):
                end = min(start + eval_bs, n_samples)
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc["attention_mask"][j] for j in batch_indices],
                }
                if "token_type_ids" in enc:
                    sl["token_type_ids"] = [enc["token_type_ids"][j] for j in batch_indices]
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                kw = {
                    "input_ids": padded["input_ids"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(dev),
                    "attention_mask": padded["attention_mask"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(dev),
                }
                if "token_type_ids" in padded:
                    kw["token_type_ids"] = padded["token_type_ids"].pin_memory().to(dev, non_blocking=True) if cuda_ok else padded["token_type_ids"].to(dev)
                with capture_intermediate_layers(model, (-1, -3, -6), target_pooler=pooler):
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        set_model_pooler_mask(model, kw["attention_mask"])
                        logits = model(**kw).logits
                all_logits.append(logits.float().cpu().numpy())
        set_model_pooler_mask(model, None)
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        return sorted_logits[inv_idx]

    model.eval()
    base_logits = _eval_single(enc_data)
    if enc_data_tta is not None:
        tta_logits = _eval_single(enc_data_tta)
        return 0.5 * base_logits + 0.5 * tta_logits
    return base_logits'''
    assert old_eval_split in text, "Could not find evaluate_transformer_split"
    text = text.replace(old_eval_split, new_eval_split, 1)

    # 7. evaluate_qwen_split: TTA support
    old_eval_qwen = '''def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes):
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
    inf_bs = 16
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
                with capture_intermediate_layers(backbone, (-1, -5, -9)) as captured_hs:
                    out = backbone(**batch_kwargs, output_hidden_states=False)
                    if -1 in captured_hs and -5 in captured_hs and -9 in captured_hs:
                        h_input = (captured_hs[-1], captured_hs[-5], captured_hs[-9])
                    else:
                        h_input = getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
                    del out
                    if isinstance(head, SpikeAndGapHierarchicalHead):
                        logits = head(h_input, batch_kwargs["attention_mask"]).float().cpu().numpy()
                    else:
                        last_h = h_input[0] if isinstance(h_input, (tuple, list)) else h_input
                        if getattr(tokenizer, "padding_side", "left") == "left":
                            pooled = last_h[:, -1, :].float()
                        else:
                            attn_mask = batch_kwargs["attention_mask"].to(last_h.device)
                            seq_lens = torch.clamp(attn_mask.sum(dim=1) - 1, min=0)
                            batch_idx = torch.arange(last_h.size(0), device=last_h.device)
                            pooled = last_h[batch_idx, seq_lens].float()
                        logits = head(pooled.to(head_dev)).float().cpu().numpy()
                    del h_input
            all_logits.append(logits)
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    new_eval_qwen = '''def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes, enc_data_tta=None):
    """
    Length-sorted batched inference for Qwen 3 8B / 2.5 7B LoRA decoder.
    Passes representations through backbone to SpikeAndGapHierarchicalHead or classification head.
    Optionally evaluates enc_data_tta and averages test predictions (Test-Time Augmentation).
    """
    def _eval_single(enc):
        n_samples = len(enc["input_ids"]) if "input_ids" in enc else 0
        if n_samples == 0:
            return np.zeros((0, num_classes), dtype=np.float64)
        lengths = [len(x) for x in enc["input_ids"]]
        sort_idx = np.argsort(lengths)
        all_logits = []
        inf_bs = 16
        autocast_ctx = amp_autocast() if cuda_ok else contextlib.nullcontext()
        with inference_context():
            for start in range(0, n_samples, inf_bs):
                end = min(start + inf_bs, n_samples)
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc["attention_mask"][j] for j in batch_indices],
                }
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                batch_kwargs = {
                    "input_ids": padded["input_ids"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(first_dev),
                    "attention_mask": padded["attention_mask"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(first_dev),
                }
                with autocast_ctx:
                    with capture_intermediate_layers(backbone, (-1, -5, -9)) as captured_hs:
                        out = backbone(**batch_kwargs, output_hidden_states=False)
                        if -1 in captured_hs and -5 in captured_hs and -9 in captured_hs:
                            h_input = (captured_hs[-1], captured_hs[-5], captured_hs[-9])
                        else:
                            h_input = getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
                        del out
                        if isinstance(head, SpikeAndGapHierarchicalHead):
                            logits = head(h_input, batch_kwargs["attention_mask"]).float().cpu().numpy()
                        else:
                            last_h = h_input[0] if isinstance(h_input, (tuple, list)) else h_input
                            if getattr(tokenizer, "padding_side", "left") == "left":
                                pooled = last_h[:, -1, :].float()
                            else:
                                attn_mask = batch_kwargs["attention_mask"].to(last_h.device)
                                seq_lens = torch.clamp(attn_mask.sum(dim=1) - 1, min=0)
                                batch_idx = torch.arange(last_h.size(0), device=last_h.device)
                                pooled = last_h[batch_idx, seq_lens].float()
                            logits = head(pooled.to(head_dev)).float().cpu().numpy()
                        del h_input
                all_logits.append(logits)
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        return sorted_logits[inv_idx]

    backbone.eval()
    head.eval()
    base_logits = _eval_single(enc_data)
    if enc_data_tta is not None:
        tta_logits = _eval_single(enc_data_tta)
        return 0.5 * base_logits + 0.5 * tta_logits
    return base_logits'''
    assert old_eval_qwen in text, "Could not find evaluate_qwen_split"
    text = text.replace(old_eval_qwen, new_eval_qwen, 1)

    # 8. precompute_token_caches: precompute te_tta
    old_precomp = '''        enc_tr = tokenize_with_head_tail(tokenizer, raw_texts_tr, max_length=max_length, preserve_newlines=preserve_newlines)
        enc_te = tokenize_with_head_tail(tokenizer, raw_texts_te, max_length=max_length, preserve_newlines=preserve_newlines)
        enc_ext = tokenize_with_head_tail(tokenizer, raw_texts_ext, max_length=max_length, preserve_newlines=preserve_newlines) if raw_texts_ext else None
        dur = time.perf_counter() - t0
        n_ext_str = f" + {len(raw_texts_ext)} ext" if raw_texts_ext else ""
        print(f"  [{model_tag.upper()} Cache] Pre-tokenized {len(raw_texts_tr)} train + {len(raw_texts_te)} test{n_ext_str} texts in {dur:.2f}s (RAM resident).", flush=True)
        return {
            "tr": enc_tr,
            "te": enc_te,
            "ext": enc_ext,
            "tokenizer": tokenizer,
            "staged_dir": staged_dir,
        }'''

    new_precomp = '''        enc_tr = tokenize_with_head_tail(tokenizer, raw_texts_tr, max_length=max_length, preserve_newlines=preserve_newlines)
        enc_te = tokenize_with_head_tail(tokenizer, raw_texts_te, max_length=max_length, preserve_newlines=preserve_newlines)
        enc_te_tta = tokenize_with_head_tail(tokenizer, raw_texts_te, max_length=max_length, head_tokens=128, tail_tokens=320, preserve_newlines=preserve_newlines)
        enc_ext = tokenize_with_head_tail(tokenizer, raw_texts_ext, max_length=max_length, preserve_newlines=preserve_newlines) if raw_texts_ext else None
        dur = time.perf_counter() - t0
        n_ext_str = f" + {len(raw_texts_ext)} ext" if raw_texts_ext else ""
        print(f"  [{model_tag.upper()} Cache] Pre-tokenized {len(raw_texts_tr)} train + {len(raw_texts_te)} test (with TTA){n_ext_str} texts in {dur:.2f}s (RAM resident).", flush=True)
        return {
            "tr": enc_tr,
            "te": enc_te,
            "te_tta": enc_te_tta,
            "ext": enc_ext,
            "tokenizer": tokenizer,
            "staged_dir": staged_dir,
        }'''
    assert old_precomp in text, "Could not find precompute_token_caches inner return"
    text = text.replace(old_precomp, new_precomp, 1)

    # 9. train_transformer_fold: multi-layer capture in training + label_smoothing=0.01 + TTA passing
    old_train_trans_sig = '''def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=12,
    grad_accum=1,
    max_length=448,
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
    n_frozen_bottom_layers=4,
):'''

    new_train_trans_sig = '''def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=12,
    grad_accum=1,
    max_length=448,
    epochs=3,
    lr_backbone=1.5e-5,
    lr_head=1.2e-4,
    weight_decay=0.01,
    label_smoothing=0.01,
    layer_decay=0.95,
    enable_grad_checkpointing=True,
    deadline=None,
    cached_enc_tr=None,
    cached_enc_va=None,
    cached_enc_te=None,
    cached_enc_te_tta=None,
    cached_tokenizer=None,
    group_ids_tr=None,
    use_topic_masking=True,
    use_sift=True,
    n_frozen_bottom_layers=4,
):'''
    assert old_train_trans_sig in text, "Could not find train_transformer_fold signature"
    text = text.replace(old_train_trans_sig, new_train_trans_sig, 1)

    old_trans_fwd = '''                try:
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        set_model_pooler_mask(model, batch_kwargs["attention_mask"])
                        if topic_prep is not None and topic_prep.sift_eps > 0:
                            emb = model.get_input_embeddings()(batch_kwargs["input_ids"])
                            emb = topic_prep.apply_sift_embedding_noise(emb, batch_kwargs["attention_mask"])
                            emb = emb.requires_grad_(True)
                            kw_forward = {
                                "inputs_embeds": emb,
                                "attention_mask": batch_kwargs["attention_mask"],
                            }
                            if "token_type_ids" in batch_kwargs:
                                kw_forward["token_type_ids"] = batch_kwargs["token_type_ids"]
                            logits = model(**kw_forward).logits
                        else:
                            logits = model(**batch_kwargs).logits
                        logits_f = logits.float()'''

    new_trans_fwd = '''                try:
                    batch_groups = batch_kwargs.pop("_batch_groups", getattr(prefetcher, "current_groups", None))
                    pooler_inst = get_model_pooler(model)
                    with capture_intermediate_layers(model, (-1, -3, -6), target_pooler=pooler_inst):
                        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                            set_model_pooler_mask(model, batch_kwargs["attention_mask"])
                            if topic_prep is not None and topic_prep.sift_eps > 0:
                                emb = model.get_input_embeddings()(batch_kwargs["input_ids"])
                                emb = topic_prep.apply_sift_embedding_noise(emb, batch_kwargs["attention_mask"])
                                emb = emb.requires_grad_(True)
                                kw_forward = {
                                    "inputs_embeds": emb,
                                    "attention_mask": batch_kwargs["attention_mask"],
                                }
                                if "token_type_ids" in batch_kwargs:
                                    kw_forward["token_type_ids"] = batch_kwargs["token_type_ids"]
                                logits = model(**kw_forward).logits
                            else:
                                logits = model(**batch_kwargs).logits
                            logits_f = logits.float()'''
    assert old_trans_fwd in text, "Could not find transformer training forward pass"
    text = text.replace(old_trans_fwd, new_trans_fwd, 1)

    # 10. In evaluate_transformer_split call at end of fold: pass cached_enc_te_tta
    old_eval_trans_call = '''    test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))'''
    new_eval_trans_call = '''    test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels), enc_data_tta=cached_enc_te_tta)'''
    assert old_eval_trans_call in text, "Could not find evaluate_transformer_split call in train_transformer_fold"
    text = text.replace(old_eval_trans_call, new_eval_trans_call, 1)

    # 11. train_qwen_lora_fold: label_smoothing=0.01 + TTA passing
    old_qwen_sig = '''def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=16,
    grad_accum=1,
    max_length=448,
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
    group_ids_tr=None,
    retain_backbone: bool = False,
):'''

    new_qwen_sig = '''def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=16,
    grad_accum=1,
    max_length=448,
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
    cached_enc_te_tta=None,
    cached_tokenizer=None,
    group_ids_tr=None,
    retain_backbone: bool = False,
):'''
    assert old_qwen_sig in text, "Could not find train_qwen_lora_fold signature"
    text = text.replace(old_qwen_sig, new_qwen_sig, 1)

    old_qwen_criterion = '''    criterion = FocalLoss(weight=weight_tensor, gamma=1.0, label_smoothing=0.05)'''
    new_qwen_criterion = '''    criterion = FocalLoss(weight=weight_tensor, gamma=1.0, label_smoothing=0.01)'''
    assert old_qwen_criterion in text, "Could not find qwen FocalLoss call"
    text = text.replace(old_qwen_criterion, new_qwen_criterion, 1)

    old_qwen_eval_call = '''    test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))'''
    new_qwen_eval_call = '''    test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels), enc_data_tta=cached_enc_te_tta)'''
    assert old_qwen_eval_call in text, "Could not find evaluate_qwen_split call"
    text = text.replace(old_qwen_eval_call, new_qwen_eval_call, 1)

    old_qwen_fwd_start = '''                try:
                    autocast_ctx = amp_autocast() if cuda_ok else contextlib.nullcontext()'''
    new_qwen_fwd_start = '''                try:
                    batch_groups = batch_kwargs.pop("_batch_groups", getattr(prefetcher, "current_groups", None))
                    autocast_ctx = amp_autocast() if cuda_ok else contextlib.nullcontext()'''
    assert old_qwen_fwd_start in text, "Could not find qwen forward start"
    text = text.replace(old_qwen_fwd_start, new_qwen_fwd_start, 1)

    # 12. FastBatchPrefetcher: include _batch_groups in gpu_kw / cpu_kw
    old_prefetch_yield = '''            if self.cuda_ok:
                gpu_kw = {k: v.to(self.dev, non_blocking=True) for k, v in kw.items()}
                gpu_labels = labels.to(self.label_dev, non_blocking=True)
                self.current_groups = groups.to(self.label_dev, non_blocking=True) if groups is not None else None
                yield gpu_kw, gpu_labels
            else:
                cpu_kw = {k: v.to(self.dev) for k, v in kw.items()}
                cpu_labels = labels.to(self.label_dev)
                self.current_groups = groups.to(self.label_dev) if groups is not None else None
                yield cpu_kw, cpu_labels'''

    new_prefetch_yield = '''            if self.cuda_ok:
                gpu_kw = {k: v.to(self.dev, non_blocking=True) for k, v in kw.items()}
                gpu_labels = labels.to(self.label_dev, non_blocking=True)
                self.current_groups = groups.to(self.label_dev, non_blocking=True) if groups is not None else None
                gpu_kw["_batch_groups"] = self.current_groups
                yield gpu_kw, gpu_labels
            else:
                cpu_kw = {k: v.to(self.dev) for k, v in kw.items()}
                cpu_labels = labels.to(self.label_dev)
                self.current_groups = groups.to(self.label_dev) if groups is not None else None
                cpu_kw["_batch_groups"] = self.current_groups
                yield cpu_kw, cpu_labels'''
    assert old_prefetch_yield in text, "Could not find FastBatchPrefetcher yield"
    text = text.replace(old_prefetch_yield, new_prefetch_yield, 1)

    # 13. Pass cached_enc_te_tta in run_quad_ensemble fold loop
    old_kw_prep = '''        deb_cache = token_caches.get("deberta")
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
            bge_kw["cached_tokenizer"] = bge_cache["tokenizer"]'''

    new_kw_prep = '''        deb_cache = token_caches.get("deberta")
        deb_kw = {}
        if deb_cache is not None:
            deb_kw["cached_enc_tr"] = {k: [deb_cache["tr"][k][i] for i in tr_idx] for k in deb_cache["tr"]}
            deb_kw["cached_enc_va"] = {k: [deb_cache["tr"][k][i] for i in va_idx] for k in deb_cache["tr"]}
            deb_kw["cached_enc_te"] = deb_cache["te"]
            deb_kw["cached_enc_te_tta"] = deb_cache.get("te_tta")
            deb_kw["cached_tokenizer"] = deb_cache["tokenizer"]

        bge_cache = token_caches.get("bge")
        bge_kw = {}
        if bge_cache is not None:
            bge_kw["cached_enc_tr"] = {k: [bge_cache["tr"][k][i] for i in tr_idx] for k in bge_cache["tr"]}
            bge_kw["cached_enc_va"] = {k: [bge_cache["tr"][k][i] for i in va_idx] for k in bge_cache["tr"]}
            bge_kw["cached_enc_te"] = bge_cache["te"]
            bge_kw["cached_enc_te_tta"] = bge_cache.get("te_tta")
            bge_kw["cached_tokenizer"] = bge_cache["tokenizer"]'''
    assert old_kw_prep in text, "Could not find deb/bge kw prep in fold loop"
    text = text.replace(old_kw_prep, new_kw_prep, 1)

    old_qwen_kw_prep = '''        qwen_cache = token_caches.get("qwen")
        qwen_kw = {}
        if qwen_cache is not None:
            qwen_kw["cached_enc_tr"] = {k: [qwen_cache["tr"][k][i] for i in tr_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_va"] = {k: [qwen_cache["tr"][k][i] for i in va_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_te"] = qwen_cache["te"]
            qwen_kw["cached_tokenizer"] = qwen_cache["tokenizer"]'''

    new_qwen_kw_prep = '''        qwen_cache = token_caches.get("qwen")
        qwen_kw = {}
        if qwen_cache is not None:
            qwen_kw["cached_enc_tr"] = {k: [qwen_cache["tr"][k][i] for i in tr_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_va"] = {k: [qwen_cache["tr"][k][i] for i in va_idx] for k in qwen_cache["tr"]}
            qwen_kw["cached_enc_te"] = qwen_cache["te"]
            qwen_kw["cached_enc_te_tta"] = qwen_cache.get("te_tta")
            qwen_kw["cached_tokenizer"] = qwen_cache["tokenizer"]'''
    assert old_qwen_kw_prep in text, "Could not find qwen kw prep in fold loop"
    text = text.replace(old_qwen_kw_prep, new_qwen_kw_prep, 1)

    path.write_text(text, encoding="utf-8")
    print("Successfully patched notebooks/kaggle_tri_ensemble.py with all hardening fixes!")

if __name__ == "__main__":
    apply_hardening()
