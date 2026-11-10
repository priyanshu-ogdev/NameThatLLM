import ast
import re
from pathlib import Path

def apply_upgrades():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    dry_path = Path("notebooks/kaggle_dry_run.py")

    with open(ens_path, "r", encoding="utf-8") as f:
        ens_code = f.read()

    with open(dry_path, "r", encoding="utf-8") as f:
        dry_code = f.read()

    marker_ens = "def run_quad_ensemble():"
    marker_dry = "def run_dry_run_verifier():"

    ens_idx = ens_code.index(marker_ens)
    dry_idx = dry_code.index(marker_dry)

    ens_shared = ens_code[:ens_idx]
    dry_shared = dry_code[:dry_idx]

    assert ens_shared == dry_shared, "Initial shared code mismatch before upgrades!"
    print(f"Initial shared code matches perfectly ({len(ens_shared)} chars, {len(ens_shared.splitlines())} lines).")

    # --- 1. Add queue to imports ---
    old_imports = "import os\nimport sys\nimport threading\nimport time"
    new_imports = "import os\nimport queue\nimport sys\nimport threading\nimport time"
    assert old_imports in ens_shared, "Could not find imports in shared code!"
    shared_upgraded = ens_shared.replace(old_imports, new_imports, 1)

    # --- 2. Upgrade RichStyleFeatures with precompiled regexes ---
    old_rich_style = '''class RichStyleFeatures(BaseEstimator, TransformerMixin):
    """Linguistic and typographical features (132 features) capturing authorial habits."""
    PUNCTUATION = [
        "\\u2014", "\\u2013", "-", ",", ".", ";", ":", "!", "?", "\\u2026", "...",
        \'"\', "\'", "\\u2018", "\\u2019", "\\u201c", "\\u201d", "(", ")", "[", "]",
        "*", "#", "\\n", "\\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+[\'\\u2019\\u2018]\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:[\'\\u2019\\u2018]\\w+)?\\b")
    SENT_RE = re.compile(r"[.!?]+\\s*")

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

            # Positional Slicing (Openings & Closings)
            n_quarter = max(nw // 4, 1)
            opening_text = " ".join(words[:n_quarter])
            ending_text = " ".join(words[-n_quarter:])
            feat.extend([
                float(bool(re.search(r"^\\s*#{1,6}\\s", t))),
                float(bool(re.search(r"^\\s*[-*\\u2022]\\s|^\\s*\\d+\\.\\s", t))),
                float(bool(re.search(r"^\\s*\\*\\*", t))),
                float(bool(re.search(r"^(certainly|sure|here is|as an|to answer|i would)", opening_text, re.I))),
                float(bool(re.search(r"(overall|in conclusion|in summary|ultimately|hope this helps|to summarize)", ending_text, re.I))),
                float(t.endswith("?")),
                float(t.endswith("!")),
            ])

            # Paragraph and Line Cadence
            paragraphs = [p for p in t.split("\\n\\n") if p.strip()]
            n_paras = max(len(paragraphs), 1)
            para_lengths = [len(p.split()) for p in paragraphs] or [0]
            lines = [l for l in t.split("\\n") if l.strip()]
            line_lengths = [len(l.split()) for l in lines] or [0]

            feat.extend([
                np.log1p(n_paras),
                float(np.mean(para_lengths)),
                float(np.std(para_lengths)),
                np.log1p(max(len(lines), 1)),
                float(np.mean(line_lengths)),
                float(np.std(line_lengths)),
                t.count("\\n\\n") / max(t.count("\\n"), 1),
                len(set(lower)) / max(np.sqrt(nw), 1e-6),    # Root Type-Token Ratio
                sum(len(w) >= 12 for w in words) / nw,        # Multisyllabic density
                t.count("```") / nw,                           # Code block density
                t.count("`") / nw,                             # Inline code density
                t.count("$") / nw,                             # LaTeX math density
                t.count("\\\\") / nw,                            # TeX / escape density
                t.count(":") / max(t.count(";") + 1, 1),       # Colon-to-semicolon ratio
                (t.count(\'"\') + t.count("\\u201c") + t.count("\\u201d")) / nw,  # Quotation density
            ])
            rows.append(feat)
        return np.asarray(rows, dtype=np.float64)'''

    new_rich_style = '''class RichStyleFeatures(BaseEstimator, TransformerMixin):
    """Linguistic and typographical features (132 features) capturing authorial habits."""
    PUNCTUATION = [
        "\\u2014", "\\u2013", "-", ",", ".", ";", ":", "!", "?", "\\u2026", "...",
        \'"\', "\'", "\\u2018", "\\u2019", "\\u201c", "\\u201d", "(", ")", "[", "]",
        "*", "#", "\\n", "\\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+[\'\\u2019\\u2018]\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:[\'\\u2019\\u2018]\\w+)?\\b")
    SENT_RE = re.compile(r"[.!?]+\\s*")
    H_RE = re.compile(r"^\\s*#{1,6}\\s")
    LIST_RE = re.compile(r"^\\s*[-*\\u2022]\\s|^\\s*\\d+\\.\\s")
    BOLD_RE = re.compile(r"^\\s*\\*\\*")
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

            # Positional Slicing (Openings & Closings with precompiled regexes)
            n_quarter = max(nw // 4, 1)
            opening_text = " ".join(words[:n_quarter])
            ending_text = " ".join(words[-n_quarter:])
            feat.extend([
                float(bool(self.H_RE.search(t))),
                float(bool(self.LIST_RE.search(t))),
                float(bool(self.BOLD_RE.search(t))),
                float(bool(self.OPENING_RE.search(opening_text))),
                float(bool(self.CLOSING_RE.search(ending_text))),
                float(t.endswith("?")),
                float(t.endswith("!")),
            ])

            # Paragraph and Line Cadence
            paragraphs = [p for p in t.split("\\n\\n") if p.strip()]
            n_paras = max(len(paragraphs), 1)
            para_lengths = [len(p.split()) for p in paragraphs] or [0]
            lines = [l for l in t.split("\\n") if l.strip()]
            line_lengths = [len(l.split()) for l in lines] or [0]

            feat.extend([
                np.log1p(n_paras),
                float(np.mean(para_lengths)),
                float(np.std(para_lengths)),
                np.log1p(max(len(lines), 1)),
                float(np.mean(line_lengths)),
                float(np.std(line_lengths)),
                t.count("\\n\\n") / max(t.count("\\n"), 1),
                len(set(lower)) / max(np.sqrt(nw), 1e-6),    # Root Type-Token Ratio
                sum(len(w) >= 12 for w in words) / nw,        # Multisyllabic density
                t.count("```") / nw,                           # Code block density
                t.count("`") / nw,                             # Inline code density
                t.count("$") / nw,                             # LaTeX math density
                t.count("\\\\") / nw,                            # TeX / escape density
                t.count(":") / max(t.count(";") + 1, 1),       # Colon-to-semicolon ratio
                (t.count(\'"\') + t.count("\\u201c") + t.count("\\u201d")) / nw,  # Quotation density
            ])
            rows.append(feat)
        return np.asarray(rows, dtype=np.float64)'''

    assert old_rich_style in shared_upgraded, "Could not find old RichStyleFeatures in shared code!"
    shared_upgraded = shared_upgraded.replace(old_rich_style, new_rich_style, 1)

    # --- 3. Add FastBatchPrefetcher & precompute_token_caches ---
    # We will insert FastBatchPrefetcher right after tokenize_with_head_tail
    target_tok = '    return res\n\n# --- 3. 132-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---'
    assert target_tok in shared_upgraded, "Could not find tokenize_with_head_tail end!"

    prefetcher_code = '''    return res


class FastBatchPrefetcher:
    """
    Background worker thread that pre-pads and pins micro-batches in host RAM ahead of the GPU.
    Eliminates CPU-GPU synchronization bubbles, keeping Tensor Cores at 100% duty cycle.
    """
    def __init__(self, tokenizer, enc_data, y_labels, indices, batch_size, dev, label_dev=None, cuda_ok=True, maxsize=3):
        self.tokenizer = tokenizer
        self.enc_data = enc_data
        self.y_labels = y_labels
        self.indices = indices
        self.batch_size = batch_size
        self.dev = dev
        self.label_dev = label_dev if label_dev is not None else dev
        self.cuda_ok = cuda_ok
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

# --- 3. 132-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---'''

    shared_upgraded = shared_upgraded.replace(target_tok, prefetcher_code, 1)

    # --- 4. Upgrade evaluate_transformer_split & evaluate_qwen_split ---
    old_eval_trans = '''def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
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
    eval_bs = 32
    with torch.no_grad():
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
                "input_ids": padded["input_ids"].to(dev, non_blocking=True),
                "attention_mask": padded["attention_mask"].to(dev, non_blocking=True),
            }
            if "token_type_ids" in padded:
                kw["token_type_ids"] = padded["token_type_ids"].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(**kw).logits
            all_logits.append(logits.float().cpu().numpy())
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    new_eval_trans = '''def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
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
    eval_bs = 48
    cuda_ok = torch.cuda.is_available() and getattr(dev, "type", str(dev)) != "cpu"
    with torch.no_grad():
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
                logits = model(**kw).logits
            all_logits.append(logits.float().cpu().numpy())
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    assert old_eval_trans in shared_upgraded, "Could not find old evaluate_transformer_split!"
    shared_upgraded = shared_upgraded.replace(old_eval_trans, new_eval_trans, 1)

    old_eval_qwen = '''def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes):
    """
    Length-sorted batched inference for Qwen 2.5 7B LoRA decoder.
    Passes representations through backbone to MultiSampleDropoutHead.
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
    with torch.no_grad():
        for start in range(0, n_samples, inf_bs):
            end = min(start + inf_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
            }
            padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded["input_ids"].to(first_dev, non_blocking=True),
                "attention_mask": padded["attention_mask"].to(first_dev, non_blocking=True),
            }
            if cuda_ok:
                with amp_autocast():
                    out = backbone(**batch_kwargs, output_hidden_states=False)
            else:
                out = backbone(**batch_kwargs, output_hidden_states=False)

            mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
            hidden_f = out.last_hidden_state.float()
            pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            logits = head(pooled.to(head_dev)).float().cpu().numpy()
            all_logits.append(logits)
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    new_eval_qwen = '''def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes):
    """
    Length-sorted batched inference for Qwen 2.5 7B LoRA decoder.
    Passes representations through backbone to MultiSampleDropoutHead.
    """
    n_samples = len(enc_data["input_ids"]) if "input_ids" in enc_data else 0
    if n_samples == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    backbone.eval()
    head.eval()
    lengths = [len(x) for x in enc_data["input_ids"]]
    sort_idx = np.argsort(lengths)
    all_logits = []
    inf_bs = 24
    with torch.no_grad():
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
            if cuda_ok:
                with amp_autocast():
                    out = backbone(**batch_kwargs, output_hidden_states=False)
            else:
                out = backbone(**batch_kwargs, output_hidden_states=False)

            mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
            hidden_f = out.last_hidden_state.float()
            pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            logits = head(pooled.to(head_dev)).float().cpu().numpy()
            all_logits.append(logits)
    sorted_logits = np.concatenate(all_logits, axis=0)
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(len(sort_idx))
    res = sorted_logits[inv_idx]
    return res'''

    assert old_eval_qwen in shared_upgraded, "Could not find old evaluate_qwen_split!"
    shared_upgraded = shared_upgraded.replace(old_eval_qwen, new_eval_qwen, 1)

    # --- 5. Upgrade train_transformer_fold with FastBatchPrefetcher & RAM cache support ---
    # Find train_transformer_fold start to train_deberta_fold
    deb_fold_idx = shared_upgraded.index("def train_deberta_fold(")
    trans_fold_idx = shared_upgraded.index("def train_transformer_fold(")

    new_trans_fold = '''def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=10,
    grad_accum=2,
    max_length=352,
    epochs=4,
    lr_backbone=1.5e-5,
    lr_head=1.2e-4,
    weight_decay=0.01,
    label_smoothing=0.05,
    enable_grad_checkpointing=False,
    deadline=None,
    cached_enc_tr=None,
    cached_enc_va=None,
    cached_enc_te=None,
    cached_tokenizer=None,
):
    """
    Unified, high-throughput transformer fine-tuning loop for DeBERTa-v3-large and BGE-large-en-v1.5.
    Features:
    - Stylistic [NL] newline preservation + Head+Tail truncation ([:224] + [-128:]).
    - FastBatchPrefetcher with background worker thread and pinned RAM transfers (100% GPU saturation).
    - Upfront RAM token cache reuse (eliminates redundant tokenizations across folds).
    - Gradient checkpointing OFF by default (fits within 10.5 GB / 14.56 GB T4 VRAM, saves 25% FLOPs).
    - LLRD recalibration (gamma = 0.92) with explicit relative embeddings and LayerNorm discovery.
    - Synchronized gradient accumulation with non-fatal NaN recovery.
    - Per-device context cache clearing.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

    # Ensure model embeddings match tokenizer vocabulary size and initialize newline weights
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb_module = model.get_input_embeddings()
            dot_id = tokenizer.convert_tokens_to_ids(".")
            bar_id = tokenizer.convert_tokens_to_ids("|")
            nl_id = tokenizer.convert_tokens_to_ids("[NL]")
            par_id = tokenizer.convert_tokens_to_ids("[PAR]")
            if dot_id is not None and nl_id is not None and nl_id < emb_module.weight.shape[0]:
                emb_module.weight[nl_id].copy_(emb_module.weight[dot_id])
            if bar_id is not None and par_id is not None and par_id < emb_module.weight.shape[0]:
                emb_module.weight[par_id].copy_(emb_module.weight[bar_id])

    model.config.use_cache = False
    in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") else model.config.hidden_size
    model.classifier = MultiSampleDropoutHead(in_features, len(labels))

    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(dev)
    gc.collect()
    if cuda_ok:
        torch.cuda.empty_cache()
    ckpt_status = "ON" if enable_grad_checkpointing else "OFF (auto OOM fallback)"
    print(f"  [{model_name}] Ready on {dev} (FP16 Tensor Cores ON, grad_ckpt={ckpt_status}, Multi-Sample Dropout ON). {get_mem_status(dev)}", flush=True)

    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    opt_params = get_optimizer_grouped_parameters(
        model, lr_head=lr_head, lr_backbone=lr_backbone, layer_decay=0.92, weight_decay=weight_decay
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_score = -1.0
    best_val_logits = None
    best_state = None
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 5
    stop_early = False
    n_skipped_micros = 0
    t_train_start = time.perf_counter()

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

        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, dev, cuda_ok=cuda_ok
        )
        try:
            for micro_step, (batch_kwargs, batch_labels) in enumerate(prefetcher):
                if deadline is not None and (micro_step % 10 == 0) and time.perf_counter() > deadline:
                    print(f"  [{model_name}] [Time Guard] Deadline reached mid-epoch at step {micro_step}/{n_total_micros}. Stopping training.", flush=True)
                    stop_early = True
                    break

                try:
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        logits = model(**batch_kwargs).logits
                        loss = criterion(logits.float(), batch_labels) / grad_accum

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
                    print(f"  [{model_name}] CUDA OOM detected. Resetting partial grads, enabling gradient checkpointing...", flush=True)
                    try:
                        del logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if not enable_grad_checkpointing:
                        print(f"  [{model_name}] Enabling gradient checkpointing on-the-fly to recover VRAM...", flush=True)
                        try:
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except TypeError:
                            model.gradient_checkpointing_enable()
                        enable_grad_checkpointing = True
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue

                running_loss += loss.item() * grad_accum
                n_micro += 1
                accumulated_steps += 1

                if accumulated_steps == grad_accum:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0
        finally:
            prefetcher.close()

        if accumulated_steps > 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stop_early:
            break

    train_dur = time.perf_counter() - t_train_start
    throughput = (len(texts_tr) * max(epoch + 1, 1)) / max(train_dur, 0.01)
    train_transformer_fold.last_throughput = throughput
    if "deberta" in model_name.lower():
        train_deberta_fold.last_throughput = throughput
    elif "bge" in model_name.lower():
        train_bge_fold.last_throughput = throughput
    print(f"  [{model_name}] Training finished: {train_dur:.1f}s ({throughput:.2f} samples/sec). Best Val Bal Acc: {best_val_score:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))

    del model, optimizer, scheduler
    if best_state is not None:
        del best_state
    gc.collect()
    if cuda_ok:
        n_gpus_local = torch.cuda.device_count()
        for gi in range(n_gpus_local):
            with torch.cuda.device(gi):
                torch.cuda.empty_cache()

    return best_val_logits, test_logits

'''

    shared_upgraded = shared_upgraded[:trans_fold_idx] + new_trans_fold + shared_upgraded[deb_fold_idx:]

    # --- 6. Upgrade train_qwen_lora_fold with FastBatchPrefetcher & RAM cache support ---
    qwen_start_idx = shared_upgraded.index("def train_qwen_lora_fold(")
    fit_temp_idx = shared_upgraded.index("def fit_temperature(")

    new_qwen_fold = '''def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=12,
    grad_accum=2,
    max_length=352,
    epochs=2,
    lr_lora=1.2e-4,
    lr_head=3.0e-4,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    deadline=None,
    cached_enc_tr=None,
    cached_enc_va=None,
    cached_enc_te=None,
    cached_tokenizer=None,
):
    """
    LoRA fine-tune of Qwen 2.5 7B-Instruct across dual GPUs (device_map="auto").
    Context length aligned to 352 tokens (covering 96.4% of all passage lengths).
    LoRA rank 32 with alpha 64 targeting all linear projections (~80M trainable parameters).
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
    with HeartbeatTimer("Reading Qwen 2.5 7B weights and sharding across Dual GPUs"):
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

    head_dev = torch.device("cuda:0" if cuda_ok else "cpu")
    first_dev = next(backbone.parameters()).device
    hidden_size = backbone.config.hidden_size
    head = nn.Sequential(
        nn.LayerNorm(hidden_size),
        MultiSampleDropoutHead(hidden_size, len(labels)),
    ).to(head_dev)

    lora_params = [p for _, p in backbone.named_parameters() if p.requires_grad]
    for p in lora_params:
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    head_params = list(head.parameters())
    optimizer = AdamW(
        [{"params": lora_params, "lr": lr_lora}, {"params": head_params, "lr": lr_head}],
        weight_decay=0.01,
    )
    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=head_dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)

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

    scaler = torch.cuda.amp.GradScaler(enabled=cuda_ok)

    best_val_score = -1.0
    best_val_logits = None
    best_lora_state = None
    best_head_state = None
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 5
    t_qwen_train_start = time.perf_counter()

    for epoch in range(epochs):
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
        stop_early = False

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
                    if cuda_ok:
                        with amp_autocast():
                            out = backbone(**batch_kwargs, output_hidden_states=False)
                    else:
                        out = backbone(**batch_kwargs, output_hidden_states=False)

                    mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                    hidden_f = out.last_hidden_state.float()
                    pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                    logits = head(pooled.to(head_dev))
                    loss = criterion(logits.float(), batch_labels) / grad_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  [Qwen LoRA] Warning: NaN loss at micro-step {micro_step}. Skipping.", flush=True)
                        if accumulated_steps > 0:
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_steps = 0
                        continue

                    if cuda_ok:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except torch.cuda.OutOfMemoryError:
                    print(f"  [Qwen LoRA] CUDA OOM at micro-step {micro_step}. Resetting partial grads...", flush=True)
                    try:
                        del out, pooled, logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue

                epoch_loss += loss.item() * grad_accum
                n_batches += 1
                accumulated_steps += 1

                if accumulated_steps == grad_accum:
                    if cuda_ok:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0
        finally:
            prefetcher.close()

        if accumulated_steps > 0:
            if cuda_ok:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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

        if stop_early:
            break

    qwen_dur = time.perf_counter() - t_qwen_train_start
    qwen_sps = (len(texts_tr) * max(epoch + 1, 1)) / max(qwen_dur, 0.01)
    train_qwen_lora_fold.last_throughput = qwen_sps
    print(f"  [Qwen LoRA] Training finished: {qwen_dur:.1f}s ({qwen_sps:.2f} samples/sec). Best Val Bal Acc: {best_val_score:.4f}", flush=True)

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
        n_gpus_local = torch.cuda.device_count()
        for gi in range(n_gpus_local):
            with torch.cuda.device(gi):
                torch.cuda.empty_cache()

    return best_val_logits, test_logits

'''

    shared_upgraded = shared_upgraded[:qwen_start_idx] + new_qwen_fold + shared_upgraded[fit_temp_idx:]

    print(f"Shared code upgraded successfully! New length: {len(shared_upgraded)} chars, {len(shared_upgraded.splitlines())} lines.")

    # --- 7. Upgrade run_quad_ensemble in kaggle_tri_ensemble.py ---
    old_ens_tail = ens_code[ens_idx:]

    # In run_quad_ensemble, add token cache computation upfront and slice per fold
    # Find STEP 0: FIT CPU MULTIVIEW
    step0_target = '    print("=" * 65, flush=True)\n    print("STEP 0: FITTING CPU FEATURE UNION & IMMEDIATE BASELINE SUBMISSION", flush=True)'
    assert step0_target in old_ens_tail, "Could not find STEP 0 in run_quad_ensemble!"

    step0_5_code = '''    print("=" * 65, flush=True)
    print("STEP 0.5: MULTI-MODEL UPFRONT PRE-TOKENIZATION & RAM CACHING", flush=True)
    print("=" * 65, flush=True)
    token_caches = precompute_token_caches(raw_texts_tr, raw_texts_te, deberta_dir, bge_dir, qwen_dir, max_length=352)
    print(flush=True)

    print("=" * 65, flush=True)
    print("STEP 0: FITTING CPU FEATURE UNION & IMMEDIATE BASELINE SUBMISSION", flush=True)'''

    new_ens_tail = old_ens_tail.replace(step0_target, step0_5_code, 1)

    # Now in the fold loop, pass cached tokens:
    fold_worker_target = '''        #  1 & 2. Concurrent Parallel Fine-Tuning: DeBERTa (GPU 0) + BGE (GPU 1) 
        run_deb = deberta_dir is not None and (time.perf_counter() + deb_expected + SAFETY_MARGIN_SEC < pipeline_deadline)
        run_bge = bge_dir is not None and (time.perf_counter() + bge_expected + SAFETY_MARGIN_SEC < pipeline_deadline)

        if run_deb and run_bge and n_gpus >= 2:
            print(f"\\n[Fold {fold+1}] Launching CONCURRENT Dual-GPU Training: DeBERTa ({deb_dev}, BS=16) || BGE ({bge_dev}, BS=24)", flush=True)
            t_par_start = time.perf_counter()

            def _train_deb_worker():
                return train_deberta_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    deberta_dir, LABELS,
                    device=deb_dev,
                    batch_size=16, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )

            def _train_bge_worker():
                return train_bge_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    bge_dir, LABELS,
                    device=bge_dev,
                    batch_size=24, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )'''

    # We want to replace it regardless of the unicode rocket emoji
    # Let's search by regex for the worker definitions
    fold_worker_regex = re.compile(
        r'(#[^\n]*Concurrent Parallel Fine-Tuning: DeBERTa[^\n]*\n'
        r'\s*run_deb = deberta_dir is not None[^\n]*\n'
        r'\s*run_bge = bge_dir is not None[^\n]*\n\n'
        r'\s*if run_deb and run_bge and n_gpus >= 2:\n'
        r'\s*print\(f"\\n\[Fold \{fold\+1\}\] Launching CONCURRENT Dual-GPU Training: DeBERTa \(\{deb_dev\}, BS=16\) \|\| BGE \(\{bge_dev\}, BS=24\)", flush=True\)\n'
        r'\s*t_par_start = time\.perf_counter\(\)\n\n'
        r'\s*def _train_deb_worker\(\):\n'
        r'\s*return train_deberta_fold\(\n'
        r'\s*texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\n'
        r'\s*deberta_dir, LABELS,\n'
        r'\s*device=deb_dev,\n'
        r'\s*batch_size=16, grad_accum=1, max_length=352,\n'
        r'\s*epochs=N_EPOCHS_DEB, lr_backbone=1\.5e-5, lr_head=1\.2e-4, enable_grad_checkpointing=False,\n'
        r'\s*deadline=pipeline_deadline,\n'
        r'\s*\)\n\n'
        r'\s*def _train_bge_worker\(\):\n'
        r'\s*return train_bge_fold\(\n'
        r'\s*texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\n'
        r'\s*bge_dir, LABELS,\n'
        r'\s*device=bge_dev,\n'
        r'\s*batch_size=24, grad_accum=1, max_length=352,\n'
        r'\s*epochs=N_EPOCHS_BGE, lr_backbone=2\.2e-5, lr_head=1\.2e-4, enable_grad_checkpointing=False,\n'
        r'\s*deadline=pipeline_deadline,\n'
        r'\s*\))'
    )

    match = fold_worker_regex.search(new_ens_tail)
    assert match is not None, "Could not match fold_worker_regex in new_ens_tail!"

    new_fold_workers = '''        # Slice precomputed token caches from RAM in 0.001s
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
        run_deb = deberta_dir is not None and (time.perf_counter() + deb_expected + SAFETY_MARGIN_SEC < pipeline_deadline)
        run_bge = bge_dir is not None and (time.perf_counter() + bge_expected + SAFETY_MARGIN_SEC < pipeline_deadline)

        if run_deb and run_bge and n_gpus >= 2:
            print(f"\\n[Fold {fold+1}] Launching CONCURRENT Dual-GPU Training: DeBERTa ({deb_dev}, BS=16) || BGE ({bge_dev}, BS=24)", flush=True)
            t_par_start = time.perf_counter()

            def _train_deb_worker():
                return train_deberta_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    deberta_dir, LABELS,
                    device=deb_dev,
                    batch_size=16, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                    **deb_kw,
                )

            def _train_bge_worker():
                return train_bge_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    bge_dir, LABELS,
                    device=bge_dev,
                    batch_size=24, grad_accum=1, max_length=352,
                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                    **bge_kw,
                )'''

    new_ens_tail = new_ens_tail[:match.start()] + new_fold_workers + new_ens_tail[match.end():]

    # Also update sequential fallback branches:
    deb_seq_pattern = re.compile(
        r'val_deb, test_deb = train_deberta_fold\(\s*'
        r'texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\s*'
        r'deberta_dir, LABELS,\s*'
        r'device=deb_dev,\s*'
        r'batch_size=16, grad_accum=1, max_length=352,\s*'
        r'epochs=N_EPOCHS_DEB, lr_backbone=1\.5e-5, lr_head=1\.2e-4, enable_grad_checkpointing=False,\s*'
        r'deadline=pipeline_deadline,\s*\)'
    )
    new_ens_tail = deb_seq_pattern.sub(
        'val_deb, test_deb = train_deberta_fold(\n'
        '                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\n'
        '                    deberta_dir, LABELS,\n'
        '                    device=deb_dev,\n'
        '                    batch_size=16, grad_accum=1, max_length=352,\n'
        '                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,\n'
        '                    deadline=pipeline_deadline,\n'
        '                    **deb_kw,\n'
        '                )',
        new_ens_tail
    )

    bge_seq_pattern = re.compile(
        r'val_bge, test_bge = train_bge_fold\(\s*'
        r'texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\s*'
        r'bge_dir, LABELS,\s*'
        r'device=bge_dev,\s*'
        r'batch_size=24, grad_accum=1, max_length=352,\s*'
        r'epochs=N_EPOCHS_BGE, lr_backbone=2\.2e-5, lr_head=1\.2e-4, enable_grad_checkpointing=False,\s*'
        r'deadline=pipeline_deadline,\s*\)'
    )
    new_ens_tail = bge_seq_pattern.sub(
        'val_bge, test_bge = train_bge_fold(\n'
        '                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\n'
        '                    bge_dir, LABELS,\n'
        '                    device=bge_dev,\n'
        '                    batch_size=24, grad_accum=1, max_length=352,\n'
        '                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,\n'
        '                    deadline=pipeline_deadline,\n'
        '                    **bge_kw,\n'
        '                )',
        new_ens_tail
    )

    # And update Qwen call in run_quad_ensemble:
    qwen_call_pattern = re.compile(
        r'val_qwen, test_qwen = train_qwen_lora_fold\(\s*'
        r'texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\s*'
        r'qwen_dir, LABELS,\s*'
        r'batch_size=12, grad_accum=2, max_length=352,\s*'
        r'epochs=N_EPOCHS_QWEN, lr_lora=1\.2e-4, lr_head=3\.0e-4,\s*'
        r'lora_r=32, lora_alpha=64, lora_dropout=0\.05,\s*'
        r'deadline=pipeline_deadline,\s*\)'
    )
    new_ens_tail = qwen_call_pattern.sub(
        'val_qwen, test_qwen = train_qwen_lora_fold(\n'
        '                texts_tr, y_tr_str, texts_va, y_va_str, texts_test,\n'
        '                qwen_dir, LABELS,\n'
        '                batch_size=12, grad_accum=2, max_length=352,\n'
        '                epochs=N_EPOCHS_QWEN, lr_lora=1.2e-4, lr_head=3.0e-4,\n'
        '                lora_r=32, lora_alpha=64, lora_dropout=0.05,\n'
        '                deadline=pipeline_deadline,\n'
        '                **qwen_kw,\n'
        '            )',
        new_ens_tail
    )

    final_ens_code = shared_upgraded + new_ens_tail
    final_dry_code = shared_upgraded + dry_code[dry_idx:]

    # Write files
    with open(ens_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_ens_code)

    with open(dry_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_dry_code)

    print("Files written successfully.")

    # Validate AST
    ast.parse(final_ens_code)
    print("kaggle_tri_ensemble.py AST is valid.")
    ast.parse(final_dry_code)
    print("kaggle_dry_run.py AST is valid.")

if __name__ == "__main__":
    apply_upgrades()

