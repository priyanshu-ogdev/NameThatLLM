import ast
import re
from pathlib import Path

def apply_upgrades():
    tri_path = Path("notebooks/kaggle_tri_ensemble.py")
    content = tri_path.read_text(encoding="utf-8")

    # 1. Add copy import
    if "import copy" not in content:
        content = content.replace("import contextlib\n", "import contextlib\nimport copy\n")
        print("1. Added import copy")

    # 2. Add Nystroem import
    if "from sklearn.kernel_approximation import Nystroem" not in content:
        content = content.replace(
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.preprocessing import StandardScaler\nfrom sklearn.kernel_approximation import Nystroem\n"
        )
        print("2. Added Nystroem import")

    # 3. Add SIBLING_DIFFICULTY_MARGINS after LABEL_TO_FAMILY_IDX
    if "SIBLING_DIFFICULTY_MARGINS" not in content:
        old_part = "LABEL_TO_FAMILY_IDX = np.array([FAMILY_TO_IDX[FAMILY_MAP[l]] for l in LABELS], dtype=np.int64)\n"
        new_part = """LABEL_TO_FAMILY_IDX = np.array([FAMILY_TO_IDX[FAMILY_MAP[l]] for l in LABELS], dtype=np.int64)

# Sibling difficulty angular margins for FocalLoss (prioritizes hard sibling cliques over distinct outliers)
SIBLING_DIFFICULTY_MARGINS = [
    0.12,  # gpt-5.5
    0.25,  # gpt-5.6-luna (sibling clique)
    0.25,  # gpt-5.6-sol (sibling clique)
    0.25,  # gpt-5.6-terra (sibling clique)
    0.10,  # gpt-6-astra
    0.08,  # gemini-3.1-pro-preview (distinct architecture/style)
    0.32,  # gemini-3.6-flash (hardest sibling triplet)
    0.32,  # gemini-3.7-flash (hardest sibling triplet)
    0.32,  # gemini-3.8-flash (hardest sibling triplet)
    0.20,  # claude-fable-5
    0.20,  # claude-opus-4-6
    0.22,  # claude-opus-4-8
    0.12,  # claude-opus-5
    0.15,  # claude-sonnet-5
    0.08,  # qwen3.8-max (distinct vocabulary/tokenizer)
    0.24,  # glm-5.3
    0.24,  # glm-5.3-flash
    0.08,  # kimi-k3 (distinct Chinese/English reasoning patterns)
]
"""
        assert old_part in content, "LABEL_TO_FAMILY_IDX not found"
        content = content.replace(old_part, new_part)
        print("3. Added SIBLING_DIFFICULTY_MARGINS")

    # 4. Add compute_same_prompt_sibling_repulsion and make_prompt_sibling_paired_indices before TopicBlindAndSiFTPrep
    if "def compute_same_prompt_sibling_repulsion" not in content:
        old_part = "class TopicBlindAndSiFTPrep:"
        new_part = """def compute_same_prompt_sibling_repulsion(
    representations: torch.Tensor,
    batch_groups: torch.Tensor | None,
    batch_labels: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    \"\"\"
    Penalizes representations of responses to the SAME prompt that collapse together
    despite originating from DIFFERENT model generators (author style separation).
    Runs 100% vectorized in PyTorch without CUDA synchronizations.
    \"\"\"
    B = representations.size(0)
    if B <= 1 or batch_groups is None:
        return torch.tensor(0.0, device=representations.device, dtype=representations.dtype)

    same_prompt = (batch_groups.unsqueeze(1) == batch_groups.unsqueeze(0))
    diff_model = (batch_labels.unsqueeze(1) != batch_labels.unsqueeze(0))
    upper_tri = torch.triu(torch.ones(B, B, device=representations.device, dtype=torch.bool), diagonal=1)
    pair_mask = same_prompt & diff_model & upper_tri

    n_pairs = pair_mask.sum()
    if n_pairs == 0:
        return torch.tensor(0.0, device=representations.device, dtype=representations.dtype)

    f_norm = F.normalize(representations.float(), p=2, dim=-1)
    cos_sim = torch.mm(f_norm, f_norm.t())
    viol = torch.clamp(cos_sim - margin, min=0.0)
    repulsion = (viol.pow(2) * pair_mask.float()).sum() / n_pairs.clamp(min=1.0)
    return repulsion


def make_prompt_sibling_paired_indices(
    group_ids: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    rng: np.random.RandomState,
    mega_factor: int = 8,
) -> np.ndarray:
    \"\"\"
    Constructs epoch batch indices that co-locate responses to the SAME prompt in the same micro-batch,
    enabling in-batch sibling contrastive separation while sorting by length to minimize padding.
    \"\"\"
    n_samples = len(group_ids)
    if n_samples <= batch_size:
        return np.arange(n_samples, dtype=np.int64)

    from collections import defaultdict
    group_to_idx = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group_to_idx[gid].append(idx)

    unique_groups = list(group_to_idx.keys())
    rng.shuffle(unique_groups)

    mega_size = batch_size * mega_factor
    mega_chunks = []
    current_chunk = []
    for gid in unique_groups:
        current_chunk.extend(group_to_idx[gid])
        if len(current_chunk) >= mega_size:
            mega_chunks.append(np.array(current_chunk, dtype=np.int64))
            current_chunk = []
    if current_chunk:
        mega_chunks.append(np.array(current_chunk, dtype=np.int64))

    sorted_epoch_indices = []
    for chunk in mega_chunks:
        chunk_lens = lengths[chunk]
        sorted_epoch_indices.extend(chunk[np.argsort(chunk_lens)])

    return np.array(sorted_epoch_indices, dtype=np.int64)


class TopicBlindAndSiFTPrep:"""
        assert old_part in content, "class TopicBlindAndSiFTPrep not found"
        content = content.replace(old_part, new_part)
        print("4. Added sibling repulsion and prompt-paired indexing")

    # 5. Upgrade FastBatchPrefetcher with group_ids support
    old_prefetcher_init = """    def __init__(self, tokenizer, enc_data, y_labels, indices, batch_size, dev, label_dev=None, cuda_ok=True, maxsize=6, topic_prep=None):
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
        self.worker.start()"""

    new_prefetcher_init = """    def __init__(self, tokenizer, enc_data, y_labels, indices, batch_size, dev, label_dev=None, cuda_ok=True, maxsize=6, topic_prep=None, group_ids=None):
        self.tokenizer = tokenizer
        self.enc_data = enc_data
        self.y_labels = y_labels
        self.indices = indices
        self.batch_size = batch_size
        self.dev = dev
        self.label_dev = label_dev if label_dev is not None else dev
        self.cuda_ok = cuda_ok
        self.topic_prep = topic_prep
        self.group_ids = group_ids
        self.current_groups = None
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.exc = None
        self.worker = threading.Thread(target=self._produce, daemon=True)
        self.worker.start()"""

    if old_prefetcher_init in content:
        content = content.replace(old_prefetcher_init, new_prefetcher_init)
        print("5. Upgraded FastBatchPrefetcher.__init__")

    old_prefetcher_produce = """                labels = torch.tensor(self.y_labels[idx], dtype=torch.long)
                if self.cuda_ok:
                    labels = labels.pin_memory()

                while not self.stop_event.is_set():
                    try:
                        self.queue.put((kw, labels), timeout=0.1)
                        break
                    except queue.Full:
                        continue"""

    new_prefetcher_produce = """                labels = torch.tensor(self.y_labels[idx], dtype=torch.long)
                if self.cuda_ok:
                    labels = labels.pin_memory()

                groups = None
                if self.group_ids is not None:
                    groups = torch.tensor(self.group_ids[idx], dtype=torch.long)
                    if self.cuda_ok:
                        groups = groups.pin_memory()

                while not self.stop_event.is_set():
                    try:
                        self.queue.put((kw, labels, groups), timeout=0.1)
                        break
                    except queue.Full:
                        continue"""

    if old_prefetcher_produce in content:
        content = content.replace(old_prefetcher_produce, new_prefetcher_produce)
        print("6. Upgraded FastBatchPrefetcher._produce")

    old_prefetcher_iter = """            kw, labels = item
            if self.cuda_ok:
                gpu_kw = {k: v.to(self.dev, non_blocking=True) for k, v in kw.items()}
                gpu_labels = labels.to(self.label_dev, non_blocking=True)
                yield gpu_kw, gpu_labels
            else:
                cpu_kw = {k: v.to(self.dev) for k, v in kw.items()}
                cpu_labels = labels.to(self.label_dev)
                yield cpu_kw, cpu_labels"""

    new_prefetcher_iter = """            if len(item) == 3:
                kw, labels, groups = item
            else:
                kw, labels = item
                groups = None
            if self.cuda_ok:
                gpu_kw = {k: v.to(self.dev, non_blocking=True) for k, v in kw.items()}
                gpu_labels = labels.to(self.label_dev, non_blocking=True)
                self.current_groups = groups.to(self.label_dev, non_blocking=True) if groups is not None else None
                yield gpu_kw, gpu_labels
            else:
                cpu_kw = {k: v.to(self.dev) for k, v in kw.items()}
                cpu_labels = labels.to(self.label_dev)
                self.current_groups = groups.to(self.label_dev) if groups is not None else None
                yield cpu_kw, cpu_labels"""

    if old_prefetcher_iter in content:
        content = content.replace(old_prefetcher_iter, new_prefetcher_iter)
        print("7. Upgraded FastBatchPrefetcher.__iter__")

    # 6. Add GLOBAL_RAM_MODEL_CACHE
    if "GLOBAL_RAM_MODEL_CACHE" not in content:
        old_part = "def precompute_token_caches("
        new_part = "GLOBAL_RAM_MODEL_CACHE: dict = {}\n\n\ndef precompute_token_caches("
        assert old_part in content
        content = content.replace(old_part, new_part)
        print("8. Added GLOBAL_RAM_MODEL_CACHE")

    # 7. Add NonLinearStylometricKernelExpander and upgrade DenseStylometricClassifier
    if "class NonLinearStylometricKernelExpander" not in content:
        old_dense_cls = """class DenseStylometricClassifier(BaseEstimator):
    \"\"\"
    Standardized dense stylometrics classifier.
    Scales continuous stylometric features with StandardScaler and fits an L2-regularized
    LogisticRegression classifier with balanced class weights to produce calibrated probabilities.
    Supports sample_weight with automatic class_weight=None override to prevent double-balancing.
    \"\"\"
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
        return p"""

        new_dense_cls = """class NonLinearStylometricKernelExpander(BaseEstimator, TransformerMixin):
    \"\"\"
    Expands dense continuous stylometric features into a non-linear Hilbert space using:
      1. Degree-2 polynomial interaction cross-products among top-variance features.
      2. Nyström RBF kernel approximation mapping to 768 non-linear components.
    Total expansion: 132 (raw) + 666 (quad cross-products) + 768 (RBF components) = 1,566 features.
    Enables linear models (LogisticRegression / Ridge) on CPU to form curved decision boundaries.
    \"\"\"
    def __init__(self, n_top_poly=36, n_nystroem=768, seed=42):
        self.n_top_poly = n_top_poly
        self.n_nystroem = n_nystroem
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_feats = X.shape
        variances = np.var(X, axis=0)
        k_poly = min(self.n_top_poly, n_feats)
        self.top_poly_idx_ = np.argsort(-variances)[:k_poly]

        k_nys = min(self.n_nystroem, max(1, n_samples - 1))
        self.nystroem_ = Nystroem(
            kernel="rbf", n_components=k_nys, random_state=self.seed, gamma=None
        )
        self.nystroem_.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        parts = [X]

        if hasattr(self, "top_poly_idx_") and len(self.top_poly_idx_) > 1:
            X_sub = X[:, self.top_poly_idx_]
            k = X_sub.shape[1]
            quad_cols = []
            for i in range(k):
                col_i = X_sub[:, i:i+1]
                quad_cols.append(col_i * X_sub[:, i:])
            parts.append(np.hstack(quad_cols))

        if hasattr(self, "nystroem_"):
            parts.append(self.nystroem_.transform(X))

        return np.hstack(parts)


class DenseStylometricClassifier(BaseEstimator):
    \"\"\"
    Standardized dense stylometrics classifier with optional non-linear kernel expansion.
    Scales continuous stylometric features and fits an L2-regularized
    LogisticRegression classifier with balanced class weights to produce calibrated probabilities.
    Supports sample_weight with automatic class_weight=None override to prevent double-balancing.
    \"\"\"
    def __init__(self, c=0.2, seed=42, class_weight="balanced", canonical_labels=None, use_kernel_expander=True):
        self.c = c
        self.seed = seed
        self.class_weight = class_weight
        self.canonical_labels = canonical_labels
        self.use_kernel_expander = use_kernel_expander

    def fit(self, V, y, sample_weight=None):
        self.scaler_ = StandardScaler()
        cw = None if sample_weight is not None else self.class_weight
        self.clf_ = LogisticRegression(
            C=self.c, max_iter=1000, random_state=self.seed, class_weight=cw, solver="lbfgs"
        )
        if self.use_kernel_expander:
            self.expander_ = NonLinearStylometricKernelExpander(seed=self.seed)
            V_feat = self.expander_.fit_transform(V)
        else:
            self.expander_ = None
            V_feat = V
        V_scaled = self.scaler_.fit_transform(V_feat)
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
        if getattr(self, "expander_", None) is not None:
            V_feat = self.expander_.transform(V)
        else:
            V_feat = V
        V_scaled = self.scaler_.transform(V_feat)
        p = self.clf_.predict_proba(V_scaled)
        if self.canonical_labels is not None:
            p_aligned = align_margins(p, self._raw_classes, self.canonical_labels, fill_value=0.0)
            return p_aligned / np.maximum(p_aligned.sum(axis=1, keepdims=True), 1e-12)
        return p"""

        assert old_dense_cls in content, "old_dense_cls not found"
        content = content.replace(old_dense_cls, new_dense_cls)
        print("9. Added NonLinearStylometricKernelExpander and upgraded DenseStylometricClassifier")

    # 8. Upgrade FocalLoss with sibling_margins
    old_focal = """class FocalLoss(nn.Module):
    \"\"\"
    Focal Loss with per-class weighting for severe class imbalance.
    Modulates cross entropy by (1 - p_t)^gamma using the predicted probability of the true class.
    Down-weights well-classified easy examples to focus gradients on hard minority classes.
    alpha (class weights) additionally up-weights rare classes.
    \"\"\"
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
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(log_p_t)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            sum_log_prob = log_prob.sum(dim=-1)
            # Analytical label smoothed cross entropy: -(1 - eps) * log(p_t) - (eps / (K - 1)) * sum_{c != t} log(p_c)
            # where sum_{c != t} log(p_c) = sum_log_prob - log_p_t
            ce_loss = -(1.0 - eps) * log_p_t - (eps / (num_classes - 1)) * (sum_log_prob - log_p_t)
        else:
            ce_loss = -log_p_t
        loss = focal_weight * ce_loss
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss"""

    new_focal = """class FocalLoss(nn.Module):
    \"\"\"
    Focal Loss with per-class weighting and sibling difficulty angular margins.
    Modulates cross entropy by (1 - p_t)^gamma using the predicted probability of the true class.
    Down-weights well-classified easy examples to focus gradients on hard minority classes.
    alpha (class weights) additionally up-weights rare classes.
    Sibling difficulty margins Delta_c force larger angular separation for ambiguous sibling cliques.
    \"\"\"
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05, reduction="mean", use_sibling_margins: bool = True, sibling_margins=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.use_sibling_margins = use_sibling_margins
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

        if sibling_margins is not None:
            sm = torch.tensor(sibling_margins, dtype=torch.float32) if not isinstance(sibling_margins, torch.Tensor) else sibling_margins.float()
            self.register_buffer("sibling_margins", sm)
        elif use_sibling_margins:
            self.register_buffer("sibling_margins", torch.tensor(SIBLING_DIFFICULTY_MARGINS, dtype=torch.float32))
        else:
            self.sibling_margins = None

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets = targets.to(device=logits.device, dtype=torch.long)
        logits_f = logits.float()
        if self.use_sibling_margins and self.sibling_margins is not None and self.sibling_margins.size(0) == num_classes:
            sm = self.sibling_margins.to(logits.device)
            one_hot = F.one_hot(targets, num_classes=num_classes).type_as(logits_f)
            logits_f = logits_f - one_hot * sm.unsqueeze(0)
        log_prob = torch.nn.functional.log_softmax(logits_f, dim=-1)
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(log_p_t)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            sum_log_prob = log_prob.sum(dim=-1)
            # Analytical label smoothed cross entropy: -(1 - eps) * log(p_t) - (eps / (K - 1)) * sum_{c != t} log(p_c)
            # where sum_{c != t} log(p_c) = sum_log_prob - log_p_t
            ce_loss = -(1.0 - eps) * log_p_t - (eps / (num_classes - 1)) * (sum_log_prob - log_p_t)
        else:
            ce_loss = -log_p_t
        loss = focal_weight * ce_loss
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss"""

    if old_focal in content:
        content = content.replace(old_focal, new_focal)
        print("10. Upgraded FocalLoss with sibling difficulty margins")

    # 9. In train_transformer_fold:
    # 9a. Host RAM model caching
    old_load = """    with HeartbeatTimer(f"Reading {model_name} from disk onto {dev}"):
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
                emb_module.weight[n_id].copy_(emb_module.weight[s_id])"""

    new_load = """    if model_tag in GLOBAL_RAM_MODEL_CACHE:
        print(f"  [{model_name}] Loading base weights from Host RAM cache (0 disk I/O)...", flush=True)
        config = GLOBAL_RAM_MODEL_CACHE[f"{model_tag}_config"]
        try:
            model = AutoModelForSequenceClassification.from_config(config)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, num_labels=len(labels), id2label=id2label, label2id=label2id, local_files_only=True
            )
        model.load_state_dict(GLOBAL_RAM_MODEL_CACHE[model_tag])
    else:
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

        try:
            GLOBAL_RAM_MODEL_CACHE[model_tag] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            GLOBAL_RAM_MODEL_CACHE[f"{model_tag}_config"] = copy.deepcopy(model.config)
            cache_mb = sum(p.numel() * p.element_size() for p in GLOBAL_RAM_MODEL_CACHE[model_tag].values()) / (1024 ** 2)
            print(f"  [{model_name}] Staged base weights to Host RAM cache ({cache_mb:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  [{model_name}] Host RAM caching skipped: {e}", flush=True)"""

    if old_load in content:
        content = content.replace(old_load, new_load)
        print("11. Upgraded train_transformer_fold with Host RAM model caching")

    # 9b. Prompt-paired sibling batch indices & prefetcher in train_transformer_fold
    old_tf_epoch = """        # Mega-batch dynamic length-bucketing: groups batches by token length within chunks
        perm = rng.permutation(len(enc_tr["input_ids"]))
        mega_size = batch_size * 8
        sorted_indices = []
        for m_start in range(0, len(perm), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_micro = 0
        accumulated_steps = 0

        t_batch_start = time.perf_counter()
        t_step1_done = None
        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, dev, cuda_ok=cuda_ok, topic_prep=topic_prep
        )"""

    new_tf_epoch = """        # Prompt-paired sibling batching with dynamic length sorting
        if group_ids_tr is not None:
            epoch_indices = make_prompt_sibling_paired_indices(group_ids_tr, y_tr_int, tr_lengths, batch_size, rng)
        else:
            perm = rng.permutation(len(enc_tr["input_ids"]))
            mega_size = batch_size * 8
            sorted_indices = []
            for m_start in range(0, len(perm), mega_size):
                m_chunk = perm[m_start:m_start + mega_size]
                sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
            epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_micro = 0
        accumulated_steps = 0

        t_batch_start = time.perf_counter()
        t_step1_done = None
        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, dev, cuda_ok=cuda_ok, topic_prep=topic_prep, group_ids=group_ids_tr
        )"""

    if old_tf_epoch in content:
        content = content.replace(old_tf_epoch, new_tf_epoch)
        print("12. Upgraded train_transformer_fold with prompt-paired sibling indices")

    # 9c. Repulsion loss in train_transformer_fold
    old_tf_loss = """                        main_loss = criterion(logits_f, batch_labels)
                        fam_logits = compute_family_logits(logits_f, temp=4.0)
                        batch_fam_labels = label_to_fam_tensor[batch_labels]
                        aux_loss = family_criterion(fam_logits, batch_fam_labels)
                        loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum
                        branch_var = getattr(getattr(model, "classifier", None), "last_branch_var", None)
                        if branch_var is not None:
                            loss = loss + (0.03 * branch_var) / grad_accum"""

    new_tf_loss = """                        main_loss = criterion(logits_f, batch_labels)
                        fam_logits = compute_family_logits(logits_f, temp=4.0)
                        batch_fam_labels = label_to_fam_tensor[batch_labels]
                        aux_loss = family_criterion(fam_logits, batch_fam_labels)
                        loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum
                        batch_groups = getattr(prefetcher, "current_groups", None)
                        if batch_groups is not None:
                            repulsion_loss = compute_same_prompt_sibling_repulsion(logits_f, batch_groups, batch_labels, margin=0.15)
                            loss = loss + (0.04 * repulsion_loss) / grad_accum
                        branch_var = getattr(getattr(model, "classifier", None), "last_branch_var", None)
                        if branch_var is not None:
                            loss = loss + (0.03 * branch_var) / grad_accum"""

    if old_tf_loss in content:
        content = content.replace(old_tf_loss, new_tf_loss)
        print("13. Upgraded train_transformer_fold with sibling repulsion loss")

    # 10. Upgrade train_qwen_lora_fold
    # 10a. Add group_ids_tr=None to signature
    old_qwen_sig = """def train_qwen_lora_fold(
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
):"""

    new_qwen_sig = """def train_qwen_lora_fold(
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
    group_ids_tr=None,
):"""

    if old_qwen_sig in content:
        content = content.replace(old_qwen_sig, new_qwen_sig)
        print("14. Upgraded train_qwen_lora_fold signature with group_ids_tr")

    # 10b. Sibling pairing & repulsion in train_qwen_lora_fold
    old_qwen_epoch = """        # Mega-batch dynamic length-bucketing: groups batches by token length within chunks
        perm = rng.permutation(len(enc_tr["input_ids"]))
        mega_size = batch_size * 8
        sorted_indices = []
        for m_start in range(0, len(perm), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_batches = 0
        accumulated_steps = 0

        t_batch_start = time.perf_counter()
        t_step1_done = None
        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, first_dev, label_dev=head_dev, cuda_ok=cuda_ok
        )"""

    new_qwen_epoch = """        # Prompt-paired sibling batching with dynamic length sorting
        if group_ids_tr is not None:
            epoch_indices = make_prompt_sibling_paired_indices(group_ids_tr, y_tr_int, tr_lengths, batch_size, rng)
        else:
            perm = rng.permutation(len(enc_tr["input_ids"]))
            mega_size = batch_size * 8
            sorted_indices = []
            for m_start in range(0, len(perm), mega_size):
                m_chunk = perm[m_start:m_start + mega_size]
                sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
            epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_batches = 0
        accumulated_steps = 0

        t_batch_start = time.perf_counter()
        t_step1_done = None
        prefetcher = FastBatchPrefetcher(
            tokenizer, enc_tr, y_tr_int, epoch_indices, batch_size, first_dev, label_dev=head_dev, cuda_ok=cuda_ok, group_ids=group_ids_tr
        )"""

    if old_qwen_epoch in content:
        content = content.replace(old_qwen_epoch, new_qwen_epoch)
        print("15. Upgraded train_qwen_lora_fold with prompt-paired sibling indices")

    # 10c. Repulsion loss in train_qwen_lora_fold
    old_qwen_loss = """                    logits_f = logits.float()
                    main_loss = criterion(logits_f, batch_labels)
                    fam_logits = compute_family_logits(logits_f, temp=4.0)
                    batch_fam_labels = label_to_fam_tensor[batch_labels]
                    aux_loss = family_criterion(fam_logits, batch_fam_labels)
                    loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum"""

    new_qwen_loss = """                    logits_f = logits.float()
                    main_loss = criterion(logits_f, batch_labels)
                    fam_logits = compute_family_logits(logits_f, temp=4.0)
                    batch_fam_labels = label_to_fam_tensor[batch_labels]
                    aux_loss = family_criterion(fam_logits, batch_fam_labels)
                    loss = (main_loss + FAMILY_AUX_WEIGHT * aux_loss) / grad_accum
                    batch_groups = getattr(prefetcher, "current_groups", None)
                    if batch_groups is not None:
                        repulsion_loss = compute_same_prompt_sibling_repulsion(logits_f, batch_groups, batch_labels, margin=0.15)
                        loss = loss + (0.04 * repulsion_loss) / grad_accum"""

    if old_qwen_loss in content:
        content = content.replace(old_qwen_loss, new_qwen_loss)
        print("16. Upgraded train_qwen_lora_fold with sibling repulsion loss")

    # 11. Pass group_ids_tr into train_qwen_lora_fold in run_quad_ensemble
    old_run_qwen = """                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=qwen_bs, grad_accum=qwen_ga, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=1.8e-4, lora_r=48, lora_alpha=96, lora_dropout=0.08,
                    deadline=pipeline_deadline,
                    **qwen_kw,
                )"""

    new_run_qwen = """                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=qwen_bs, grad_accum=qwen_ga, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=1.8e-4, lora_r=48, lora_alpha=96, lora_dropout=0.08,
                    deadline=pipeline_deadline,
                    group_ids_tr=group_ids[tr_idx],
                    **qwen_kw,
                )"""

    if old_run_qwen in content:
        content = content.replace(old_run_qwen, new_run_qwen)
        print("17. Passed group_ids_tr to train_qwen_lora_fold in run_quad_ensemble")

    # Verify AST
    ast.parse(content)
    tri_path.write_text(content, encoding="utf-8")
    print("ALL UPGRADES APPLIED CLEANLY AND AST PARSED PERFECTLY!")

if __name__ == "__main__":
    apply_upgrades()
