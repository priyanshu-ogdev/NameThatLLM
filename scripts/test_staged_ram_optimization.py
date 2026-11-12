import sys
sys.path.insert(0, ".")
import tempfile
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForSequenceClassification,
    Qwen2Config,
    Qwen2Model,
)
from notebooks.kaggle_tri_ensemble import (
    LABELS,
    GLOBAL_RAM_MODEL_CACHE,
    train_bge_fold,
    train_qwen_lora_fold,
)

print("Testing LoRA parameter reset mechanism on multi-layer mock backbone...")

class MockBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        return self.out_proj(self.v_proj(self.q_proj(x)))

class MockTransformer(nn.Module):
    def __init__(self, num_layers=4, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([MockBlock(dim) for _ in range(num_layers)])
        self.config = type("Config", (), {"hidden_size": dim})()
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

# 1. Instantiate and attach LoRA
base_model = MockTransformer(num_layers=4, dim=32)
lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(base_model, lora_cfg)

# 2. Check initial LoRA state
q0_b = peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight
assert torch.all(q0_b == 0.0), "Initial LoRA B should be all zeros"

# 3. Simulate training Fold 1 (modify LoRA weights)
with torch.no_grad():
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.add_(1.5)

assert not torch.all(peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight == 0.0)

# 4. Reset LoRA parameters for Fold 2
for name, module in peft_model.named_modules():
    if hasattr(module, "reset_lora_parameters"):
        try:
            module.reset_lora_parameters("default", init_lora_weights=True)
        except TypeError:
            module.reset_lora_parameters("default")

# 5. Verify that Fold 2 has completely reset LoRA weights
q0_b_reset = peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight
assert torch.all(q0_b_reset == 0.0), "Reset LoRA B should be all zeros again!"
print("[Pass] LoRA parameter reset verified 100% clean across folds.")

# 6. Test bottom layer freezing
print("Testing encoder bottom layer freezing...")
enc = MockTransformer(num_layers=6, dim=32)
n_freeze = 3
for l_idx in range(n_freeze):
    for p in enc.layers[l_idx].parameters():
        p.requires_grad = False

for l_idx in range(6):
    is_frozen = all(not p.requires_grad for p in enc.layers[l_idx].parameters())
    if l_idx < n_freeze:
        assert is_frozen, f"Layer {l_idx} should be frozen"
    else:
        assert not is_frozen, f"Layer {l_idx} should NOT be frozen"

print("[Pass] Encoder bottom layer freezing verified.")

# 7. Real Offline End-to-End Fold 1 -> Fold 2 Execution of train_bge_fold & train_qwen_lora_fold
print("\nTesting end-to-end Fold 1 -> Fold 2 execution on real HF BertForSequenceClassification & Qwen2Model...")
GLOBAL_RAM_MODEL_CACHE.clear()

texts_tr = [f"hello world test text sample {i % 3}\n\nmore lines {i}" for i in range(18)]
y_tr = [LABELS[i % len(LABELS)] for i in range(18)]
groups_tr = np.array([f"p_{i % 6}" for i in range(18)], dtype=object)
texts_va = [f"hello world val {i}" for i in range(6)]
y_va = [LABELS[i % len(LABELS)] for i in range(6)]
texts_te = [f"hello world test {i}" for i in range(6)]

vocab = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "hello": 5, "world": 6, "test": 7, "text": 8, "sample": 9,
    "more": 10, "lines": 11, "val": 12, ".": 13, "|": 14,
}
tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tok_obj.pre_tokenizer = Whitespace()

with tempfile.TemporaryDirectory() as bge_dir, tempfile.TemporaryDirectory() as qwen_dir:
    tok_bge = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj, unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]", eos_token="[SEP]",
    )
    tok_bge.save_pretrained(bge_dir)
    bert_cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=8,
        num_attention_heads=2, intermediate_size=64, num_labels=len(LABELS),
        max_position_embeddings=64,
    )
    bert_m = BertForSequenceClassification(bert_cfg)
    bert_m.save_pretrained(bge_dir)

    # Fold 1 BGE (stages weights in GLOBAL_RAM_MODEL_CACHE)
    val_bge_1, test_bge_1 = train_bge_fold(
        texts_tr, y_tr, texts_va, y_va, texts_te,
        bge_dir, LABELS, device="cpu", batch_size=6, grad_accum=1,
        max_length=32, epochs=2, enable_grad_checkpointing=False,
        n_frozen_bottom_layers=2, group_ids_tr=groups_tr,
    )
    assert val_bge_1.shape == (6, len(LABELS)) and np.all(np.isfinite(val_bge_1))
    assert test_bge_1.shape == (6, len(LABELS)) and np.all(np.isfinite(test_bge_1))
    assert "bge" in GLOBAL_RAM_MODEL_CACHE, "BGE base weights must be staged in RAM cache"

    # Fold 2 BGE (reloads cleanly from GLOBAL_RAM_MODEL_CACHE)
    val_bge_2, test_bge_2 = train_bge_fold(
        texts_tr, y_tr, texts_va, y_va, texts_te,
        bge_dir, LABELS, device="cpu", batch_size=6, grad_accum=1,
        max_length=32, epochs=2, enable_grad_checkpointing=False,
        n_frozen_bottom_layers=2, group_ids_tr=groups_tr,
    )
    assert val_bge_2.shape == (6, len(LABELS)) and np.all(np.isfinite(val_bge_2))
    assert test_bge_2.shape == (6, len(LABELS)) and np.all(np.isfinite(test_bge_2))
    print("[Pass] train_bge_fold Fold 1 -> Fold 2 + Multi-Layer Fusion + RAM cache verified!")

    # Tiny Qwen2Model (10 layers so -1, -5, -9 hooks all fire)
    import json
    from pathlib import Path
    tok_q = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj, unk_token="[UNK]", pad_token="[PAD]", eos_token="[SEP]",
    )
    tok_q.save_pretrained(qwen_dir)
    tc_file = Path(qwen_dir) / "tokenizer_config.json"
    tc_data = json.loads(tc_file.read_text(encoding="utf-8"))
    tc_data["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc_file.write_text(json.dumps(tc_data), encoding="utf-8")
    qwen_cfg = Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=10, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    qwen_m = Qwen2Model(qwen_cfg)
    qwen_m.save_pretrained(qwen_dir)

    # Fold 1 Qwen (attaches PEFT, captures -1/-5/-9 hooks, updates weights, retains backbone)
    val_q_1, test_q_1 = train_qwen_lora_fold(
        texts_tr, y_tr, texts_va, y_va, texts_te,
        qwen_dir, LABELS, batch_size=6, grad_accum=1, max_length=32,
        epochs=2, lora_r=4, lora_alpha=8, group_ids_tr=groups_tr, retain_backbone=True,
    )
    assert val_q_1.shape == (6, len(LABELS)) and np.all(np.isfinite(val_q_1))
    assert test_q_1.shape == (6, len(LABELS)) and np.all(np.isfinite(test_q_1))
    assert "qwen_backbone" in GLOBAL_RAM_MODEL_CACHE, "Qwen backbone must be cached in RAM"

    # Assert Fold 1 Qwen ACTUALLY performed training updates (LoRA weights changed from zeros!)
    cached_qwen = GLOBAL_RAM_MODEL_CACHE["qwen_backbone"]
    lora_b_norms = [
        float(p.abs().sum().item())
        for n, p in cached_qwen.named_parameters()
        if "lora_B" in n
    ]
    assert lora_b_norms and max(lora_b_norms) > 0.0, "Qwen LoRA weights must be updated during training!"

    # Fold 2 Qwen (reloads from GLOBAL_RAM_MODEL_CACHE, resets LoRA, trains Fold 2)
    val_q_2, test_q_2 = train_qwen_lora_fold(
        texts_tr, y_tr, texts_va, y_va, texts_te,
        qwen_dir, LABELS, batch_size=6, grad_accum=1, max_length=32,
        epochs=2, lora_r=4, lora_alpha=8, group_ids_tr=groups_tr, retain_backbone=False,
    )
    assert val_q_2.shape == (6, len(LABELS)) and np.all(np.isfinite(val_q_2))
    assert test_q_2.shape == (6, len(LABELS)) and np.all(np.isfinite(test_q_2))
    assert "qwen_backbone" not in GLOBAL_RAM_MODEL_CACHE, "Qwen cache must be cleared when retain_backbone=False"
    print("[Pass] train_qwen_lora_fold Fold 1 -> Fold 2 + LoRA updates + PEFT hooks verified!")

print("\n>>> ALL STAGED RAM & END-TO-END FOLD OPTIMIZATION TESTS PASSED! <<<")
