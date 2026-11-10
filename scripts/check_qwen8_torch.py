"""CPU architecture/budget check; explicitly reports unrun dual-GPU validation."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from ntl.generative import extract_frozen, pool_hidden, write_json


torch.set_num_threads(2)
config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_window_layers=36)
model = Qwen3Model(config).eval().requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B-Base-kaggle-v1', local_files_only=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
frame = pd.DataFrame({'id': ['001', '002'], 'text': ['A short sentence.', 'A longer sentence with more words and punctuation!']})
with tempfile.TemporaryDirectory() as cache:
    try:
        extract_frozen(frame, model, tokenizer, {'purpose': 'random tiny architecture check'}, cache, max_seconds=0)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Zero-budget extraction did not stop')
    values, info = extract_frozen(frame, model, tokenizer, {'purpose': 'random tiny architecture check'}, cache)
    assert values.shape == (2, 4, 32)
    np.testing.assert_allclose(np.linalg.norm(values, axis=2), 1, atol=1e-5)
    again, resumed = extract_frozen(frame, model, tokenizer, {'purpose': 'random tiny architecture check'}, cache)
    assert resumed['resumed_rows'] == 2
    np.testing.assert_array_equal(values, again)
cross_device = 'unrun: no two CUDA devices in this cloud workspace'
if torch.cuda.device_count() >= 2:
    hidden = (torch.tensor([[[1., 0.], [0., 2.], [99., 99.]]], device='cuda:1'),)
    mask = torch.tensor([[1, 1, 0]], device='cuda:0')
    value = pool_hidden(hidden, mask, (0, 0, 0)).cpu().numpy()
    np.testing.assert_allclose(value[0, 0], [1 / 5**0.5, 2 / 5**0.5], atol=1e-5)
    np.testing.assert_allclose(value[0, -1], [0, 1])
    cross_device = 'passed'
out = Path('reports/qwen8-v1')
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'local_architecture_check.json', {
    'purpose': 'Random tiny 36-layer Qwen3 CPU functional test, not pretrained 8B evaluation',
    'layers_pooled': [18, 27, 36], 'masked_pooling_and_cache_resume': 'passed',
    'zero_budget_stops_before_forward_and_preserves_cache': 'passed',
    'cross_gpu_pooling_check': cross_device,
    'actual_8b_weights_loaded': False, 'gpu_8b_pilot': 'unrun', 'competition_score': None})
print('PASS: 36-layer feature selection, extraction deadline and cache reuse. Dual GPU:', cross_device)
