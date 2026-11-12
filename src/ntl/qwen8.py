"""Pinned, offline two-GPU Qwen3-8B-Base experiment with a bounded pilot."""
import importlib.metadata
import importlib.util
import hashlib
import json
import time
from pathlib import Path

from ntl.generative import extract_frozen, sha256_file, write_json
from ntl.combined import run_combined


QWEN8_REVISION = '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'
QWEN8_SHARDS = {
    'model-00001-of-00005.safetensors': '9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b',
    'model-00002-of-00005.safetensors': '9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f',
    'model-00003-of-00005.safetensors': 'c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc',
    'model-00004-of-00005.safetensors': 'ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe',
    'model-00005-of-00005.safetensors': 'fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550',
}
EXPERIMENT_BUDGET_SECONDS = 2 * 60 * 60


def discover_qwen8(input_root):
    matches = []
    for path in Path(input_root).rglob('config.json'):
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (config.get('model_type') == 'qwen3' and config.get('hidden_size') == 4096
                and config.get('num_hidden_layers') == 36 and '8b-base' in str(path).lower()):
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError('Attach exactly one qwen-lm/qwen-3/transformers/8b-base/1 input, not the 1.7B or instruct model.')
    return matches[0]


def validate_shard_index(model_dir):
    index = json.loads((Path(model_dir) / 'model.safetensors.index.json').read_text())
    files = set(index.get('weight_map', {}).values())
    if files != set(QWEN8_SHARDS):
        raise ValueError('Unexpected Qwen8 shard index; expected all five pinned upstream shards.')


def gpu_memory_plan(devices):
    """Keep 3 GiB/device for activations, pooling and CUDA overhead."""
    if len(devices) < 2 or any(free < 12 * 2**30 for free in devices[:2]):
        raise RuntimeError('This experiment requires GPU T4 ×2, with at least 12 GiB free on each GPU. Restart a clean two-GPU session.')
    return {i: int(devices[i] - 3 * 2**30) for i in range(2)}


def validate_device_map(device_map):
    values = list(device_map.values())
    if not values or any(value not in (0, 1, 'cuda:0', 'cuda:1') for value in values):
        raise RuntimeError('Model would require CPU/disk offload; stop and share the memory report instead of running slowly.')


def pilot_projection(elapsed, seconds, pilot_rows, train_rows, hidden_rows=4000):
    if pilot_rows <= 0 or seconds <= 0:
        raise ValueError('Pilot timing must be positive')
    # Longest passages give a deliberately conservative extraction estimate.
    return elapsed + seconds / pilot_rows * (train_rows + hidden_rows) + 30 * 60


def load_qwen8(model_dir, allow_cpu=False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    if allow_cpu or torch.cuda.device_count() < 2:
        raise RuntimeError('Select GPU T4 ×2. Full 8B CPU extraction is deliberately disabled.')
    if importlib.util.find_spec('accelerate') is None:
        raise RuntimeError('This Kaggle runtime lacks Accelerate. Share this error; compatible offline dependencies must be attached.')
    free = [torch.cuda.mem_get_info(i)[0] for i in range(2)]
    plan = gpu_memory_plan(free)
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    print('GPUs:', names, 'free GiB:', [round(x / 2**30, 2) for x in free], flush=True)
    model_dir = Path(model_dir)
    validate_shard_index(model_dir)
    for filename, expected_blob in {'config.json': 'c5c5b1c67be484bcdb225c060f942736988c3c66',
                                    'model.safetensors.index.json': '041fe9a0d3ab9c8895f264e7a3a60aa563398d7c'}.items():
        data = (model_dir / filename).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != expected_blob:
            raise ValueError('Pinned upstream configuration/index mismatch: ' + filename)
    for name, expected in QWEN8_SHARDS.items():
        if sha256_file(model_dir / name) != expected:
            raise ValueError('Upstream weight checksum mismatch: ' + name)
        print('Verified weight shard:', name, flush=True)
    resources = {name: sha256_file(model_dir / name) for name in
                 ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors.index.json')}
    native_bf16 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(2))
    dtype = torch.bfloat16 if native_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    version = tuple(int(part) for part in importlib.metadata.version('transformers').split('.')[:2])
    dtype_option = {'dtype': dtype} if version >= (4, 56) else {'torch_dtype': dtype}
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation='sdpa',
                                     device_map='balanced', max_memory=plan, **dtype_option)
    validate_device_map(model.hf_device_map)
    model.requires_grad_(False).eval()
    model._ntl_started = started
    provenance = {'repo': 'Qwen/Qwen3-8B-Base', 'upstream_weight_revision': QWEN8_REVISION,
                  'source': 'qwen-lm/qwen-3/transformers/8b-base/1', 'license': 'Apache-2.0',
                  'weight_sha256': QWEN8_SHARDS, 'resource_sha256': resources, 'dtype': str(dtype),
                  'device': 'two_gpu_balanced', 'device_map': dict(model.hf_device_map),
                  'gpu_names': names, 'seed': 42, 'quantization': None,
                  'packages': {name: importlib.metadata.version(name) for name in
                               ('torch', 'transformers', 'accelerate', 'numpy', 'pandas', 'scikit-learn', 'safetensors', 'pyarrow')}}
    print('Frozen model loaded; layer pools: 18, 27, 36, final token. dtype:', dtype, flush=True)
    return model, tokenizer, provenance


def extract_qwen8(frame, model, tokenizer, provenance, cache_root):
    import torch

    out = Path(cache_root).parent
    out.mkdir(parents=True, exist_ok=True)
    if not getattr(model, '_ntl_pilot_done', False):
        write_json(out / 'provenance.json', provenance)
        lengths = [len(ids) for ids in tokenizer(frame.text.tolist(), add_special_tokens=False)['input_ids']]
        indices = sorted(range(len(frame)), key=lambda i: lengths[i], reverse=True)[:16]
        pilot_frame = frame.iloc[indices].reset_index(drop=True)
        for device in range(2):
            torch.cuda.reset_peak_memory_stats(device)
        before = time.perf_counter()
        # Fresh tiny cache ensures pilot measures real computation on every run.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='qwen8-pilot-', dir=out) as temporary:
            vectors, info = extract_frozen(pilot_frame, model, tokenizer, provenance, temporary,
                                           max_seconds=10 * 60)
        seconds = time.perf_counter() - before
        elapsed = time.perf_counter() - model._ntl_started
        projection = pilot_projection(elapsed, seconds, len(pilot_frame), len(frame))
        report = {'pilot_rows': len(pilot_frame), 'pilot_seconds': seconds, 'elapsed_seconds': elapsed,
                  'projected_total_seconds_with_4000_test_rows': projection,
                  'budget_seconds': EXPERIMENT_BUDGET_SECONDS,
                  'peak_allocated_gib': [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)],
                  'note': 'Conservative longest-passage extraction extrapolation plus 30-minute classifier reserve; not measured hidden-test runtime.',
                  'passed': projection <= EXPERIMENT_BUDGET_SECONDS}
        write_json(out / 'pilot.json', report)
        print('PILOT RESULT:', json.dumps(report), flush=True)
        if not report['passed']:
            raise RuntimeError('Pilot exceeds the two-hour experiment budget. Stop this GPU session and share pilot.json; no full extraction has started.')
        model._ntl_pilot_done = True
    remaining = EXPERIMENT_BUDGET_SECONDS - (time.perf_counter() - model._ntl_started)
    if remaining <= 0:
        raise TimeoutError('Experiment budget reached; feature checkpoints are retained.')
    return extract_frozen(frame, model, tokenizer, provenance, cache_root, max_seconds=remaining)


def run_qwen8():
    return run_combined(model_loader=load_qwen8, model_discovery=discover_qwen8,
                        feature_extractor=extract_qwen8, experiment_name='qwen8-v1', cache_name='qwen8-v1/cache')
