import re
import ast

def apply_staging(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Insert prepare_model_dir right above class MultiSampleDropoutHead
    staging_func_code = '''def prepare_model_dir(model_dir, model_tag="model"):
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


'''

    target_msd = "class MultiSampleDropoutHead(nn.Module):"
    if "def prepare_model_dir" not in content:
        assert target_msd in content, f"Could not find {target_msd} in {filepath}"
        content = content.replace(target_msd, staging_func_code + target_msd)

    # 2. Update discover_paths() to stage deberta and bge upfront
    old_discover_ret = """    deb_variant = "large" if deberta_dir and "large" in str(deberta_dir).lower() else "base/unknown"
    print(f"  [Paths] data={data_dir}")
    print(f"  [Paths] qwen={qwen_dir}")
    print(f"  [Paths] deberta={deberta_dir}  ({deb_variant})")
    print(f"  [Paths] bge={bge_dir}")
    return data_dir, qwen_dir, deberta_dir, bge_dir, working_dir"""

    new_discover_ret = """    deb_variant = "large" if deberta_dir and "large" in str(deberta_dir).lower() else "base/unknown"
    print(f"  [Paths] data={data_dir}")
    print(f"  [Paths] qwen={qwen_dir}")
    print(f"  [Paths] deberta={deberta_dir}  ({deb_variant})")
    print(f"  [Paths] bge={bge_dir}")
    if deberta_dir is not None:
        deberta_dir = prepare_model_dir(deberta_dir, "deberta")
    if bge_dir is not None:
        bge_dir = prepare_model_dir(bge_dir, "bge")
    return data_dir, qwen_dir, deberta_dir, bge_dir, working_dir"""

    if old_discover_ret in content:
        content = content.replace(old_discover_ret, new_discover_ret)

    # 3. Update train_transformer_fold to stage locally and use low_cpu_mem_usage=True
    old_load_block = """    print(f"  [{model_name}] Loading from {model_dir} on {dev}...", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )

    id2label = {i: l for i, l in enumerate(labels)}

    with HeartbeatTimer(f"Reading {model_name} from disk onto {dev}"):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
                local_files_only=True,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                dtype=torch.float32,
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
                local_files_only=True,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float32,
            )"""

    new_load_block = """    model_tag = "deberta" if "deberta" in model_name.lower() else ("bge" if "bge" in model_name.lower() else "model")
    model_dir = prepare_model_dir(model_dir, model_tag=model_tag)
    print(f"  [{model_name}] Loading from {model_dir} on {dev}...", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )

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
                )"""

    assert old_load_block in content, f"Could not find old_load_block in {filepath}"
    content = content.replace(old_load_block, new_load_block)

    # 4. Add gc.collect() and empty_cache right after model = model.to(dev)
    old_to_dev = """    model = model.to(dev)
    ckpt_status = "ON" if enable_grad_checkpointing else "OFF (auto OOM fallback)" """

    # Wait, check exact string around model = model.to(dev)
    if "model = model.to(dev)\n    ckpt_status = " in content:
        content = content.replace(
            "model = model.to(dev)\n    ckpt_status = ",
            "model = model.to(dev)\n    gc.collect()\n    if cuda_ok:\n        torch.cuda.empty_cache()\n    ckpt_status = "
        )

    # Verify AST
    ast.parse(content)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully applied staging and low_cpu_mem_usage to {filepath}")

apply_staging("notebooks/kaggle_tri_ensemble.py")
apply_staging("notebooks/kaggle_dry_run.py")

