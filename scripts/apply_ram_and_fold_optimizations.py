import re
from pathlib import Path

def apply_optimizations():
    target_path = Path("notebooks/kaggle_tri_ensemble.py")
    code = target_path.read_text(encoding="utf-8")

    # 1. Update get_optimizer_grouped_parameters to filter and p.requires_grad
    old_opt_block = """    # Head and pooler modules (receive lr_head)
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

    # Embeddings layer: decayed geometrically below bottom encoder layer to prevent memorizing rare prompt nouns
    n_layers = len(layers) if layers is not None else 24
    emb_lr = lr_backbone * (layer_decay ** n_layers)
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
        })"""

    new_opt_block = """    # Head and pooler modules (receive lr_head)
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
        p_decay = [p for n, p in head_params if not is_no_decay(n, p) and p.requires_grad]
        p_nodecay = [p for n, p in head_params if is_no_decay(n, p) and p.requires_grad]
        if p_decay:
            params_group.append({"params": p_decay, "lr": lr_head, "weight_decay": weight_decay})
        if p_nodecay:
            params_group.append({"params": p_nodecay, "lr": lr_head, "weight_decay": 0.0})
        for n, p in head_params:
            if p.requires_grad:
                assigned_param_ids.add(id(p))

    # Encoder layers with geometric decay (ξ = 0.95)
    if layers is not None:
        n_layers = len(layers)
        for i in reversed(range(n_layers)):
            layer = layers[i]
            decay_factor = layer_decay ** (n_layers - 1 - i)
            layer_lr = lr_backbone * decay_factor
            l_params = list(layer.named_parameters())
            p_decay = [p for n, p in l_params if not is_no_decay(n, p) and p.requires_grad]
            p_nodecay = [p for n, p in l_params if is_no_decay(n, p) and p.requires_grad]
            if p_decay:
                params_group.append({"params": p_decay, "lr": layer_lr, "weight_decay": weight_decay})
            if p_nodecay:
                params_group.append({"params": p_nodecay, "lr": layer_lr, "weight_decay": 0.0})
            for n, p in l_params:
                if p.requires_grad:
                    assigned_param_ids.add(id(p))

    # Embeddings layer: decayed geometrically below bottom encoder layer to prevent memorizing rare prompt nouns
    n_layers = len(layers) if layers is not None else 24
    emb_lr = lr_backbone * (layer_decay ** n_layers)
    emb_params = []
    if embeddings is not None:
        emb_params.extend(list(embeddings.named_parameters()))
    if encoder is not None and hasattr(encoder, "rel_embeddings"):
        emb_params.extend(list(encoder.rel_embeddings.named_parameters()))

    if emb_params:
        p_decay = [p for n, p in emb_params if not is_no_decay(n, p) and p.requires_grad]
        p_nodecay = [p for n, p in emb_params if is_no_decay(n, p) and p.requires_grad]
        if p_decay:
            params_group.append({"params": p_decay, "lr": emb_lr, "weight_decay": weight_decay})
        if p_nodecay:
            params_group.append({"params": p_nodecay, "lr": emb_lr, "weight_decay": 0.0})
        for n, p in emb_params:
            if p.requires_grad:
                assigned_param_ids.add(id(p))

    # Encoder final LayerNorm assigned to top layer LR
    if encoder is not None and hasattr(encoder, "LayerNorm"):
        ln_params = list(encoder.LayerNorm.named_parameters())
        p_ln = [p for n, p in ln_params if p.requires_grad]
        if p_ln:
            params_group.append({"params": p_ln, "lr": lr_backbone, "weight_decay": 0.0})
            for n, p in ln_params:
                if p.requires_grad:
                    assigned_param_ids.add(id(p))

    # Fallback for any leftover parameters (ensures zero frozen parameters accidentally included)
    leftover = [(n, p) for n, p in model.named_parameters() if id(p) not in assigned_param_ids and p.requires_grad]
    if leftover:
        p_decay = [p for n, p in leftover if not is_no_decay(n, p)]
        p_nodecay = [p for n, p in leftover if is_no_decay(n, p)]
        if p_decay:
            params_group.append({"params": p_decay, "lr": lr_backbone, "weight_decay": weight_decay})
        if p_nodecay:
            params_group.append({"params": p_nodecay, "lr": lr_backbone, "weight_decay": 0.0})"""

    assert old_opt_block in code, "Failed to find old_opt_block in code"
    code = code.replace(old_opt_block, new_opt_block)

    # 2. Add n_frozen_bottom_layers to train_transformer_fold
    old_tf_def = """    group_ids_tr=None,
    use_topic_masking=True,
    use_sift=True,
):"""
    new_tf_def = """    group_ids_tr=None,
    use_topic_masking=True,
    use_sift=True,
    n_frozen_bottom_layers=4,
):"""
    assert old_tf_def in code, "Failed to find old_tf_def"
    code = code.replace(old_tf_def, new_tf_def, 1)

    # Add freezing logic inside train_transformer_fold
    old_tf_freeze_anchor = """    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(dev)"""

    new_tf_freeze_anchor = """    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if n_frozen_bottom_layers > 0:
        base_m = getattr(model, "deberta", getattr(model, "bert", getattr(model, "model", None)))
        encoder = getattr(base_m, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            n_freeze = min(n_frozen_bottom_layers, len(layers))
            for l_idx in range(n_freeze):
                for p in layers[l_idx].parameters():
                    p.requires_grad = False
            print(f"  [{model_name}] Frozen bottom {n_freeze} layers (skips backward pass on lower syntax, prevents prompt memorization).", flush=True)

    model = model.to(dev)"""

    assert old_tf_freeze_anchor in code, "Failed to find old_tf_freeze_anchor"
    code = code.replace(old_tf_freeze_anchor, new_tf_freeze_anchor)

    # 3. Update train_qwen_lora_fold signature and body to support resident backbone reuse and CPU offload
    old_qwen_def = """    cached_tokenizer=None,
    group_ids_tr=None,
):
    \"\"\"
    LoRA fine-tune of Qwen 3 8B / 2.5 7B across dual GPUs (device_map="auto")."""

    new_qwen_def = """    cached_tokenizer=None,
    group_ids_tr=None,
    retain_backbone: bool = False,
):
    \"\"\"
    LoRA fine-tune of Qwen 3 8B / 2.5 7B across dual GPUs (device_map="auto").
    Supports zero-disk resident backbone reuse via Host RAM caching & PCIe staging."""

    assert old_qwen_def in code, "Failed to find old_qwen_def"
    code = code.replace(old_qwen_def, new_qwen_def)

    # Update backbone loading in train_qwen_lora_fold
    old_qwen_load = """    n_gpus = torch.cuda.device_count() if cuda_ok else 0
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
    backbone.print_trainable_parameters()"""

    new_qwen_load = """    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    qwen_desc = "Qwen 3 8B" if ("qwen-3" in str(model_dir).lower() or "qwen3" in str(model_dir).lower()) else "Qwen 2.5 7B"
    if "qwen_backbone" in GLOBAL_RAM_MODEL_CACHE:
        print(f"  [Qwen LoRA] Loading resident backbone from Host RAM cache (0 disk I/O, fast PCIe transfer)...", flush=True)
        backbone = GLOBAL_RAM_MODEL_CACHE["qwen_backbone"]
        dev_map = GLOBAL_RAM_MODEL_CACHE.get("qwen_device_map")
        if dev_map is not None and n_gpus >= 2:
            try:
                from accelerate import dispatch_model
                backbone = dispatch_model(backbone, device_map=dev_map)
            except Exception as e:
                print(f"  [Qwen LoRA] accelerate dispatch notice: {e}; moving to GPU", flush=True)
        elif cuda_ok:
            backbone = backbone.to("cuda:0")

        # Reset LoRA adapter parameters in 0.01s (clean Kaiming/zeros re-initialization)
        for name, module in backbone.named_modules():
            if hasattr(module, "reset_lora_parameters"):
                try:
                    module.reset_lora_parameters("default", init_lora_weights=True)
                except TypeError:
                    module.reset_lora_parameters("default")
        print("  [Qwen LoRA] Reset LoRA adapter weights to initial distribution in 0.01s [OK]", flush=True)
    else:
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
        if hasattr(backbone, "hf_device_map"):
            GLOBAL_RAM_MODEL_CACHE["qwen_device_map"] = copy.deepcopy(backbone.hf_device_map)"""

    assert old_qwen_load in code, "Failed to find old_qwen_load"
    code = code.replace(old_qwen_load, new_qwen_load)

    # Update backbone cleanup at the end of train_qwen_lora_fold
    old_qwen_cleanup = """    test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))

    del backbone, head, optimizer, scheduler, epoch_lora_states, epoch_head_states
    if "swa_lora" in locals():
        del swa_lora
    if "swa_head" in locals():
        del swa_head
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

    return best_val_logits, test_logits"""

    new_qwen_cleanup = """    test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))

    del head, optimizer, scheduler, epoch_lora_states, epoch_head_states
    if "swa_lora" in locals():
        del swa_lora
    if "swa_head" in locals():
        del swa_head
    if best_lora_state is not None:
        del best_lora_state
    if best_head_state is not None:
        del best_head_state

    if retain_backbone:
        # Offload backbone to host RAM so GPU 0 and GPU 1 are 100% free for DeBERTa/BGE next fold!
        # Host RAM has >20 GB free; moving Qwen (15.2 GB) to CPU RAM takes ~1.3s via PCIe
        # and eliminates 180s Kaggle FUSE network disk re-read!
        try:
            GLOBAL_RAM_MODEL_CACHE["qwen_backbone"] = backbone.to("cpu")
            print("  [Qwen LoRA] Offloaded resident backbone to Host RAM (Dual GPUs 100% free for next fold)", flush=True)
        except Exception as e:
            print(f"  [Qwen LoRA] Host RAM offload notice: {e}; keeping backbone in cache", flush=True)
            GLOBAL_RAM_MODEL_CACHE["qwen_backbone"] = backbone
        del backbone
    else:
        GLOBAL_RAM_MODEL_CACHE.pop("qwen_backbone", None)
        GLOBAL_RAM_MODEL_CACHE.pop("qwen_device_map", None)
        del backbone

    gc.collect()
    if cuda_ok:
        for gi in (first_dev, head_dev):
            if str(gi).startswith("cuda"):
                dev_idx = torch.device(gi).index if torch.device(gi).index is not None else 0
                with torch.cuda.device(dev_idx):
                    torch.cuda.synchronize(dev_idx)
                    torch.cuda.empty_cache()

    return best_val_logits, test_logits"""

    assert old_qwen_cleanup in code, "Failed to find old_qwen_cleanup"
    code = code.replace(old_qwen_cleanup, new_qwen_cleanup)

    # 4. In run_quad_ensemble, pass retain_backbone=(fold < n_splits - 1)
    old_call_qwen = """                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=qwen_bs, grad_accum=qwen_ga, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=1.8e-4, lora_r=48, lora_alpha=96, lora_dropout=0.08,
                    deadline=pipeline_deadline,
                    group_ids_tr=group_ids[tr_idx],
                    **qwen_kw,
                )"""

    new_call_qwen = """                retain_qwen = (fold < n_splits - 1)
                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=qwen_bs, grad_accum=qwen_ga, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=1.8e-4, lora_r=48, lora_alpha=96, lora_dropout=0.08,
                    deadline=pipeline_deadline,
                    group_ids_tr=group_ids[tr_idx],
                    retain_backbone=retain_qwen,
                    **qwen_kw,
                )"""

    assert old_call_qwen in code, "Failed to find old_call_qwen"
    code = code.replace(old_call_qwen, new_call_qwen)

    target_path.write_text(code, encoding="utf-8")
    print("Successfully updated notebooks/kaggle_tri_ensemble.py with RAM and fold optimizations!")

if __name__ == "__main__":
    apply_optimizations()

