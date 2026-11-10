def patch_script(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    
    # Patch 1: DeBERTa load
    old_deb = """        model_deb = AutoModelForSequenceClassification.from_pretrained(
            deberta_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            device_map="cuda:0",
        )
        model_deb.train()"""
    new_deb = """        model_deb = AutoModelForSequenceClassification.from_pretrained(
            deberta_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cuda:0",
        )
        model_deb.config.use_cache = False
        model_deb = model_deb.to(torch.float32)
        model_deb.train()"""
    assert old_deb in text, f"old_deb not found in {path}"
    text = text.replace(old_deb, new_deb)

    # Patch 2: BGE load
    old_bge = """        model_bge = AutoModelForSequenceClassification.from_pretrained(
            bge_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            device_map=bge_dev_str,
        )
        model_bge.train()"""
    new_bge = """        model_bge = AutoModelForSequenceClassification.from_pretrained(
            bge_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=bge_dev_str,
        )
        model_bge.config.use_cache = False
        model_bge = model_bge.to(torch.float32)
        model_bge.train()"""
    assert old_bge in text, f"old_bge not found in {path}"
    text = text.replace(old_bge, new_bge)

    # Patch 3: Concurrent check worker
    old_worker = """                m = AutoModelForSequenceClassification.from_pretrained(
                    m_dir, num_labels=18, ignore_mismatched_sizes=True, local_files_only=True, device_map=dev_name
                )"""
    new_worker = """                m = AutoModelForSequenceClassification.from_pretrained(
                    m_dir, num_labels=18, ignore_mismatched_sizes=True, local_files_only=True, torch_dtype=torch.float32, device_map=dev_name
                ).to(torch.float32)"""
    assert old_worker in text, f"old_worker not found in {path}"
    text = text.replace(old_worker, new_worker)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Successfully patched {path}")

patch_script("notebooks/kaggle_dry_run.py")
patch_script("scripts/kaggle_dry_run_verifier.py")

