import re

def update_file(path, old_sub, new_sub):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_sub in content:
        content = content.replace(old_sub, new_sub)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"Updated {path}")
    else:
        print(f"Warning: pattern not found in {path}")

# In dry run scripts, ensure DeBERTa and BGE have torch_dtype=torch.float32 and .to(torch.float32)
for script_path in ["notebooks/kaggle_dry_run.py", "scripts/kaggle_dry_run_verifier.py"]:
    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Update DeBERTa load in Check 3
    text = re.sub(
        r'(\[DeBERTa\] Loading weights directly onto GPU 0 \(device_map=\'cuda:0\'\)\.\.\.", flush=True\)\n\s+model_deb = AutoModelForSequenceClassification\.from_pretrained\(\n\s+deberta_dir,\n\s+num_labels=18,\n\s+ignore_mismatched_sizes=True,\n\s+local_files_only=True,\n\s+device_map="cuda:0",\n\s+\)\n\s+model_deb\.train\(\)',
        '''[DeBERTa] Loading weights directly onto GPU 0 (device_map='cuda:0', torch_dtype=float32)...", flush=True)
        model_deb = AutoModelForSequenceClassification.from_pretrained(
            deberta_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cuda:0",
        )
        model_deb.config.use_cache = False
        model_deb = model_deb.to(torch.float32)
        model_deb.train()''',
        text
    )

    # 2. Update BGE load in Check 4
    text = re.sub(
        r'(\[BGE\] Loading weights directly onto \{bge_dev_str\} \(device_map=\'\{bge_dev_str\}\'\)\.\.\.", flush=True\)\n\s+model_bge = AutoModelForSequenceClassification\.from_pretrained\(\n\s+bge_dir,\n\s+num_labels=18,\n\s+ignore_mismatched_sizes=True,\n\s+local_files_only=True,\n\s+device_map=bge_dev_str,\n\s+\)\n\s+model_bge\.train\(\)',
        '''[BGE] Loading weights directly onto {bge_dev_str} (device_map='{bge_dev_str}', torch_dtype=float32)...", flush=True)
        model_bge = AutoModelForSequenceClassification.from_pretrained(
            bge_dir,
            num_labels=18,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=bge_dev_str,
        )
        model_bge.config.use_cache = False
        model_bge = model_bge.to(torch.float32)
        model_bge.train()''',
        text
    )

    # 3. Update concurrent check worker
    text = re.sub(
        r'm = AutoModelForSequenceClassification\.from_pretrained\(\s*m_dir,\s*num_labels=18,\s*ignore_mismatched_sizes=True,\s*local_files_only=True,\s*device_map=dev_name\s*\)',
        'm = AutoModelForSequenceClassification.from_pretrained(m_dir, num_labels=18, ignore_mismatched_sizes=True, local_files_only=True, torch_dtype=torch.float32, device_map=dev_name).to(torch.float32)',
        text
    )

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Patched {script_path}")

