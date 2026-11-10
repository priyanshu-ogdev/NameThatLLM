import ast

print("Applying Qwen 3 8B Priority & Multi-Candidate Discovery...")

with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    ens_text = f.read()

with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    dry_text = f.read()

old_disco_init = '''    data_dir = None
    qwen_dir = None
    bge_dir = None
    deberta_candidates = []'''

new_disco_init = '''    data_dir = None
    qwen_dir = None
    bge_dir = None
    qwen_candidates = []
    deberta_candidates = []'''

old_disco_qwen = '''            is_qwen = "qwen" in mtype or "qwen" in p_str or "qwen" in arch
            if is_qwen and qwen_dir is None:
                qwen_dir = p.parent'''

new_disco_qwen = '''            is_qwen = "qwen" in mtype or "qwen" in p_str or "qwen" in arch
            if is_qwen:
                # Priority: Qwen 3 (tier 0) > Qwen 2.5 (tier 1) > other (tier 2)
                # Size: 8b/7b (size 0) > other (size 1)
                tier = 0 if ("qwen-3" in p_str or "qwen3" in p_str) else (1 if ("qwen2.5" in p_str or "qwen-2.5" in p_str) else 2)
                size = 0 if ("8b" in p_str or "7b" in p_str) else 1
                qwen_candidates.append(((tier, size, len(p.parts)), p.parent))'''

old_disco_sort = '''    deberta_dir = None
    if deberta_candidates:
        deberta_candidates.sort(key=lambda x: x[0])
        deberta_dir = deberta_candidates[0][1]

    deb_variant = "large" if deberta_dir and "large" in str(deberta_dir).lower() else "base/unknown"
    print(f"  [Paths] data={data_dir}")
    print(f"  [Paths] qwen={qwen_dir}")
    print(f"  [Paths] deberta={deberta_dir}  ({deb_variant})")'''

new_disco_sort = '''    if qwen_candidates:
        qwen_candidates.sort(key=lambda x: x[0])
        qwen_dir = qwen_candidates[0][1]

    deberta_dir = None
    if deberta_candidates:
        deberta_candidates.sort(key=lambda x: x[0])
        deberta_dir = deberta_candidates[0][1]

    deb_variant = "large" if deberta_dir and "large" in str(deberta_dir).lower() else "base/unknown"
    qwen_variant = "qwen-3-8b" if qwen_dir and ("qwen-3" in str(qwen_dir).lower() or "qwen3" in str(qwen_dir).lower()) else ("qwen-2.5-7b" if qwen_dir and "2.5" in str(qwen_dir).lower() else "custom/unknown")
    print(f"  [Paths] data={data_dir}")
    print(f"  [Paths] qwen={qwen_dir}  ({qwen_variant})")
    print(f"  [Paths] deberta={deberta_dir}  ({deb_variant})")'''

for label, target in [
    ("old_disco_init", old_disco_init),
    ("old_disco_qwen", old_disco_qwen),
    ("old_disco_sort", old_disco_sort),
]:
    assert target in ens_text, f"Target {label} not found in ens_text"
    assert target in dry_text, f"Target {label} not found in dry_text"

ens_text = ens_text.replace(old_disco_init, new_disco_init, 1)
ens_text = ens_text.replace(old_disco_qwen, new_disco_qwen, 1)
ens_text = ens_text.replace(old_disco_sort, new_disco_sort, 1)

dry_text = dry_text.replace(old_disco_init, new_disco_init, 1)
dry_text = dry_text.replace(old_disco_qwen, new_disco_qwen, 1)
dry_text = dry_text.replace(old_disco_sort, new_disco_sort, 1)

ast.parse(ens_text)
ast.parse(dry_text)

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(ens_text)

with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8") as f:
    f.write(dry_text)

print("Applied Qwen 3 multi-candidate discovery successfully!")

