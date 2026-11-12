with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text_ens = f.read()

with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    text_dry = f.read()

marker = "def run_quad_ensemble():"
ens_shared = text_ens[:text_ens.index(marker)]
dry_shared = text_dry[:text_dry.index("def run_dry_run_verifier():")]

print(f"Shared code length in tri_ensemble: {len(ens_shared)} chars, {len(ens_shared.splitlines())} lines")
print(f"Shared code length in dry_run:      {len(dry_shared)} chars, {len(dry_shared.splitlines())} lines")
print(f"Exact byte match: {ens_shared == dry_shared}")

if ens_shared != dry_shared:
    ens_lines = ens_shared.splitlines()
    dry_lines = dry_shared.splitlines()
    for i, (l1, l2) in enumerate(zip(ens_lines, dry_lines)):
        if l1 != l2:
            print(f"First mismatch at line {i+1}:")
            print(f"  ens: {repr(l1)}")
            print(f"  dry: {repr(l2)}")
            break

