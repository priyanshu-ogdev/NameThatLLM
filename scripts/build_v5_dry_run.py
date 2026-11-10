from pathlib import Path

def build_v5_dry_run():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    ens_content = ens_path.read_text(encoding="utf-8")

    marker = "def run_quad_ensemble():"
    shared_part = ens_content[:ens_content.index(marker)]

    verifier_code = Path("scripts/dry_run_verifier_template.py").read_text(encoding="utf-8")
    dry_run_code = shared_part + verifier_code

    with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(dry_run_code)
    print("Updated notebooks/kaggle_dry_run.py successfully with all 9 throughput and exception checks.")

if __name__ == "__main__":
    build_v5_dry_run()
