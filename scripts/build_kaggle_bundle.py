"""Package one trusted model and compatible wheels for offline Kaggle inference.

Run with the Python version/platform intended for Kaggle. This never uploads data
or models and never submits a notebook. Output must be a new directory.
"""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ntl.data import fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if fingerprint(args.artifact / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    wheels = args.out / "wheels"
    wheels.mkdir()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--wheel-dir", str(wheels.resolve()),
                    "--constraint", str(root / "requirements.lock"), str(root)], check=True)
    shutil.copy2(args.artifact / "model.joblib", args.out / "model.joblib")
    shutil.copy2(args.artifact / "manifest.json", args.out / "training_manifest.json")
    write_json(args.out / "ntl_bundle.json", {"python_minor": ".".join(platform.python_version().split(".")[:2]),
        "platform": platform.platform(), "package": f"name-that-llm=={importlib.metadata.version('name-that-llm')}",
        "files": {str(p.relative_to(args.out)): fingerprint(p) for p in args.out.rglob("*") if p.is_file()}})
    print(f"Bundle ready locally: {args.out}. Verify Kaggle Python compatibility before uploading.")


if __name__ == "__main__":
    main()
