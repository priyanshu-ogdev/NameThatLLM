"""Fetch or verify the pinned public MIT-licensed Kaggle encoder copy."""
import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://www.kaggle.com/models/happenmass/bge-small-en-v1.5/Transformers/bge/2/download"
ARCHIVE_SHA256 = "952f6b1903d5a00a38fcbe676fb3c6a454c15a3629a92b9ccd3860a9d97cfbb5"
FILES = {".gitattributes", "README.md", "config.json", "config_sentence_transformers.json",
         "model.safetensors", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json"}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Reuse a locally downloaded archive after checksum verification")
    args = parser.parse_args()
    if args.out.exists():
        provenance = json.loads((args.out / "provenance.json").read_text())
        if provenance["archive_sha256"] != ARCHIVE_SHA256 or set(provenance["files"]) != FILES:
            raise ValueError("Unexpected encoder provenance; refusing to overwrite")
        for relative, expected in provenance["files"].items():
            if checksum(args.out / relative) != expected:
                raise ValueError(f"Encoder checksum mismatch: {relative}")
        print(f"Existing encoder verified: {args.out}")
        return
    with tempfile.TemporaryDirectory(prefix="ntl-encoder-") as temporary:
        archive = args.archive or Path(temporary) / "encoder.tar.gz"
        if args.archive is None:
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("xb") as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
        if checksum(archive) != ARCHIVE_SHA256:
            raise ValueError("Archive checksum mismatch; no encoder files extracted")
        with tarfile.open(archive) as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            if not FILES <= set(members) or any(not members[name].isfile() for name in FILES):
                raise ValueError("Unexpected archive layout")
            args.out.mkdir(parents=True, exist_ok=False)
            for name in sorted(FILES):
                target = args.out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(members[name]) as source, target.open("xb") as out:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        out.write(chunk)
        provenance = {"kaggle_ref": "happenmass/bge-small-en-v1.5/transformers/bge/2",
            "license": "MIT", "archive_sha256": ARCHIVE_SHA256,
            "files": {name: checksum(args.out / name) for name in sorted(FILES)}}
        (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Verified encoder saved: {args.out}")


if __name__ == "__main__":
    main()
