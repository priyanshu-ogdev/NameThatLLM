"""Download the public official Kaggle Qwen copy with archive/weight verification."""
import argparse
import tarfile
import urllib.request
from pathlib import Path

from ntl.generative import WEIGHT_SHA256, sha256_file, write_json


ARCHIVE_SHA256 = "56c57b14bf9b9cc17cdb05c0366ba97fab5e566f71d3bfbf7ddfb256367f225b"
URL = "https://www.kaggle.com/api/v1/models/qwen-lm/qwen-3/transformers/1.7b-base/1/download"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive = args.archive or args.out.parent / "qwen3-kaggle-download.archive"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        if sha256_file(temporary) != ARCHIVE_SHA256:
            raise ValueError("Official model archive checksum changed; review provenance before using it")
        temporary.replace(archive)
    elif sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Existing archive failed checksum verification")
    args.out.mkdir(parents=True, exist_ok=True)
    expected = {"README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    if not (args.out / "model.safetensors").is_file():
        with tarfile.open(archive) as package:
            members = package.getmembers()
            if {member.name for member in members} != expected or any(not member.isfile() for member in members):
                raise ValueError("Unexpected model archive contents")
            package.extractall(args.out, filter="data")
    if sha256_file(args.out / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("Official model weights failed upstream checksum verification")
    hashes = {name: sha256_file(args.out / name) for name in sorted(expected)}
    write_json(args.out / "download_provenance.json", {"source": URL, "archive_sha256": ARCHIVE_SHA256,
                                                      "license": "Apache-2.0", "file_sha256": hashes})
    print("Verified official model:", args.out)


if __name__ == "__main__":
    main()
