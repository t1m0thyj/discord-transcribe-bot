#!/usr/bin/env python3
"""
Download an ONNX model from Hugging Face into a local `models/` subfolder.

Usage:
    python download_model.py <repo_id> [--files FILE1 FILE2 ...] [--quant SUBSTRING] [--out models] [--revision main]

Examples:
    # Download the whole repo (all files, e.g. encoder/decoder/tokenizer)
    python download_model.py Mazino0/moonshine-streaming-medium-onnx

    # Download only specific files instead of the whole repo
    python download_model.py Mazino0/moonshine-streaming-medium-onnx \\
        --files encoder_model_int8.onnx decoder_model_int8.onnx tokenizer.json

    # Download everything matching a quantization variant, without listing exact filenames
    # (quantization isn't a separate HF parameter -- it's just part of the filename,
    # e.g. encoder_model_int8.onnx vs encoder_model_fp16.onnx vs the plain fp32 name)
    python download_model.py Mazino0/moonshine-streaming-medium-onnx --quant int8

Requires:
    pip install huggingface_hub
"""
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    sys.exit("Missing dependency. Install with: pip install huggingface_hub")


def download_full_repo(repo_id: str, out_dir: Path, revision: str, quant: str = None) -> Path:
    dest = out_dir / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    if quant:
        # Non-onnx metadata files (tokenizer, config) are always fetched too, since
        # they're required alongside whichever quantized weights you pick and rarely
        # come in multiple variants themselves.
        allow_patterns = [f"*{quant}*", "*.json", "*.txt"]
        print(f"Downloading repo '{repo_id}' (quant match: '{quant}') -> {dest}")
    else:
        allow_patterns = None
        print(f"Downloading full repo '{repo_id}' -> {dest}")
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest,
        allow_patterns=allow_patterns,
    )
    return Path(local_path)


def download_files(repo_id: str, filenames: list, out_dir: Path, revision: str) -> Path:
    dest = out_dir / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print(f"Downloading '{filename}' from '{repo_id}' -> {dest}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=dest,
        )
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Download an ONNX model from Hugging Face into a local models/ folder."
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repo id, e.g. Mazino0/moonshine-streaming-medium-onnx",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific filenames to download instead of the whole repo (default: download everything)",
    )
    parser.add_argument(
        "--quant",
        default=None,
        help=(
            "Substring to match against filenames, e.g. 'int8' or 'fp16'. "
            "Useful for picking one quantization variant out of a repo that hosts several, "
            "without listing every filename by hand. Ignored if --files is given."
        ),
    )
    parser.add_argument(
        "--out",
        default="models",
        help="Base output directory; a subfolder named after the repo is created inside it (default: ./models)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision, branch, or tag to download from (default: main)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.files:
            dest = download_files(args.repo_id, args.files, out_dir, args.revision)
        else:
            dest = download_full_repo(args.repo_id, out_dir, args.revision, args.quant)
    except Exception as e:
        sys.exit(f"Download failed: {e}")

    print(f"\nDone. Model files are in: {dest.resolve()}")
    print("Contents:")
    for f in sorted(dest.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
