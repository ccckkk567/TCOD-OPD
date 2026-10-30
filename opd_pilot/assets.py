"""Download reusable model assets with fixed revisions and resumable metadata."""

import argparse
import fnmatch
import json
from pathlib import Path

MODELS = {
    "s17": "Qwen/Qwen3-1.7B",
    "s04": "Qwen/Qwen3-4B",
    "t08": "Qwen/Qwen3-8B",
    "t30": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}
PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja")


def write_json(path, value):
    """Publish metadata only after its complete contents have been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_model(directory, manifest):
    """Check expected sizes and all referenced weight shards without hashing large weights."""
    files = manifest["files"]
    for name, size in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or incomplete model file: {path}")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if name not in files:
            raise ValueError(f"Required Qwen metadata not listed: {name}")
        json.loads((directory / name).read_text(encoding="utf-8"))
    index_path = directory / "model.safetensors.index.json"
    if "model.safetensors.index.json" in files:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        if not shards or not shards.issubset(files):
            raise ValueError("Weight index references unlisted shards")
    elif "model.safetensors" not in files:
        raise ValueError("No model.safetensors or weight index")


def download_model(key, models_root, api=None, download=None, validate_only=False):
    """Reuse verified assets; resume partial downloads at their original revision."""
    repo_id = MODELS[key]
    directory = Path(models_root) / repo_id.split("/")[1]
    manifest_path = directory / "asset.json"
    ready_path = directory / "READY.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["repo_id"] != repo_id:
            raise ValueError(f"Model identity conflict in {directory}")
        if ready_path.exists() or validate_only:
            validate_model(directory, manifest)
            ready = json.loads(ready_path.read_text()) if ready_path.exists() else None
            if ready != manifest:
                raise ValueError("Missing or inconsistent READY metadata")
            return directory
    else:
        if validate_only:
            raise ValueError(f"No verified asset manifest: {manifest_path}")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                f"Existing unmanaged model at {directory}; inspect its provenance before reuse. "
                "No files were overwritten or downloaded."
            )
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        info = api.model_info(repo_id, revision="main", files_metadata=True, timeout=30)
        files = {
            f.rfilename: f.size
            for f in info.siblings
            if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in PATTERNS)
        }
        if not info.sha or not files or any(not isinstance(size, int) for size in files.values()):
            raise ValueError("Hub response missing revision or file sizes")
        manifest = {"repo_id": repo_id, "revision": info.sha, "files": files}
        directory.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    if download is None:
        from huggingface_hub import snapshot_download

        download = snapshot_download
    download(
        repo_id=repo_id,
        revision=manifest["revision"],
        local_dir=str(directory),
        allow_patterns=list(manifest["files"]),
        max_workers=4,
    )
    validate_model(directory, manifest)
    write_json(ready_path, manifest)
    return directory


def main():
    """Serialize downloads of the same shared model on the Linux cluster."""
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", choices=MODELS, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    lock_dir = args.models_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{MODELS[args.key].split('/')[1]}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = download_model(args.key, args.models_root, validate_only=args.validate_only)
    print(f"Model verified: {path}")


if __name__ == "__main__":
    main()
