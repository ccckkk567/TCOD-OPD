"""Build reusable WebShop indexes with the repository's own conversion/index scripts."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def link_existing_or_new(link, target):
    """Never replace an existing data directory or an unrelated link."""
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError(f"Existing path needs inspection: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def validate(spec):
    repo, data = Path(spec["repo"]).resolve(), Path(spec["data_root"]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != spec["revision"]:
        raise ValueError("WebShop checkout revision differs from spec")
    files = {}
    for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
        path = data / name
        json.loads(path.read_text(encoding="utf-8"))
        files[name] = dict(bytes=path.stat().st_size, modified_ns=path.stat().st_mtime_ns)
    return repo, data, revision, files


def prepare(spec):
    repo, data, revision, files = validate(spec)
    for name in files:
        path = data / name
        link_existing_or_new(repo / "data" / name, path)
    search = data / "search_engine"
    search.mkdir(parents=True, exist_ok=True)
    marker = search / "INDEX_READY.json"
    if marker.exists():
        result = json.loads(marker.read_text(encoding="utf-8"))
        if result["source_revision"] != revision or result.get("data_files") != files:
            raise ValueError("Existing index uses different source/data; inspect before rebuilding")
    else:
        # Original converter creates the same four corpus views as setup.sh.
        for suffix in ("", "_100", "_1k", "_100k"):
            (search / ("resources" + suffix)).mkdir(exist_ok=True)
            index = search / ("indexes" + suffix)
            if index.exists():
                raise ValueError(
                    f"Partial/unmanaged index exists: {index}; inspect before rebuilding"
                )
        env = {**os.environ, "PYTHONPATH": str(repo)}
        subprocess.run(
            [sys.executable, str(repo / "search_engine/convert_product_file_format.py")],
            cwd=search,
            env=env,
            check=True,
        )
        subprocess.run(
            ["bash", str(repo / "search_engine/run_indexing.sh")], cwd=search, env=env, check=True
        )
        result = dict(source_revision=revision, variant="small", data_files=files, indexes={})
    from pyserini.search.lucene import LuceneSearcher

    for suffix in ("", "_100", "_1k", "_100k"):
        name = "indexes" + suffix
        index = search / name
        reader = LuceneSearcher(str(index))
        try:
            count = reader.num_docs
            if count <= 0:
                raise ValueError("Empty search index")
            result["indexes"][name] = count
        finally:
            reader.close()
        link_existing_or_new(repo / "search_engine" / name, index)
    marker.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.validate_only:
        _, _, revision, files = validate(spec)
        print(json.dumps(dict(source_revision=revision, data_files=files), indent=2))
    else:
        # The GPU preparation stage serializes use of this shared resource.
        import fcntl

        with (Path(spec["data_root"]) / "index-preparation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(prepare(spec), indent=2))


if __name__ == "__main__":
    main()
