"""Download the official WebShop small dataset without installing its simulator."""

import argparse
import json
from pathlib import Path

FILES = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}
MIRROR = "HongbangYuan/webshop"
MIRROR_REVISION = "0129d4a81dbdb827e76afd20a1e2c38b61098613"


def download(root, source="official"):
    """Community files are explicitly restricted to the approved small simulator smoke."""
    root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    previous = root / "asset.json"
    old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    if source == "official" and old.get("usage_scope") == "small_environment_smoke_only":
        raise ValueError("Do not relabel community smoke assets as official data")
    for name, file_id in FILES.items():
        target = root / name
        if target.exists():
            with target.open(encoding="utf-8") as stream:
                json.load(stream)
            provenance[name] = old.get("provenance", {}).get(
                name, {"source": "existing_local_unverified"}
            )
            continue
        if source == "hf-smoke":
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=MIRROR,
                filename=name,
                repo_type="dataset",
                revision=MIRROR_REVISION,
                local_dir=str(root),
            )
            with target.open(encoding="utf-8") as stream:
                json.load(stream)
            provenance[name] = dict(source=MIRROR, revision=MIRROR_REVISION)
        else:
            import gdown

            temporary = root / (name + ".partial")
            result = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
            if result is None:
                raise RuntimeError(f"Download failed: {name}")
            with temporary.open(encoding="utf-8") as stream:
                json.load(stream)
            temporary.replace(target)
            provenance[name] = dict(source="official_google_drive", file_id=file_id)
    manifest = dict(
        source="princeton-nlp/webshop setup.sh" if source == "official" else MIRROR,
        variant="small",
        files=FILES,
        provenance=provenance,
        search_index_ready=False,
        usage_scope="small_environment_smoke_only" if source == "hf-smoke" else "official_download",
        official_byte_equivalence_verified=False,
    )
    (root / "asset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    """Reuse valid JSON files and publish downloads only after parsing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", choices=("official", "hf-smoke"), default="official")
    args = parser.parse_args()
    print(json.dumps(download(args.root, args.source), indent=2))


if __name__ == "__main__":
    main()
