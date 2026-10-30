"""Download the official WebShop small dataset without installing its simulator."""

import json
import sys
from pathlib import Path

import gdown

FILES = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}


def main():
    """Reuse valid JSON files and publish downloads only after parsing."""
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        target = root / name
        if target.exists():
            with target.open(encoding="utf-8") as stream:
                json.load(stream)
            continue
        temporary = root / (name + ".partial")
        result = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
        if result is None:
            raise RuntimeError(f"Download failed: {name}")
        with temporary.open(encoding="utf-8") as stream:
            json.load(stream)
        temporary.replace(target)
    with (root / "asset.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "source": "princeton-nlp/webshop setup.sh",
                "variant": "small",
                "files": FILES,
                "search_index_ready": False,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
