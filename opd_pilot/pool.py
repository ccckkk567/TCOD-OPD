"""Freeze a common diagnostic prefix pool from already selected student dev banks."""

import argparse
import json
import random
from pathlib import Path

from opd_pilot.evidence import publish, separate_output


def combine(spec):
    sources, groups = [], []
    for directory in spec["banks"]:
        root = Path(directory)
        source_spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
        if source_spec["split"] != "analysis_dev":
            raise ValueError("Common exploratory pools cannot use train or confirm inputs")
        bank = json.loads((root / "prefix_bank.json").read_text(encoding="utf-8"))
        rows = list(bank["prefixes"])
        if not rows or len({row["prefix_uid"] for row in rows}) != len(rows):
            raise ValueError("Require nonempty unique prefixes per source")
        sources.append(dict(bank=str(root.resolve()), records_root=source_spec["records_root"]))
        groups.append(rows)
    if not groups or len(set(spec["banks"])) != len(groups):
        raise ValueError("Require distinct source banks")
    limit = spec["max_prefixes"]
    if type(limit) is not int or limit < 1:
        raise ValueError("max_prefixes must be positive")
    rng = random.Random(spec["seed"])
    for rows in groups:
        rng.shuffle(rows)
    selected = []
    for depth in range(max(map(len, groups))):
        for source, rows in zip(sources, groups):
            if depth < len(rows) and len(selected) < limit:
                row = rows[depth]
                selected.append(
                    {
                        **row,
                        "source_prefix_uid": row["prefix_uid"],
                        "source_bank": source["bank"],
                        "records_root": source["records_root"],
                        "prefix_uid": f"common-prefix-{len(selected) + 1:05d}",
                    }
                )
    return dict(
        schema_version=1,
        split="analysis_dev",
        selection="seeded_source_balanced_union",
        seed=spec["seed"],
        sources=sources,
        prefixes=selected,
        note="Equal text does not establish equal simulator state; state identities remain distinct.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], *spec["banks"])
    bank = combine(spec)
    if not args.validate_only:
        publish(
            output,
            {
                "prefix_bank.json": bank,
                "spec.json": spec,
                "status.json": dict(status="completed", prefixes=len(bank["prefixes"])),
            },
        )
    print(json.dumps(dict(prefixes=len(bank["prefixes"]))))


if __name__ == "__main__":
    main()
