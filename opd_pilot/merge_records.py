"""Merge closed evidence shards offline without changing source evidence."""

import argparse
import json
import os
import shutil
from pathlib import Path

from opd_pilot.records import audit


def merge(source, destination):
    """Validate links and stream records into a separate, readable output directory.

    Run only after every producer has stopped. Source shards must remain immutable.
    One merge job owns the destination; failed staging directories are retained for
    inspection and never automatically deleted or overwritten.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directory trees")
    if destination.exists():
        raise FileExistsError(destination)
    summary = audit(source)
    shards = sorted(path for path in source.iterdir() if path.is_dir())
    staging = destination.with_name(destination.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    merged = staging / "merged"
    merged.mkdir()
    with (merged / "events.jsonl").open("xb") as output:
        for shard in shards:
            with (shard / "events.jsonl").open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    with (merged / "COMPLETE.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "records": sum(summary["records"].values())}, stream)
    merged_summary = audit(staging)
    if merged_summary["records"] != summary["records"]:
        raise ValueError("Merged record counts differ from the validated source")
    with (staging / "merge.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "source": str(source),
                "source_shards": [shard.name for shard in shards],
                "summary": summary,
                "ordering": "source shard name, then original line order; not chronological",
            },
            stream,
            indent=2,
        )
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    return merged_summary


def main():
    """Merge validated shards with bounded copying memory and no GPU imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
