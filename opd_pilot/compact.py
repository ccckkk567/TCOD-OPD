"""Losslessly archive record arrays in NPZ and deduplicate stored message payloads."""

import argparse
import json
from pathlib import Path

from opd_pilot.evidence import read_records, separate_output
from opd_pilot.records import Recorder, audit

ARRAY_FIELDS = {"tokens", "student_logprobs", "teacher_logprobs", "mask", "advantage"}


def pack(source, destination):
    import numpy as np

    records = read_records(source)
    output = separate_output(destination, source)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    arrays, messages, message_ids, rows = {}, {}, {}, []
    for kind, items in records.items():
        for original in items:
            row = dict(original)
            references = {}
            for key in ARRAY_FIELDS & row.keys():
                value = row[key]
                if value is None:
                    continue
                array = np.asarray(value)
                if array.dtype.kind not in "biuf":
                    raise ValueError("Only rectangular numeric arrays can be compacted")
                name = f"array-{len(arrays) + 1:08d}"
                arrays[name] = array
                references[key] = name
                del row[key]
            message_ref = None
            if "messages" in row:
                serialized = json.dumps(row.pop("messages"), ensure_ascii=False, allow_nan=False)
                if serialized not in message_ids:
                    name = f"messages-{len(messages) + 1:08d}"
                    message_ids[serialized] = name
                    messages[name] = json.loads(serialized)
                message_ref = message_ids[serialized]
            rows.append(dict(kind=kind, evidence=row, arrays=references, messages=message_ref))
    np.savez_compressed(staging / "arrays.npz", **arrays)
    (staging / "records.json").write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (staging / "messages.json").write_text(
        json.dumps(messages, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    metadata = dict(
        schema_version=1,
        records=len(rows),
        arrays=len(arrays),
        unique_messages=len(messages),
        source_audit=audit(source),
    )
    (staging / "COMPLETE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    staging.rename(output)
    return metadata


def unpack(source, destination):
    import numpy as np

    source = Path(source)
    metadata = json.loads((source / "COMPLETE.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError("Unknown compact schema")
    rows = json.loads((source / "records.json").read_text(encoding="utf-8"))
    messages = json.loads((source / "messages.json").read_text(encoding="utf-8"))
    if len(rows) != metadata["records"] or len(messages) != metadata["unique_messages"]:
        raise ValueError("Compact metadata count mismatch")
    output = separate_output(destination, source)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    with np.load(source / "arrays.npz", allow_pickle=False) as arrays:
        if len(arrays.files) != metadata["arrays"]:
            raise ValueError("Array count mismatch")
        with Recorder(staging, "restored-records") as writer:
            for item in rows:
                evidence = dict(item["evidence"])
                for key, name in item["arrays"].items():
                    evidence[key] = arrays[name].tolist()
                if item["messages"] is not None:
                    evidence["messages"] = messages[item["messages"]]
                writer.record(item["kind"], **evidence)
    result = audit(staging)
    if result["records"] != metadata["source_audit"]["records"]:
        raise ValueError("Restored evidence count mismatch")
    staging.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "unpack"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = (
        pack(args.source, args.destination)
        if args.operation == "pack"
        else unpack(args.source, args.destination)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
