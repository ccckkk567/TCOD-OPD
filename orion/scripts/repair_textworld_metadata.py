"""Repair the observed TextWorld 1.7.0 WHEEL header separator, without reinstalling."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from email.parser import Parser
from pathlib import Path


def normalized_wheel(text):
    """Only move misplaced Tag headers out of the body; keep their values unchanged."""
    message = Parser().parsestr(text)
    if message.get_all("Tag"):
        return text
    body = [line for line in message.get_payload().splitlines() if line.strip()]
    if not body or any(not line.startswith("Tag: ") for line in body):
        raise ValueError("Not the observed misplaced-Tag metadata defect")
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


def repair(wheel, backup, supports):
    original = wheel.read_bytes()
    fixed = normalized_wheel(original.decode("utf-8")).encode("utf-8")
    tags = Parser().parsestr(fixed.decode("utf-8")).get_all("Tag", [])
    if not supports(tags):
        raise ValueError("Actual platform tags do not match; metadata-only repair is insufficient")
    if fixed == original:
        return dict(status="already_valid", tags=tags)
    record = wheel.parent / "RECORD"
    old_record = record.read_bytes()
    rows = list(csv.reader(io.StringIO(old_record.decode("utf-8"))))
    name = wheel.parent.name + "/WHEEL"
    matches = [row for row in rows if row[0] == name]
    if len(matches) != 1:
        raise ValueError("Expected exactly one WHEEL entry in installed RECORD")
    digest = base64.urlsafe_b64encode(hashlib.sha256(fixed).digest()).rstrip(b"=").decode()
    matches[0][1:] = ["sha256=" + digest, str(len(fixed))]
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    backup.mkdir(parents=True, exist_ok=False)
    (backup / "WHEEL.before").write_bytes(original)
    (backup / "RECORD.before").write_bytes(old_record)
    try:
        wheel.write_bytes(fixed)
        record.write_text(output.getvalue(), encoding="utf-8", newline="")
    except Exception:
        wheel.write_bytes(original)
        record.write_bytes(old_record)
        raise
    result = dict(
        status="metadata_repaired",
        version="1.7.0",
        tags=tags,
        changed_files=[str(wheel), str(record)],
        code_and_binaries_changed=False,
    )
    (backup / "repair.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    from pip._vendor.packaging.tags import parse_tag, sys_tags

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    dist = importlib.metadata.distribution("textworld")
    if dist.version != "1.7.0":
        raise ValueError("This repair targets only the observed TextWorld 1.7.0 package")
    entry = next(path for path in dist.files if str(path).endswith(".dist-info/WHEEL"))
    supported = set(sys_tags())
    result = repair(
        Path(dist.locate_file(entry)),
        args.backup,
        lambda tags: any(supported.intersection(parse_tag(tag)) for tag in tags),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
