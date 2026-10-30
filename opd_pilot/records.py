"""Independent evidence writer; importing this module does not change TCOD behavior."""

import argparse
import json
import os
import queue
import re
import threading
from pathlib import Path

REQUIRED = {
    "episode": ("episode_uid", "task_id", "outcome"),
    "turn": (
        "turn_uid",
        "episode_uid",
        "sampling_student_revision",
        "scoring_teacher_revision",
        "messages",
        "tokens",
        "prompt_length",
        "response",
        "action",
        "raw_reward",
        "student_logprobs",
        "teacher_logprobs",
    ),
    "update": (
        "update_uid",
        "update_student_revision",
        "consumed_turn_uids",
        "mask",
        "advantage",
        "effective_token_count",
    ),
}


class Recorder:
    """Write one exclusive worker/session shard with bounded, blocking backpressure.

    The caller supplies actual evidence; this class does not reconstruct prompts,
    infer success, compute advantages or decide which samples enter training.
    Use a new readable shard name after restart. Only close publishes COMPLETE.
    """

    def __init__(self, root, shard, capacity=32):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", shard):
            raise ValueError("Use a readable worker-session name without path separators")
        if capacity < 1:
            raise ValueError("Queue capacity must be positive")
        self.path = Path(root) / shard
        self.path.mkdir(parents=True, exist_ok=False)
        self._queue = queue.Queue(maxsize=capacity)
        self._error = None
        self._closed = False
        self._count = 0
        self._thread = threading.Thread(target=self._write, name=f"records-{shard}", daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f"Evidence writer failed: {self.path}") from self._error

    def _enqueue(self, item):
        while True:
            self._check()
            try:
                self._queue.put(item, timeout=0.1)
                self._check()
                return
            except queue.Full:
                continue

    def record(self, kind, **evidence):
        """Snapshot JSON evidence before queuing; never silently drop a record."""
        if self._closed:
            raise RuntimeError("Recorder is closed")
        if kind not in REQUIRED:
            raise ValueError(f"Unknown record kind: {kind}")
        missing = set(REQUIRED[kind]) - evidence.keys()
        if missing:
            raise ValueError(f"Missing {kind} fields: {sorted(missing)}")
        # Reject non-JSON values and NaNs rather than stringifying tensors silently.
        line = (
            json.dumps(
                {"schema_version": 1, "kind": kind, "evidence": evidence},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self._enqueue(line)

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("x", encoding="utf-8") as stream:
                while True:
                    line = self._queue.get()
                    if line is None:
                        break
                    stream.write(line)
                    stream.flush()
                    self._count += 1
                os.fsync(stream.fileno())
        except BaseException as error:
            self._error = error

    def close(self):
        """Drain accepted records and mark completion only after successful writes."""
        if self._closed:
            self._check()
            return
        self._closed = True
        self._enqueue(None)
        self._thread.join()
        self._check()
        with (self.path / "COMPLETE.json").open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "records": self._count}, stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        # COMPLETE describes the writer drain, not successful episode/training execution.
        self.close()


def audit(root):
    """Check completed shards and cross-worker episode/turn/update references offline."""
    counts = dict.fromkeys(REQUIRED, 0)
    identifiers = {kind: set() for kind in REQUIRED}
    episode_refs = set()
    turn_refs = set()
    shards = sorted(path for path in Path(root).iterdir() if path.is_dir())
    if not shards:
        raise ValueError("No record shards found")
    for shard in shards:
        complete = json.loads((shard / "COMPLETE.json").read_text(encoding="utf-8"))
        count = 0
        with (shard / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                kind, evidence = item["kind"], item["evidence"]
                if item["schema_version"] != 1 or kind not in REQUIRED:
                    raise ValueError(f"Unsupported record schema in {shard}")
                if set(REQUIRED[kind]) - evidence.keys():
                    raise ValueError(f"Incomplete {kind} evidence in {shard}")
                uid = evidence[f"{kind}_uid"]
                if not isinstance(uid, str) or not uid or uid in identifiers[kind]:
                    raise ValueError(f"Invalid or duplicate {kind} ID: {uid}")
                identifiers[kind].add(uid)
                counts[kind] += 1
                count += 1
                if kind == "turn":
                    episode_refs.add(evidence["episode_uid"])
                if kind == "update":
                    turn_refs.update(evidence["consumed_turn_uids"])
        if complete["records"] != count:
            raise ValueError(f"Record count mismatch in {shard}")
    if episode_refs - identifiers["episode"] or turn_refs - identifiers["turn"]:
        raise ValueError("Dangling episode or consumed-turn references")
    return {"shards": len(shards), "records": counts}


def main():
    """Audit evidence without importing models, simulators or TCOD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
