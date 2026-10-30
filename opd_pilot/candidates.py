"""Generate or cross-score immutable diagnostic response banks from fixed prefixes."""

import argparse
import json
import math
from pathlib import Path

from opd_pilot.diagnostic_backend import (
    TransformersBackend,
    model_manifest,
    sample_seed,
    token_identity,
)
from opd_pilot.evidence import publish, separate_output


def validate(spec):
    """Validate assets, token-ID compatibility and operation without GPU imports."""
    operation = spec["operation"]
    if operation not in ("generate", "score"):
        raise ValueError("operation must be generate or score")
    manifest = model_manifest(spec["model_path"])
    source = model_manifest(spec["prefix_tokenizer_path"])
    if token_identity(spec["model_path"]) != token_identity(spec["prefix_tokenizer_path"]):
        raise ValueError("Different token-ID mappings; cross-tokenizer scoring is unsupported")
    if not math.isfinite(spec["temperature"]) or spec["temperature"] <= 0:
        raise ValueError("temperature must be finite and positive")
    bank = json.loads(Path(spec["input_bank"]).read_text(encoding="utf-8"))
    rows = bank["prefixes"] if operation == "generate" else bank["candidates"]
    id_key = "prefix_uid" if operation == "generate" else "candidate_uid"
    if not rows or len({row[id_key] for row in rows}) != len(rows):
        raise ValueError("Input bank must contain nonempty, unique record identities")
    if operation == "generate":
        if not 0 < spec["top_p"] <= 1 or spec["top_k"] != 0:
            raise ValueError("Diagnostic generation requires top_p in (0,1] and top_k=0")
        if any(
            isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 1
            for key in ("samples_per_prefix", "max_new_tokens")
        ):
            raise ValueError("Positive integer sample count and token budget required")
        if spec["owner"] not in ("student", "teacher"):
            raise ValueError("Candidate owner must be student or teacher")
        if any(row["sampling_student_revision"] != source["revision"] for row in bank["prefixes"]):
            raise ValueError("Prefix tokenizer asset revision differs from recorded student")
    elif bank["token_identity"] != token_identity(spec["model_path"]):
        raise ValueError("Candidate bank token identity does not match scoring model")
    return manifest, bank


def generate(spec, manifest, bank, backend):
    """Retain all samples, including duplicate responses and generation truncations."""
    rows = []
    for prefix in bank["prefixes"]:
        for index in range(spec["samples_per_prefix"]):
            seed = sample_seed(spec["seed"], prefix["prefix_uid"], spec["owner"], index)
            result = backend.generate(prefix["prefix_tokens"], spec, seed)
            if not result["response_tokens"]:
                raise ValueError("Empty backend output; no synthetic candidate substituted")
            rows.append(
                {
                    **result,
                    "candidate_uid": f"{spec['owner']}-{prefix['prefix_uid']}-{index + 1:02d}",
                    "prefix_uid": prefix["prefix_uid"],
                    "task_id": prefix["task_id"],
                    "source_turn_uid": prefix["source_turn_uid"],
                    "messages": prefix["messages"],
                    "prefix_tokens": prefix["prefix_tokens"],
                    "owner": spec["owner"],
                    "generation_revision": manifest["revision"],
                    "seed": seed,
                }
            )
    return {
        "schema_version": 1,
        "candidates": rows,
        "token_identity": token_identity(spec["model_path"]),
        "generation_settings": {
            key: spec[key] for key in ("temperature", "top_p", "top_k", "max_new_tokens")
        },
        "environment_execution_complete": False,
    }


def score(spec, manifest, bank, backend):
    """Score the same exact IDs under one model; retain token-level evidence."""
    rows = []
    for candidate in bank["candidates"]:
        response = candidate["response_tokens"]
        values = backend.score(candidate["prefix_tokens"], response, spec["temperature"])
        if len(values) != len(response) or not values or not all(math.isfinite(v) for v in values):
            raise ValueError("Missing, misaligned or nonfinite candidate scores")
        rows.append(
            {
                "candidate_uid": candidate["candidate_uid"],
                "prefix_uid": candidate["prefix_uid"],
                "response_tokens": response,
                "token_logprobs": values,
                "sum_nll": -sum(values),
                "mean_token_nll": -sum(values) / len(values),
                "token_count": len(values),
            }
        )
    return {
        "schema_version": 1,
        "scoring_revision": manifest["revision"],
        "temperature": spec["temperature"],
        "normalization": "full_vocabulary_temperature_scaled",
        "scores": rows,
        "tcod_backend_equivalence_validated": False,
    }


def main():
    """Run a single-model GPU diagnostic; validation never loads model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(
        spec["output"],
        Path(spec["input_bank"]).parent,
        spec["model_path"],
        spec["prefix_tokenizer_path"],
    )
    manifest, bank = validate(spec)
    if args.validate_only:
        print(json.dumps({"status": "spec_validated", "revision": manifest["revision"]}))
        return
    backend = TransformersBackend(spec["model_path"])
    result = (
        generate(spec, manifest, bank, backend)
        if spec["operation"] == "generate"
        else score(spec, manifest, bank, backend)
    )
    publish(
        output,
        {
            "bank.json": result,
            "spec.json": spec,
            "status.json": {"status": "completed", "stage": spec["operation"]},
        },
    )


if __name__ == "__main__":
    main()
