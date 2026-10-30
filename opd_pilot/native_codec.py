"""Lossless, typed JSON envelopes for trusted native Experience payloads."""

from dataclasses import fields, is_dataclass


def encode(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return {"type": "numpy_scalar", "dtype": str(value.dtype), "value": value.item()}
    if isinstance(value, torch.dtype):
        return {"type": "dtype", "value": str(value).removeprefix("torch.")}
    if is_dataclass(value):
        if type(value).__name__ not in ("EID", "CustomField"):
            raise TypeError("Unsupported native dataclass")
        return {
            "type": type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [encode(v) for v in value]}
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported native value: {type(value).__name__}")


def decode(value):
    import numpy as np
    import torch

    from trinity.common.experience import EID, CustomField

    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "tensor":
        return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"])).reshape(
            value["shape"]
        )
    if kind == "ndarray":
        return np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
    if kind == "numpy_scalar":
        return np.asarray(value["value"], dtype=value["dtype"])[()]
    if kind == "dtype":
        return getattr(torch, value["value"])
    if kind in ("EID", "CustomField"):
        return {"EID": EID, "CustomField": CustomField}[kind](
            **{k: decode(v) for k, v in value["fields"].items()}
        )
    if kind == "dict":
        return {decode(k): decode(v) for k, v in value["items"]}
    if kind in ("list", "tuple"):
        return {"list": list, "tuple": tuple}[kind](decode(v) for v in value["items"])
    raise ValueError("Unknown native field encoding")


def capture(experiences, metadata):
    """Called explicitly by a fresh-data producer; never transforms diagnostic records."""
    from opd_pilot.update_probe import validate_batch

    rows = []
    for exp in experiences:
        rows.append(
            dict(
                sample_id=exp.eid.uid,
                tokens=exp.tokens.tolist(),
                prompt_length=exp.prompt_length,
                response_mask=exp.action_mask.tolist(),
                native_fields=encode(vars(exp)),
            )
        )
    bundle = dict(metadata, rows=rows)
    validate_batch(bundle)
    return bundle


def restore(batch, scores):
    import torch

    from opd_pilot.update_probe import alignment
    from trinity.common.experience import Experience

    report = alignment(batch, scores)
    if not report["aligned"]:
        raise ValueError(str(report))
    experiences = []
    for row, scored in zip(batch["rows"], scores["rows"]):
        native = decode(row["native_fields"])
        exp = Experience.__new__(Experience)
        exp.__dict__.update(native)
        if encode(vars(exp)) != row["native_fields"]:
            raise ValueError("Native Experience roundtrip changed fields")
        if (
            exp.eid.uid != row["sample_id"]
            or exp.tokens.tolist() != row["tokens"]
            or exp.prompt_length != row["prompt_length"]
            or exp.action_mask.tolist() != row["response_mask"]
        ):
            raise ValueError("Native Experience disagrees with bundle identity/tokens/mask")
        if exp.teacher_logprobs is None:
            raise ValueError("Need original teacher-score tensor dtype/shape")
        valid = getattr(exp, "teacher_logprobs_valid_mask", None)
        if valid is not None and valid.tolist() != row["response_mask"]:
            raise ValueError("Original teacher mask differs; no mask intersection is authorized")
        replacement = torch.tensor(scored["teacher_log_probs"], dtype=exp.teacher_logprobs.dtype)
        if replacement.shape != exp.teacher_logprobs.shape or not torch.isfinite(replacement).all():
            raise ValueError("Replacement scores cannot preserve native shape/dtype")
        exp.teacher_logprobs = replacement
        experiences.append(exp)
    return experiences
