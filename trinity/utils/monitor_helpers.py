"""Dependency-free redaction for experiment configuration and metrics."""

import os


def redact_secrets(value):
    """Remove credential fields and known runtime keys without hiding token counts."""
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(
                    marker in str(key).lower()
                    for marker in (
                        "api_key",
                        "password",
                        "secret",
                        "hf_token",
                        "access_token",
                        "authorization",
                    )
                )
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        for name in ("HF_TOKEN", "SWANLAB_API_KEY"):
            credential = os.environ.get(name)
            if credential:
                value = value.replace(credential, "[REDACTED]")
    return value
