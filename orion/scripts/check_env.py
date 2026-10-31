"""Small real GPU check; no model downloads and no claim of OPD readiness."""

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    for name in ("vllm", "verl", "ray", "swanlab", "trinity", "datasets", "pyarrow"):
        importlib.import_module(name)
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.03)
    actual.float().square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    torch.cuda.synchronize()
    result = {
        "status": "basic_gpu_check_passed",
        "opd_ready": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "verl", "flash-attn", "transformers", "ray", "swanlab")
        },
        "checks": ["imports", "flash_attention_vs_math", "attention_backward"],
        "pending": ["model_generation", "teacher_token_scoring", "opd_update", "recovery"],
    }
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
