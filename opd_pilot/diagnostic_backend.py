"""Local-only Transformers backend for independent GPU diagnostics, not TCOD training."""

import hashlib
import json
import math
from pathlib import Path

from opd_pilot.assets import validate_model


def model_manifest(directory):
    """Require a completed downloaded asset and preserve its exact revision."""
    directory = Path(directory)
    manifest = json.loads((directory / "asset.json").read_text(encoding="utf-8"))
    if json.loads((directory / "READY.json").read_text(encoding="utf-8")) != manifest:
        raise ValueError("Model READY does not match manifest")
    validate_model(directory, manifest)
    return manifest


def token_identity(directory):
    """Read Qwen BPE token-ID mappings, including added special tokens."""
    data = json.loads((Path(directory) / "tokenizer.json").read_text(encoding="utf-8"))
    if data["model"]["type"] != "BPE":
        raise ValueError("This diagnostic currently validates BPE tokenizer mappings only")
    return {
        "vocab": data["model"]["vocab"],
        "added": sorted([item["id"], item["content"]] for item in data["added_tokens"]),
    }


def sample_seed(seed, prefix_uid, owner, sample):
    """Derive repeatable seeds independent of job/worker iteration order."""
    value = json.dumps([seed, prefix_uid, owner, sample]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def prediction_positions(prefix, response):
    """Map response token j to logits at absolute input position len(prefix)-1+j."""
    if not prefix or not response:
        raise ValueError("Nonempty prefix and response required for conditional scoring")
    return list(range(len(prefix) - 1, len(prefix) + len(response) - 1))


class TransformersBackend:
    """Load one local BF16 model on one allocated GPU; never download or train."""

    def __init__(self, directory):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Run this diagnostic in a GPU allocation")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                directory,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda:0")
            .eval()
        )

    def encode_messages(self, messages, enable_thinking=None):
        """Apply the tokenizer's chat template with the TCOD user-final convention."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.tokenizer.get_chat_template(),
            enable_thinking=enable_thinking,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def generate(self, prefix, settings, seed):
        """Generate using actual recorded prefix IDs and a fresh explicit sampling policy."""
        from transformers import GenerationConfig

        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
        config = GenerationConfig(
            do_sample=True,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            max_new_tokens=settings["max_new_tokens"],
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            repetition_penalty=1.0,
        )
        with torch.inference_mode():
            tokens = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=config,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                max_new_tokens=settings["max_new_tokens"],
                do_sample=True,
                repetition_penalty=1.0,
            )
        ids = tokens[0, len(prefix) :].tolist()
        eos = config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        return {
            "response_tokens": ids,
            "response_text": self.tokenizer.decode(ids, skip_special_tokens=True),
            "response_text_with_special_tokens": self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
            "termination": "eos" if ids and ids[-1] in eos_ids else "length",
        }

    def score(self, prefix, response, temperature):
        """Return full-vocabulary temperature-normalized response-token logprobs."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Scoring temperature must be finite and positive")
        torch = self.torch
        positions = prediction_positions(prefix, response)
        inputs = torch.tensor([prefix + response], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            logits = self.model(
                inputs, attention_mask=torch.ones_like(inputs), use_cache=False
            ).logits
            # Normalize only response-position rows, in FP32. No top-k/top-p truncation.
            rows = logits[0, positions].float() / temperature
            targets = torch.tensor(response, device=rows.device).unsqueeze(-1)
            values = torch.log_softmax(rows, dim=-1).gather(-1, targets).squeeze(-1)
        result = values.cpu().tolist()
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Nonfinite conditional token score")
        return result
