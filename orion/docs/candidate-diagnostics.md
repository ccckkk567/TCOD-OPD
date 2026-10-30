# Independent candidate generation and cross-scoring

This is diagnostic code, not TCOD's training rollout/scoring backend. It performs
real Transformers model generation/scoring when run on an allocated GPU, using
completed local model assets only. The GPU paths have not yet been exercised on
Orion. Local mock tests check plumbing and fail-visible contracts, not model
numerics or agreement with TCOD/veRL/vLLM.

Entry point:

```bash
python -m opd_pilot.candidates --spec /absolute/path/spec.json --validate-only
python -m opd_pilot.candidates --spec /absolute/path/spec.json
```

Both operations require operation, model_path, prefix_tokenizer_path, input_bank,
output and temperature. Output must be a new directory outside input/model trees.
model_path and prefix_tokenizer_path must have matching asset.json/READY.json and
pass asset checks. prefix_tokenizer_path is the actual student asset that produced
the stored prefix IDs. Token-ID vocabularies and added special tokens must match
the model being used; differing token-ID spaces are rejected rather than retokenized.
This is a BPE mapping check, not proof that tokenizers reconstruct identical chats.

## Generate

Set operation to generate. input_bank is the prefix_bank.json from fixed-prefix
selection. Additional fields: owner (student or teacher), seed, samples_per_prefix,
max_new_tokens, top_p and top_k. top_k must be zero; temperature must be finite
and positive. Set top_p=1 for untruncated-support diagnostic sampling. All generation
settings are recorded explicitly. Each sample seed derives from the supplied seed,
prefix identity, owner and sample index, independent of worker ordering. This does
not imply cross-GPU bitwise reproducibility.

Actual recorded prefix IDs are passed directly to generation. Messages are copied
for evidence, never templated again. A fresh sampling GenerationConfig avoids
silently inheriting model-specific repetition penalties or sampling defaults;
local EOS IDs are retained. This is an explicit diagnostic sampling protocol and
must not be called equivalent to TCOD's native sampling without comparison.

bank.json retains the full response token IDs, decoded text with and without
special tokens, generation revision, source turn, seed and EOS/length termination.
Duplicate responses remain separate frequency observations. Empty generation fails
instead of inserting dummy tokens. No action parser, simulator execution, success
label or candidate-value conclusion is fabricated.

## Score

Set operation to score. input_bank is a generated bank.json. Run separately with
student and teacher model_path values against the same immutable input bank.
No generation settings other than temperature are needed. The model receives
the exact prefix IDs followed by the exact candidate response IDs, including any
generated EOS token. Response token j is scored from causal logits at absolute
position len(prefix)-1+j. Normalization is FP32 log-softmax after dividing raw logits
by temperature, over the full vocabulary, without top-k/top-p truncation.

Output bank.json contains candidate IDs, response IDs, each token's conditional
logprob, token count, total NLL and mean token NLL. These are conditional likelihood
measurements, not entropy, action probability mass, environment value or proof of
learning. TCOD-backend equivalence remains explicitly false pending a real paired
token-position/temperature comparison. Per-model score banks can be joined on
candidate_uid only after verifying they refer to the same original candidate bank.

## Runtime and publication

orion/slurm/diagnostic_candidates.sbatch is a one-GPU, one-hour testqueue entry point.
Create logs/tcod-opd/diagnostics before submission and submit from a checkout that
contains this implementation. Do not update a checkout/venv used by active jobs.
The backend uses local BF16 weights and SDPA on cuda:0, with no CPU offload,
quantization or automatic distributed placement. A resource/context failure exits
nonzero; it does not change model, budget or precision automatically.

Validation checks assets/spec/bank without importing Torch or loading weights.
Actual execution publishes bank.json, spec.json and status.json only after all
items finish. It currently reruns an incomplete bank rather than supporting
per-candidate resume. Existing complete outputs are not overwritten. No diagnostic
bank is admitted to student training by these tools.

Backend references: Transformers 4.57 generation/configuration and AutoModel APIs:
https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation
https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto
