# Native E5 engineering replay (opt-in)

Approval: isolated replay integration and engineering verification preparation only.
No jobs were submitted by Codex. Formal A/B, E6 and algorithm interventions are not
enabled by this approval. `native_replay` defaults to offline checks and requires
both `--execute` and an eight-GPU Slurm allocation for an actual update.

1. After real training/recovery smoke, use a stopped full checkpoint and its saved
   resolved-config.yaml (with translated trainer_config, not an unvalidated template).
   Preserve actor topology, source configuration and the 100-step scheduler horizon.
2. A fresh-data producer can call native_codec.capture(experiences, metadata) and
   update_probe.freeze to retain every Experience instance field, dtype and shape.
   This API does not itself run a sampler. Do not use diagnostic trajectory records
   as a substitute. Batch metadata must identify train-only fresh student data and
   the checkpoint revision; each native info.model_version must match its step.
3. Freeze scores against that batch. Engineering A/A requires exact cached native
   teacher scores in both repetitions. The typed restore replaces only teacher scores;
   it rejects native token, boundary, mask or original valid-mask inconsistencies.
   No constructor is allowed to silently regenerate masks during roundtrip.
4. Save `update_probe plan`'s JSON audit and fill orion/templates/native-replay.json.
   Use distinct output/checkpoint_output/logs paths for A and AA-repeat, under the
   user's storage_ext outputs/checkpoints and root logs/tcod-opd/replay respectively.
   Source file digests are metadata only. All output directories must be new and
   outside source trees. Keep checkpoint writers and environment installers stopped.
5. `python -m opd_pilot.native_replay --spec PATH` performs offline checks. Hashing
   large checkpoint files belongs on a compute node. It does not initialize models
   or prove the resolved worker layout/restoration. An explicit `--execute` invokes
   the native loader, original train_step and original checkpoint saver once.

Prepared Slurm wrapper: native_replay.sbatch SPEC [--validate-only|--execute]. The
default is validation only. It retains the approved eight-GPU allocation (even though
only the original trainer workers are instantiated), 32 CPUs/256G, with a provisional
30-minute upper limit. Create logs/tcod-opd/replay before user submission. Do not use
this template to launch a formal A/B comparison. No model/package downloads occur.

Artifacts include prepared-batch.json (native processed tensors including actual
old probabilities/advantages/masks), per-rank restoration and update evidence,
checkpoint structure, native metrics, and Ray logs. Rank-local IDs and dynamic/fixed
micro-batch order are recorded before execution; all ranks must report one actual
optimizer step and one scheduler step. Count synchronized ranks as one logical step.
If original settings yield multiple mini-batches/epochs, preflight fails without
modifying them. A skipped optimizer update fails engineering acceptance; no branch
checkpoint is published as complete in that case.

COMPLETE means the isolated native call returned, each worker reported the expected
step counts, checkpoint archives were saved and source digests remained unchanged.
It does not certify A/A agreement, optimizer tensor equivalence, exported-model
behavior or scientific validity. Compare the two repetitions with the existing
evaluation/probe tools before making those claims. RNG equality checks do not imply
bitwise GPU reproducibility. Checkpoint contents are trusted experiment artifacts;
the native loader uses PyTorch deserialization and must not load untrusted files.

Only output paths, resume path/delete-source protection, isolated namespace and log
locations change in the copied config. Sampling/loss/precision/batch settings are
inherited. The replay preparation omits explorer sync because there is no online
producer. It never calls the main Trainer loop or schedules a follow-up experiment.
