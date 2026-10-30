# Benchmark preparation recovery

The user supplied logs show ALFWorld data and 256/64/32/4 task lists prepared,
ScienceWorld 1.3.0 installed with its JAR/tasks/object IDs copied, and WebShop Python
dependencies installed. The shared pip check stopped on TextWorld's malformed
WHEEL header; the separate metadata repair preserves all package versions.
WebShop's official Drive request failed before its first product file completed.

After the metadata repair and after other jobs using the shared environment/checkout
finish, use three jobs from the user's existing checkout:

- scienceworld_smoke.sbatch: 15 minutes, 1 GPU, no installation/download. Invoke
  unchanged TCOD creation/reset/action helpers on boil/melt variation 0 with the
  existing shared JAR. Keep the existing easy simplification/default limits, close
  each JVM, save observations and exceptions under reports/tcod-opd/benchmarks.
- download_webshop_smoke.sbatch: 20 minutes, 1 GPU. Download exactly three small
  JSON files from HongbangYuan/webshop, fixed revision recorded in the downloader.
  Existing JSON files are reused with their recorded provenance. No clone occurs.
  User approval is solely for small environment smoke, not formal experiments.
- webshop_smoke.sbatch: 45 minutes, 1 GPU, afterok the download. Add the official
  en_core_web_lg model only if missing, preserving installed dependency constraints;
  build the existing WebShop source's own indexes into shared data/webshop;
  call TCOD's unchanged constructor and run reset plus one search.

The first two jobs can run in parallel. The index/smoke stage exclusively locks the
shared installation while it may add the language resource. ScienceWorld and the
download hold shared locks. Existing training jobs do not necessarily use these
locks, so do not submit environment mutations alongside such jobs.

Source data, mirror revision and the restricted usage scope are retained in
data/webshop/asset.json, the index marker and WebShop smoke report. Community
mirrors are not claimed byte-identical to the official Drive files. Original TCOD
source, reward rules, version choices and training pipelines are unchanged.

Local tests cover mirror file selection/reuse/provenance and simulator cleanup
with doubles. Actual reset/step success requires copied Orion job results. A
successful smoke is neither task success nor validation of training readiness.

## Missing small spaCy model after successful indexing

Observed log: en_core_web_lg 3.8.0 installed, pip check passed, all four indexes built
and reopened (1000/100/1000/1000 documents, zero indexing errors). Index log span
was approximately nine seconds, not 45 minutes. `indexes_100k` is still only 1000
documents for this small source, not a full dataset. Runtime then failed in WebShop
engine/goal.py loading en_core_web_sm. Both small and large spaCy packages are needed;
the scripts previously installed only the large package. No Gym version change is
justified by the warning preceding this specific missing-model exception.

Both preparation paths now install missing lg and sm models using spaCy's existing
download command under the frozen package constraints. The recovery submission
extracts the necessary scripts/helpers from the existing Git object store (no clone
or checkout mutation), waits afterany for existing opd-* jobs to stop using the shared
environment, and takes its exclusive lock. Do not submit new environment readers
while this repair is running. Existing indexes must have INDEX_READY and are reused
after metadata/open checks. No data download or forced index rebuild occurs.
The new upper time limit is 15 minutes. Runtime success still needs reset/search
evidence; community data remains restricted to small environment smoke only.

## Three-turn interface checks

Observed after the recovery jobs: TextWorld metadata repair and all three imports
passed; ScienceWorld 1.3.0 passed real boil/melt reset and step. WebShop small mirror
files downloaded with the restricted provenance marker. Index/smoke evidence is
still pending; ALFWorld real simulator evidence is also pending.

`submit_environment_interfaces.sh REVISION` submits two independent 10-minute,
1-GPU testqueue jobs for ALFWorld and ScienceWorld. It extracts only the required
scripts/helpers from the existing Git object store into a readable dated temporary
directory. This is not another clone and does not modify the active checkout.
Known queued/running shared-environment writers are afterany dependencies of both
jobs. Both readers use a shared environment lock; a later installer causes a clear
nonzero exit rather than a concurrent read or a long GPU wait.

ALFWorld uses two tasks from the previously reserved smoke split. ScienceWorld
uses boil/melt variation 0 and the existing JAR. Both exercise original TCOD
constructors, templates and action parsers through the independent diagnostic
Episode for three turns, covering the history template. Responses are scripted
look/look-around actions, not model outputs. Reports retain the exact prompts,
responses and transitions and explicitly mark model interaction, production
workflow execution and training as unvalidated. No package install, download,
reward modification or training-buffer write occurs.
