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
