# vLLM-learn

## Methods Decision Log (2026-07-12)
- Pinned to vLLM v0.25.0 (commit 702f4814fe54fabff350d43cb753ae3e47c0c276);
  instrumentation lives on the `instrumentation` branch of the fork IsseiHasegawa/vllm
- Environment: RunPod A40 48GB, uniform across all experiments (1x; 2x for tp=2).
  The benchmark client runs on the same Pod as the server (localhost)
- D1: `--ignore-eos` on every run (controls the number of generated tokens across models)
- D2: `--num-warmups 10`. Discard one throwaway run right after server startup
- D3: The server is started per (model x TP configuration). Rate, dataset, and repetitions run against the same server
- D4: Run attribution uses the manifest method (slice the phase JSONL by start/end timestamps)
- D5: Primary metric is p95. p99 is a reference value with error bars over 3 repetitions (n=200)
- D6: Repetitions use the same seed=42 (workload fixed; error bars reflect systematic noise only)
- D7: Single-GPU experiments (S1, S2, S3, I1, I2) are consolidated into one session on the same instance
- D8: `--no-enable-prefix-caching` required / `--disable-log-stats` forbidden /
  async scheduling disabled (to keep step-instrumentation attribution unambiguous)
