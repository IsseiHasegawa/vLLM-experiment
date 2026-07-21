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
- D9 (2026-07-19): v0.25.0 enables async scheduling by default; all servers are launched
  with `--no-async-scheduling` so that step-level timing is attributable. The runner aborts
  if the PHASE-INSTR batch-queue warning appears in the server log
- D10 (2026-07-19): The RunPod PyTorch 2.8.0 template ships datasets 1.1.1, which is
  incompatible with pyarrow 25. Each session starts with `uv pip install -U "datasets>=3.0"`
- D11 (2026-07-20): EngineCore does not run atexit handlers on termination, so the phase
  logger flushes on a 1s background timer; the runner waits 3s before stopping a server
- D12 (2026-07-20): D7 revised for a 3h/day schedule: the 7B block (S1, S2, I1, I2; 42 runs)
  and the 0.5B block (S3; 24 runs) run on separate instances. Day 2 begins by re-measuring
  the S1 r=5 anchor point (3 reps) to demonstrate cross-instance consistency
