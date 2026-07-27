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
- D13 (2026-07-27): Rate grid revised to {1,2,3,4,5,6,8,inf} after the pilot measured a
  sustainable capacity of 3.2 req/s (7B/ShareGPT) and 4.5 req/s (7B/random)
- D14 (2026-07-27): **Revises D6.** Repetitions use seeds 1/2/3 instead of a fixed 42, with
  the same seed set reused for every condition. Comparisons stay paired, and the error bars
  now include prompt sampling and arrival jitter rather than system noise alone
- D15 (2026-07-27): `--temperature 0` on every run. The server's generation_config would
  otherwise apply sampling; output length is pinned by `--ignore-eos`, so the amount of work
  is unchanged and runs become reproducible
- D16 (2026-07-27): `--num-warmups` raised 10 -> 30; the pilot showed an elevated first
  quarter of prefill times at 10
- D17 (2026-07-27): C1 control — the same condition is measured with the phase logger
  enabled (A1a) and disabled (C1off), adjacent in time, to bound the instrumentation's
  effect on what it measures
- D18 (2026-07-27): C2 closed-loop control with a fixed concurrency limit. Above capacity an
  open-loop system has no steady state, so open-loop overload points are reported as
  burst transients and C2 supplies the steady-state latency-throughput curve
- D19 (2026-07-27): GPU count is compared **within one multi-GPU instance** (G1/G2/G4),
  replacing the rev. 1 plan of comparing S1 on a 1-GPU pod with S4 on a 2-GPU pod
- D20 (2026-07-27): `HF_HOME=/root/hf_cache` (container disk). `/workspace` is a 50 GB
  network volume of which the venv occupies 21 GB; model weights there fail with
  `Disk quota exceeded`
- D21 (2026-07-27): The runner keeps launching one `vllm bench serve` process per run even
  though ~78 % of wall time is interpreter startup. Process isolation and the exact
  reproducibility of each recorded command outweigh the ~4 h saved by an in-process loop
