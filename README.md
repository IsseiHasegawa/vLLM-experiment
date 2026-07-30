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
  and the 0.5B block (S3; 27 runs in rev. 2) run on separate instances. Day 2 begins by re-measuring
  the S1 r=5 anchor point (3 reps) to demonstrate cross-instance consistency
- D13 (2026-07-27): Rate grid revised to {1,2,3,4,5,6,8,inf} after the pilot measured a
  sustainable capacity of 3.2 req/s (7B/ShareGPT) and 4.5 req/s (7B/random).
  **Superseded in part by D24/D28 (2026-07-29):** the 4.5 figure was an artefact of
  including drain time in the denominator. Session A reaches 6.83 req/s at rate 8 and
  13.8 req/s at rate inf on 7B/random, so the grid tops out near 50 % utilisation and
  contains no knee for that dataset. The 3.2 figure for ShareGPT is confirmed
  (A1a/A1b: 3.01/3.21/3.36)
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
- D21 (2026-07-27, figure corrected 2026-07-29): The runner keeps launching one
  `vllm bench serve` process per run even though a large fraction of wall time is
  process start-up. Process isolation and the exact reproducibility of each recorded
  command outweigh the hours saved by an in-process loop.
  **Measured on session A:** 60.0 % of wall clock is fixed cost (15 176 s wall,
  6 066 s measured across 91 runs), not the 75-82 % seen in the pilot, which sampled
  only short runs. Fixed cost is 91-105 s per run and is independent of the dataset
  (S1/ShareGPT 95.8 s vs S2/random 100.1 s), so it is `import vllm` plus tokenizer
  load plus warm-up, not the 642 MB ShareGPT parse. The share of a run that is
  overhead ranges from 31 % (rate 1) to 92 % (S3 rate 32)
- D22 (2026-07-29): `--torch-backend=cu128` is unsatisfiable (torchcodec>=0.14 has no
  cu128 wheel). Using `auto`, so torch's CUDA build follows the host driver:
  session A got driver 580.95.05 -> torch 2.11.0+cu130 (the pilot ran on 570.x).
  Recorded in env_freeze.txt; anchors A1c/A1d detect any resulting difference.
- D23 (2026-07-29): `enable_chunked_prefill=True` (vLLM default). Prefill is split
  and mixes with decode in the same step, so "prefill time" is not one contiguous
  interval. Affects figure 8 and the Methods definition of the phase split.
- D24 (2026-07-29): RATE_SHORTFALL is a definitional artefact, not saturation.
  `achieved = 200 / (arrival span + drain)`, so the flag fires at 58 % utilisation
  (S2 r=8: 6.83 vs capacity 13.8). Arrival span / expected span = 1.00 in all 78
  open-loop runs, so arrivals were always delivered as specified. The §6.4 gate
  becomes "span ratio == 1.00". Figure 3 still plots the client's own
  `request_throughput`, which contains the drain bias: the operational definition of
  the achieved rate and of the saturation point is deliberately left to the 8/3
  analysis task rather than fixed in code now, and the figure must not be read as a
  capacity measurement until then.
- D25 (2026-07-29): Do not quote rinf as capacity for ShareGPT. Output length p95/p50
  is 5.7x (max 1642 tokens), so a run ends when its longest request ends; S1_rinf
  varies 3.23/4.57/3.56 across seeds (41 %). Dataset effect is reported at matched
  rates or from closed-loop C2.
- D26 (2026-07-29): The ShareGPT sampler admits only prompts <= ~1024 tokens
  (max realised n_prompt = 1010 vs 66 076 in the raw file). The "realistic
  conversation" claim is bounded accordingly. **Implemented 2026-07-30:** figure 5 now
  overlays the nominal distribution (results/dataset_stats.json, from the source file)
  with the realised one (n_prompt / n_gen from the phase logs of S1 and S2, n=4800 each).
  ShareGPT nominal input p50/p95/max = 145/938/66076 against realised 135/767/1010, so
  plotting the nominal alone would overstate the served tail by 65x. Confirm the exact
  filter in benchmarks/datasets.py before writing Methods.
- D27 (2026-07-29): 0.5B saturates at ~18-20 req/s with SM <= 59 % and memory
  controller <= 45 %, while server CPU doubles vs 7B (169 % vs 55 %). Hypothesis:
  per-step framework overhead, not the GPU, is the ceiling at small model size.
  Test with sched_s / (sched_s + exec_s) from the step logs.
- D28 (2026-07-29): **Analysis windows.** `resource_rows()` and the step-log slice now
  use a *measured* window, not the manifest window. The manifest window is the whole
  `vllm bench serve` lifetime, of which ~100 s is start-up with the GPU near idle, so
  averaging over it understates utilisation by 1.4x (rate 1) to 8.2x (S3 rate 32) -
  and because the distortion grows with rate, figure 9 would have shown utilisation
  *falling* under load. The window is recovered from the request log (records inside
  the manifest window, ordered by completion, first `--num-warmups` dropped); runs with
  no phase log (C1off) fall back to `end_ts - duration`. Corrected mean SM utilisation,
  averaged over the three repetitions: 86.7 % not 39.3 % at 7B/ShareGPT rate 5 (2.2x),
  and 53.3 % not 5.3 % at 0.5B rate 32 (10.0x)
- D29 (2026-07-29): **Warm-up requests are excluded from phase-log analysis.** The
  window contains `num_prompts + num_warmups` = 230 records per run; the leading 30 are
  dropped by completion order, which yields exactly 200 for all 88 instrumented runs.
  Keeping them would reimport the contamination D16 was raised to avoid
- D30 (2026-07-29): **run_ids are not unique across campaigns.** The pilot and session A
  both contain I1_rep1, I2_rep1, S1_r5_rep1, S2_r5_rep1. Keying the manifest on run_id
  alone let a pilot timing window be applied to a session A bench file, which silently
  emptied the phase-log slice for exactly the runs figure 8 is built from. Manifest rows
  are now keyed by (run_id, campaign directory) taken from `result_json`, and a shadowed
  run_id prints a NOTE
- D31 (2026-07-29): **C1 is analysed paired by seed, and it detects an effect.** The arms
  share seeds (D14), so the paired difference isolates logging: TTFT p50 +2.97 %
  (p=0.010), TPOT p95 +1.05 %, request throughput -0.26 %, and 7/7 metrics move in the
  same direction. The unpaired Welch test on n=3 has almost no power (all p>0.5) and is
  retained for completeness only. Reported claim: logging costs <=3 % on latency and
  <=0.3 % on throughput, so instrumented latencies are upper bounds and instrumented
  throughput a lower bound. This revises the session-A quality gate in PLAN 6.4, which
  required the string "no instrumentation effect detected": a small, bounded, correctly
  signed effect is the expected outcome, not a failure
- D32 (2026-07-29): **S2b** adds rates {5,10,12,16,20} for 7B/random in session B, since
  S2's grid stopped at ~50 % utilisation (D13). r=5 overlaps session A's S2_r5 so the two
  instances can be cross-checked; the group is named separately because it is a different
  instance and must not be silently merged into the S2 series
- D33 (2026-07-30): `cpu_total`, `cpu_max_core` and `ram_used_gb` in the resource logs
  are **host-wide** (`psutil` sees all 96 cores of the physical machine), not the
  container's 9 vCPUs, so they include other tenants: `ram_used_gb` reads 60 GB on a
  50 GB pod and `cpu_max_core` sits at 100 % while idle. Bottleneck claims use
  `cpu_server_pct` / `cpu_client_pct`, which are per-process and clean (100 % = one
  core). Measured on session A the client never exceeded ~210 % of 900 %, so the
  harness was never the bottleneck
- D34 (2026-07-30): The runner's `HF_HOME` fallback was `/workspace/hf_cache`, which is
  exactly the path D20 forbids; launching the runner from a shell that had not exported
  `HF_HOME` would have reproduced `Disk quota exceeded` on the 7B download. The default
  is now `/root/hf_cache`
- D35 (2026-07-30): **Figure 9's CPU panel plots the per-process counters.** It previously
  drew `cpu_total` and `cpu_max_core`, which D33 shows are host-wide: measured across S1,
  `cpu_total` is flat at 5-8 % and `cpu_max_core` sits at 99-100 % at every rate, so the
  panel carried two meaningless lines while omitting the series the bottleneck argument
  rests on. `cpu_server_pct` rises 28 % -> 40 % across S1's grid and 65 % -> 144 % across
  S3's, i.e. the 0.5B server spends 3.6x the CPU of the 7B server while leaving the GPU at
  53 % (D27). The 900 % ceiling (9 vCPU) is stated in the panel as text rather than drawn,
  which would flatten both series
- D36 (2026-07-30): **The resource logger counts itself as a server process.** It is
  launched with `sys.executable` = `/workspace/vllm/.venv/bin/python`, whose path contains
  "vllm", so `classify_procs` files it under `server`. Measured from the samples after each
  server is stopped, the offset is **1-4 % of one core**. Left uncorrected on purpose:
  changing the instrumentation between sessions would break comparability for a 2-3 %
  offset on a secondary metric, and the idle tail of every resources CSV gives the baseline
  if it is ever needed. It does not affect D27 - subtracting it makes the 0.5B/7B server-CPU
  ratio slightly larger (3.8x), not smaller
- D37 (2026-07-30): **The smoke test could not fail, and did not.** `make_figures` writes an
  empty PNG when a figure receives no data, and the test only counted files, so when the
  D30 manifest keying broke `load_runs` for the synthetic campaign (`result_json` there is a
  bare filename, giving an empty campaign key) it reported "rc=0, 8/9 figures" while every
  panel was blank and 0 runs had loaded. `load_runs` now falls back to the run_id when it is
  unambiguous and skips with a WARN only when several campaigns claim it; the smoke test
  asserts a non-zero loaded-run count and the full figure set, and returns non-zero
  otherwise. Verified: it fails loudly when `load_runs` is broken again on purpose
- D38 (2026-07-30): `--ready-timeout` default raised 900 s -> 1800 s. Cold-pod boot is
  dominated by `import vllm` off the network volume (session A's first boot: 588 s total,
  of which ~400 s import, with engine init only 138 s), and session C adds a 15 GB weight
  download and NCCL init across 2-4 workers. A false timeout costs a whole boot's worth of
  runs; a generous one costs nothing unless the server is genuinely broken
