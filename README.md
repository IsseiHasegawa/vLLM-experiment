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
- D27 (2026-07-29, revised 2026-07-31): 0.5B saturates at ~18-20 req/s with SM
  <= 59 % and memory controller <= 45 %, while server CPU doubles vs 7B (169 %
  vs 55 %). The original hypothesis blamed the *scheduler*; the step log refutes
  that specific claim: sched_s/(sched_s+exec_s) is only 8.0 % on S3_r32 (2.4 %
  on 7B). The correct statement is about step granularity: the 0.5B mean exec_s
  is 6.3 ms vs 34.6 ms on 7B, so per-step fixed costs outside the timed sections
  (step loop, kernel launch, sampling, detokenisation) occupy a far larger share
  of each step, the CPU works ~5x as many steps per second, and the GPU is left
  idle between launches. Evidence: high server CPU + low SM + short exec_s;
  the scheduler-specific share is measured and small.
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
- D31 (2026-07-29, corrected 2026-08-03): **C1 is analysed paired by seed.** The arms
  share seeds (D14), so the paired difference isolates logging: TTFT p50 +2.97 %,
  TPOT p95 +1.05 %, request throughput -0.26 %, and all seven metrics move in the same
  direction. Reported claim: logging costs <=3 % on latency and <=0.3 % on throughput, so
  instrumented latencies are upper bounds and instrumented throughput a lower bound.
  This revises the session-A quality gate in PLAN 6.4, which required the string "no
  instrumentation effect detected": a small, bounded, correctly signed effect is the
  expected outcome, not a failure.
  **The original entry claimed a detection at p=0.010; that was wrong** - see D60. With
  three seed pairs no metric reaches significance (smallest two-sided p is 0.125). The
  claim this control supports is the *bound*, not a detected effect
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
- D39 (2026-07-31): **Environment pinning across sessions.** `env_freeze.txt` cannot be
  used directly as a constraints file: it records the state *after* the `datasets>=3.0`
  upgrade, which is not a single consistent resolution (numpy 2.5.1 violates
  mistral-common's `<2.4` and numba's `<2.5`; session A ran fine regardless). The working
  recipe, used for session B: take env_freeze.txt, substitute the three post-upgrade
  packages back to their install-time values (numpy==2.3.5, datasets==1.1.1,
  fsspec==2026.7.0), drop the `-e` line, install with `uv pip install -r`, add build
  deps (setuptools_rust setuptools_scm cmake wheel), then
  `VLLM_USE_PRECOMPILED=1 uv pip install --editable . --no-deps --no-build-isolation`,
  then re-run the datasets upgrade. Verified equivalent: A1c within 0.67 % of A1a on
  throughput and within 1.9 % on every latency metric
- D40 (2026-07-31): **Editable-install checks must run outside the source tree.** Inside
  /workspace/vllm the CWD shadows site-packages, so `import vllm` succeeds even when
  nothing is installed; session B lost one runner start to this
  (`FileNotFoundError: 'vllm'`). Canonical check: `cd /tmp && which vllm && python -c
  "import vllm; print(vllm.__file__)"` expecting the .venv bin path and the source-tree
  __init__.py
- D41 (2026-07-31): **Per-run fixed cost is instance-dependent.** Session A: 91-105 s
  (eu-se-1); session B: ~260 s (ca-mtl-1), same commands, 3.4x. The measured section is
  unaffected (A1c matched A1a to 1 %); only wall-clock forecasts change. D21's "60 %"
  is a session A figure, not a constant. Set UV_CACHE_DIR=/root/.cache/uv and
  TMPDIR=/root/tmp on every pod: /workspace quota exhaustion killed one install attempt
  (D20's failure mode through a different path)
- D42 (2026-07-31): **C2x (c=128, 3 runs) added to session C** to settle figure 10's
  right edge: c=64 measured cv 17.8 % on throughput (output-length tail), so flat-vs-
  rising was not decided. C2x runs on the session C instance inside the tp=1 boot;
  figure 10 draws it as a separate open-marker series joined via the A1d/A1c anchor
- D44 (2026-08-01): **Pipeline parallelism (P1, pp=2) added.** The task lists "test
  performance using varying numbers of GPUs" and "enable and evaluate parallel processing
  options" as separate requirements; sessions A-C answered both with tensor parallelism
  alone. P1 runs the G1/G2/G4 rate grid at pp=2, i.e. the same two GPUs as G2 but a
  different communication pattern: TP all-reduces at every layer, PP hands the activation
  across once per stage boundary. With P2P disabled (D43) every transfer stages through
  host memory, so message count dominates and the tp=2 vs pp=2 pair is a clean comparison
  of communication strategy at fixed device count. Matrix is now 232 rows; the runner
  gained a `pp` column, includes pp in the boot key, and applies the D43 NCCL workaround
  whenever tp>1 **or** pp>1. tp=1/pp=1 server commands are unchanged, verified by dry-run
- D45 (2026-08-01): **Why tensor parallelism helps, measured at step granularity - and the
  KV-cache hypothesis is refuted.** The session C write-up first attributed the tp speedup
  to "memory bandwidth plus doubled KV cache allowing larger batches". The step log rules
  the second half out: at rate 8 `kv_usage` averages 2.1 % (tp=1), 0.8 % (tp=2), 0.3 %
  (tp=4) and the running batch is essentially unchanged (25.7 / 24.5 / 23.6). KV capacity
  was never the constraint. The entire effect is a shorter model-execution step at constant
  batch size:
    * decode-only steps (n_ctx_toks == 0): 32.2 -> 22.4 -> 18.1 ms, i.e. 1.44x and 1.78x
    * steps carrying prefill (n_ctx_toks > 0): 73.3 -> 64.5 -> 58.2 ms, i.e. 1.14x and 1.26x
  The two phases respond differently because decode is memory-bandwidth bound (each GPU
  streams 1/N of the weights per step, so aggregate bandwidth converts directly into time)
  while prefill is already compute-dense at ~300 context tokens per step and gains far less
  relative to the added all-reduce.
  Speedup also shrinks as the batch grows (decode-only, matched batch): 1.54x/2.07x at
  batch 8-16, 1.35x/1.67x at 16-32, 1.28x/1.48x at 32-64 - at small batch nearly the whole
  step is weight streaming, which sharding divides by N; at large batch the weights are
  amortised over more tokens while the all-reduce volume grows with the batch.
  Finally, step speedup only becomes throughput once the system saturates: at rate 1 the
  tp=4 step is 2.5x faster yet throughput rises 4.6 %, because below saturation the arrival
  rate sets throughput and a faster step only buys idle time. At rate 8 and inf the two
  converge (+52 %, +55 %). Figure 11 carries all three panels
- D46 (2026-08-01): Figure 11 added (`scripts/plots/fig11_step_parallelism.py`). It is the
  only figure drawing on the step log's phase split, and it is where the assignment's
  "time and resource usage during the prefill phase and the decode phase" requirement meets
  the "parallel processing options" requirement. The smoke test now expects 11 figures
- D47 (2026-08-01): **Latency panels moved to a log y-axis; figure 5 rebuilt as a CDF.**
  Reviewing the rendered figures against the underlying numbers exposed two presentation
  faults that changed what the figures appeared to claim.
  *Latency panels (fig01, fig02, fig04-left, fig06-left/middle, fig07-left/middle).* Once
  the offline (`inf`) point shares an axis with the finite rates, the range is ~90x
  (151 ms at rate 1 to 6767 ms offline on 7B/ShareGPT). On a linear axis the whole finite
  grid occupied ~1 % of the panel and rendered as a flat line, so figure 1 read as "TTFT is
  constant until capacity, then explodes". It is not: p95 TTFT rises 62 % from rate 1 to
  rate 8, and figure 4's dataset gap at rate 8 (ShareGPT 245 ms vs random 211 ms) was
  invisible for the same reason. A log axis with ticks at 1/2/5 per decade raises the
  finite-rate share to ~10-14 % and separates p50 from p95. Throughput panels stay linear:
  they span well under one order of magnitude, and the ShareGPT/random crossing near rate 5
  in figure 4 and the tp ordering in figure 7 read best on a linear scale.
  *Figure 5.* The density histogram could not show these two workloads together. `random`
  is a fixed length, so its density is a delta spike that owned the y-axis and flattened
  both ShareGPT curves to invisibility - four series were drawn and one was visible - while
  the nominal ShareGPT tail (66 076 tokens) stretched the x-axis until the served data
  occupied 1.5 % of the panel, and the numeric caption collided with the axis labels.
  Rebuilt as a cumulative distribution on a log x-axis: bounded y, fixed lengths render as
  clean steps, four orders of magnitude fit, and the nominal/realised separation at the
  ~1024-token admission cutoff (D26) is now the visible feature it should always have been
- D48 (2026-08-01): **Figure 3 now plots two definitions of "achieved rate", and the
  saturation argument is moved off it.** D24 established that the client's
  `request_throughput` is completed / measured-duration, and that the measured duration
  runs to the *last* completion. Measured on S1 that drain is 12.2 s at rate 1 and 32.1 s
  at rate 8, so the reported rate is biased low at every point - 0.95 at an offered rate of
  1, where the server is plainly keeping up. A reader taking a capacity off that curve
  reads a definitional artefact.
  The alternative - completions at or before the last arrival, over the arrival span -
  gives a materially different curve: 0.97 / 1.90 / 2.75 / 3.48 / 4.08 / 4.59 / 5.18 at
  rates 1-8, against 0.95 / 1.71 / 2.33 / 2.83 / 3.20 / 3.37 / 3.54 for the client metric.
  The first plateaus near 3.5; the second is still climbing at rate 8. It is not unbiased
  either: requests in flight when arrivals stop are never counted, a loss of roughly
  rate x latency that grows with load.
  Both are drawn. Neither is the capacity. The substantive point is that on this system
  achieved throughput is a weak saturation detector at all: vLLM admits every arrival into
  the running batch, so backlog appears as batch growth and rising latency rather than as a
  queue or a throughput ceiling (queueing delay is ~0 everywhere in session A). The
  saturation claim therefore rests on latency (figure 1: p95 TTFT +62 % across the finite
  grid) and on the closed-loop curve (figure 10), with figure 3 reporting what the standard
  tool reports and showing how far that depends on the definition. This closes the
  operational-definition task that PLAN scheduled for 8/3
- D49 (2026-08-01): **Overlays with different rate grids were plotting the offline point
  at two different x positions.** `xpos` placed 'inf' one of each series' own steps past
  that series' own maximum, so in figure 6 the 7B offline point (grid 1-8) landed near
  x=9 while the 0.5B one (grid 1-32) landed near x=36. The 7B curve therefore appeared to
  spike at "rate 9" - a rate that was never run - and the panel carried two offline
  marker lines. `plot_series` also called `set_xticks` per series, so the last series'
  grid overwrote the axis and the other model's rates lost their labels; on a linear axis
  the surviving 0.5B grid crowded 1, 2 and 4 into an unreadable clump.
  Callers now pin a shared `inf_x` for the whole figure and own the tick set. Figure 6
  additionally switches to a log x-axis, since its union grid spans 1-32; the labelled
  ticks are the powers of two and the remaining rates (3, 5, 6, 12, 24) get unlabelled
  minor ticks. Figures 4 and 7 span only 1-8 and stay linear, where every rate including
  3, 5 and 6 can be labelled. Figures 1-3 draw a single grid and are unaffected
- D50 (2026-08-01): **The offline point is drawn detached from the finite-rate line.**
  'inf' is not the next step of the sweep: it sends all requests at once, so it is a
  different arrival process, and on ShareGPT it is not a stable measurement either (seed
  spread 41 %, D25). Connecting it invited two misreadings. In figure 6 the 7B grid stops
  at 8 while the shared axis now runs to 32 (D49), so a connecting line swept across
  rates 16 and 24 that were never run for that model - a reader could take a value off
  it. More generally a joined line presents the offline point as "the curve continued",
  which is exactly the reading D25 rules out. Every rate sweep now draws the finite grid
  as a line and the offline point as a bare marker at the shared position, with the
  existing dotted vertical rule separating them
- D51 (2026-08-01): **Figure 6's throughput panel is log-scaled; the scale is chosen per
  figure, not by a rule.** D47 decided log axes by metric type - latency log, throughput
  linear - which is too crude. In figure 6 the two models' output throughput spans
  181-3783 tok/s (21x), so on a linear axis the 7B curve occupied 13 % of the panel and
  its saturation near 675 tok/s was unreadable; that saturation is half of what a
  model-size comparison is for. On a log axis it occupies 38 %. Figure 4's throughput
  panel stays linear: its two series overlap and cross near rate 5, and a log axis would
  flatten that crossing. A numeric threshold separating 14x (figure 4) from 21x
  (figure 6) would have been arbitrary, so `_overlay` takes an explicit `log_fields`
  argument and figure 6 passes `output_throughput`
- D52 (2026-08-01): **Figure 6 compares the two models at equal offered rate, not equal
  utilisation.** 7B saturates near 3.5 req/s and 0.5B near 18-20, so at rate 4 the 7B is
  at ~114 % of its capacity while the 0.5B is at ~22 % of its. The panel therefore
  contrasts a saturated system with an idle one, and the resulting TTFT gap (195 ms vs
  52 ms at rate 4) is not a pure model-speed difference. The comparison is still the one
  a deployer wants - "at this load, which model?" - but the caption must state that
  utilisation is not matched, or "the 0.5B is 4x faster on TTFT" will be read as an
  intrinsic property
- D53 (2026-08-01): **Figure 8's invisible segments are now labelled, and both legends
  moved out of the plot area.** The stacked phase bars have the same failure mode the
  first version of figure 5 had: decode is 98.0 % of a prefill-heavy request (512 in /
  128 out) and 99.7 % of a decode-heavy one, so the queued and prefill segments cannot be
  seen at all - three colours in the legend, one visible bar. That dominance *is* the
  result, and a strong one: quadrupling the input and quartering the output still leaves
  98 % of the request in decode, because 512 prompt tokens take 126 ms of prefill against
  6304 ms of decode. But a segment the reader cannot see reports no number, so each bar
  now carries the decode value inline and the queued/prefill values on a leader above the
  bar (0.9 ms / 126 ms and 0.0 ms / 70 ms). On the right panel the three-line label for
  the residual was drawn inside the axes and sat on top of the prefill-heavy bar,
  colliding with its own percentage annotation; both panels now put the legend below the
  axes
- D54 (2026-08-01): **Figure 10 gains x error bars, and the pair of them is a result.**
  Only the latency spread was drawn, which showed the noise at low concurrency and hid it
  at high concurrency - the two are anti-correlated. Throughput CV is 0.4 % at
  concurrency 1 and 12.9 % at 128; p95 latency CV runs the other way, 20.6 % down to
  6.8 %. With one request in flight the aggregate rate is just 1 / mean latency over 60
  requests and is very stable, while p95 is decided by whichever long ShareGPT completion
  landed in that small sample. Saturated, the run's duration is set by the output-length
  tail so throughput becomes the noisy axis, while p95 over 200 requests is well
  determined. Both bars are now drawn at 45 % opacity behind the line so the two regimes
  of the curve - flat from c=1 to 8 (19.2 to 20.6 s), rising from c=16 (23.0 to 27.8 s) -
  stay readable. The throughput axis starts near the leftmost point instead of at zero
- D55 (2026-08-01): Figure 9's CPU headroom note read "ceiling: 9 vCPU = 900 %" on a
  panel whose axis tops out near 45 %, a number 20x off the scale that the reader cannot
  relate to anything drawn. It now states the ratio: the vLLM server peaks at 43 % of one
  core, which is 4.7 % of the 900 % the container has. That is the bottleneck claim the
  panel exists to support - on 7B the host CPU is nowhere near the constraint, which is
  what makes the 0.5B contrast in D27 meaningful
- D56 (2026-08-01): Figure 11 presentation fixes. Its right panel carries a two-line title,
  which collided with the suptitle because this figure never got the `tight_layout` change
  D47 applied to the sweep figures; the title is also shortened. The left panel printed
  "1.00x" on both tp=1 bars - a ratio that is 1 by construction, but which reads as a
  measured speed-up of one - and now prints "baseline". The right panel's legend has five
  entries spanning two y-axes and sat over the middle of the plot, which is exactly where
  the step-time and throughput-gain curves cross; it moves below the axes
- D57 (2026-08-03): **Session D: pipeline parallelism (P1, pp=2), 24 runs.** Instance
  chosen so GPU0-GPU1 is PXB on one NUMA node, the same interconnect class session C used
  for G2; a first 4-GPU host offering PIX was discarded because it would have changed the
  interconnect and the parallelism strategy at once. Records complete: 5570 = 50 warm-up
  + 24 x 230, n_cached all zero, 24/24 runs at 200 completed.
  Two operational findings. First, **vLLM's pipeline-parallel executor requires the
  batch-queue step path**, so the instrumented step function is never called and
  steps-*.jsonl is not written at all; the D9 guard aborted the first attempt on this and
  was relaxed for pp>1 only. Per-request records are unaffected, which is what this
  analysis needs. Second, on the same host where **tensor parallelism hung in NCCL
  communicator setup until P2P was disabled (D43), pipeline parallelism came up**: PP
  exchanges activations with point-to-point send/recv rather than all-reduce, and the
  server log shows NCCL creating an unbatched P2P communicator. The communication pattern
  decided not just performance but whether the configuration started at all
- D58 (2026-08-03): **The two parallelism strategies act on different phases, and figure
  12 shows it.** At rate 5, against tp=1: tp=2 cuts prefill 13 % and decode 28.5 %; pp=2
  cuts prefill 10.5 % and decode 1.7 %. The pattern holds across the whole rate grid
  (pp=2 decode: -1 % to -5 %; tp=2 decode: -27 % to -40 %).
  Mechanism: a decode step emits one token per request, and under PP that token crosses
  stage 0 then stage 1 in sequence, so there is no parallelism within a step. TP splits
  every layer, so both GPUs work on the same token and aggregate memory bandwidth - what
  decode is bound by (D45) - doubles. Prefill processes hundreds of tokens at once, so
  pipeline stages have work to overlap and PP does help there.
  Consequences, all measured: decode is ~98 % of a request (D53), so only TP moves
  throughput (pp=2 is within 0.6 % of tp=1 up to rate 4). TTFT is prefill plus queueing
  plus the frontend path outside the server timestamps (D-, §4.2),
  so PP does move it: -9 % to -12 % on the finite grid. Under burst arrival the effect is
  much larger - at rate inf the mean queueing delay falls from 2809 ms (tp=1) to 1270 ms
  (pp=2), a 55 % reduction, and TTFT p95 from 6940 ms to 3884 ms. The earlier one-line
  summary "pp=2 does not help" was wrong: it looked only at achieved req/s
- D59 (2026-08-03): The synthetic campaign only ever wrote phase records for I1 and I2, so
  figure 11 rendered an empty panel while still counting as "made" - the exact failure D37
  was written to stop, surviving in a corner the D37 fix did not reach. It now writes
  per-request records for every group and step records for every group with pp=1, which
  also mirrors the real constraint that pipeline parallelism produces no step log (D57).
  Figure 12 skips with a message instead of raising when a group has no phase records
- D60 (2026-08-03): **The paired p-values were computed with a normal approximation and
  were wrong.** `analyze_c1.py` used `erfc(|t|/sqrt(2))`, i.e. the z-test. With three seed
  pairs there are two degrees of freedom, where the t distribution has far heavier tails,
  so every p-value was understated by a factor of 4 to 12:

  | metric | t | reported | correct (t, df=2) |
  |---|---|---|---|
  | TTFT p50 | 2.56 | 0.010 | **0.125** |
  | TPOT p95 | 2.28 | 0.023 | **0.150** |
  | Request throughput | -1.98 | 0.048 | **0.186** |

  Corrected, **no metric reaches p<0.05** and the verdict flips from "instrumentation
  effect detected on TTFT p50" to "no effect above the +-2 % practical bound". D31 and D45
  both quoted the 0.010 figure and are amended. The script now computes the two-sided
  p-value from Student's t via the regularised incomplete beta (verified against the
  textbook critical value t=4.303 at df=2 -> p=0.050), and Welch's test uses
  Welch-Satterthwaite degrees of freedom instead of the same approximation.
  The "7/7 metrics agree in direction" line was also overstated and is now qualified in
  the output: run duration is 200 / request throughput, output throughput is request
  throughput times a seed-fixed output length, and the p50/p95 pairs describe one
  distribution, so there are roughly three distinct quantities, not seven independent
  ones. The consistent sign is worth reporting; it is not a significance test.
  What the control supports is an observed mean difference, not a bound: **+2.97 % on
  latency and −0.26 % on throughput**, with no metric reaching significance. At n=3 the
  95 % interval on the largest difference spans roughly −1.8 % to +7.7 %, so the control
  rules out a large effect but does not establish a tight upper bound.
