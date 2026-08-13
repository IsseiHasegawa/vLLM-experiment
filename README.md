# vLLM Serving Performance Analysis: Phase-Level Measurement of Prefill and Decode

An instrumented fork of vLLM V1 that separates prefill from decode timing, and the
measurement campaign built on it: 232 runs over arrival rate, dataset, model size, GPU
count and parallelism strategy, on NVIDIA A40 GPUs.

**Report** — [`report/report.pdf`](report/report.pdf) (short) ·
[`report/main.pdf`](report/main.pdf) (full, with appendices). Appendix B maps each
requirement of the task to where it is addressed.

**Instrumented fork** —
[IsseiHasegawa/vllm @ `instrumentation`](https://github.com/IsseiHasegawa/vllm/tree/instrumentation):
three files, ~190 lines, based on `702f4814`.

## Instrumentation

vLLM V1 already computes per-phase timestamps internally. The patch writes them out
rather than adding new timers, which keeps the numbers cross-checkable against vLLM's own
Prometheus histograms. Three layers, logged independently:

| Layer | Output | Fields |
|---|---|---|
| Request axis | `requests.jsonl` | `queued_s`, `prefill_s`, `decode_s`, `inference_s`, `e2e_s`, token counts |
| Step axis | `steps.jsonl` | `sched_s`, `exec_s`, batch composition, `n_waiting`, `kv_usage` |
| Resource | external logger, 1 Hz | GPU SM / memory-controller / VRAM / power, per-process CPU |

A phase is an attribute of a request; a step is a batch that mixes both, because chunked
prefill is on by default. Neither axis alone separates time and resource use by phase.
The logger is gated on `VLLM_PHASE_LOG_DIR`, so one binary runs instrumented or not —
which is what makes the C1 overhead control possible.

## What was measured

| Factor | Levels |
|---|---|
| Arrival rate | 1–8 req/s (7B), 1–32 req/s (0.5B), offline, and closed loop at concurrency 1–128 |
| Dataset | ShareGPT (conversational) and random (256/128 fixed, plus 512/128 and 128/512) |
| Model | Qwen2.5-7B-Instruct, Qwen2.5-0.5B-Instruct |
| GPUs | 1, 2 and 4 × A40 48GB, compared within a single instance |
| Parallelism | tp = 1/2/4, and pp = 2 against tp = 2 at equal device count |

232 runs across four sessions, 231 analysed. Two controls: instrumentation on versus off
(C1), and an anchor condition repeated in every session (A1, spread 0.97 %).

## Layout

```
report/          main.md (full paper), report.md (short), appendix/, build scripts
scripts/         make_matrix.py, run_experiments.py, resource_logger.py, analysis, verification
scripts/plots/   make_figures.py regenerates all thirteen figures
configs/         matrix.csv — the 232-run matrix, generated rather than hand-written
results/         raw/ per-run JSONL and CSV, manifest.csv, c1_control.txt
figures/         generated figures
docs/            decisions.md — the decision log kept during the project
```

## Reproduce

```bash
python3 scripts/make_matrix.py
python3 scripts/plots/make_figures.py
python3 scripts/verify_report_numbers.py
```

`verify_report_numbers.py` re-checks every number in the report against the raw logs.
Re-running the measurements themselves needs GPUs; Appendix C of the report gives the
full command and the environment.
