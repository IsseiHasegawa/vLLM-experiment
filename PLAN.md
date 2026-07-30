# PLAN.md — vLLM Benchmarking Assignment (working plan, rev. 2)

> Internal working document. **Delete before the repo is made public / submitted.**
> Rev. 2 written 2026-07-27 after the pilot; rev. 1 is in git history.
> Owner: Issei Hasegawa (GitHub: IsseiHasegawa).

---

## 0. How to resume in a new chat

1. Open a chat **inside the same Claude project**.
2. Say `"<date> セッション開始"`.
3. If the repo is public, give the URL and ask Claude to read `PLAN.md`,
   `README.md`, `configs/matrix.csv`. If private, paste this file.
4. Claude produces that day's runbook from §6.

Claude writes tooling, runbooks, analysis and the report draft, and audits pushed
state through the public fork. Claude does not run the experiments. AI assistance
is explicitly permitted by the assignment author and is acknowledged in the report.

---

## 1. What changed in rev. 2, and why

The pilot (2026-07-27, 5 runs) produced three findings that invalidated parts of
rev. 1. The redesign is a direct response to each.

**Finding 1 — the rate grid missed the operating region.** Sustainable capacity
on one A40 is **3.2 req/s** (7B, ShareGPT) and **4.5 req/s** (7B, random
256/128). [Session A: the 3.2 figure is confirmed; the 4.5 figure was a drain
artefact and the true value is ~13.8 req/s - see D24/D28/D32.] The rev. 1 grid {1,2,5,10,20} put three of five points deep in the
saturated plateau. New grid: **{1,2,3,4,5,6,8,inf}**.

**Finding 2 — above capacity there is no steady state.** Queueing delay is ~0
everywhere (p50 = 0.0 ms): vLLM admits every arrival into the running batch, so
overload appears as batch growth, not as a queue. Splitting each run into
quarters shows the consequence — at rate 20 the median decode time grows
2.4 s → 22.6 s across the run while prefill falls 180 → 105 ms. For offered load
above capacity an open-loop system has no stationary distribution, so "the p95
latency at rate 8" is a property of the 200-request burst, not of the rate.
Two responses: overload points are reported as *burst transients* with that
caveat stated, and a **closed-loop control (C2)** with a fixed concurrency limit
is added, where every point does have a steady state.

**Finding 3 — per-run overhead dominates wall clock, and is accepted.** 60 %
(measured on session A: 15 176 s wall vs 6 066 s measured over 91 runs; the pilot's
75–82 % sampled only short runs, and the share ranges 31 % at rate 1 to 92 % at
S3 rate 32). Fixed cost is 91–105 s per run, independent of dataset. Formerly: 75–82 %
of each run's wall time is `vllm bench serve` process startup (`import vllm`),
not measurement. Driving the benchmark in-process would cut the campaign from
~6 h to ~2 h. **Rejected**: separate processes give complete isolation between
runs, and the command recorded in the manifest is exactly what executed, so any
single run is reproducible by copy-paste. Buying speed by weakening isolation
and reproducibility is the wrong trade here; the cost of the slower path is ~$3.

Two further changes come from methodological review rather than from the data.

**Seeds now vary across repetitions** (1/2/3 instead of 42/42/42), with the same
seed set reused for every condition. Fixed seeds made the error bars represent
system noise only, understating experimental uncertainty; varying them keeps
comparisons paired while the bars now also cover prompt sampling and arrival
jitter. This revises decision D6.

**GPU count is measured within one instance.** Rev. 1 compared S1 (1-GPU pod)
with S4 (2-GPU pod), confounding parallelism with the instance. Rev. 2 measures
tp=1, tp=2 and — if capacity allows — tp=4 **on the same multi-GPU pod** (groups
G1/G2/G4), so the only difference between the series is the parallelism setting.

---

## 2. Assignment and context

**From**: Dr. Minlan Yu (Harvard). **Nature**: screening task attached to a
request to join her group. Her group publishes on LLM serving (DUCHESS, HACK,
prefix-aware data parallelism), so the report targets a serving-systems reader.

| Requirement | Satisfied by |
|---|---|
| Deploy from source; clone, change, recompile | Fork + editable install (session 0) |
| Instrument for latency (E2E + per-phase) and throughput | 3-file patch, validated in session 0, controlled in C1 |
| Time **and resource** usage in prefill and decode | Phase logs + 1 Hz resource logger; figures 8, 9 |
| >=2 datasets, documented | ShareGPT + random; figures 4, 5 |
| Vary request arrival rate | Open-loop sweeps S1-S3, G1-G4; figures 1-3 |
| >=2 models / sizes | 7B vs 0.5B on the same instance; figure 6 |
| Vary GPU count | tp=1/2/4 on one instance; figure 7 |
| Document CPU performance | Resource logger + `sched_s`; figure 9 |
| Enable/evaluate parallelism | Tensor parallel, interconnect recorded |
| Bottleneck analysis | Figures 3 + 8 + 9 + 10 cross-read |
| Figures comparing metrics | 10 figures, error bars over 3 repetitions |
| "Send me your results" | Report PDF + public repo + email |

**Target submission 2026-08-12; hard limit 2026-08-15.** (Rev. 1 targeted 08-10;
the expanded design costs two days and buys the controls in §4.)

**Report**: IMRaD, 8-12 pages + appendix, ~12 references, written in English by
the author from a Claude draft, with a one-line AI-assistance acknowledgment.

---

## 3. Experiment matrix (205 runs)

Generated by `scripts/make_matrix.py`, which carries the rationale in code.
Fixed for every run: `--ignore-eos`, `--num-warmups 30`, `--temperature 0`,
percentiles 50/95/99, 3 repetitions with seeds 1/2/3.

| Group | Runs | Model | Dataset | tp | Sweep | Purpose |
|---|---|---|---|---|---|---|
| A1a | 3 | 7B | sharegpt | 1 | r=5 | Session-start anchor **and** enabled arm of C1 |
| C1off | 3 | 7B | sharegpt | 1 | r=5 | Same condition, phase logger **disabled** |
| S1 | 24 | 7B | sharegpt | 1 | r in {1,2,3,4,5,6,8,inf} | Primary rate sweep |
| S2 | 24 | 7B | random 256/128 | 1 | same | Dataset effect |
| I1 | 3 | 7B | random 512/128 | 1 | r=5 | Prefill-heavy phase characterisation |
| I2 | 3 | 7B | random 128/512 | 1 | r=5 | Decode-heavy phase characterisation |
| A1b | 3 | 7B | sharegpt | 1 | r=5 | Session-end anchor (drift vs A1a) |
| P0 | 1 | 0.5B | sharegpt | 1 | inf | Capacity probe; confirms/extends the S3 grid |
| S3 | 27 | 0.5B | sharegpt | 1 | r in {1,2,4,8,12,16,24,32,inf} | Model size, same instance as S1 |
| A1c | 3 | 7B | sharegpt | 1 | r=5 | Anchor for session B |
| C2 | 21 | 7B | sharegpt | 1 | concurrency in {1,2,4,8,16,32,64} | Closed-loop control |
| S2b | 15 | 7B | random 256/128 | 1 | r in {5,10,12,16,20} | Extends S2 past 50 % utilisation (D32) |
| A1d | 3 | 7B | sharegpt | 1 | r=5 | Anchor for session C |
| C2x | 3 | 7B | sharegpt | 1 | concurrency=128 | Right edge of figure 10 (D42) |
| G1 | 24 | 7B | sharegpt | 1 | full grid | GPU-count baseline, on the multi-GPU instance |
| G2 | 24 | 7B | sharegpt | 2 | full grid | tp=2 |
| G4 | 24 | 7B | sharegpt | 4 | full grid | tp=4, only if 4 GPUs are secured |

`num_prompts` is 200 except in C2, where the concurrency limit already bounds the
run (60 at c<=2, 120 at c in {4,8}, 200 above).

Anchors appear in **every** session and at both ends of session A. Together they
support three claims: no drift within a session (A1a vs A1b), consistency across
boots of one instance (A1a vs S1 r=5), and consistency across instances and days
(A1a vs A1c vs A1d).

### Figures

| # | Content | Source |
|---|---|---|
| 1 | rate -> TTFT p50/p95 | S1 |
| 2 | rate -> TPOT, ITL | S1 |
| 3 | rate -> throughput, achieved vs offered (knee) | S1 |
| 4 | figs 1+3 overlaid, ShareGPT vs random | S1 vs S2 |
| 5 | token-length distributions of both datasets | `dataset_stats.py` + realised `n_prompt` from phase logs |
| 6 | rate -> TTFT/TPOT/throughput, 7B vs 0.5B | S1 vs S3 |
| 7 | rate -> TTFT/TPOT/throughput, tp=1/2/4 | G1/G2/G4 |
| 8 | per-request phase split + TTFT decomposition | I1, I2 phase logs |
| 9 | resource utilisation vs rate (SM, memory controller, CPU) | resource logger + manifest slices |
| 10 | closed-loop latency vs throughput | C2 |

The C1 control is a Methods table from `scripts/analyze_c1.py`, not a figure.
Figure 8's "queued" layer will be near zero — a result, not a defect, and is
discussed rather than hidden.

---

## 4. Controls and threats to validity

| Threat | Control |
|---|---|
| Instrumentation perturbs the system | **C1**: same condition with logging on/off, adjacent in time, same seeds; Welch's t plus a +/-2 % practical-equivalence bound |
| Overload points have no steady state | **C2** closed-loop sweep; open-loop overload points labelled burst transients |
| Drift within a session (thermal, noisy neighbour) | **A1a vs A1b** at the two ends of session A |
| Cross-instance / cross-day differences | **A1c, A1d** anchors in sessions B and C |
| GPU count confounded with instance | **G1/G2/G4 on one instance** |
| Client becomes the bottleneck | `RATE_SHORTFALL` + `cpu_client_pct`; a shortfall with an unsaturated client indicates the server, not the harness |
| Sampling non-determinism | `--temperature 0`; output length pinned by `--ignore-eos` |
| Warm-up insufficiency | `--num-warmups 30` (the pilot showed an elevated first quarter at 10) plus a discarded boot warm-up per server |
| Prompts longer than `max_model_len` | ShareGPT reaches 66 076 tokens vs a 32 768 limit; confirm and document the harness's handling, and report the realised `n_prompt` distribution beside the nominal one |
| Cross-run state leakage | Every run is a fresh process; the manifest records the exact command |

---

## 5. Schedule

| Date | Work | GPU |
|---|---|---|
| 7/27 Mon | Rev. 2 tooling: matrix generator, runner changes, C1 analysis, figure 10 | — |
| 7/28 Tue | Conference. Optional: review the diff, read the vLLM paper | — |
| 7/29 Wed | **Session A** — 91 runs, 1xA40, ~7 h, mostly unattended | 1x |
| 7/30 Thu | **Session B** — 39 runs (C2 + S2b), ~3.5 h; start analysis | 1x |
| 7/31 Fri | **Session C** — 54 runs (tp=1/2) or 78 (tp=1/2/4, incl. C2x), 5-7 h | 2x/4x |
| 8/1 Sat | Analysis: C1 verdict, anchors/drift, figures 1-4, 6 | — |
| 8/2 Sun | Analysis: figures 7, 8, 9, 10 | — |
| 8/3 Mon | Analysis: bottleneck synthesis; operational definition of the saturation point | — |
| 8/4 Tue | Buffer / re-measurement day 1 | maybe |
| 8/5 Wed | Buffer day 2. **Data freeze** at end of day | maybe |
| 8/6 Thu | Hand results to Claude -> English draft -> read, list questions | — |
| 8/7 Fri | Write: Abstract + Introduction + Background (thin, ~0.5 p) | — |
| 8/8 Sat | Write: Methodology (controls, C1 table, steady-state caveat) | — |
| 8/9 Sun | Write: Results + Bottleneck Analysis | — |
| 8/10 Mon | Write: Discussion + Limitations + Conclusion + acknowledgment | — |
| 8/11 Tue | Mock Q&A (15 questions); repo cleanup, secret scan, README, tag | — |
| 8/12 Wed | Final read, PDF, make public, **submit** | — |
| 8/13-15 | Reserve before the hard limit | — |

Session A exceeds the nominal 3 h/day because the runner is unattended: start it,
check in every couple of hours with `tmux attach`, collect at the end. If the day
must be split, run the 7B block (A1a...A1b) and the 0.5B block (P0, S3)
separately — but keeping them on one instance is what makes figure 6 free of
instance confounding, so split only if necessary.

### Budget

| Item | Hours | Rate | Cost |
|---|---|---|---|
| Session A | 7.0 | $0.45 | $3.15 |
| Session B | 3.5 | $0.45 | $1.58 |
| Session C, 2xA40 | 4.5 | $0.89 | $4.01 |
| Session C, 4xA40 instead | 6.1 | $1.78 | $10.86 |
| Reserve / re-measurement | 3.0 | $0.45 | $1.35 |
| **Total** | | | **$10-17** |

Balance is $23.95; no top-up needed. Cost is not a constraint — the real risks
are forgetting to terminate a pod and losing a long session to a mistake.

---

### Report format (decided 2026-07-31)

Paper-style (IMRaD) but sized for a screening task, not a venue: 6-8 pages of
body text plus the 10 figures and an appendix. Markdown -> PDF; no LaTeX.
Related Work stays thin (~0.5 p: vLLM/PagedAttention, Orca, DistServe).
Methodology keeps the validity material (instrumentation design, Prometheus
cross-check, C1, anchors) but states it compactly; it is the differentiator,
not the headline. Results let the figures speak - one paragraph per figure.
References: hand-written list, <= 10 entries.

## 6. Runbooks

### 6.1 Start of every GPU session

1. RunPod -> Billing: note the balance.
2. Pods -> Deploy. **Clear the Filter** (a stale filter makes A40 look "Out of
   capacity"), **Available** tab, A40, template **Runpod Pytorch 2.8.0**, Any
   region, disks 30 GB / 50 GB. Session C needs 2 (or 4) GPUs.
   Fallback order: **L40S -> RTX 6000 Ada -> RTX A6000** (all 48 GB); S1-S3 must
   share one GPU type.
3. Connect -> **Enable web terminal**. SSH keys are not used.
4. `nvidia-smi`; for tp>1 also `nvidia-smi topo -m`.
5. **Session C only:** `nvidia-smi -L | wc -l` before starting. Run
   `--only A1d,C2x,G1,G2,G4` only if it prints 4; with 2 GPUs use
   `--only A1d,C2x,G1,G2` and take the tp=4 fallback in section 8. The runner does
   not pre-check the GPU count, so a tp=4 boot on a 2-GPU pod just fails after
   the ready timeout.

```bash
tmux new -s sess
cd /workspace
git clone -b instrumentation https://github.com/IsseiHasegawa/vllm.git
read -p "PAT: " GH_PAT          # visible on purpose; the web terminal truncates read -s
git clone https://IsseiHasegawa:${GH_PAT}@github.com/IsseiHasegawa/vLLM-experiment.git
git -C vLLM-experiment remote set-url origin https://github.com/IsseiHasegawa/vLLM-experiment.git
clear; history -c
cd vllm
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.12 --seed && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable ".[bench]" --torch-backend=auto
uv pip install -U "datasets>=3.0"                        # D10
export HF_HOME=/root/hf_cache && mkdir -p /root/hf_cache # D20
python -c "import psutil, vllm; print('deps ok')"
cd /workspace && wget -q --show-progress https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

**D20 matters**: `/workspace` is a network volume with a 50 GB quota and the venv
alone occupies 21 GB, so model weights must go to the container disk or the 7B
download dies with `Disk quota exceeded` (this killed the first pilot attempt).

### 6.2 Running

```bash
cd /workspace/vLLM-experiment
python3 scripts/run_experiments.py \
  --matrix configs/matrix.csv --manifest results/manifest.csv \
  --session A --results-dir results/raw/sessionA \
  --sharegpt-path /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
  --only <groups> [--dry-run]
```

| Session | `--only` | Runs | Boots |
|---|---|---|---|
| A | `P0` then `A1a,C1off,S1,S2,I1,I2,A1b,S3` | 91 | 5 |
| B | `A1c,S2b,C2` | 39 | 1 |
| C | `A1d,C2x,G1,G2` (+`,G4`) | 54 (78) | 2 (3) |

Always `--dry-run` first. The runner boots one server per consecutive
(model, tp, instr) block, aborts if the PHASE-INSTR batch-queue warning appears,
records a manifest row per completed run, flags `RATE_SHORTFALL`, and resumes
where it stopped if interrupted (rows already `ok` are skipped).

**In session A, stop after P0 and read the probe.** If the measured capacity is
far from ~24 req/s, adjust the S3 grid in `make_matrix.py`, regenerate, and
continue with `--only S3`. (Done on 2026-07-29: probe read 17.27 req/s, the grid
was kept - see D13.)

**Validate the instrumentation on every new pod (2 min, second terminal).** The
torch CUDA build follows the host driver (D22), so the phase logger has to be
re-checked per instance. A few minutes after boot 1's warm-up, while the server
is still up and the phase log holds only warm-up records:

```bash
python3 scripts/verify_session0.py \
  --phase-log-dir results/raw/sessionX/phase_logs \
  --bench-json results/raw/sessionX/bench/BOOT1_Qwen2.5-7B-Instruct_tp1.json \
  --metrics-url http://localhost:8000/metrics --expect-output-len 64 \
  | tee results/raw/sessionX/verification.txt
```

V2 cross-checks the phase log against vLLM's own Prometheus histograms, which is
an independent oracle; it only works while the server is running. Expect 0 FAIL.
Run it later in the session and V1d will fail harmlessly, because by then the log
also holds ShareGPT requests whose output lengths vary.

**Session C: pass `--ready-timeout 2400`** if the first boot is slower than
expected; the default is 1800 s (D38).

### 6.3 End of every session (never skip)

```bash
cd /workspace/vLLM-experiment
# session A only: run the C1 analysis before gzip, while the pod is still up,
# so a problem can still be re-measured
python3 scripts/analyze_c1.py --repo . --out results/c1_control.txt
date -u > results/raw/sessionX/SESSION_END.txt
git -C /workspace/vllm log --oneline -3 > results/raw/sessionX/vllm_commit.txt
gzip -f results/raw/sessionX/phase_logs/*.jsonl
nvidia-smi > results/raw/sessionX/nvidia_smi.txt
nvidia-smi topo -m > results/raw/sessionX/topo.txt      # tp>1
lscpu > results/raw/sessionX/lscpu.txt
uv pip freeze > results/raw/sessionX/env_freeze.txt
git add -A && git commit -m "Session X: <what ran>"
read -p "PAT: " GH_PAT
git push https://IsseiHasegawa:${GH_PAT}@github.com/IsseiHasegawa/vLLM-experiment.git HEAD:main
clear; history -c
```

Then **Stop Pod -> Terminate**, confirm it disappears, check the balance.

### 6.4 Quality gates

After session A: `python3 scripts/analyze_c1.py --repo . --out results/c1_control.txt`
must report a paired bound of a few percent or less with a consistent direction
(D31 — a small bounded effect is the expected outcome, not a failure; the old
"no instrumentation effect detected" wording is retired); `n_cached == 0`
everywhere; A1a and A1b agree within their error bars; **arrival span / expected
span == 1.00 for every open-loop run** (D24 — `RATE_SHORTFALL` fires at 58 %
utilisation and is not a validity signal).

After every analysis day: for each figure the author writes three sentences of
interpretation **before** reading Claude's. Those sentences become the Results
text and the mock-Q&A answers.

---

## 7. Known gotchas

- **Disk quota (D20)** — see §6.1. Symptom: `RuntimeError: ... Disk quota
  exceeded (os error 122)` during model download.
- **The web terminal truncates pasted input into `read -s`** (8 of 93 characters
  arrived once). Use `read -p`, then `clear; history -c`. Never screenshot a token.
- **Multi-line commands with trailing backslashes** may not receive the final
  newline when pasted; the shell then waits at `>`. Prefer single-line commands.
- **`Ctrl-b` does not always reach tmux** in the web terminal. Open a second tab
  instead of splitting panes. The runner survives disconnects; `tmux attach -t sess`.
- **Editing in Cursor without saving** means git sees no change.
- **zsh on macOS** treats a pasted `#` comment as a command.
- **`import vllm` takes 1-3 minutes**; a bench run that appears stuck for two
  minutes at startup is normal.

---

## 8. Open risks

| Risk | Mitigation |
|---|---|
| 4xA40 unavailable for tp=4 | Fall back to tp=1/2; figure 7 keeps two points and the scaling claim is stated more cautiously |
| 0.5B capacity far above the S3 grid | P0 probe runs first; regenerate the grid before S3 |
| Session A interrupted mid-way | Rows already `ok` are skipped on restart; worst case is one boot re-done |
| C1 detects a real instrumentation effect | Report it; treat instrumented latencies as an upper bound and throughput as a lower bound. Effect size matters more than existence |
| Schedule slip | Buffers on 8/4 and 8/13-15; drop G4 first, then C2's low-concurrency points |

---

## 9. Submission package (8/11-12 checklist)

- [ ] Report PDF: 8-12 pages + appendix (instrumentation diff, full config table, supplementary figures)
- [ ] `vLLM-experiment` cleaned, **PLAN.md deleted**, README rewritten for readers, tagged `v1.0`, made public
- [ ] Fork link, `instrumentation` branch (2 commits, +192 lines)
- [ ] 10 figures with error bars, consistent styling
- [ ] `results/c1_control.txt` referenced from Methods
- [ ] `results/raw/` retained and offered on request
- [ ] Secret scan across full history before going public
- [ ] "Reproduce in 10 minutes" section in the README (0.5B path)
- [ ] One-line AI-assistance acknowledgment
- [ ] Submission email: three-line summary + two links
- [ ] Mock Q&A completed
