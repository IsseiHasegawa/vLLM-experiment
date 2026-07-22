# PLAN.md — vLLM Benchmarking Assignment (working plan)

> Internal working document. **Delete before the repo is made public / submitted.**
> Last updated: 2026-07-21. Owner: Issei Hasegawa (GitHub: IsseiHasegawa).

---

## 0. How to resume this project in a new chat

1. Open a chat **inside the same Claude project**.
2. Say: `"<date> セッション開始"` (e.g. `7/23 セッション開始`).
3. Give Claude access to this file:
   - If the repo is **public**: paste the URL `https://github.com/IsseiHasegawa/vLLM-experiment`
     and ask Claude to read `PLAN.md` + `README.md` + `configs/matrix.csv`.
   - If the repo is **private** (default state, decision 3): **paste the contents of this file**
     into the chat. Claude cannot read private repos.
4. Claude should then produce that day's runbook from §5 and §6 below.

**Claude's role**: produces runbooks, scripts, analysis and the report draft; audits pushed
state via the public fork / public repo. **Does not** run the experiments. AI use is
explicitly permitted by the assignment author.

---

## 1. Assignment and context

**From**: Dr. Minlan Yu (Harvard). **Nature**: screening task after the author asked to join
her research group. Her group publishes actively on LLM serving (DUCHESS, HACK,
prefix-aware data parallelism), so the report is written for a serving-systems researcher.

**Required deliverables** (verbatim requirement → where it is satisfied):

| Requirement | Satisfied by |
|---|---|
| Deploy vLLM from source; clone, change, recompile | Fork + editable install, demonstrated in Session 0 |
| Instrument for latency (E2E + per-phase) and throughput | 3-file instrumentation, validated in Session 0 |
| Time **and resource** usage in prefill and decode | Phase logs + 1 Hz resource logger; figures 8, 9 |
| ≥2 datasets, documented | ShareGPT + random; figures 4, 5 |
| Vary request arrival rate | Rate sweeps in S1–S4; figures 1–3 |
| ≥2 models / sizes | Qwen2.5-7B vs 0.5B; figure 6 |
| Vary GPU count | 1× vs 2× (tp=2); figure 7 |
| Document CPU performance | Resource logger (`cpu_*` columns) + `sched_s`; figure 9 |
| Enable/evaluate parallelism | Tensor parallel tp=2 (tp=4 if capacity allows); figure 7 |
| Bottleneck analysis | Figures 3 + 8 + 9 cross-read; report §Bottleneck Analysis |
| Figures comparing metrics | 9 figures, error bars over 3 repetitions |
| "Send me your results" | Report PDF + public repo link + email |

**Deadlines**: target submission **2026-08-10**; hard limit **2026-08-15**.
**Capacity**: 3 h/day. **Blocked**: 2026-07-25 → 07-28 (conference, no work at all).

**Report spec**: IMRaD, 8–12 pages body + appendix (instrumentation diff, full config table,
supplementary figures), ~12 references, written in English by the author from a Claude draft.
Includes a one-line AI-assistance acknowledgment.

---

## 2. Current state (verified 2026-07-21)

### Repositories

| Repo | Purpose | State |
|---|---|---|
| `github.com/IsseiHasegawa/vllm` (public fork) | Instrumented vLLM | branch `instrumentation`, HEAD `d4e0675` |
| `github.com/IsseiHasegawa/vLLM-experiment` (private) | Everything else | HEAD `e1478af` |

Local: `~/dev/vllm` and `~/dev/vLLM-experiment` (siblings, never nested).

### Instrumentation (fork)

Pinned to **vLLM v0.25.0** = commit `702f4814fe54fabff350d43cb753ae3e47c0c276`.

| Commit | Content |
|---|---|
| `019e5d1` | 3 files, +166 lines, additions only: `vllm/phase_logger.py` (new, 107L), `vllm/v1/metrics/stats.py` (+21L), `vllm/v1/engine/core.py` (+38L) |
| `d4e0675` | +26L: 1 s background flush thread in `phase_logger.py` (fixes Session 0 tail loss) |

- **C1** `phase_logger.py` — stdlib-only JSONL writer, enabled by env `VLLM_PHASE_LOG_DIR`,
  one file per (kind, pid), buffered (200 records) + 1 s timer + atexit, never raises.
- **C2** `stats.py` — one record per finished request at the end of
  `update_from_finished_request()`: `queued_s, prefill_s, decode_s, inference_s, e2e_s,
  n_prompt, n_gen, n_cached, mean_tpot_s, finish, ts, req_id, arrival_ts`.
- **C3** `core.py` — one record per engine step in `step()`: `sched_s, exec_s,
  n_ctx_reqs, n_ctx_toks, n_gen_reqs, n_gen_toks, n_running, n_waiting, kv_usage, ts`;
  plus a startup warning if the un-instrumented batch-queue step path is active.

Anchors if line numbers drift: `grep -n "update_from_finished_request" vllm/v1/metrics/stats.py`,
`grep -n "self.step_fn = " vllm/v1/engine/core.py`, `grep -n "def step(self)" vllm/v1/engine/core.py`.

### Experiment repo contents

```
README.md                     Methods decision log D1–D12
configs/matrix.csv            90 rows (see §3)
scripts/run_experiments.py    runner: boots servers, runs matrix, writes manifest
scripts/resource_logger.py    1 Hz GPU/CPU sampler (needs psutil + nvidia-smi)
scripts/verify_session0.py    instrumentation validation (V1–V4)
scripts/plots/                EMPTY — plotting scripts not written yet
results/manifest.csv          header only, 0 data rows
results/raw/session0/         Session 0 artifacts (10 files)
figures/                      empty
report/main.md                IMRaD headings only
```

### Session 0 (2026-07-19, A40 ×1, ~$1.0) — instrumentation validated

Ran Qwen2.5-0.5B-Instruct, random 256/128, 10 warmups + 50 measured, rate 5, seed 42.

Result: **16 PASS, 1 FAIL** (`results/raw/session0/verification.txt`).

- `prefill_s + decode_s == inference_s` exact; `n_cached == 0` for all; `n_gen == 128` for all.
- **Cross-check vs vLLM's own Prometheus histograms matched to the printed precision**:
  queue 1.07 ms, prefill 25.43 ms, decode 714.32 ms (ours == theirs, n=60 both).
- Client TPOT 5.568 ms vs server 5.573 ms.
- The single FAIL was step-token accounting short by ~3 requests' worth — tail records lost
  because EngineCore was killed without running atexit. **Fixed in `d4e0675`**; the fix was
  proven with a SIGKILL test (50/50 records survive).

**Findings worth reporting**:
1. **TTFT decomposition**: client-observed TTFT 30.50 ms vs server `queued+prefill` 11.74 ms
   → **18.77 ms (62%) sits outside prefill compute** (HTTP, serialization, tokenization,
   streaming) on localhost with a 0.5B model. Feeds figure 8; expect the ratio to shrink at 7B.
2. **v0.25.0 enables async scheduling by default** → the un-instrumented step path. Must pass
   `--no-async-scheduling` (D9). The built-in warning caught this before any real measurement.
3. **`datasets` 1.1.1 in the RunPod template is incompatible with pyarrow 25** → `vllm bench`
   crashes on import. Fix each session with `uv pip install -U "datasets>=3.0"` (D10).

### Budget

Loaded $25.00, remaining **$23.98**. A40 1× = $0.44/hr (+~$0.01 disk); 2× = $0.88/hr.
Estimated remaining spend $8–12; hard ceiling agreed with the author is $100, so cost is
not a constraint. **The only real cost risk is forgetting to terminate a pod.**

---

## 3. Experiment matrix (`configs/matrix.csv`, 90 rows)

Fixed for every row: `num_prompts=200`, `seed=42`, 3 repetitions, `--ignore-eos`,
`--num-warmups 10`, percentiles 50/95/99.

| Group | Rows | Model | Dataset | tp | Rates |
|---|---|---|---|---|---|
| S1 | 18 | 7B | sharegpt | 1 | 1, 2, 5, 10, 20, inf |
| S2 | 18 | 7B | random 256/128 | 1 | 1, 2, 5, 10, 20, inf |
| S3 | 24 | 0.5B | sharegpt | 1 | 1, 2, 5, 10, 20, 50, 100, inf |
| S4 | 21 | 7B | sharegpt | 2 | 1, 2, 5, 10, 20, 30, inf |
| I1 | 3 | 7B | random 512/128 | 1 | 5 |
| I2 | 3 | 7B | random 128/512 | 1 | 5 |
| A1 | 3 | 7B | sharegpt | 1 | 5 (anchor re-measurement, see D12) |

`A1` duplicates the S1 r=5 condition under a **different run_id** so it can be re-measured on a
second instance without overwriting day-1 results. Never re-run S1 rows with `--force`.

The runner groups **consecutive** rows by `(model, tp)` into server boots. File order is
S1, S2, S3, S4, I1, I2, A1 → full matrix = 4 boots.

### Figures and their data sources

| # | Content | Source |
|---|---|---|
| 1 | rate → TTFT p50/p95 (error bars over reps) | bench JSON, S1 |
| 2 | rate → TPOT and ITL | bench JSON, S1 |
| 3 | rate → request/token throughput (saturation knee) | bench JSON, S1 |
| 4 | figs 1+3 with ShareGPT vs random overlaid | S1 vs S2 |
| 5 | input/output token-length distributions of both datasets | tokenizer, Mac-side, no GPU |
| 6 | figs 1–3 with 7B vs 0.5B overlaid | S1 vs S3 |
| 7 | fig 3 with 1 GPU vs tp=2 overlaid | S1 vs S4 |
| 8 | per-request time split queued/prefill/decode (I1 vs I2) + TTFT decomposition | phase logs + bench JSON |
| 9 | resource utilization vs rate (GPU util, memory-controller util, VRAM, CPU) | resources CSV joined to manifest time slices |

Bottleneck analysis is the cross-read of 3 + 8 + 9, not a separate figure.

---

## 4. Operating decisions

D1–D12 live in `README.md` and are the authoritative record. Operationally the ones that
change commands are:

- **D8/D9**: every server launch needs `--no-enable-prefix-caching --no-async-scheduling`.
  Never `--disable-log-stats` (it removes the timestamps the instrumentation reads).
- **D10**: every pod session starts with `uv pip install -U "datasets>=3.0"`.
- **D2**: `--num-warmups 10` on every run; the runner also does one throwaway bench per boot.
- **D3/D7/D12**: one server per (model, tp); the 7B block and the 0.5B block run on separate
  instances, bridged by the A1 anchor.
- **D4**: runs are attributed to phase-log records offline by `[start_ts, end_ts]` from
  `results/manifest.csv`.
- **D5**: p95 is the headline metric; p99 is shown with error bars as a reference.
- **D6**: all repetitions use seed 42, so error bars represent system noise only.
- **Decision 7 (planning)**: no network volume; pods are disposable, models re-downloaded.

---

## 5. Schedule

Assumes 3 h/day. Conference 7/25–7/28 is immovable. **The pilot and the 7B block must land
before the conference.**

| Date | Day | Work | GPU |
|---|---|---|---|
| 7/22 | Wed | Plotting scripts (9) + figure 5 + download ShareGPT (Mac + pod) | — |
| 7/23 | Thu | **Pilot** (2 rates × 1 rep) + lock GPU type + start daily 2×A40 stock checks | ~1.5 h |
| 7/24 | Fri | **Main run A**: 7B block `--only S1,S2,I1,I2` (42 runs) | ~2.5 h |
| 7/25–28 | Sat–Tue | **Conference — no work** | — |
| 7/29 | Wed | **Main run B**: `--only A1` then `--only S3` (27 runs) | ~1.5 h |
| 7/30 | Thu | S4 attempt (2×A40; add tp=4 if capacity allows), else pull analysis ① forward | ~2 h |
| 7/31 | Fri | Analysis ①: figures 1–4, 6 + write own 3-sentence reading of each | — |
| 8/1 | Sat | Analysis ②: figures 8, 9 + instrumentation-validation table for Methods | — |
| 8/2 | Sun | Analysis ③: bottleneck synthesis; define "saturation point" operationally. **S4 hard deadline** | maybe |
| 8/3 | Mon | Buffer / re-measurement / catch-up | — |
| 8/4 | Tue | **Data freeze.** Hand everything to Claude → receive English draft → read it, list questions | — |
| 8/5 | Wed | Write ①: Abstract + Introduction | — |
| 8/6 | Thu | Write ②: Background & Related Work | — |
| 8/7 | Fri | Write ③: Methodology (instrumentation, validation table, D12 bridging) | — |
| 8/8 | Sat | Write ④: Results + Bottleneck Analysis | — |
| 8/9 | Sun | Write ⑤: Discussion + Limitations + Conclusion + AI acknowledgment | — |
| 8/10 | Mon | Mock Q&A (15 questions) + repo cleanup/public/tag + PDF + submission email | — |
| 8/11–15 | — | Reserve band before the hard deadline | — |

If the 7/22 plotting work gets finished earlier, everything shifts one day earlier and 8/9
becomes the submission day.

---

## 6. Session runbooks

### 6.1 Every GPU session — start

1. RunPod → Billing: note the balance.
2. Pods → Deploy. **Clear the Filter** (a stale filter makes A40 look "Out of capacity"),
   use the **Available** tab, pick **A40 ×1**, template **Runpod Pytorch 2.8.0**
   (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), Any region, disks 30 GB / 50 GB.
   Fallback order if A40 is unavailable: **L40S → RTX 6000 Ada → RTX A6000** (all 48 GB).
   S1–S3 must all use the same GPU type; lock it at the pilot.
3. Connect → **Enable web terminal** → open it. (SSH keys are not used.)
4. `nvidia-smi` to confirm the GPU and ~46 GB.

Environment (proven in Session 0; ~15 min, plus model download):

```bash
tmux new -s sN
cd /workspace
git clone -b instrumentation https://github.com/IsseiHasegawa/vllm.git
read -p "PAT: " GH_PAT          # visible on purpose, see §7
git clone https://IsseiHasegawa:${GH_PAT}@github.com/IsseiHasegawa/vLLM-experiment.git
git -C vLLM-experiment remote set-url origin https://github.com/IsseiHasegawa/vLLM-experiment.git
clear; history -c
cd vllm
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.12 --seed && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable ".[bench]" --torch-backend=auto
uv pip install -U "datasets>=3.0"                       # D10
python -c "import psutil; import vllm; print('deps ok')" # resource logger needs psutil
```

ShareGPT (needed by S1, S3, S4, A1; ~600 MB):

```bash
cd /workspace
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Fallback if the editable install fails twice: install the official v0.25.0 wheel and apply the
same three-file diff inside `site-packages` (equivalent, faster).

### 6.2 Running the matrix

```bash
cd /workspace/vLLM-experiment
python3 scripts/run_experiments.py \
  --matrix configs/matrix.csv --manifest results/manifest.csv \
  --session sN --results-dir results/raw/sessionN \
  --sharegpt-path /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
  --only <groups>
```

Add `--dry-run` first to print the plan without executing. Useful flags: `--runs id1,id2`,
`--force`, `--iteration-details` (pilot only), `--only`.

Expected selections: pilot `--runs` a few ids; 7/24 `--only S1,S2,I1,I2` → 42 rows, 1 boot;
7/29 `--only A1` then `--only S3` → 3 + 24 rows; S4 `--only S4` → 21 rows.

The runner: boots one server per (model, tp) with the D8/D9 flags and `VLLM_PHASE_LOG_DIR`
pointed at `<results-dir>/phase_logs`; **aborts if the PHASE-INSTR batch-queue warning appears**;
does one throwaway warmup bench per boot (recorded as `boot_warmup`, excluded from analysis);
appends a manifest row when each run finishes (so an interrupted run simply re-runs on resume);
flags `RATE_SHORTFALL` when achieved rate < 90% of requested; starts/stops the resource logger.

### 6.3 Every GPU session — end (never skip)

```bash
cd /workspace/vLLM-experiment
gzip results/raw/sessionN/phase_logs/*.jsonl
nvidia-smi > results/raw/sessionN/nvidia_smi.txt
nvidia-smi topo -m > results/raw/sessionN/topo.txt      # tp>1 sessions: interconnect matters
lscpu > results/raw/sessionN/lscpu.txt
uv pip freeze > results/raw/sessionN/env_freeze.txt
git add -A && git commit -m "Session N: <what ran>" 
git push https://IsseiHasegawa:${GH_PAT}@github.com/IsseiHasegawa/vLLM-experiment.git HEAD:main
```

Then **Stop Pod → Terminate**, confirm the pod disappears from the list, and check the balance.

### 6.4 Quality gates

**Pilot (7/23) passes only if**: (a) all 9 plotting scripts render a figure from pilot data,
(b) the runner completes unattended, (c) the manifest rows are correct and slice the phase logs
sensibly, (d) per-run wall time is measured and the 7/24 estimate re-derived, (e) GPU type locked.

**During main runs**: `n_cached == 0` everywhere; no `RATE_SHORTFALL` at low rates; manifest
row count matches the selection; server log free of PHASE-INSTR warnings.

**Understanding protocol**: for every figure, the author writes 3 sentences of interpretation
*before* reading Claude's. These become the Results text and the mock-Q&A answers.

---

## 7. Known gotchas

- **RunPod web terminal swallows pasted text into `read -s`** (only 8 of 93 characters arrived).
  Use `read -p "PAT: "` (visible), then `clear; history -c`. Never screenshot the token.
- **Multi-line commands with trailing backslashes** may not receive the final newline when
  pasted; the shell then waits at `>`. Prefer single-line commands in the web terminal.
- **`Ctrl-b` (tmux prefix) does not always reach tmux** in the web terminal. Open a second web
  terminal tab instead of splitting panes.
- **Editing a file in Cursor without saving** means git sees no change (`nothing to commit`).
- **zsh on macOS** treats a pasted `#` comment as a command → `command not found: #`.
- **Session 0 pattern**: request records stay buffered until the server stops; the 1 s flusher
  now bounds this, but still stop the server before running verification.

---

## 8. Open risks

| Risk | Status | Mitigation |
|---|---|---|
| 2×A40 capacity for S4 | Out of capacity on 7/12; not re-checked since | Check every morning from 7/23. Hard deadline 8/2 → then run a self-contained {1×, 2×} pair on another 48 GB type, same day, same type, and note it in Methods |
| 7B may not saturate by rate 20 | Unknown until the pilot | Inspect at pilot; extend the S1/S2 grid if the knee is not visible |
| 0.5B at rates 50/100 may saturate the *client* (9 vCPU shared) | Detected automatically | `RATE_SHORTFALL` flag; drop unreachable points and document in Limitations — this is a strength, not a defect |
| tp=2 may be *slower* at low load | Expected on PCIe | This is a finding, not a failure; record `nvidia-smi topo -m` and discuss allreduce overhead |
| Schedule slip | Currently 1 day behind the original plan | Buffers on 8/3 and 8/11–15; if needed, drop tp=4 and the chunked-prefill bonus first |

---

## 9. Submission package (8/10 checklist)

- [ ] Report PDF: 8–12 pages body + appendix (instrumentation diff, full config table, extra figures)
- [ ] `vLLM-experiment` cleaned, **PLAN.md deleted**, README rewritten for readers, tagged `v1.0`, made public
- [ ] Fork link with the `instrumentation` branch (2 commits, +192 lines total)
- [ ] 9–10 figures with error bars, consistent styling
- [ ] `results/raw/` retained and offered on request
- [ ] Secret scan across the full history before making the repo public
- [ ] "Reproduce in 10 minutes" section in the README (0.5B path)
- [ ] One-line AI-assistance acknowledgment
- [ ] Submission email: 3-line summary of what was measured and found + two links
- [ ] Mock Q&A completed (15 questions) — a follow-up conversation with the author is likely
