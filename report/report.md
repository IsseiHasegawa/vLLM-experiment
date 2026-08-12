# Summary

<!-- 3-4 sentences: fork base 702f4814, 3 files ~190 lines,
232 runs on A40s, 5 factors. Plain register. -->

<!-- Finding 1: load -> batch growth, not queue growth;
capacity 711 tok/s ~ 3.7 req/s (closed loop) -->

<!-- Finding 2: client TTFT 38% (7B) / 48-71% (0.5B)
outside server-recorded interval -->

<!-- Finding 3: phase-selective parallelism, 1.80x decode vs 1.26x prefill;
pp=2 at same 2 GPUs: TTFT -12%, throughput -0.3% -->

<!-- Requirement checklist, 12 rows:
Requirement | Section | Figure | Key number
rows 3 and 4 (prefill / decode resource use) cite
sections 2+4 and figures R2+R6 -->
| # | Requirement | Section | Figure | Key number |
|---|---|---|---|---|
| 1 | Deploy vLLM; make changes and recompile | 1 | — | fork of 702f4814, 3 files, ~190 lines |
| 2 | Instrument end-to-end and per-phase latency, throughput | 1, 3.1 | R1 | 51,808 request + 756,661 step records |
| 3 | Time and resource use during prefill | 2, 4 | R2, R6 | 0.30–1.94 % of e2e; prefill-carrying step 1.26× at tp=4 |
| 4 | Time and resource use during decode | 2, 4 | R2, R6 | 98.7 % of e2e; decode-only step 1.80× at tp=4 |
| 5 | At least two documented datasets | 3.2 | paper Fig. 2 | ShareGPT p95 767 tok vs random fixed 256 |
| 6 | Vary request arrival rate | 3.1 | R3 | 1–8 req/s open loop, 1–128 closed loop |
| 7 | At least two models or model sizes | 3.3 | R4 | Qwen2.5-7B and 0.5B, 14:1 parameters |
| 8 | Vary the number of GPUs | 3.4 | R5 | 1 / 2 / 4 GPUs, +32.5 % and +52.0 % at rate 8 |
| 9 | Document CPU performance | 3.3 | R4 | server group 28 % → 144 % of one core |
| 10 | Enable and evaluate parallelism options | 3.4 | R5 | tp = 1/2/4 and pp = 2 at equal device count |
| 11 | Analyse results to determine bottlenecks | 4 | R6 | memory controller 93.6 % → 39.5 % |
| 12 | Generate figures | all | R1–R6 | 6 figures here, 12 in the attached paper |

# 1. Setup and instrumentation

<!-- ~0.5p. Fig R1 = flat-style instrumentation diagram (redrawn).
fork 702f4814 / 3 files ~190 lines / editable install.
Three layers: request axis, step axis, 1 Hz external logger.
Verification, one sentence each:
prefill_s + decode_s = inference_s over all records;
Prometheus agreement (72.04 / 238.24 / 2135.56 ms);
C1 on/off = +2.97% latency, -0.26% throughput, not significant (n=3).
Pointer to paper section 3. -->

# 2. Where the time goes: prefill vs decode

<!-- ~0.5p. Fig R2 = paper Fig 6 right panel (TTFT decomposition).
prefill 0.30-1.94% of e2e at every finite rate, 2.7% only in offline burst.
Client TTFT: 38% unattributed on 7B at rate 5; 48-71% on 0.5B, rising with rate.
Closing bridge sentence: which resource limits each phase -> section 4. -->

# 3. Factor analysis

## 3.1 Arrival rate and capacity

<!-- ~0.75p. Fig R3 = paper Fig 11 (closed-loop curve).
Rate 1->8: TTFT p50 76->120 ms, ITL p95 2.8x while p50 flat.
Scheduler dwell <=0.021 ms, n_waiting = 0 at every step -> batch growth
(decode batch 6.0 -> 23.0), not queue growth.
Achieved throughput definition-dependent -> capacity from closed loop:
knee between 32 and 64, 711 tok/s ~ 3.7 req/s at 191.6 output tokens. -->

## 3.2 Dataset

<!-- ~0.45p, no new figure; cite paper Fig 2 in checklist.
Documentation first: why these two (ShareGPT = real conversations, wide
spread; random = fixed length, controllable). Realised stats:
ShareGPT input p50 136 / p95 767 / max 1,010 under the harness 1,024 cap,
output mean 191.6; random fixed 256/128. sha256 recorded.
Then results: random 12.10 req/s at offered 20 vs ShareGPT 3.54 at rate 8
= combined prompt+output effect, not prompt length alone.
Prefill efficiency 4,070 (I1) vs 1,829 (I2) prompt tok/s, ShareGPT 3,385.
I1/I2 metric reversal: total token throughput +35% vs output 3.0x. -->

## 3.3 Model size and CPU

<!-- ~0.5p. Fig R4 = paper Fig 13.
14:1 parameter ratio but TTFT p95 2.9x vs TPOT p95 5.8x -> phase-dependent.
0.5B: SM 54.9%, memory controller 39.3% at rate 8 -> GPU half idle,
while server-side process group 65.3% -> 144.4% of one core.
System-wide CPU 5-8% -> framework-bound, not CPU-exhausted. -->

## 3.4 GPU count and parallelism strategy

<!-- ~0.75p. Fig R5 = paper Fig 10.
tp: +32.5% at rate 8 (2 GPUs), +52.0% (4 GPUs), increment falls to 14.7%.
Phase-selective at rate 5: decode -28.5% vs prefill -13.2% (tp=2).
pp=2 at same device count: throughput -0.3% (t = -0.43, 23 seed-matched
pairs) but TTFT p95 -12.0% (t = -12.1), on par with tp=2's -10.0%.
One sentence each: NCCL p2p and custom all-reduce disabled -> lower bound;
Session D cross-host, +-3% tolerance. -->

# 4. Bottlenecks

<!-- ~0.75p, the core of the report. Fig R6 = paper Fig 12.
Chain: KV hypothesis rejected (2.13% at rate 8, falls to 0.33% at tp=4)
-> decode batch constant across tp (22.9 / 22.3 / 21.7)
-> shorter steps instead: decode-only 32.27 -> 17.95 ms (1.80x),
prefill-carrying 72.34 -> 57.33 ms (1.26x)
-> resource log agrees: SM 87.9 -> 82.1% but memory controller
93.6 -> 39.5%, power 285 -> 187 W
-> arithmetic: 15.2 GB weights / 696 GB/s = 21.8 ms = 68% of the tp=1 step
-> batch dependence: 2.29x (batch 4-8) down to 1.48x (32-64)
-> residual after weight transfer grows 10.43 -> 12.49 ms = diminishing return.
Closing: the limiting resource is not fixed -- memory bandwidth on 7B,
serving layer on 0.5B (see section 3.3). This is the RQ4 answer. -->

# 5. Limitations and reproduction

<!-- ~0.3p. Harness caps prompts at 1,024 tokens, so realised max is 1,010
against a nominal 66,076 -- no long-context claim.
NCCL p2p disabled -> tp gains are lower bounds.
pp=2 measured on a separate host, +-3% tolerance; anchor spread 0.97%
across sessions A-C. Session D has no step-axis log (batch-queue path).
One A40 model, one model family, two datasets.
Repo link + verify_report_numbers.py re-checks every number from raw logs. -->