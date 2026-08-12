# Summary

I forked vLLM and added instrumentation to three files, about 190 lines
in total, so that the per-phase timestamps the engine already computes
are written out as logs. Using that build I ran 232 experiments on
NVIDIA A40 GPUs, varying arrival rate, dataset, model size, GPU count,
and parallelism strategy. All numbers below come from those runs; a
script in the repository re-checks every one of them against the raw
logs.

**Load shows up as bigger batches, not longer queues.** vLLM admits a
waiting request into the running batch at the next scheduling step, so
time spent in the scheduler queue stayed at or below 0.021 ms at every
rate I tested, while the decode batch grew from 6 to 23 requests. This
means achieved throughput is a poor signal for saturation, so I measured
capacity with a closed-loop sweep instead: about 711 output tokens per
second, or 3.7 requests per second, for the 7B model on one GPU.

**Most of what the client waits for is not prefill compute.** For the 7B
model at rate 5, the server accounts for 69 ms of a 112 ms observed
TTFT; the remaining 38 % happens outside the interval the engine
records. On the 0.5B model that share is 48-71 % and grows with load.
Prefill compute itself never exceeded 2 % of end-to-end time.

**Parallelism helps one phase at a time.** Going from one GPU to four
makes decode-only steps 1.80x faster but steps carrying prefill only
1.26x faster. Holding the device count at two and changing only the
strategy separates the effect: pipeline parallelism turns none of the
second GPU into throughput (-0.3 %) yet cuts TTFT p95 by 12 %, about as
much as tensor parallelism does.

The limiting resource is not the same in every configuration. On the 7B
model the GPU memory controller is busy 94 % of the time; on the 0.5B
model the GPU sits half idle while the serving-layer processes exceed
one CPU core. Sections 3.3 and 4 work through this.

| # | Requirement | Where | Key number |
|:--|:-----------------------------|:---------|:------------------------------|
| 1 | Deploy vLLM; recompile | §1 | fork of 702f4814, 3 files, ~190 lines |
| 2 | Instrument latency and throughput | §1, §3.1, R1 | 51,808 request + 756,661 step records |
| 3 | Prefill: time and resource use | §2, §4, R2, R6 | 0.30–1.94 % of e2e; step 1.26× at tp=4 |
| 4 | Decode: time and resource use | §2, §4, R2, R6 | 98.7 % of e2e; step 1.80× at tp=4 |
| 5 | Two documented datasets | §3.2 | ShareGPT p95 767 tok vs random 256 |
| 6 | Vary arrival rate | §3.1, R3 | 1–8 req/s open, 1–128 closed loop |
| 7 | Two model sizes | §3.3, R4 | Qwen2.5-7B and 0.5B, 14:1 parameters |
| 8 | Vary GPU count | §3.4, R5 | 1/2/4 GPUs, +32.5 % and +52.0 % |
| 9 | Document CPU performance | §3.3, R4 | server group 28 % → 144 % of one core |
| 10 | Evaluate parallelism options | §3.4, R5 | tp = 1/2/4 and pp = 2 at 2 GPUs |
| 11 | Determine bottlenecks | §4, R6 | memory controller 93.6 % → 39.5 % |
| 12 | Generate figures | R1–R6 | 6 here, 12 in the attached paper |

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

Tensor parallelism raises throughput by up to 52 %. The obvious
explanation is KV cache capacity: more GPUs means more room for
concurrent requests, so batches grow and each step does more work. The
measurement rules this out. At rate 8 on one GPU, KV utilisation
averaged 2.13 % per step and peaked at 6.43 %. It does not rise with
more GPUs either — it falls, to 0.77 % at tp=2 and 0.33 % at tp=4. A
resource that is 97 % free is not the constraint. The decode batch size
confirms it: 22.9 requests per step at tp=1, 22.3 at tp=2, 21.7 at
tp=4. Sharding does not process more requests at once.

What it does is process them faster, and the step-axis log shows by how
much. At rate 8, decode-only steps fall from 32.27 ms at tp=1 to
17.95 ms at tp=4, a factor of 1.80. Steps carrying prefill fall from
72.34 ms to 57.33 ms, a factor of only 1.26 (Figure R6, left). The same
configuration change helps one phase far more than the other.

The resource log explains the asymmetry. Across the same tp sweep, GPU
utilisation barely moves — 87.9 % at tp=1 against 82.1 % at tp=4 — while
memory-controller utilisation falls from 93.6 % to 39.5 %, and power
from 285 W to 187 W. Sharding relieves the memory side and leaves the
compute side where it was. This matches the standard account: decode
reads every weight to produce one token per request and is limited by
memory bandwidth, while prefill processes hundreds of tokens per step
and amortises that read, so it is limited by arithmetic instead. The
arithmetic agrees. The 7B model's bf16 weights total 15.2 GB, and the
A40 moves 696 GB/s, so streaming the weights once takes 21.8 ms — 68 %
of the 32.27 ms step at tp=1. At tp=4 each GPU holds 3.8 GB, so that
term drops to 5.5 ms, only 30 % of the measured 17.95 ms.

The same principle predicts two further observations, and both hold.
First, the gain shrinks as batches grow, because a larger decode batch
spreads one weight read over more tokens and so behaves more like
prefill: the tp=4 speedup falls from 2.29x at batches of 4-8 to 1.48x
at 32-64 (Figure R6, centre). Second, pipeline parallelism does not
help decode at all. Splitting the model by stages leaves every weight
on exactly one GPU, so each device still streams its own share once per
step; measured change in decode time is -1.7 %, within tolerance of
zero, against -28.5 % for tensor parallelism at the same device count.

Diminishing returns come from the part sharding cannot divide.
Subtracting the theoretical weight-transfer time from the step time
leaves 10.43 ms at tp=1, 11.35 ms at tp=2, and 12.49 ms at tp=4. The
weight term falls exactly in proportion to the shard count; the
remainder does not fall at all, it grows by 20 %. That remainder is not
purely communication — at tp=1 there is no inter-GPU traffic and it is
already 10.43 ms, so it also holds attention compute, kernel launches,
and framework overhead. But its share rises with every doubling, which
is why the gain from two to four GPUs (14.7 %) is less than half the
gain from one to two (32.5 %).

Finally, the limiting resource is not fixed. Everything above concerns
the 7B model, where the memory controller is busy 94 % of the time. On
the 0.5B model the same counter reads 39 %, the GPU sits half idle, and
the serving-layer processes exceed one CPU core instead (§3.3). The
model is no longer what the server is waiting for.

# 5. Limitations and reproduction

<!-- ~0.3p. Harness caps prompts at 1,024 tokens, so realised max is 1,010
against a nominal 66,076 -- no long-context claim.
NCCL p2p disabled -> tp gains are lower bounds.
pp=2 measured on a separate host, +-3% tolerance; anchor spread 0.97%
across sessions A-C. Session D has no step-axis log (batch-queue path).
One A40 model, one model family, two datasets.
Repo link + verify_report_numbers.py re-checks every number from raw logs. -->