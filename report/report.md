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
second GPU into throughput (-0.3 %) yet cuts TTFT p95 by 12 % against
one GPU, about as much as tensor parallelism does.

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

I forked vLLM at commit 702f4814 and built it from source with an
editable install, so that changing a file and restarting the server is a
one-step operation. All measurement changes live on an `instrumentation`
branch: three files, about 190 lines. No inference path was modified.

The design principle was to write out numbers vLLM already computes
rather than add new timers. The V1 metrics layer already holds the
queued, prefill and decode intervals for each request when it builds
`FinishedRequestStats`; the patch serialises those to JSONL instead of
recalculating them. This keeps the risk of introducing timing bugs low
and makes the output cross-checkable against vLLM's own Prometheus
histograms.

The instrumentation has three layers (Figure R1), and the split matters.
Chunked prefill is always on in vLLM V1, so a single engine step can
contain some requests doing prefill and others decoding. A phase is a
property of a request; a step is a unit of batching. Neither view alone
can separate time and resource use by phase, so I log both:

- **Request axis** (`requests.jsonl`) — queued, prefill, decode,
  inference and end-to-end time per request, with token counts.
- **Step axis** (`steps.jsonl`) — scheduling and execution time per
  engine step, batch composition, and KV cache usage.
- **Resource layer** — an external process sampling GPU counters and
  per-process CPU at 1 Hz, outside the server, so it is independent of
  the code being measured.

Three checks support the numbers that follow. The identity
`prefill + decode = inference` holds for every record. Against
Prometheus, the queued, prefill and decode means agree exactly (72.04,
238.24 and 2,135.56 ms), and token counts match on both axes. Finally,
because the logger is switched on by an environment variable, I could
run the same binary with instrumentation off: latency was 2.97 % lower
and throughput 0.26 % higher without it, and with three seed pairs
neither difference reaches significance (smallest p = 0.125). That
bounds the overhead as small but does not pin it precisely.

Two limitations belong here. Session D, which measured pipeline
parallelism, produces no step-axis log at all: vLLM's pipeline executor
takes a different step path that the patch does not hook, so §3.4 and
§4 rest on request-axis data for pp=2. And the client-observed TTFT
contains an interval the server never sees, which §2 takes up next.

Full details of the patch, the run matrix and the validity checks are in
the attached paper, §3.

# 2. Where the time goes: prefill vs decode

The two phases are wildly unequal in wall-clock terms. For ShareGPT at
rate 5, a request spends 7,334 ms of its 7,434 ms end-to-end time in
decode and 69 ms in prefill — 0.93 %. This is not an artefact of the
workload: with a prefill-heavy mix (512 in, 128 out) the prefill share
rises only to 1.94 %, and with a decode-heavy mix (128 in, 512 out) it
falls to 0.30 %. Across every finite rate and every group I measured,
prefill stayed under 2 % of end-to-end time. It reaches 2.7 % only in
the offline burst, where all 200 requests arrive at once.

Prefill still matters, because it is what the user waits for before the
first token appears. But most of that wait is not prefill compute
either. For the 7B model at rate 5, the client observes a mean TTFT of
111.6 ms, while the queue and prefill intervals the server records total
69.0 ms. The remaining 38 % happens somewhere the instrumentation does
not reach (Figure R2): HTTP receipt and response, serialisation,
tokenisation, the hand-off from frontend to engine, and any wait before
the request reaches the scheduler. I did not measure which of these
dominates; separating them would need a timestamp at each hand-off.

That share grows as the model gets smaller. On the 0.5B model it is
48 % at rate 1, 59 % at rate 8, and 71 % at rate 32, because the prefill
interval itself stays flat — 17.2 ms down to 14.4 ms — while the
client-observed TTFT climbs from 33.1 ms to 49.2 ms. For a small model,
then, most of the room for improving TTFT is outside the model
computation.

Two things follow for the rest of this report. Since decode dominates
request time, anything that changes overall throughput has to act on
decode; §3.4 shows that only one of the two parallelism strategies
does. And since the phases differ this much in structure, they are
limited by different resources — which is what §4 works out.

# 3. Factor analysis

## 3.1 Arrival rate and capacity

Raising the arrival rate eightfold, from 1 to 8 req/s, moves latency
much less than that. TTFT p50 goes from 76 ms to 120 ms and p95 from
151 ms to 245 ms, a factor of about 1.6. On the decode side TPOT p50
goes from 33.0 ms to 44.2 ms. The exception is the tail of inter-token
latency: ITL p95 grows 2.8-fold, from 34.3 ms to 95.0 ms, while its p50
stays essentially flat at 32-36 ms. Most tokens keep arriving on
schedule under load; a few are delayed a great deal.

Where load does not show up is the queue. Time spent in the scheduler's
waiting queue averaged 0.018-0.021 ms at every finite rate, with a
maximum of 0.08 ms — three to four orders of magnitude below TTFT. The
scheduler's own waiting count was 0 after every scheduling decision of
every finite-rate run. This is not because the server was idle: vLLM
admits a waiting request into the running batch at the next scheduling
step rather than holding it until capacity frees, so backlog appears as
a larger batch instead. The decode batch grows from 6.0 requests per
step at rate 1 to 18.8 at rate 4 and 23.0 at rate 8, with the maximum
reaching 76. Note that this covers the scheduler only — as §2 showed,
38 % of the client's wait falls outside the intervals the server records
at all.

This has a practical consequence: achieved throughput is a weak signal
for saturation. It also depends on how you define it. Counting
completions over the whole measured duration includes the drain after
arrivals stop, so even at rate 1 the figure reads 0.95 req/s; counting
them over the arrival window instead excludes requests still in flight
and reads 5.2 req/s at rate 8. Neither is unbiased, and the open-loop
series never flattens within the grid — it still climbs 5.0 % from rate
6 to rate 8, ending at 3.54 req/s.

So I measured capacity a different way. In a closed-loop run the number
of in-flight requests is fixed, so every point has a steady state and
the latency-throughput trade-off can be read directly (Figure R3).
Doubling the concurrency limit from 1 up to 32 buys 62-95 % more
throughput for 2-12 % more p95 latency. From 32 to 64 the two come into
balance, 14.5 % against 10.3 %. From 64 to 128 throughput gains 0.9 %
while latency costs 5.4 %. The knee is therefore between 32 and 64, at
about 711 output tokens per second — roughly 3.7 req/s at this
workload's mean output length of 191.6 tokens. That the open loop is
still climbing at 3.54 req/s is consistent with the same ceiling
approached from the other side.

## 3.2 Dataset

I used two workloads with deliberately opposite properties. ShareGPT is
a public collection of real conversations with dialogue models, so
prompt lengths vary the way they would in production; it gives external
validity but cannot be controlled. The random workload is generated by
vLLM's own benchmark harness with input and output lengths fixed, which
is unrealistic but lets me vary one thing at a time. Since prefill work
is proportional to prompt length, the length distribution is itself an
experimental variable, not a background detail.

What the server actually sees is not what the source file contains. The
harness admits a conversation only if the prompt is at most 1,024
tokens, so although the source file reaches 66,076 tokens, the longest
prompt actually served was 1,010. Reporting the nominal distribution
would overstate the processed tail by a factor of 65. The realised
ShareGPT prompts have a median of 136 tokens and a p95 of 767, against
a fixed 256 for random; outputs average 191.6 tokens against a fixed
128. I recorded the source file's SHA-256 so the sampling can be
reproduced.

At matched rates, random is faster on TTFT — p95 of 109 ms against
151 ms at rate 1, and 211 ms against 245 ms at rate 8. The prompt-length
tail is the likely reason, though the two workloads also differ in
output length, so this comparison does not isolate it. On delivered
throughput the difference is larger: random was still climbing at an
offered rate of 20, reaching 12.10 req/s, where ShareGPT reached
3.54 req/s at rate 8. That is more than threefold, but it bounds the
combined effect of prompt and output length rather than separating them,
and no closed-loop sweep was run for random, so its steady-state ceiling
is unmeasured.

The input-to-output ratio produces a sharper lesson. Holding everything
else fixed and comparing 512-in/128-out against 128-in/512-out, the
first is 35 % higher on total token throughput (2,891 against
2,144 tok/s) while the second is 3.0 times higher on output token
throughput (1,715 against 578 tok/s). The two metrics rank the same two
runs in opposite orders, because one counts prompt tokens as work and
the other does not. Reporting only one of them would support either
conclusion.

Prefill efficiency follows prompt length as expected: 4,070 prompt
tokens per second during the prefill interval for the long-prompt
workload against 1,829 for the short-prompt one, with ShareGPT between
them at 3,385. Longer prompts amortise the per-step cost better — the
same amortisation that §4 identifies as the reason prefill benefits less
from sharding than decode does.

## 3.3 Model size and CPU

Qwen2.5-7B and Qwen2.5-0.5B differ by 14:1 in parameter count, but
neither phase scales by that ratio, and the two phases do not scale
alike. At rate 1, TTFT p95 is 151 ms against 52 ms — a factor of 2.9 —
while TPOT p95 is 34.8 ms against 6.1 ms, a factor of 5.8. Decode
tracks model size far more closely than TTFT does, which follows from §2:
TTFT carries a component outside the server-recorded interval that
reaches 48-71 % on the 0.5B model, and that component does not shrink
when the model does.

The two models also respond to load differently. TTFT p95 for the 0.5B
model stays flat at 51-53 ms from rate 1 to rate 8, where the 7B model
climbs from 151 ms to 245 ms. The 0.5B model only starts responding at
rate 12 and above, reaching 78 ms at rate 32. Because the two were swept
over different rate grids, figures up to rate 8 are a like-for-like
comparison and anything beyond that describes the 0.5B model alone. Peak
output throughput is 675 tok/s for the 7B at rate 8 against 3,371 tok/s
for the 0.5B at rate 32, a factor of 5.0.

What makes the small model interesting is that it stops being limited by
the GPU. At rate 8 the 7B model runs the GPU at 88.8 % utilisation and
the memory controller at 94.2 %. The 0.5B model reaches only 54.9 % and
39.3 %, and raising the rate to 32 leaves those at 53.3 % and 38.8 %
(Figure R4, left). Nearly half the GPU stays unused no matter how hard I
push. Power says the same thing: 173-182 W against 273 W, a 100 W gap
that is computation not being performed.

The CPU side moves in the opposite direction. Summed over the processes
belonging to the server, CPU usage for the 7B model rises only from
28.1 % of one core at rate 1 to 39.4 % at rate 8. For the 0.5B model it
starts at 65.3 %, reaches 120.8 % at rate 8, and peaks at 144.4 % at
rate 24 (Figure R4, right). Values above 100 % mean the group spans more
than one core; vLLM V1 runs at least a frontend and an engine-core
process, so this is an aggregate, not a single-threaded figure.

This is not the machine running out of CPU. System-wide usage stayed
between 5 % and 8 % throughout, on a 96-core host, so 1.4
core-equivalents leaves plenty of headroom. What it shows is that
host-side work in the serving layer — scheduling, tokenisation, HTTP
handling, and the hand-offs between them — grows into the space a short
model step leaves. The step-axis log agrees: scheduling takes 2.2 % of
per-step time for the 7B model at rate 8 (0.801 ms against 36.17 ms of
execution), 5.4 % for the 0.5B model, and 8.2 % at rate 32. The shorter
the model step, the larger the fixed serving cost looms.

I did not resolve which part of that work is the constraint — the CPU
counter is an aggregate over processes, and a profile would be needed to
separate them. The regime is framework-bound rather than CPU-bound, and
it is the reason §4 ends by saying the limiting resource is not fixed.

## 3.4 GPU count and parallelism strategy

Adding GPUs raises throughput, but with a clear diminishing return. At
rate 8, two GPUs deliver 32.5 % more than one and four deliver 52.0 %,
so the second doubling is worth less than half the first (14.7 %). Per
GPU the picture is starker: 3.51 req/s at tp=1, 2.33 at tp=2, and 1.34
at tp=4. All three series were measured on the same four-GPU machine
with only the parallelism setting changed, so host differences cannot
contribute. Two caveats apply throughout. This host has no NVLink, and
NCCL peer-to-peer transport and the custom all-reduce kernel had to be
disabled for the communicator to start at all, so these are lower bounds
on what the same GPUs could achieve.

The gain is not spread evenly over the two phases. At rate 5, tp=2 cuts
mean decode time per request by 28.5 % but prefill by only 13.2 %; at
tp=4 the figures are 42.5 % and 16.9 % (Figure R5). Tensor parallelism
is mostly a decode optimisation. §4 explains why.

Holding the device count at two and changing only the strategy separates
throughput from latency completely. Pipeline parallelism converts none
of the second GPU into throughput: across 23 seed-matched pairs the mean
difference against a single GPU is -0.3 %, with t = -0.43. Individual
points swing by up to ±9.2 %, but the sign is inconsistent, whereas
tensor parallelism is positive at all 23 points in the same test. On
throughput, then, the second GPU is worth 32.5 % under tp=2 and nothing
under pp=2.

Latency goes the other way. TTFT p95 under pp=2 falls below the
single-GPU baseline at all 20 seed-matched points, averaging -12.0 %
(t = -12.1) — on par with the -10.0 % that tp=2 achieves. The mean
TTFT moves by a similar -11.9 %, so this is not a tail effect. The
instrumented prefill interval accounts for under half of it: 7.4 ms of
the 15.9 ms reduction at rate 5, with the rest coming from the
unattributed interval of §2, which shrinks from 44.1 ms to 35.6 ms.
Running two worker processes instead of one plausibly changes contention
on the frontend path, but this instrumentation cannot resolve that.

One caveat is specific to this comparison. The pp=2 runs were measured
in a separate session on a different host, matched in GPU model, driver,
CPU and interconnect class, but not anchored by a repeated condition as
the other sessions were. A single boot warm-up puts that host about 3 %
faster, so I treat ±3 % as the working tolerance here. The prefill and
TTFT reductions exceed it by a factor of three or more and survive. The
decode change of -1.7 % does not, which is why §4 reads it as "no
effect" rather than a small one.

So the two strategies are not interchangeable, and the choice depends on
which metric matters. If the goal is serving more requests with the same
hardware, only tensor parallelism does that. If the goal is a faster
first token, either works.

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

The tensor-parallel numbers are lower bounds. This host has no NVLink,
and NCCL peer-to-peer transport and the custom all-reduce kernel had to
be disabled for the communicator to start, so faster paths were never
used. The diminishing return in §4 may be steeper here than the same
GPUs would show with those enabled.

The pp=2 comparison crosses hosts. Sessions A through C are tied
together by a repeated anchor condition that agrees to within 0.97 %,
but Session D has no anchor; a single boot warm-up puts that host about
3 % faster, which is the tolerance §3.4 applies. Session D also has no
step-axis log, so the step-level decomposition covers tensor
parallelism only.

The residual arithmetic in §4 rests on the A40's theoretical peak
bandwidth of 696 GB/s. Effective bandwidth is lower, so the residual's
20 % growth depends on that assumption — at 10 % below peak it would be
closer to 50 %. The residual is also not a measurement of communication:
at tp=1 there is no inter-GPU traffic and it is already 10.43 ms, so it
holds attention compute and framework overhead as well.

The workload is capped at 1,024-token prompts by the harness sampler,
so nothing here speaks to long-context behaviour. Everything was
measured on one GPU model (A40 48GB), one model family (Qwen2.5 at 7B
and 0.5B) and two datasets; other architectures, precisions or
length distributions are outside what I tested. Of 232 runs, 231
completed; one G1 point rests on two repetitions instead of three.

Everything needed to reproduce this is at
[github.com/IsseiHasegawa/vLLM-experiment](https://github.com/IsseiHasegawa/vLLM-experiment).
The run matrix is generated by a script rather than written by hand,
`make_figures.py` regenerates every figure from the raw logs, and
`verify_report_numbers.py` re-checks each number in this report against
those logs — it passes from a clean clone.