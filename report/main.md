# vLLM Serving Performance Analysis: Phase-Level Measurement of Prefill and Decode

Issei Hasegawa — Allegheny College
Repository: https://github.com/IsseiHasegawa/vLLM-experiment

<!--
TARGET: 6-8 pages. Budget per section is noted in each heading comment.
FIGURE NUMBERING is by document order, not by filename. Source file is noted at each slot.
MARKERS:
  <!-- TODO --> = must fix before submission
  <!-- P1 insert --> = add only if the pp=2 session is executed; delete the marker otherwise
WRITING ORDER: 3 -> 4 -> 5 -> 6 -> 1 -> Abstract -> 2
-->

## Abstract

<!-- BUDGET 0.2p / ~150 words. Write LAST.
Contents: what was measured, three headline numbers, one-sentence conclusion. -->

---

## 1. Introduction

<!-- BUDGET 0.8p. Write after section 5 is finished. -->

<!-- Paragraph 1: why serving-system performance is measured per phase at all. -->

<!-- Paragraph 2: what this report does. 208 runs, 3 sessions, instrumented fork. -->

<!-- Paragraph 3: the research questions. These map 1:1 onto 4.1-4.4 and onto section 7. -->

**RQ1.** As the request arrival rate increases, how do per-phase latency and throughput change, and where and in what form does the capacity limit appear?

**RQ2.** How do workload characteristics (dataset, input/output ratio) and model size change capacity and phase composition?

**RQ3.** How large is the gain from additional GPUs and parallelism strategy, and to which phase is that gain attributable?

**RQ4.** Which resource determines the limits observed above?

---

## 2. Background & Related Work

<!-- BUDGET 0.8p total. Write LAST. -->

### 2.1 Prefill and decode

<!-- ~0.3p. Minimum needed to read the results: two-phase structure, KV cache,
     continuous batching, chunked prefill, and the metric mapping
     (TTFT <-> prefill, TPOT/ITL <-> decode). Cite Vaswani, Orca, vLLM, Sarathi-Serve. -->

### 2.2 Related work

<!-- ~0.5p. One paragraph. One-line attribution per citation; do not summarise papers.
     Order: Orca (continuous batching) -> vLLM/PagedAttention (the system measured here)
     -> Sarathi-Serve (chunked prefill) -> DistServe, Splitwise (phase asymmetry as a
     design premise) -> Megatron-LM (tensor parallelism) -> DUCHESS, HACK (Yu group).
     Close by stating what is NOT covered by prior work: an end-to-end phase-level
     measurement across all five factors on a single instrumented build. -->

---

## 3. Methodology

<!-- BUDGET 1.5p total across 3.1-3.5. -->

### 3.1 Instrumentation
For this measurement, we forked vLLM from GitHub (vllm-project/vllm, base commit 702f4814), made changes to three files totaling approximately 190 lines on the instrumentation branch, and then built it using an editable install from the source. Inference paths have never been changed except for instrumentation. The main purpose of this research is not optimaization but measurement. All of these changes are intended to make the internal time readable from the outside. The fork, the instrumentation commit (019e5d1), and the buffered-flush commit (d4e0675) are available at github.com/IsseiHasegawa/vllm.

The design of Instrumentation is not to  build a new timer, but to write out the value that vLLM V1 has already calculated. The V1 metrics layer retains the times for queuing, prefilling, and decoding for each request at the time FinishedRequestStats is constructed. By adopting an approach that outputs this data directly to JSONL without recalculating it, we minimized the risk of introducing bugs through custom code and, at the same time, enabled cross-validation with the Prometheus histograms published by vLLM itself.

The core of this design lies in dividing the instrumentation into two axes. In vLLM V1, chunked prefill is always enabled, and a single engine step contains a mix of requests undergoing prefill and requests undergoing decoding. In other words, phase refers to a request attribute, while step refers to a batch containing a mix of both phases. Since the two have different levels of granularity, relying on only one of them would not allow for the separation of time and resource usage to be separated by phase.. Therefore, we adopted a three-layer architecture that independently logs the request and step axes and samples resources from outside the process at 1 Hz (Figure 1).

| Layer | Output | Fields |
|:------------|:-----------------|:----------------------------------------------------------------------|
| Request axis | `requests.jsonl` | `queued_s`, `prefill_s`, `decode_s`, `inference_s`, `e2e_s`, `n_prompt`, `n_gen`, `n_cached` |
| Step axis | `steps.jsonl` | `sched_s`, `exec_s`, `n_ctx_reqs`, `n_ctx_toks`, `n_gen_reqs`, `n_gen_toks`, `n_running`, `n_waiting`, `kv_usage` |
| Resource | external logger | SM utilisation, memory-controller utilisation, VRAM, power (pynvml); per-process CPU utilisation for the server and the benchmark client (psutil) |

: Schema of the three instrumentation layers.

The definitions of each time period were adopted directly from the vLLM’s internal definitions. queued_s covers the period from the QUEUED event to the first SCHEDULED event; prefill_s covers the period from the first SCHEDULED event to the first token (including chunk segmentation and any waiting time in between); decode_s covers the period from the first token to the last token; and e2e_s covers the period from arrival at the front end to completion. Since the frontend and engine core are separate processes, the output destinations are also separated by process.

The logger is enabled only when the environment variable VLLM_PHASE_LOG_DIR is set; if it is not set, the function immediately returns at the beginning of the hook. Because instrumentation can be disabled without changing the binary, this enables the C1 control experiment described later (§3.5). Writing is performed via a buffer, and a background flush is scheduled every second to ensure that the last record is not lost in the event of a server crash.

The correctness of the instrumentation was verified through actual measurements. First, we verified the identity prefill_s + decode_s = inference_s—which should hold given the configuration—for all records, and then cross-checked the results against vLLM’s Prometheus metrics. In the verification using a 0.5B model, the queue time (1.07 ms), prefill time (25.43 ms), and decode time (714.32 ms) matched down to the decimal places. Token accounting also matched exactly: 6,400 vs. 6,400 for prefill and 3,150 vs. 3,150 for decode. Across all three sessions of this measurement, a total of 46,229 records were logged, the count matching the expected number of records exactly, with no tail loss.

Additionally, we implemented two safeguards to ensure that the attribution of timing measurements remained unambiguous. First, there are two execution paths for vLLM steps: step and step_with_batch_queue; in the latter, execution overlaps with the next step. In this measurement, we explicitly disabled async scheduling and ensured that all runs followed the former path by having the patch log which path was selected at startup once. Second, we did not use the --disable-log-stats option. This is because phase timestamps are carried as EngineCoreEvents, and disabling statistics would cause the measurement targets themselves to disappear.

### 3.2 Workloads and datasets

<!-- BUDGET 0.3p. WRITE THIS NEXT.
Materials: figures/fig05, configs/, decision D26.
Contents:
  - ShareGPT (real conversation, high length variance) and random (synthetic, fixed length)
  - why two: they differ in prompt-length distribution, which is the input to prefill
  - provenance: sha256 of the dataset file, retrieval date
  - the nominal vs realised divergence: the harness truncates at ~1024 tokens, so the
    nominal distribution overstates the tail by a large factor. State this plainly. -->

![](../figures/fig05_dataset_distributions.png)

**Figure 2.** <!-- TODO: caption. Must be self-contained: name both series, explain that
solid = realised (what was actually submitted) and dashed = nominal, and explain the
divergence near 1024 tokens. -->

### 3.3 Experiment matrix

<!-- BUDGET 0.4p.
Materials: configs/matrix.csv, figures/fig00b_sessions.svg.
Contents:
  - 208 runs, 3 sessions, 3 repetitions per condition with seeds 1/2/3
  - rate grids: 7B {1,2,3,4,5,6,8,inf}, 0.5B {1,2,4,8,12,16,24,32,inf}
  - fixed flags and why: --temperature 0, --num-warmups 30, --no-enable-prefix-caching
    (prefix caching would make repetitions non-independent; n_cached==0 verifies this)
  - hardware: A40 48GB, driver 580 series / CUDA 13.0
  - table of groups -> what is varied -->

| Group | Model | Dataset | tp | Varied | Runs |
|---|---|---|---|---|---|
| S1 | 7B | ShareGPT | 1 | arrival rate | 24 |
| S2 / S2b | 7B | random | 1 | arrival rate | 24 / 15 |
| S3 | 0.5B | ShareGPT | 1 | arrival rate | 27 |
| I1 / I2 | 7B | random | 1 | input/output ratio | 3 / 3 |
| G1 / G2 / G4 | 7B | ShareGPT | 1 / 2 / 4 | GPU count | 24 / 24 / 24 |
| C2 / C2x | 7B | ShareGPT | 1 | concurrency (closed loop) | 21 / 3 |
| C1off | 7B | ShareGPT | 1 | instrumentation on/off | 3 |
| A1a–A1d | 7B | ShareGPT | 1 | anchor (repeated across sessions) | 12 |
<!-- P1 insert: add a row - P1 | 7B | ShareGPT | pp=2 | parallelism strategy | 24 -->
<!-- TODO: verify every run count against `cut -d, -f1 configs/matrix.csv | sort | uniq -c`. -->

### 3.4 Metrics

<!-- BUDGET 0.2p.
Contents:
  - TTFT, TPOT, ITL, E2EL and their mapping to phases
  - the TWO definitions of achieved throughput used in Figure 5 (completed / measured
    duration, and completed / arrival span). Define both here so 4.1 can just refer to them.
  - percentiles reported (p50, p95) and why the mean is not used -->

### 3.5 Measurement validity

<!-- BUDGET 0.2p. Keep it SHORT. These are controls, not results.
Contents:
  - C1: same condition with the phase logger on and off, adjacent in time.
    Welch's t-test found a real but small effect: +2.97% on TTFT p50, p=0.010.
  - A1 anchors: the same condition repeated in all three sessions, across three
    instances and two regions, agreed within 0.63%.
  - one sentence: these establish that cross-session comparison is admissible. -->

---

## 4. Results

<!-- BUDGET 2.5p total. FACTS ONLY.
Rule: if a sentence contains "because", "due to", or "this is explained by",
it belongs in section 5. Check this before committing the section. -->

### 4.1 Effect of arrival rate

<!-- BUDGET 0.8p. Answers RQ1. -->

![](../figures/fig01_ttft_vs_rate.png)

**Figure 3.** <!-- TODO: caption. Note that the offline (inf) point is detached and is
not a continuation of the finite-rate series. -->

![](../figures/fig02_decode_latency_vs_rate.png)

**Figure 4.** <!-- TODO: caption. -->

![](../figures/fig03_throughput_vs_rate.png)

**Figure 5.** <!-- TODO: caption. Both throughput definitions from 3.4 appear here. -->

<!-- Facts to state:
  - TTFT p50 76 -> 124 ms and p95 151 -> 245 ms over rate 1 -> 8
  - sustainable capacity 3.2 req/s (7B, ShareGPT)
  - queueing delay is approximately zero at all rates; the running batch grows instead
  - DO NOT write "saturation explosion" for the rate 8 -> inf jump. The inf point is a
    burst transient, not a steady state. Say so explicitly. -->

### 4.2 Phase-level behaviour

<!-- BUDGET 0.6p. Answers RQ1 (phase part) and feeds RQ4. -->

![](../figures/fig08_phase_breakdown.png)

**Figure 6.** <!-- TODO: caption. -->

<!-- Facts to state:
  - 38% (7B) and 62% (0.5B) of client-observed TTFT lies outside prefill compute.
    This answers the task's "initial processing, input handling" directly.
  - I1 (512 in / 128 out) 2,891 tok/s vs I2 (128 in / 512 out) 2,144 tok/s.
    Same server, only the input/output ratio differs. -->

### 4.3 Effect of dataset and model size

<!-- BUDGET 0.5p. Answers RQ2. Figures 7 and 8 may go to the appendix if space is tight;
     if so, keep the numbers in the text and reference the appendix figures. -->

![](../figures/fig04_dataset_comparison.png)

**Figure 7.** <!-- TODO: caption. -->

![](../figures/fig06_model_comparison.png)

**Figure 8.** <!-- TODO: caption. -->

<!-- Facts to state:
  - capacity 4.5 req/s (random 256/128) vs 3.2 req/s (ShareGPT)
  - 0.5B vs 7B: where the curves cross and where they do not -->

### 4.4 Effect of GPU count and parallelism strategy

<!-- BUDGET 0.6p. Answers RQ3. -->

![](../figures/fig07_gpu_count_comparison.png)

**Figure 9.** <!-- TODO: caption. Must state that P2P was disabled; see below. -->

<!-- Facts to state:
  - tp=2: +32.5% throughput at rate 8, +43.9% at rinf
  - tp=4: a further +15%, i.e. sublinear
  - EVERY tp number must carry the qualifier "with NCCL P2P and custom all-reduce
    disabled" (D43). These are conservative lower bounds. -->

<!-- P1 insert: one paragraph comparing pp=2 against tp=2 at equal GPU count, plus a new
     Figure. Do NOT modify Figure 9 - add a separate figure. Note that pp>1 switches vLLM
     to step_with_batch_queue, so step-axis data is not comparable with the tp series;
     the pp comparison rests on request-axis metrics only. -->

---

## 5. Bottleneck Analysis

<!-- BUDGET 1.2p. REASONS. This section carries the originality of the report.
     Write it in this order - the order is the argument. -->

### 5.1 The KV cache is not the constraint

<!-- KV utilisation 2.1% at rate 8; batch size constant across tp.
     State the hypothesis, then reject it with the measurement. -->

### 5.2 The mechanism: shorter steps at constant batch

![](../figures/fig11_step_parallelism.png)

**Figure 10.** <!-- TODO: caption. Three panels. -->

<!-- decode-only steps 32.18 -> 22.39 -> 18.07 ms (tp=1/2/4), a 1.78x improvement
     prefill-carrying steps 73.34 -> 64.48 -> 58.20 ms, a 1.26x improvement
     Same parallelisation, different effect per phase. -->

### 5.3 Why decode benefits more

<!-- Per-GPU weight traffic falls with tensor parallelism, relieving a
     memory-bandwidth-bound phase; prefill is compute-bound and gains less.
     Connect back to 4.2. -->

### 5.4 Sublinear scaling and topology

<!-- tp=4 crosses NUMA (SYS) whereas tp=2 stays within one NUMA node (PXB).
     Reference the recorded `nvidia-smi topo -m` output in Appendix C. -->

### 5.5 CPU and framework-bound behaviour

![](../figures/fig09_resources_vs_rate.png)

**Figure 11.** <!-- TODO: caption. -->

<!-- The 0.5B model reaches a framework-bound regime (D27). Per-process CPU from the
     resource logger. This is where the task's "Document CPU performance" is answered. -->

---

## 6. Threats to Validity

<!-- BUDGET 0.4p. A plain list. Honesty here is worth more than polish. -->

<!--
  - tp=2/4 measured with NCCL P2P and custom all-reduce disabled; the reported gains are
    conservative lower bounds
  - no NVLink on the test host; an NVLink system would likely scale differently
  - G1_r1_rep1 returned 199/200 completions and was not re-run, by design
  - the harness truncates ShareGPT prompts at ~1024 tokens
  - rate=inf is not a continuation of the finite-rate series
  - a single GPU model (A40) and a single model family (Qwen2.5)
-->

---

## 7. Conclusion

<!-- BUDGET 0.3p. One short paragraph per RQ, answering it directly.
     No new numbers that have not appeared above. -->

<!-- P1 insert: if pp=2 was NOT executed, add exactly one future-work sentence:
     "Pipeline parallelism was outside the scope of this measurement; because layer-wise
     and tensor-wise partitioning differ in the frequency and granularity of inter-GPU
     communication, a like-for-like comparison at equal GPU count remains future work." -->

---

## References

<!-- See references.md. Target 10-15 entries. -->

---

## Appendix

- **A.** Timing-attribution safeguards — `appendix/A_safeguards.md`
- **B.** Task requirement map — `appendix/B_requirements_map.md`
- **C.** Environment and reproduction — `appendix/C_environment.md`
