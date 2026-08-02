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
For this measurement, we selected two datasets with contrasting characteristics. ShareGPT is a public dataset comprising actual interactions between humans and dialogue models, and the prompt lengths vary widely. random is a synthetic workload generated by the vLLM benchmark harness, where the input and output lengths can be fixed. We chose these two because, since the amount of work required for prefill depends on the prompt length, the distribution of prompt lengths itself becomes an independent variable. ShareGPT is realistic but uncontrollable, while random is unrealistic but fully controllable. The former provides external validity, while the latter provides internal validity.

ShareGPT used ShareGPT_V3_unfiltered_cleaned_split.json (672,837,942 bytes; first 16 digits of SHA-256 hash: 35f0e213ce091ed9). Distribution statistics were calculated using the Qwen2.5-7B-Instruct tokenizer, with 5,000 samples extracted using seed 42. The random workload used a default of 256 input and 128 output tokens, with range_ratio set to 0 to eliminate length variability. In groups I1 and I2, which isolate the effects of the input-to-output ratio, this fixed length was changed to 512/128 and 128/512, respectively.

The nominal distribution does not match the observed distribution. The ShareGPT sampler in the benchmark harness does not accept prompts exceeding approximately 1,024 tokens. While the maximum input length of the source files was 66,076 tokens, the maximum length actually processed was 1,010 tokens. The summaries of both are shown side by side.

![](../figures/fig05_dataset_distributions.png)

**Figure 2.** Cumulative distribution of prompt lengths for ShareGPT and random
(log x-axis). Solid lines are realised values taken from the phase logs; dashed
lines are nominal values from the source file. The divergence of the two ShareGPT
curves near 1,024 tokens is caused by the benchmark harness sampler, which does
not admit longer prompts. The random workload is fixed-length and therefore steps.

| Dataset | Source | Input p50 | p95 | max |
|:-------------|:--------------------------------|----------:|----:|------:|
| ShareGPT | nominal (source file, n=5,000) | 145 | 938 | 66,076 |
| ShareGPT | realised (S1 phase logs, n=4,800) | 136 | 767 | 1,010 |
| random (S2 default) | nominal = realised (fixed) | 256 | 256 | 256 |

: Prompt-length distributions, nominal versus realised.

Showing only the nominal distribution would result in an overestimation of the actual processed tail by a factor of 65. Therefore, in Figure 2, both distributions are overlaid, and the upper limit of 1,024 tokens is explicitly indicated. The claim of a realistic conversational workload in this report is limited to the range within this upper bound.

Caution is also required regarding output length. ShareGPT's realised output lengths have a p95/p50 ratio of 5.8 with a maximum of 1,642 tokens (S1, n = 4,800), i.e. a heavy tail. Under offline conditions (rate = inf), a run ends when its longest request completes, so S1's achieved throughput at rate = inf varies by 41 % across seeds (3.23 / 4.57 / 3.56 req/s). For this reason, we do not derive ShareGPT's capacity from rate = inf; instead, dataset comparisons are made at matched rates or from the closed-loop group C2.

We specified --ignore-eos for all runs, which pins the generation length to the requested value. This ensures that the output length is no longer influenced by the model’s decision to stop, thereby standardizing the workload across conditions. With prefix caching disabled (§3.3), no KV state is shared between requests even when prompts repeat across repetitions, so the repetitions remain independent. We verified n_cached = 0 for all 39,480 request records.

<!-- TODO: confirm exact filter -->


### 3.3 Experiment matrix



### 3.4 Metrics



### 3.5 Measurement validity


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
