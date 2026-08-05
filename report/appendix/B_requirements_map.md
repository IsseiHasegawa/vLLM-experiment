# Appendix B. Task requirement map

Each requirement of the assignment, and where it is addressed.

| Requirement | Section | Figure | Evidence |
|---|---|---|---|
| Deploy vLLM from the repository; be able to change and recompile it | 3.1 | — | Fork at base `702f4814`, `instrumentation` branch, editable install from source |
| Capture end-to-end and per-phase latency, and throughput | 3.1, 3.4 | 1 | `requests.jsonl`, `steps.jsonl`; cross-checked against Prometheus |
| Time and resource usage during the prefill phase | 4.2, 5.2 | 6, 12 | Group I1 (512 in / 128 out); `n_ctx_toks`-dominated steps; SM utilisation |
| Time and resource usage during the decode phase | 4.2, 5.2 | 6, 12 | Group I2 (128 in / 512 out); `n_gen_toks`-dominated steps; memory-controller utilisation |
| At least two documented datasets | 3.2 | 2 | ShareGPT and random; sha256 recorded; nominal vs realised distributions |
| Vary the request arrival rate | 3.3, 4.1, 4.5 | 3, 4, 5, 11 | Open loop S1/S2/S2b/S3; closed loop C2/C2x |
| Deploy at least two models or model sizes | 3.3, 4.3 | 8 | Qwen2.5-7B and Qwen2.5-0.5B, compared within one instance |
| Vary the number of GPUs | 3.3, 4.4 | 9 | G1/G2/G4 = 1/2/4 GPUs, all within a single 4-GPU instance |
| Document CPU performance | 3.1, 5.5 | 13 | Per-process CPU from the external resource logger; framework-bound regime on the 0.5B model |
| Enable and evaluate parallel processing options | 4.4 | 9, 10 | Tensor parallelism tp=1/2/4 and pipeline parallelism pp=2 at equal GPU count |
| Analyse results to determine bottlenecks | 5 | 12, 13 | KV hypothesis rejected; step-level decomposition; memory-bandwidth attribution |
| Generate figures | 4, 5 | 1–13 | Thirteen figures, all generated from measured data |
