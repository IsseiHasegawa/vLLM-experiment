# Appendix A. Timing-attribution safeguards

Two safeguards were applied so that the attribution of step timings remained unambiguous.

**Single execution path.** vLLM exposes two step routines, `step` and
`step_with_batch_queue`. In the latter, execution of one step overlaps the next, which
makes it ambiguous which step a measured duration belongs to. Async scheduling was
explicitly disabled for every run, and the instrumentation patch logs the selected path
once at server startup, so that the choice is recorded rather than assumed. All runs in
Sessions A-C followed the `step` path.

<!-- P1 insert: pipeline parallelism (pp>1) forces the `step_with_batch_queue` path.
     If the P1 session is included, state here that its step-axis data is therefore not
     comparable with the tp series, and that the P1 analysis uses request-axis metrics only. -->

**Statistics left enabled.** The `--disable-log-stats` flag was not used. Phase timestamps
are carried as `EngineCoreEvents`; disabling statistics would remove the very quantities
being measured. This is noted because benchmarking guides commonly recommend the flag as a
speed optimisation.
