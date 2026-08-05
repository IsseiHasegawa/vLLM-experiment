# Appendix A. Timing-attribution safeguards

Two safeguards were applied so that the attribution of step timings remained unambiguous.

**Single execution path.** vLLM exposes two step routines, `step` and
`step_with_batch_queue`. In the latter, execution of one step overlaps the next, which
makes it ambiguous which step a measured duration belongs to. Async scheduling was
explicitly disabled for every run, and the instrumentation patch logs the selected path
once at server startup, so that the choice is recorded rather than assumed. All runs in
Sessions A-C followed the `step` path.

Pipeline parallelism is the documented exception: pp > 1 requires the
`step_with_batch_queue` path, so the instrumented step function is never called and
Session D produces no step-axis log. The startup log records this path selection for
the Session D boot, so the exception is itself measured rather than assumed. The P1
analysis therefore rests on request-axis metrics only, and the step-level
decomposition of §5.2 covers the tensor-parallel series alone.

**Statistics left enabled.** The `--disable-log-stats` flag was not used. Phase timestamps
are carried as `EngineCoreEvents`; disabling statistics would remove the very quantities
being measured. This is noted because benchmarking guides commonly recommend the flag as a
speed optimisation.
