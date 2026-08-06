#!/usr/bin/env python3
"""Three internal-consistency checks a careful reader would run on the report.

These are not number-vs-data checks (verify_report_numbers.py does that and
passes). They ask whether claims in different sections can all be true at once.

    python scripts/check_report_consistency.py
"""

import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "plots"))
import common  # noqa: E402


def runs(R, group, rate=None):
    rs = common.select(R, group=group)
    if rate is not None:
        rs = [r for r in rs if str(r.get("request_rate")) == str(rate)]
    return rs


def req_mean(R, group, rate, field, scale=1.0):
    per_run = []
    for run in runs(R, group, rate):
        recs = common.phase_records(run, "requests")
        if recs:
            per_run.append(st.mean(r[field] * scale for r in recs))
    return st.mean(per_run) if per_run else float("nan")


def bench_mean(R, group, rate, field):
    vals = [(r.get("bench") or {}).get(field) for r in runs(R, group, rate)]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else float("nan")


# --- Check 1 -----------------------------------------------------------------
# Section 4.4 reports two numbers for pp = 2 against tp = 1: mean prefill down
# 10.4 % and mean TTFT down 11.9 %. Section 4.2 says prefill is only part of
# TTFT -- the rest is the unattributed frontend residual. If TTFT fell by more
# than the prefill saving can account for, the residual fell too, and that is a
# host effect the +/-3 % tolerance does not cover. This decomposes both sides.
def check_ttft_decomposition(R):
    print("=== 1. pp=2 の TTFT 改善は prefill 短縮で説明できるか (rate 5) ===")
    rows = {}
    for g in ("G1", "P1"):
        ttft = bench_mean(R, g, 5, "mean_ttft_ms")
        pre = req_mean(R, g, 5, "prefill_s", 1000)
        que = req_mean(R, g, 5, "queued_s", 1000)
        rows[g] = (ttft, que, pre, ttft - que - pre)
        print(f"  {g}: TTFT {ttft:7.2f}  queued {que:5.3f}  prefill {pre:6.2f}"
              f"  residual {ttft - que - pre:7.2f} ms")
    a, b = rows["G1"], rows["P1"]
    d_ttft = b[0] - a[0]
    d_pre = b[2] - a[2]
    d_res = b[3] - a[3]
    print(f"\n  TTFT の変化    {d_ttft:+7.2f} ms  ({d_ttft/a[0]*100:+5.1f} %)")
    print(f"  うち prefill   {d_pre:+7.2f} ms  ({d_pre/a[2]*100:+5.1f} %)")
    print(f"  うち residual  {d_res:+7.2f} ms  ({d_res/a[3]*100:+5.1f} %)")
    share = abs(d_pre) / abs(d_ttft) * 100 if d_ttft else float("nan")
    print(f"\n  prefill が説明する割合: {share:.0f} %")
    if share < 80:
        print("  -> TTFT の改善は prefill 短縮だけでは説明できない。")
        print("     residual も縮んでおり、これはホスト差である可能性が高い。")
        print("     §3.5 の ±3 % 許容はこの大きさをカバーしない。")
    else:
        print("  -> prefill 短縮でおおむね説明できる。記述と整合。")


# --- Check 2 -----------------------------------------------------------------
# The abstract says prefill "never exceeds 2 % of end-to-end time in any
# condition". Section 4.2 supports that for ShareGPT, I1 and I2 at rate 5 and
# for S1 across rates. "Any condition" also covers the 0.5B model and the
# multi-GPU groups, which the text does not report.
def check_prefill_share(R):
    print("\n=== 2. 「prefill は全条件で 2 % 未満」は全条件で成り立つか ===")
    worst = []
    for g in ("S1", "S2", "S2b", "S3", "I1", "I2", "G1", "G2", "G4", "P1"):
        rs = runs(R, g)
        rates = sorted({str(r.get("request_rate")) for r in rs})
        for rate in rates:
            try:
                pre = req_mean(R, g, rate, "prefill_s")
                e2e = req_mean(R, g, rate, "e2e_s")
            except Exception:
                continue
            if e2e and e2e == e2e:
                worst.append((pre / e2e * 100, g, rate))
    worst.sort(reverse=True)
    for v, g, rate in worst[:6]:
        print(f"  {v:6.2f} %   {g} @ rate {rate}")
    top = worst[0][0] if worst else float("nan")
    print(f"\n  最大 {top:.2f} %")
    print("  -> 2 % 未満で成立" if top < 2.0
          else "  -> 2 % を超える条件がある。Abstract の全称主張を要修正。")


# --- Check 3 -----------------------------------------------------------------
# Section 5.1 rules out the KV cache as the constraint from 2.13 % mean and
# 6.43 % peak at rate 8. The offline point submits all 200 requests at once and
# section 4.1 reports the batch growing to 76, so that is where KV pressure
# would show if anywhere.
def check_kv_ceiling(R):
    print("\n=== 3. KV キャッシュは全条件で余っていたか ===")
    worst = []
    for g in ("S1", "S2", "S2b", "S3", "G1", "G2", "G4", "C2", "C2x", "I1", "I2"):
        for run in runs(R, g):
            recs = common.phase_records(run, "steps")
            if not recs:
                continue
            peak = max(s.get("kv_usage", 0) for s in recs) * 100
            worst.append((peak, g, str(run.get("request_rate")),
                          str(run.get("max_concurrency"))))
    worst.sort(reverse=True)
    for v, g, rate, conc in worst[:6]:
        print(f"  {v:6.2f} %   {g} rate={rate} conc={conc}")
    top = worst[0][0] if worst else float("nan")
    print(f"\n  全ラン中の最大 KV 利用率: {top:.2f} %")
    print("  -> どの条件でも余裕あり。§5.1 の反証は成立。" if top < 30
          else "  -> 高い条件がある。§5.1 は rate 8 に限定して述べる必要がある。")


def main():
    R = common.load_runs()
    check_ttft_decomposition(R)
    check_prefill_share(R)
    check_kv_ceiling(R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
