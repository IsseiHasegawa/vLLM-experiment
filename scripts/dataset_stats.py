#!/usr/bin/env python3
"""Compute token-length statistics for the two datasets (input for figure 5).

Run this wherever a tokenizer is available - the pod already has `transformers`
as a vLLM dependency, so no extra install is needed there:

    python3 scripts/dataset_stats.py \
        --sharegpt /workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
        --model Qwen/Qwen2.5-7B-Instruct \
        --out results/dataset_stats.json

Output JSON:
    {"sharegpt": {"input_lens": [...], "output_lens": [...], "summary": {...}},
     "random":   {"input_lens": [...], "output_lens": [...], "summary": {...}}}

Notes
-----
* Sampling mirrors what `vllm bench serve --dataset-name sharegpt` feeds the
  server: the first human turn is the prompt and the following assistant turn
  is the reference output; conversations with fewer than two turns are dropped.
* The `random` workload is fixed-length by construction (256 in / 128 out for
  S2), so its "distribution" is a spike. That contrast is the point of the
  figure, and the fixed lengths are read from the CLI rather than sampled.
* The commit of the dataset file is recorded via its size and sha256 prefix so
  the figure can be tied to an exact input in the report appendix.
"""

import argparse
import hashlib
import json
import os
import random
import statistics as st


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return s[i]


def summarize(inp, out):
    return {
        "n": len(inp),
        "input_mean": round(st.mean(inp), 1) if inp else None,
        "input_p50": percentile(inp, 0.50),
        "input_p95": percentile(inp, 0.95),
        "input_max": max(inp) if inp else None,
        "output_mean": round(st.mean(out), 1) if out else None,
        "output_p50": percentile(out, 0.50),
        "output_p95": percentile(out, 0.95),
        "output_max": max(out) if out else None,
    }


def file_fingerprint(path, chunk=1 << 20):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return {"bytes": n, "sha256_prefix": h.hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/dataset_stats.json")
    ap.add_argument("--sample", type=int, default=5000,
                    help="conversations to tokenize (0 = all; 5000 is plenty)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-input-len", type=int, default=256)
    ap.add_argument("--random-output-len", type=int, default=128)
    ap.add_argument("--random-n", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoTokenizer  # imported late: only needed here

    print(f"loading tokenizer {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"reading {args.sharegpt} ...")
    with open(args.sharegpt) as f:
        data = json.load(f)
    convs = [c for c in data if len(c.get("conversations", [])) >= 2]
    print(f"  {len(data)} entries, {len(convs)} usable conversations")

    random.seed(args.seed)
    if args.sample and len(convs) > args.sample:
        convs = random.sample(convs, args.sample)
        print(f"  sampling {len(convs)} (seed {args.seed})")

    inp, out = [], []
    for i, c in enumerate(convs):
        t = c["conversations"]
        try:
            prompt, answer = t[0]["value"], t[1]["value"]
        except (KeyError, IndexError, TypeError):
            continue
        inp.append(len(tok(prompt).input_ids))
        out.append(len(tok(answer).input_ids))
        if (i + 1) % 1000 == 0:
            print(f"  tokenized {i + 1}/{len(convs)}")

    stats = {
        "sharegpt": {
            "input_lens": inp,
            "output_lens": out,
            "summary": summarize(inp, out),
            "source": {"path": os.path.abspath(args.sharegpt),
                       **file_fingerprint(args.sharegpt),
                       "tokenizer": args.model,
                       "sampled": len(inp), "seed": args.seed},
        },
        "random": {
            "input_lens": [args.random_input_len] * args.random_n,
            "output_lens": [args.random_output_len] * args.random_n,
            "summary": summarize([args.random_input_len] * args.random_n,
                                 [args.random_output_len] * args.random_n),
            "source": {"synthetic": True,
                       "random_input_len": args.random_input_len,
                       "random_output_len": args.random_output_len,
                       "range_ratio": 0.0},
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f)
    print(f"\nwrote {args.out}")
    for k, v in stats.items():
        s = v["summary"]
        print(f"  {k:9s} input p50={s['input_p50']} p95={s['input_p95']} "
              f"max={s['input_max']} | output p50={s['output_p50']} "
              f"p95={s['output_p95']} max={s['output_max']}")


if __name__ == "__main__":
    main()
