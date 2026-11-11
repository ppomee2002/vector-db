#!/usr/bin/env python3
"""Optional: how long does the generation step take, next to retrieval?

Runs a small open LLM on one of the server's GPUs with a RAG-shaped prompt
(k retrieved chunks of ~chunk_tokens tokens each + a question) and measures
  - prefill time (time to first token, TTFT)
  - total time for --gen new tokens
at batch size 1. Writes llm_probe.csv; analyze.py puts it next to the
retrieval latency so the share of retrieval in end-to-end latency is visible.

usage: python llm_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --k 10 --chunk-tokens 200 --gen 128 --runs 10
"""
import argparse
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk-tokens", type=int, default=200)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default="llm_probe.csv")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("no CUDA GPU visible", file=sys.stderr)
        return 1
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
    g = torch.Generator().manual_seed(0)
    plen = a.k * a.chunk_tokens + 32
    lo, hi = 1000, min(tok.vocab_size, 30000)
    ids = torch.randint(lo, hi, (1, plen), generator=g).to(dev)  # stand-in for retrieved text
    name = torch.cuda.get_device_name(0)
    print(f"model {a.model} on {name}; prompt {plen} tokens, generate {a.gen}", flush=True)

    def sync():
        torch.cuda.synchronize()

    pre, tot = [], []
    with torch.no_grad():
        for r in range(a.runs + 2):
            sync()
            t = time.perf_counter()
            model(ids)
            sync()
            p = time.perf_counter() - t
            t = time.perf_counter()
            model.generate(ids, max_new_tokens=a.gen, min_new_tokens=a.gen, do_sample=False,
                           pad_token_id=tok.eos_token_id)
            sync()
            e = time.perf_counter() - t
            if r >= 2:  # first two are warm-up
                pre.append(p * 1000)
                tot.append(e * 1000)
    with open(a.out, "w") as f:
        f.write("model,gpu,prompt_tokens,gen_tokens,runs,prefill_ms_median,total_ms_median,ms_per_token\n")
        mp, mt = statistics.median(pre), statistics.median(tot)
        f.write(f"{a.model},{name},{plen},{a.gen},{a.runs},{mp:.2f},{mt:.2f},{(mt - mp) / a.gen:.3f}\n")
    print(f"prefill (TTFT) median {mp:.1f} ms, total {mt:.1f} ms for {a.gen} tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
