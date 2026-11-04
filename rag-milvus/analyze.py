#!/usr/bin/env python3
"""Summarize a run_trend.sh directory into the four motivation metrics.

(1) duplicate S3 fetches      : GETs of an object already fetched earlier in the same case
(2) cache-structure gap       : Milvus text bytes fetched from storage vs. a chunk-granular
                                LRU / Belady cache with the same text byte budget (replayed
                                on the same per-request results, cold start)
(3) context-read cost         : QPS/p99 with text vs without text (zipf, c32)
(4) read amplification        : Milvus text miss bytes / text bytes actually returned
"""
import argparse, collections, heapq, json, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def text_lengths(ds):
    f = ROOT / f"datasets/emb/{ds}/text_len.npy"
    if not f.exists():
        t = pq.read_table(next((ROOT / f"datasets/raw/hf/{ds}").glob("corpus-*.parquet")), columns=["title", "text"]).to_pydict()
        np.save(f, np.array([len(((ti + " " + x).strip() if ti else x).encode()) for ti, x in zip(t["title"], t["text"])], dtype=np.int32))
    return np.load(f)


def replay(seq, size, cap):
    """Return (lru_miss_bytes, belady_miss_bytes) for a byte-capacity chunk cache."""
    # LRU
    od, used, lru_miss = collections.OrderedDict(), 0, 0
    for k in seq:
        if k in od:
            od.move_to_end(k); continue
        lru_miss += size[k]
        od[k] = True; used += size[k]
        while used > cap and od:
            v, _ = od.popitem(last=False); used -= size[v]
    # Belady (evict farthest next use; variable sizes → greedy approximation)
    nxt, last = [0] * len(seq), {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
    cache, heap, used, bel_miss = {}, [], 0, 0
    for i, k in enumerate(seq):
        if k in cache:
            cache[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
        bel_miss += size[k]
        if size[k] > cap:
            continue
        cache[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); used += size[k]
        while used > cap:
            negn, v = heapq.heappop(heap)
            if v in cache and cache[v] == -negn:
                del cache[v]; used -= size[v]
    return lru_miss, bel_miss


def s3_stats(path):
    if not path.exists():
        return None, None
    keys = [l.split()[1] for l in path.read_text(errors="ignore").splitlines() if l.startswith("GET ")]
    seen, dup = set(), 0
    for k in keys:
        dup += k in seen; seen.add(k)
    return len(keys), (dup / len(keys) if keys else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    a = p.parse_args()
    sizes = json.loads((a.dir / "sizes.json").read_text())
    rows, by = [], {}
    for f in sorted(a.dir.glob("*.json")):
        if f.name in ("sizes.json", "analysis.json"):
            continue
        s = json.loads(f.read_text())
        by[s["case"]] = s
    ds = next(iter(by.values()))["dataset"]
    L = text_lengths(ds).tolist()
    for case, s in sorted(by.items()):
        m = re.match(r".*-([BCQ])-b([0-9.]+)-(natural|zipf)-c(\d+)(-notext)?$", case)
        if not m:
            continue
        eng, b, tr, cc, notext = m.group(1), float(m.group(2)), m.group(3), int(m.group(4)), bool(m.group(5))
        md = s.get("metrics_delta", {})
        if eng == "Q":  # Qdrant: bytes read from disk by the container (payload pages via page cache)
            tmiss, vmiss = s["storage_read_bytes"], None
        elif eng == "B":  # Milvus mmap: bytes read from disk by the container (mmap'd text via page cache)
            tmiss, vmiss = s["proc_delta"]["io_rbytes"], None
        else:
            tmiss = md.get("internal_cache_cell_access_miss_bytes_total:scalar_field", s["cache_miss_bytes"])
            vmiss = md.get("internal_cache_cell_access_miss_bytes_total:vector_index")
        r = dict(case=case, budget=b, trace=tr, c=cc, text=not notext, qps=s["qps"], p95=s["p95_ms"], p99=s["p99_ms"],
                 returned=s["returned_text_bytes"], milvus_text_miss=tmiss, vec_index_miss=vmiss,
                 evicted=md.get("internal_cache_evicted_bytes_total", 0))
        if not notext:
            seq = []
            for line in (a.dir / f"{case}.jsonl").open():
                seq += json.loads(line)["pks"]
            cap = b * sizes["text_bytes"]
            r["chunk_lru_miss"], r["chunk_belady_miss"] = replay(seq, L, cap)
            r["s3_gets"], r["s3_dup_rate"] = s3_stats(a.dir / f"{case}.s3.log")
        rows.append(r)
    (a.dir / "analysis.json").write_text(json.dumps(rows, indent=2) + "\n")

    mb = lambda x: "-" if x is None else f"{x/1e6:.1f}"
    eng_name = ("Qdrant (on_disk_payload, page-cache budget via memory limit)" if "payload_disk_bytes" in sizes
                else "Milvus v2.6.23 mmap (config B, page-cache budget via memory limit)" if sizes.get("mmap")
                else "Milvus v2.6.23 tiered storage (config C)")
    extra = (f"payload on disk {sizes['payload_disk_bytes']/1e6:.1f} MB" if "payload_disk_bytes" in sizes
             else f"container memory after vector-only probe {sizes['m0_bytes']/1e6:.1f} MB" if sizes.get("mmap")
             else f"vector index {sizes['vector_index_bytes']/1e6:.1f} MB")
    out = [f"# Trend: {ds}, {eng_name}", "",
           f"text {sizes['text_bytes']/1e6:.1f} MB, {extra}. "
           "Budget = text cache share (Qdrant: of payload disk size). Chunk LRU/Belady = same byte budget, chunk-granular, cold replay of the same results.", "",
           "| budget | trace | C | QPS | p95 ms | p99 ms | returned MB | storage text read MB | amplification | chunk LRU MB | chunk Belady MB | Milvus/Belady | S3 GETs | dup GET % |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in rows if r["text"]), key=lambda r: (r["trace"], -r["budget"], r["c"])):
        amp = r["milvus_text_miss"] / r["returned"] if r["returned"] else 0
        ratio = r["milvus_text_miss"] / r["chunk_belady_miss"] if r["chunk_belady_miss"] else float("nan")
        out.append(f"| {r['budget']:.0%} | {r['trace']} | {r['c']} | {r['qps']:.0f} | {r['p95']:.1f} | {r['p99']:.1f} | "
                   f"{mb(r['returned'])} | {mb(r['milvus_text_miss'])} | {amp:.1f}x | {mb(r['chunk_lru_miss'])} | "
                   f"{mb(r['chunk_belady_miss'])} | {ratio:.1f}x | {r['s3_gets'] or 0} | {100*(r['s3_dup_rate'] or 0):.0f} |")
    out += ["", "## Context-read cost (zipf, C=32): text vs no text", "", "| budget | QPS text | QPS no text | p99 text | p99 no text |", "|---:|---:|---:|---:|---:|"]
    for b in sorted({r["budget"] for r in rows}, reverse=True):
        t = next((r for r in rows if r["budget"] == b and r["trace"] == "zipf" and r["c"] == 32 and r["text"]), None)
        n = next((r for r in rows if r["budget"] == b and not r["text"]), None)
        if t and n:
            out.append(f"| {b:.0%} | {t['qps']:.0f} | {n['qps']:.0f} | {t['p99']:.1f} | {n['p99']:.1f} |")
    (a.dir / "TREND.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
