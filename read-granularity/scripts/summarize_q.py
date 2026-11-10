import json, glob, re, sys, statistics as st, collections
d = sys.argv[1]
rows = collections.defaultdict(list)
for f in glob.glob(f"{d}/*.json"):
    if f.endswith(".gran.json") or "/probe" in f: continue
    s = json.load(open(f)); c = s["case"]
    m = re.match(r"(\w+)-(stockmmap|g(\d+))-(c\d+|b[\d.]+-c\d+|n\d+-c\d+)(?:-r\d+)?$", c)
    if not m: continue
    ds, kind, g, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    gr = json.load(open(f[:-5] + ".gran.json")).get("after", {})
    tot = s["requests"]
    rows[(ds, rest, "stock" if kind == "stockmmap" else int(g))].append(dict(
        qps=s["qps"], p50=s["p50_ms"], p99=s["p99_ms"],
        dev_amp=s["storage_read_bytes"] / max(1, s["returned_text_bytes"]),
        dev_ios_q=s["proc_delta"]["io_rios"] / tot,
        dev_kb_io=s["storage_read_bytes"] / max(1, s["proc_delta"]["io_rios"]) / 1024,
        app_amp=(gr.get("io_bytes", 0) / gr["value_bytes"]) if gr.get("value_bytes") else float("nan")))
def key(k): return (k[0], int(re.findall(r"c(\d+)", k[1])[0]), k[1], -1 if k[2] == "stock" else k[2])
def fmt(g): return "stock(mmap)" if g == "stock" else ("io_uring 4K" if g == 0 else (f"{g//1024}K" if g < 1<<20 else f"{g>>20}M"))
print("| ds | cond | read unit | n | QPS | p50 ms | p99 ms | dev amp | dev IO/query | dev KB/IO | app amp |")
print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for k in sorted(rows, key=key):
    v = rows[k]; md = lambda x: st.median(r[x] for r in v)
    print(f"| {k[0]} | {k[1]} | {fmt(k[2])} | {len(v)} | {md('qps'):.0f} | {md('p50'):.1f} | {md('p99'):.1f} | {md('dev_amp'):.1f}x | {md('dev_ios_q'):.1f} | {md('dev_kb_io'):.1f} | {md('app_amp'):.0f}x |")
