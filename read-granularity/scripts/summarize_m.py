import json, re, sys, glob, os, statistics as st
d = sys.argv[1]; WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = re.compile(r'[Rr]ange: bytes=(\d+)-(\d+)')
print("| case | QPS | p50 ms | p99 ms | GET/req | KB/GET | S3 MB/req | amp | cell MB (avg) |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
order = {"r1": 1, "r2": 2, "r4": 4, "rstock": 4.5, "r8": 8, "r16": 16}
rows = []
for f in glob.glob(f"{d}/*-c32.json"):
    j = json.load(open(f)); c = j["case"]
    ds, r, b, tr = c.split("-")[:4]
    lines = [json.loads(l) for l in open(f[:-5] + ".jsonl")]
    n = len(lines); tb = sum(x["text_bytes"] for x in lines) / n
    s3 = open(f[:-5] + ".s3.log", errors="ignore").read()
    gets = s3.count("GET "); rs = [int(b2) - int(a) + 1 for a, b2 in rng.findall(s3)]
    tot = n + WARM  # capture covers warmup + measured
    cells = open(f"{d}/{ds}-{r}-{b}.cells.txt", errors="ignore").read()
    cb = [float(x) for x in re.findall(r'cell bytes avg ([0-9.]+)', cells)]
    rows.append((ds, b, tr, order.get(r, 99), r, j["qps"], j["p50_ms"], j["p99_ms"], gets / tot,
                 st.mean(rs) / 1e3 if rs else 0, sum(rs) / tot / 1e6, sum(rs) / tot / tb if tb else 0,
                 st.mean(cb) / 1e6 if cb else float("nan")))
for x in sorted(rows):
    print(f"| {x[0]} {x[1]} {x[2]} {x[4]} | {x[5]:.0f} | {x[6]:.0f} | {x[7]:.0f} | {x[8]:.2f} | {x[9]:.0f} | {x[10]:.2f} | {x[11]:.0f}x | {x[12]:.2f} |")
