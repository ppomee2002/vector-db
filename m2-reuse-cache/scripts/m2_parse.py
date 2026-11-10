#!/usr/bin/env python3
"""Parse the Qdrant M2 trace into per-query access arrays (one entry per distinct query, in run order).

Page model (4 KB pages, one global page-id space):
  vectors  : segment matrix.dat, 4-byte header + internal_id * dim*4 bytes (a vector may span 2 pages)
  links    : segment links_compressed.bin; level-0 list of node i at i * (links bytes / points) (uniform approx.),
             upper level l in its own region of size links/16^l, node i at rank i*n_l/points
  context  : gridstore payload page_<p>.dat (32 MB) at byte offset, length = stored (LZ4) length
Chunk ids (for experiment 1): (segment index, internal offset) -> seg*2^22 + offset.
Writes <out>/trace.npz and layout.json."""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np

TR, CLIENT, SEGDIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
PG = 4096; DIM4 = 384 * 4
uuid_re = re.compile(r"segments/([0-9a-f-]{36})/")

def size(p):
    r = subprocess.run(["sudo", "-n", "stat", "-c", "%s", str(p)], capture_output=True, text=True)
    return int(r.stdout) if r.returncode == 0 else 0

segs = sorted(x.strip() for x in subprocess.run(["sudo", "-n", "ls", str(SEGDIR)], capture_output=True, text=True).stdout.split())
lay, base = {}, 0
for s in segs:
    mat = size(SEGDIR / s / "vector_storage/matrix.dat")
    if mat == 0:
        continue
    links = size(SEGDIR / s / "vector_index/links_compressed.bin")
    npg = len([x for x in subprocess.run(["sudo", "-n", "ls", str(SEGDIR / s / "payload_storage")], capture_output=True, text=True).stdout.split() if x.startswith("page_")])
    n = (mat - 4) // DIM4
    L = dict(idx=len(lay), points=n, vec_bytes=mat, links_bytes=links, payload_pages=npg)
    L["vec_base"] = base; base += -(-mat // PG)
    L["l0_base"] = base; base += -(-links // PG)
    L["up_base"] = []
    for lvl in range(1, 8):
        L["up_base"].append(base); base += max(1, -(-links // 16 ** lvl // PG))
    L["ann_end"] = base
    lay[s] = L
ann_pages = sum(L["ann_end"] - L["vec_base"] for L in lay.values())
for s, L in lay.items():
    L["ctx_base"] = base; base += L["payload_pages"] * (32 << 20) // PG
ctx_pages = base - ann_pages
meta = dict(page=PG, total_pages=base, ann_pages=ann_pages, ctx_pages=ctx_pages, segments=lay)

def vec_pages(L, i):
    a = 4 + i * DIM4
    return range(L["vec_base"] + a // PG, L["vec_base"] + (a + DIM4 - 1) // PG + 1)

def link_page(L, i, lvl):
    per = L["links_bytes"] / L["points"]
    if lvl == 0:
        return L["l0_base"] + int(i * per) // PG
    return L["up_base"][min(lvl, 7) - 1] + int(i / 16 ** lvl * per) // PG

queries, cur, state = [], None, "P"
for line in TR.open():
    t = line.split()
    if not t:
        continue
    if t[0] == "H":
        if state == "P":
            cur = dict(h=0, vp=[], lp=[], vn=[], cp=[], cn=[], cb=0); queries.append(cur)
        state = "H"; cur["h"] += 1
        L = lay[uuid_re.search(t[1]).group(1)]
        seen = set()
        for tok in t[2:]:
            if tok[0] == "L":
                i, lvl = tok[1:].split("."); cur["lp"].append(link_page(L, int(i), int(lvl)))
            else:
                i = int(tok)
                cur["vp"].extend(vec_pages(L, i))
                if i not in seen:
                    seen.add(i); cur["vn"].append(L["idx"] << 22 | i)
    elif t[0] == "P":
        state = "P"
        L = lay[uuid_re.search(t[1]).group(1)]
        off, page, bo, ln = map(int, t[2:6])
        a = page * (32 << 20) + bo
        cur["cp"].extend(range(L["ctx_base"] + a // PG, L["ctx_base"] + (a + ln - 1) // PG + 1))
        cur["cn"].append(L["idx"] << 22 | off); cur["cb"] += ln

client = [json.loads(l) for l in CLIENT.open()]
nh = np.array([q["h"] for q in queries]); nc = np.array([len(q["cn"]) for q in queries])
print(f"trace queries {len(queries)} client {len(client)}; H/query {np.unique(nh, return_counts=True)}; ctx/query {np.unique(nc, return_counts=True)}")
assert len(queries) == len(client), "query grouping mismatch"
qid = np.array([c["q"] for c in client])

def pack(key, dt=np.int64):
    lens = np.array([len(q[key]) for q in queries]); off = np.concatenate([[0], np.cumsum(lens)])
    return np.concatenate([np.array(q[key], dtype=dt) for q in queries]), off

arrs = {}
for k in ("vp", "lp", "vn", "cp", "cn"):
    arrs[k], arrs[k + "_off"] = pack(k)
np.savez(OUT / "trace.npz", qid=qid, ctx_stored_bytes=np.array([q["cb"] for q in queries]),
         ctx_text_bytes=np.array([sum(c["tl"]) for c in client]), **arrs)
(OUT / "layout.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ("total_pages", "ann_pages", "ctx_pages")}),
      "per query: vec pages %.0f, link pages %.0f, vec nodes %.0f, ctx pages %.1f" % tuple(
          np.diff(arrs[k + "_off"]).mean() for k in ("vp", "lp", "vn", "cp")))
