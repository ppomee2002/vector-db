#!/usr/bin/env python3
"""Compact the M2 results (res/*.json) into data embedded in report/m2-report.html (template: m2-template.html)."""
import json
from pathlib import Path
R = Path(__file__).parent / "res"
data = {"sim": {}, "exp1": {}}
for sc in ("S1", "S2", "S3"):
    f = R / f"sim-{sc}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text()); out = {"data_pages": d["data_pages"]}
    for tr in ("natural", "zipf1.2", "zipf0.9"):
        if tr not in d: continue
        x = d[tr]; n = x["requests"]; b = {}
        for k, v in x["budgets"].items():
            b[k] = {kk: (round(vv / n, 2) if isinstance(vv, int) and kk != "pages" else vv) for kk, vv in v.items() if "curve" not in kk}
            b[k]["lru_curve"] = [round(c / n, 2) for c in v["lru_split_curve"]]
            b[k]["lfu_curve"] = [round(c / n, 2) for c in v["lfu_split_curve"]]
        out[tr] = dict(acc=x["accesses_per_query"], budgets=b)
    data["sim"][sc] = out
f = R / "exp1.json" if (R / "exp1.json").exists() else R / "exp1-test.json"
e = json.loads(f.read_text())
for k, v in e.items():
    keep = {kk: vv for kk, vv in v.items() if kk not in ("hist2d", "sample", "rankfreq")}
    if k.startswith("distinct/"):
        keep["rankfreq"] = v["rankfreq"]
        s = v["sample"]; keep["sample"] = [[a, c] for a, c in zip(s["ann"], s["ctx"])]
    data["exp1"][k] = keep
tpl = (Path(__file__).parent / "m2-template.html").read_text()
out = Path(__file__).parent / "report/m2-report.html"; out.parent.mkdir(exist_ok=True)
out.write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
print(out, out.stat().st_size, list(data["sim"]), list(data["exp1"]))
