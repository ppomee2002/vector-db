#!/usr/bin/env python3
"""Pick m3bench search settings (nlist, nprobe) from the calibration measurements -> calib.env

Three search-cost levels, each = the Go IVF setting whose single-thread time is closest to a
FAISS HNSW operating point measured on this server:
  LIGHT : synthetic data (easy) at recall >= --light-recall      -> lower bound of search cost
  MAIN  : real embeddings (NYTimes-256, scaled) at recall >= --main-recall   (production-like)
  HEAVY : real embeddings at recall >= --heavy-recall                         (strict accuracy)
Without usable real-data measurements MAIN = HEAVY = LIGHT (flagged in CALIB_SOURCE).
Re-runnable on its own: python3 calib_choose.py   (uses the CSVs already in this folder)
"""
import argparse
import csv
import os
import sys


def rows(path, index="hnsw_flat"):
    if not os.path.exists(path):
        return []
    return [r for r in csv.DictReader(open(path)) if r["index"] == index]


def target(rs, recall, label):
    """Smallest efSearch reaching the recall target; None if no setting reaches it."""
    ok = sorted((r for r in rs if float(r["recall"]) >= recall), key=lambda r: int(r["value"]))
    if not ok:
        if rs:
            print(f"WARNING: {label}: no setting reached recall {recall} "
                  f"(best {max(float(r['recall']) for r in rs):.3f}) -> not used")
        return None
    return ok[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--light-recall", type=float, default=0.95)
    ap.add_argument("--main-recall", type=float, default=0.90)
    ap.add_argument("--heavy-recall", type=float, default=0.95)
    a = ap.parse_args()
    go = list(csv.DictReader(open("calib_go.csv")))
    closest = lambda ms: min(go, key=lambda r: abs(float(r["ms_per_query"]) - ms))
    synth, real = rows("calib_faiss_synth.csv"), rows("calib_faiss_real.csv")
    light = target(synth, a.light_recall, "synthetic")
    if light is None:
        sys.exit("synthetic calibration did not reach the recall target; keeping the defaults (no calib.env written)")
    main_t = target(real, a.main_recall, "real/main")
    heavy_t = target(real, a.heavy_recall, "real/heavy")
    if main_t is None:
        main_t, src = light, "synthetic (real data unavailable or unusable)"
    else:
        src = f"real (NYTimes-256, scaled) main@recall{a.main_recall} heavy@recall{a.heavy_recall}"
    if heavy_t is None:
        heavy_t = main_t
    lv = {}
    for name, t in (("LIGHT", light), ("MAIN", main_t), ("HEAVY", heavy_t)):
        g = closest(float(t["ms_scaled"]))
        lv[name] = (t, g)
    with open("calib.env", "w") as f:
        g = lv["MAIN"][1]
        f.write(f"NLIST={g['nlist']}\nNPROBE={g['nprobe']}\n")
        for name in ("LIGHT", "HEAVY"):
            g = lv[name][1]
            f.write(f"NLIST_{name}={g['nlist']}\nNPROBE_{name}={g['nprobe']}\n")
        f.write(f"CALIB_SOURCE='{src}'\n")
        for name in ("LIGHT", "MAIN", "HEAVY"):
            t, g = lv[name]
            f.write(f"# {name.lower():5s}: FAISS HNSW efSearch={t['value']} recall={float(t['recall']):.3f} -> "
                    f"{float(t['ms_scaled']):.4f} ms; Go nlist={g['nlist']} nprobe={g['nprobe']} = {float(g['ms_per_query']):.4f} ms\n")
    print(open("calib.env").read())
    fastest = min(float(r["ms_per_query"]) for r in go)
    for name in ("LIGHT", "MAIN", "HEAVY"):
        t = float(lv[name][0]["ms_scaled"])
        if fastest > 1.3 * t:
            print(f"NOTE: {name.lower()} target {t:.3f} ms is below the cheapest Go setting ({fastest:.3f} ms); "
                  "that level over-states search CPU (the nosearch family is the lower bound).")


if __name__ == "__main__":
    main()
