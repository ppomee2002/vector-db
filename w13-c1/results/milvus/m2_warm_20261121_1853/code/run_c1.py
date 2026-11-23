# run_c1.py: C1 실험 (E1 배치 x 읽기 방식, E2 병합 간격, E3 Rerank 전 후보 읽기)  김지현 2024134018
#
# 쿼리: NQ test 3,452개. 읽을 chunk 목록은 재은이 결과 파일에서 만든다 (읽기만).
#   top10   Rerank 후 최종 top-10 (rerank_top100_real.csv, ce_rank < 10)
#   bc1     top10 + C2 규칙 B+C1로 고른 이웃 (임준 pf3_rules.py 사본)
#   all1    top10 + 모든 결과의 앞뒤 이웃
#   ann30   Rerank 전 ANN 후보 상위 30개 (ann_top100_real.csv)
#   ann100  ANN 후보 100개 전부
# 읽기: O_DIRECT, 4KB 블록, 단일 스레드. O_DIRECT라 페이지 캐시를 거치지 않으므로 매번 디스크에서 읽는다.
# 반복: 조건 26개 x 3회, 회차마다 조건 순서를 섞는다 (seed = 회차 번호).
#
# 결과 폴더: {WORK}/results/run_YYYYMMDD_HHMM/
#   per_query.csv.gz   쿼리 하나당 한 줄 (읽기 횟수, 읽은 바이트, 필요한 바이트, 지연)
#   per_run.csv        실행(조건 x 회차) 하나당 한 줄 (앱 합계 + /proc/diskstats 합계)
#   read_sets.json     읽는 범위별 쿼리당 평균 chunk 수, 정답 chunk 포함률
#   env.json           실행 환경, 입력 파일 크기·md5, 조건 목록
#   run.log            진행 기록 (run_all.sh가 남김)
import os, sys, csv, gzip, json, time, random, socket, argparse
import numpy as np
import pandas as pd
from common import WORK, ANN_CSV, RERANK_CSV, QRELS, log, safe_path, out_dir, open_read, file_info, write_json, now_kst
from c1_store import Store
from check_store import device_of, diskstats
import nbr_rules

SETS = ("top10", "bc1", "all1", "ann30", "ann100")


def conditions():
    c = []
    for lay in ("rand", "doc"):                                   # E1: 12개
        for merge in (0, 1):
            for s in ("top10", "bc1", "all1"):
                c.append(("E1", lay, merge, 0, s))
    for gap in (4096, 16384, 65536):                              # E2: 6개 (gap 0은 E1에 있음)
        for s in ("bc1", "all1"):
            c.append(("E2", "doc", 1, gap, s))
    for lay in ("rand", "doc"):                                   # E3: 8개
        for merge in (0, 1):
            for s in ("ann30", "ann100"):
                c.append(("E3", lay, merge, 0, s))
    return c


def build_read_sets(doc, nq_limit=None):
    ann = pd.read_csv(open_read(ANN_CSV), usecols=["q_idx", "rank", "chunk_id"])
    rr = pd.read_csv(open_read(RERANK_CSV), usecols=["q_idx", "chunk_id", "ce_rank"])
    qs = sorted(ann.q_idx.unique())
    if nq_limit:
        qs = qs[:nq_limit]
    ann = ann.sort_values(["q_idx", "rank"]).groupby("q_idx").chunk_id.apply(list).to_dict()
    fin = rr[rr.ce_rank < 10].sort_values(["q_idx", "ce_rank"]).groupby("q_idx").chunk_id.apply(list).to_dict()
    sets = {s: [] for s in SETS}
    for q in qs:
        R = [int(c) for c in fin[q]]
        sets["top10"].append(R)
        sets["bc1"].append(R + nbr_rules.select("BC1", R, doc))
        sets["all1"].append(R + nbr_rules.select("all", R, doc))
        sets["ann30"].append([int(c) for c in ann[q][:30]])
        sets["ann100"].append([int(c) for c in ann[q][:100]])
    return qs, sets


def coverage(qs, sets):
    """정답 chunk(qrels_chunk.tsv, 정답 문단 기준)가 읽은 범위에 하나라도 있는 쿼리 비율."""
    out = {}
    rel = None
    if os.path.exists(QRELS):
        qr = pd.read_csv(open_read(QRELS), sep="\t", usecols=["q_idx", "chunk_id"])
        rel = qr.groupby("q_idx").chunk_id.apply(set).to_dict()
    for s, lst in sets.items():
        d = {"mean_chunks_per_query": round(float(np.mean([len(x) for x in lst])), 2)}
        if rel is not None:
            d["gold_chunk_hit_rate"] = round(float(np.mean([len(set(x) & rel.get(q, set())) > 0 for q, x in zip(qs, lst)])), 4)
        out[s] = d
    return out


def quiet_check(dev, sec=5, max_reads_per_s=50):
    """다른 사람이 같은 장치를 많이 읽는 중인지 본다 (아무것도 바꾸지 않음)."""
    if not dev:
        return {"device": None, "quiet": None}
    a = diskstats(dev); time.sleep(sec); b = diskstats(dev)
    rps = (b[0] - a[0]) / sec
    return {"device": dev, "reads_per_s": round(rps, 1), "MB_per_s": round((b[1] - a[1]) / sec / 1e6, 2), "quiet": rps <= max_reads_per_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(WORK, "store"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="쿼리 300개, 1회만 (점검용)")
    ap.add_argument("--only", default="", help="예: E1  (쉼표로 여러 개)")
    ap.add_argument("--no-direct", action="store_true", help="O_DIRECT 없이 (로컬 테스트용)")
    ap.add_argument("--out", default="")
    ap.add_argument("--wait-quiet-min", type=int, default=30, help="장치가 바쁘면 최대 몇 분 기다릴지")
    a = ap.parse_args()
    reps = 1 if a.quick else a.reps
    stamp = time.strftime("%Y%m%d_%H%M")
    R = safe_path(a.out) if a.out else out_dir("results", f"run_{stamp}" + ("_quick" if a.quick else ""))
    os.makedirs(R, exist_ok=True)

    doc = np.load(os.path.join(a.store, "doc_idx.npy"))
    qs, sets = build_read_sets(doc, 300 if a.quick else None)
    cov = coverage(qs, sets)
    write_json(os.path.join(R, "read_sets.json"), cov)
    log("read sets:", json.dumps(cov, ensure_ascii=False))

    conds = conditions()
    if a.only:
        keep = set(a.only.split(","))
        conds = [c for c in conds if c[0] in keep]
    dev = device_of(os.path.join(a.store, "doc.bin"))
    stores = {lay: Store(a.store, lay, direct=not a.no_direct) for lay in ("rand", "doc")}

    waited = 0
    while True:
        q = quiet_check(dev)
        if q["quiet"] in (True, None) or waited >= a.wait_quiet_min:
            break
        log(f"device busy ({q['reads_per_s']} reads/s), waiting 1 min"); time.sleep(60); waited += 1
    log("quiet check:", q)

    write_json(os.path.join(R, "env.json"), {
        "started_at_kst": now_kst(), "host": socket.gethostname(), "user": os.environ.get("USER"),
        "python": sys.version.split()[0], "numpy": np.__version__, "device": dev, "o_direct": not a.no_direct,
        "block": 4096, "threads": 1, "reps": reps, "n_queries": len(qs), "quick": a.quick,
        "quiet_check_before": q, "waited_min": waited,
        "inputs": {**{k: file_info(p) for k, p in (("ann", ANN_CSV), ("rerank", RERANK_CSV))},
                   **({"qrels": file_info(QRELS, md5=False)} if os.path.exists(QRELS) else {})},
        "store_meta": json.load(open_read(os.path.join(a.store, "meta.json"), encoding="utf-8")),
        "conditions": [dict(zip(("exp", "layout", "merge", "gap", "read_set"), c)) for c in conds]})

    fq = gzip.open(safe_path(os.path.join(R, "per_query.csv.gz")), "wt", newline="")
    wq = csv.writer(fq)
    wq.writerow(["run_id", "exp", "layout", "merge", "gap", "read_set", "rep", "q_idx", "n_chunks", "n_reads", "bytes_read", "useful_bytes", "latency_us"])
    fr = open(safe_path(os.path.join(R, "per_run.csv")), "w", newline="")
    wr = csv.writer(fr)
    wr.writerow(["run_id", "exp", "layout", "merge", "gap", "read_set", "rep", "n_queries", "app_reads", "app_bytes",
                 "disk_reads", "disk_bytes", "wall_s", "start_kst", "end_kst"])
    rid, total = 0, len(conds) * reps
    for rep in range(reps):
        order = conds[:]
        random.Random(rep).shuffle(order)
        for exp, lay, merge, gap, s in order:
            rid += 1
            st = stores[lay]
            t_start = now_kst(); d0 = diskstats(dev) if dev else None; t0 = time.perf_counter(); ar = ab = 0
            for q, ids in zip(qs, sets[s]):
                _, x = st.read(ids, merge=bool(merge), gap=gap, want_text=False)
                ar += x["n_reads"]; ab += x["bytes_read"]
                wq.writerow([rid, exp, lay, merge, gap, s, rep, q, x["n_chunks"], x["n_reads"], x["bytes_read"], x["useful_bytes"], x["latency_us"]])
            wall = time.perf_counter() - t0; d1 = diskstats(dev) if dev else None
            dr, db = (d1[0] - d0[0], d1[1] - d0[1]) if dev else ("", "")
            wr.writerow([rid, exp, lay, merge, gap, s, rep, len(qs), ar, ab, dr, db, round(wall, 2), t_start, now_kst()])
            fr.flush()
            log(f"run {rid}/{total} {exp} {lay} merge={merge} gap={gap} {s}: reads/q {ar / len(qs):.2f}, {wall:.1f}s"
                + (f", disk/app reads {dr / max(ar, 1):.3f}" if dev else ""))
    fq.close(); fr.close()
    for st in stores.values():
        st.close()
    env = json.load(open_read(os.path.join(R, "env.json"), encoding="utf-8"))
    env["finished_at_kst"] = now_kst()
    write_json(os.path.join(R, "env.json"), env)
    log("ALL DONE ->", R)
    print(R)


if __name__ == "__main__":
    main()
