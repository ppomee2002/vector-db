# c1_milvus.py: Milvus 안에서 C1 (문서 순 PK 배치 + 결과·이웃 한 번에 조회) 측정  김지현 2024134018
#
# 명령
#   python c1_milvus.py check            pymilvus 접속, 메트릭 읽기, 내 DB 확인 (아무것도 만들지 않음)
#   python c1_milvus.py probe            2만 행 테스트 컬렉션으로 콜드 조회 시 메트릭이 움직이는지 확인 후 삭제
#   python c1_milvus.py load             w13c1_doc, w13c1_rand 컬렉션 만들기 (각 100만 행)
#   python c1_milvus.py m1 [--nq 100]    콜드 조회 (쿼리마다 release/load)
#   python c1_milvus.py m1p --results <results.jsonl> [--nq 100]
#                                        파이프라인 결과의 Chunk 집합으로 콜드 조회 (query(pk in) / get, 한 번에 / 개별)
#   python c1_milvus.py m2 [--nq 500]    웜 조회 (한 번 load, 3회 반복)
#   python c1_milvus.py release          내 컬렉션 메모리에서 내리기
#   python c1_milvus.py drop --yes       내 컬렉션(w13c1_*) 삭제
#
# 지키는 것 (서버 수칙)
#   - database "jihyeon"만 쓴다. 이름이 w13c1_로 시작하는 컬렉션만 만들고, release하고, 지운다.
#   - 다른 database, 다른 컬렉션은 조회·release·삭제하지 않는다.
#   - 입력 파일(준이 chunks.jsonl, 재은이 CSV)은 읽기만 한다. 결과는 /workspace/jihyeon 아래에만 쓴다.
#   - 측정이 끝나면 내 컬렉션을 release한다.
#
# 배치
#   doc   pk = chunks.jsonl 줄 번호 (문서 번호, 문서 안 순서대로). 준이 nq_seq, 재은이 nq1m_sel과 같은 번호.
#   rand  pk = 줄 번호를 무작위로 섞은 값 (seed 1). 기준선.
#   두 컬렉션 모두 pk 순서대로 넣는다. Milvus는 pk 순서로 저장하므로(W12 확인) pk가 곧 저장 위치 순서다.
#   벡터는 8차원 가짜 벡터(FLAT). 검색 결과는 재은이 파일을 쓰고, 여기서는 본문 조회만 잰다.
#   warmup.scalarField=disable: load할 때 본문을 미리 올리지 않아서, 처음 조회할 때 저장소에서 읽는다 (W12와 같음).
import os, sys, csv, json, time, random, socket, argparse, re, urllib.request
import numpy as np
from common import WORK, CHUNKS, log, safe_path, out_dir, open_read, file_info, write_json, now_kst
from run_c1 import build_read_sets, coverage

URI = os.environ.get("C1_MILVUS_URI", "http://localhost:19530")
MET = os.environ.get("C1_MILVUS_METRICS", "http://localhost:9091/metrics")
DB = "jihyeon"
PREFIX = "w13c1_"
LAYOUTS = ("doc", "rand")
DIM = 8
META = os.path.join(WORK, "milvus_meta")


# ---------------------------------------------------------------- 안전장치
def mine(name):
    if not name.startswith(PREFIX):
        sys.exit(f"[중단] 내 컬렉션이 아닙니다: {name}")
    return name


def client():
    from pymilvus import MilvusClient
    root = MilvusClient(uri=URI, timeout=1800)
    if DB not in root.list_databases():
        root.create_database(DB)
        log("created database", DB)
    root.close()
    return MilvusClient(uri=URI, db_name=DB, timeout=1800)


# ---------------------------------------------------------------- 메트릭 (서버 전체 값)
KEYS = ["internal_cache_cell_access_miss_bytes_total", "internal_cache_cell_access_hit_bytes_total",
        "internal_cache_load_latency_microseconds_count", "internal_storage_op_count",
        "internal_storage_kv_size_sum", "milvus_proxy_req_count", "process_cpu_seconds_total"]


def metrics():
    out = {}
    for line in urllib.request.urlopen(MET, timeout=10).read().decode().splitlines():
        if line.startswith("#"):
            continue
        for k in KEYS:
            if line.startswith(k + "{") or line.startswith(k + " "):
                try:
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def delta(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in KEYS}


def quiet_check(sec=10):
    """다른 사람이 Milvus를 쓰는 중인지 본다 (아무것도 바꾸지 않음)."""
    a = metrics(); time.sleep(sec if not os.environ.get("C1_FAST") else 0); b = metrics(); d = delta(a, b)
    r = {"proxy_req_per_s": round(d["milvus_proxy_req_count"] / sec, 2),
         "milvus_cpu_cores": round(d["process_cpu_seconds_total"] / sec, 2),
         "miss_MB_per_s": round(d["internal_cache_cell_access_miss_bytes_total"] / sec / 1e6, 3)}
    r["quiet"] = r["proxy_req_per_s"] < 5 and r["milvus_cpu_cores"] < 1.0
    return r


def wait_quiet(max_min):
    for i in range(max_min + 1):
        q = quiet_check()
        if q["quiet"]:
            return q, i
        log(f"Milvus busy {q}, waiting 1 min")
        time.sleep(50)
    return q, max_min


# ---------------------------------------------------------------- 데이터
def read_chunks():
    texts, doc, cix = [], [], []
    with open_read(CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"]); doc.append(r["doc_idx"]); cix.append(r["chunk_idx"])
    return texts, np.array(doc, np.int64), np.array(cix, np.int32)


def pk_maps(n):
    """chunk_id(줄 번호) -> pk. doc은 그대로, rand는 seed 1 무작위."""
    return {"doc": np.arange(n, dtype=np.int64), "rand": np.random.default_rng(1).permutation(n).astype(np.int64)}


# ---------------------------------------------------------------- 만들기
def create(c, name, rows_iter, n):
    mine(name)
    from pymilvus import DataType
    if c.has_collection(name):
        sys.exit(f"{name}이 이미 있습니다. 다시 만들려면 먼저 'drop --yes'를 실행하세요.")
    s = c.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("pk", DataType.INT64, is_primary=True)
    s.add_field("chunk_id", DataType.INT64)
    s.add_field("doc_idx", DataType.INT64)
    s.add_field("chunk_idx", DataType.INT32)
    s.add_field("text", DataType.VARCHAR, max_length=65535)
    s.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    c.create_collection(name, schema=s, properties={"warmup.scalarField": "disable"})
    t0, done, buf = time.time(), 0, []
    for row in rows_iter:
        buf.append(row)
        if len(buf) >= 5000:
            c.insert(name, buf); done += len(buf); buf = []
            if done % 100000 == 0:
                log(f"{name}: inserted {done}/{n} ({done / (time.time() - t0):.0f} rows/s)")
    if buf:
        c.insert(name, buf); done += len(buf)
    c.flush(name)
    ip = c.prepare_index_params()
    ip.add_index(field_name="vec", index_type="FLAT", metric_type="L2")
    c.create_index(name, ip)
    for _ in range(360):
        d = c.describe_index(name, "vec")
        if d.get("state") == "Finished" or (int(d.get("pending_index_rows", 1) or 0) == 0 and int(d.get("indexed_rows", 0) or 0) >= n):
            break
        time.sleep(5)
    log(f"{name}: {done} rows, index {d.get('state')}, {time.time() - t0:.0f}s")
    return done


def rows_for(layout, texts, doc, cix, pk_of, limit=None):
    n = len(texts) if limit is None else limit
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((n, DIM)).astype(np.float32)
    order = np.argsort(pk_of[:n]) if layout == "rand" else np.arange(n)   # pk 순서대로 넣기
    for i in order:
        yield {"pk": int(pk_of[i]), "chunk_id": int(i), "doc_idx": int(doc[i]), "chunk_idx": int(cix[i]),
               "text": texts[i], "vec": vec[i].tolist()}


def cmd_load(a):
    texts, doc, cix = read_chunks()
    n = len(texts)
    log(f"chunks {n}, docs {int(doc.max()) + 1}")
    pks = pk_maps(n)
    os.makedirs(safe_path(META), exist_ok=True)
    for lay in LAYOUTS:
        np.save(safe_path(os.path.join(META, f"pk_{lay}.npy")), pks[lay])
    c = client()
    for lay in LAYOUTS:
        create(c, PREFIX + lay, rows_for(lay, texts, doc, cix, pks[lay]), n)
    write_json(os.path.join(META, "meta.json"), {"created_at_kst": now_kst(), "db": DB, "collections": [PREFIX + l for l in LAYOUTS],
                                                 "n": n, "dim": DIM, "index": "FLAT L2 (dummy vectors)",
                                                 "properties": {"warmup.scalarField": "disable"}, "input": file_info(CHUNKS)})
    log("load done. 컬렉션은 아직 load(메모리 적재) 하지 않은 상태입니다.")


# ---------------------------------------------------------------- 조회
def reload(c, name):
    c.release_collection(mine(name))
    c.load_collection(name)
    time.sleep(1)


def fetch(c, name, pks, impl="get"):
    """impl: get = get(ids) (w13 M1·M2와 같음) / query = query(filter="pk in [...]") (파이프라인 B0와 같음)"""
    pks = [int(p) for p in pks]
    if impl == "get":
        r = c.get(mine(name), ids=pks, output_fields=["chunk_id", "text"])
    elif impl == "query":
        r = c.query(mine(name), filter=f"pk in [{','.join(map(str, pks))}]", output_fields=["chunk_id", "text"])
    else:
        raise ValueError(impl)
    return {int(x["chunk_id"]): x["text"] for x in r}


def run_fetch(c, name, pk_of, parts, impl="get"):
    """parts: 요청 단위로 나눈 chunk_id 목록들. 반환: (요청 수, 지연 ms, 결과 dict)"""
    got = {}
    t = time.perf_counter()
    for ids in parts:
        if ids:
            got.update(fetch(c, name, pk_of[np.array(ids, np.int64)], impl))
    return sum(1 for p in parts if p), (time.perf_counter() - t) * 1e3, got


def parts_for(mode, top, ids):
    if mode == "one":
        return [ids]
    if mode == "split":                       # top-10 먼저, 이웃은 두 번째 요청
        return [top, [x for x in ids if x not in set(top)]]
    if mode == "each":                        # chunk마다 따로 요청
        return [[x] for x in ids]
    raise ValueError(mode)


def m1_conditions():
    c = [(lay, s, "one") for lay in LAYOUTS for s in ("top10", "bc1", "all1", "ann30", "ann100")]
    c += [(lay, s, "split") for lay in LAYOUTS for s in ("bc1", "all1")]
    return c


def m2_conditions():
    return [(lay, s, m) for lay in LAYOUTS for s in ("top10", "bc1", "all1", "ann30", "ann100") for m in ("one", "each")]


def setup_run(a, tag, n_q):
    stamp = time.strftime("%Y%m%d_%H%M")
    R = out_dir("milvus", f"{tag}_{stamp}" + ("_quick" if a.quick else ""))
    texts, doc, _ = read_chunks()
    qs, sets = build_read_sets(doc, n_q)
    write_json(os.path.join(R, "read_sets.json"), coverage(qs, sets))
    pks = {lay: np.load(os.path.join(META, f"pk_{lay}.npy")) for lay in LAYOUTS}
    return R, texts, qs, sets, pks


def check_texts(got, ids, texts):
    return all(got.get(i) == texts[i] for i in ids)


def cmd_m1(a):
    nq = 5 if a.quick else a.nq
    R, texts, qs, sets, pks = setup_run(a, "m1_cold", nq)
    c = client()
    conds = m1_conditions()
    q0, waited = wait_quiet(a.wait_quiet_min)
    log("quiet check:", q0)
    write_json(os.path.join(R, "env.json"), {"started_at_kst": now_kst(), "host": socket.gethostname(), "db": DB,
               "kind": "M1 cold: release/load before every fetch", "n_queries": len(qs), "conditions": conds,
               "quiet_check_before": q0, "waited_min": waited, "meta": json.load(open_read(os.path.join(META, "meta.json")))})
    f = open(safe_path(os.path.join(R, "per_query.csv")), "w", newline="")
    w = csv.writer(f)
    w.writerow(["q_idx", "layout", "read_set", "mode", "n_chunks", "n_requests", "lat_ms", "miss_bytes", "hit_bytes",
                "cell_loads", "storage_ops", "storage_bytes", "proxy_req", "ok"])
    t0 = time.time(); total = len(qs) * len(conds); k = 0
    try:
        for qi, q in enumerate(qs):
            order = conds[:]; random.Random(qi).shuffle(order)      # 쿼리마다 조건 순서를 섞음
            for lay, s, mode in order:
                k += 1
                name = PREFIX + lay
                ids = sets[s][qi]; top = sets["top10"][qi]
                reload(c, name)
                m0 = metrics()
                nreq, lat, got = run_fetch(c, name, pks[lay], parts_for(mode, top, ids))
                time.sleep(0.3)
                d = delta(m0, metrics())
                w.writerow([q, lay, s, mode, len(ids), nreq, round(lat, 2), int(d["internal_cache_cell_access_miss_bytes_total"]),
                            int(d["internal_cache_cell_access_hit_bytes_total"]), int(d["internal_cache_load_latency_microseconds_count"]),
                            int(d["internal_storage_op_count"]), int(d["internal_storage_kv_size_sum"]),
                            int(d["milvus_proxy_req_count"]), int(check_texts(got, ids, texts))])
            f.flush()
            el = time.time() - t0
            log(f"query {qi + 1}/{len(qs)} done ({k}/{total} fetches, {el / 60:.1f} min, left ~{el / k * (total - k) / 60:.0f} min)")
    finally:
        f.close()
        for lay in LAYOUTS:
            c.release_collection(mine(PREFIX + lay))
        log("released my collections")
    env = json.load(open_read(os.path.join(R, "env.json"))); env["finished_at_kst"] = now_kst()
    env["quiet_check_after"] = quiet_check(); write_json(os.path.join(R, "env.json"), env)
    log("M1 DONE ->", R); print(R)


# ---------------------------------------------------------------- M1P: 파이프라인 결과의 Chunk 집합으로 cold 비교
def m1p_conditions():
    """(layout, 읽는 범위, 요청 방식, 조회 방식). 배치마다 6개, 모두 12개."""
    c = []
    for lay in LAYOUTS:
        c += [(lay, s, "one", "query") for s in ("top10", "bc1", "all1")]   # 파이프라인 기본: 한 번에, query(pk in)
        c += [(lay, "all1", "one", "get")]                                  # w13 M1(get)과 이어 보기
        c += [(lay, s, "each", "query") for s in ("top10", "all1")]         # 개별 읽기 (chunk마다 요청)
    return c


def sets_from_results(path, doc, nq):
    """파이프라인 results.jsonl에서 질문별 top-10(context_ids, 순위 순)을 읽고 이웃 정책을 적용한다.
    이웃 규칙은 임준 pf3_rules.py 사본(nbr_rules) = 파이프라인 C2와 같은 규칙. 오류·예열 결과는 뺀다."""
    import nbr_rules
    qs, sets, skipped = [], {"top10": [], "bc1": [], "all1": []}, 0
    with open_read(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or r.get("is_warmup") or len(r.get("context_ids") or []) == 0:
                skipped += 1
                continue
            R = [int(x) for x in r["context_ids"]]
            qs.append((r["query_set"], int(r["q_idx"])))
            sets["top10"].append(R)
            sets["bc1"].append(R + nbr_rules.select("BC1", R, doc))
            sets["all1"].append(R + nbr_rules.select("all", R, doc))
            if nq and len(qs) >= nq:
                break
    return qs, sets, skipped


def cmd_m1p(a):
    if not a.results or not os.path.exists(a.results):
        sys.exit("--results <파이프라인 results.jsonl 경로>가 필요합니다")
    nq = 3 if a.quick else a.nq
    stamp = time.strftime("%Y%m%d_%H%M")
    R = out_dir("milvus", f"m1p_cold_{stamp}" + ("_quick" if a.quick else ""))
    texts, doc, _ = read_chunks()
    qs, sets, skipped = sets_from_results(a.results, doc, nq)
    if not qs:
        sys.exit("results.jsonl에서 쓸 수 있는 질문이 없습니다")
    write_json(os.path.join(R, "read_sets.json"), {k: {"mean_chunks_per_query": round(float(np.mean([len(x) for x in v])), 2)}
                                                  for k, v in sets.items()})
    pks = {lay: np.load(os.path.join(META, f"pk_{lay}.npy")) for lay in LAYOUTS}
    c = client()
    conds = m1p_conditions()
    q0, waited = wait_quiet(a.wait_quiet_min)
    log("quiet check:", q0)
    write_json(os.path.join(R, "env.json"), {
        "started_at_kst": now_kst(), "host": socket.gethostname(), "db": DB,
        "kind": "M1P cold: pipeline results.jsonl chunk sets, release/load before every fetch",
        "n_queries": len(qs), "skipped_results": skipped, "conditions": conds,
        "results_input": file_info(a.results), "neighbor_rule": "nbr_rules.py (= junlim pf3_rules.py, md5 2754643f)",
        "quiet_check_before": q0, "waited_min": waited, "meta": json.load(open_read(os.path.join(META, "meta.json")))})
    f = open(safe_path(os.path.join(R, "per_query.csv")), "w", newline="")
    w = csv.writer(f)
    w.writerow(["q_idx", "layout", "read_set", "mode", "impl", "n_chunks", "n_requests", "lat_ms", "miss_bytes", "hit_bytes",
                "cell_loads", "storage_ops", "storage_bytes", "proxy_req", "ok"])
    t0 = time.time(); total = len(qs) * len(conds); k = 0
    try:
        for qi, (qset, q) in enumerate(qs):
            order = conds[:]; random.Random(qi).shuffle(order)
            for lay, s, mode, impl in order:
                k += 1
                name = PREFIX + lay
                ids = sets[s][qi]; top = sets["top10"][qi]
                reload(c, name)
                m0 = metrics()
                nreq, lat, got = run_fetch(c, name, pks[lay], parts_for(mode, top, ids), impl)
                time.sleep(0.3)
                d = delta(m0, metrics())
                w.writerow([q, lay, s, mode, impl, len(ids), nreq, round(lat, 2), int(d["internal_cache_cell_access_miss_bytes_total"]),
                            int(d["internal_cache_cell_access_hit_bytes_total"]), int(d["internal_cache_load_latency_microseconds_count"]),
                            int(d["internal_storage_op_count"]), int(d["internal_storage_kv_size_sum"]),
                            int(d["milvus_proxy_req_count"]), int(check_texts(got, ids, texts))])
            f.flush()
            el = time.time() - t0
            log(f"query {qi + 1}/{len(qs)} done ({k}/{total} fetches, {el / 60:.1f} min, left ~{el / k * (total - k) / 60:.0f} min)")
    finally:
        f.close()
        for lay in LAYOUTS:
            c.release_collection(mine(PREFIX + lay))
        log("released my collections")
    env = json.load(open_read(os.path.join(R, "env.json"))); env["finished_at_kst"] = now_kst()
    env["quiet_check_after"] = quiet_check(); write_json(os.path.join(R, "env.json"), env)
    log("M1P DONE ->", R); print(R)


def cmd_m2(a):
    nq = 50 if a.quick else a.nq
    reps = 1 if a.quick else a.reps
    R, texts, qs, sets, pks = setup_run(a, "m2_warm", nq)
    c = client()
    conds = m2_conditions()
    q0, waited = wait_quiet(a.wait_quiet_min)
    log("quiet check:", q0)
    for lay in LAYOUTS:
        c.load_collection(mine(PREFIX + lay))
    log("warm-up: 모든 조건을 한 번씩 조회 (기록 안 함)")
    for lay, s, mode in conds:
        for qi in range(len(qs)):
            run_fetch(c, PREFIX + lay, pks[lay], parts_for("one", sets["top10"][qi], sets[s][qi]))
    write_json(os.path.join(R, "env.json"), {"started_at_kst": now_kst(), "host": socket.gethostname(), "db": DB,
               "kind": "M2 warm: load once, warm-up pass, then measure", "n_queries": len(qs), "reps": reps,
               "conditions": conds, "quiet_check_before": q0, "waited_min": waited})
    fq = open(safe_path(os.path.join(R, "per_query.csv")), "w", newline=""); wq = csv.writer(fq)
    wq.writerow(["rep", "layout", "read_set", "mode", "q_idx", "n_chunks", "n_requests", "lat_ms", "ok"])
    fr = open(safe_path(os.path.join(R, "per_run.csv")), "w", newline=""); wr = csv.writer(fr)
    wr.writerow(["rep", "layout", "read_set", "mode", "n_queries", "wall_s", "miss_bytes", "cell_loads", "proxy_req"])
    try:
        for rep in range(reps):
            order = conds[:]; random.Random(rep).shuffle(order)
            for lay, s, mode in order:
                name = PREFIX + lay
                m0 = metrics(); t0 = time.perf_counter(); nreq_all = 0
                for qi, q in enumerate(qs):
                    ids = sets[s][qi]
                    nreq, lat, got = run_fetch(c, name, pks[lay], parts_for(mode, sets["top10"][qi], ids))
                    nreq_all += nreq
                    wq.writerow([rep, lay, s, mode, q, len(ids), nreq, round(lat, 3), int(check_texts(got, ids, texts))])
                wall = time.perf_counter() - t0; d = delta(m0, metrics())
                wr.writerow([rep, lay, s, mode, len(qs), round(wall, 2), int(d["internal_cache_cell_access_miss_bytes_total"]),
                             int(d["internal_cache_load_latency_microseconds_count"]), int(d["milvus_proxy_req_count"])])
                fq.flush(); fr.flush()
                log(f"rep {rep} {lay} {s} {mode}: {wall:.1f}s")
    finally:
        fq.close(); fr.close()
        for lay in LAYOUTS:
            c.release_collection(mine(PREFIX + lay))
        log("released my collections")
    env = json.load(open_read(os.path.join(R, "env.json"))); env["finished_at_kst"] = now_kst()
    write_json(os.path.join(R, "env.json"), env)
    log("M2 DONE ->", R); print(R)


# ---------------------------------------------------------------- 점검
def cmd_check(a):
    import pymilvus
    log("pymilvus", pymilvus.__version__)
    m = metrics()
    log("metrics readable:", {k: m.get(k) is not None for k in KEYS})
    c = client()
    log("my database:", DB, "collections:", c.list_collections())
    log("quiet check:", quiet_check())


def cmd_probe(a):
    texts, doc, cix = read_chunks()
    n = 20000
    pk = np.arange(len(texts), dtype=np.int64)
    c = client()
    name = PREFIX + "probe"
    if c.has_collection(name):
        c.drop_collection(mine(name))
    create(c, name, rows_for("doc", texts, doc, cix, pk, limit=n), n)
    ok = True
    try:
        res = {}
        for label, ids in (("cold 10 adjacent", list(range(100, 110))), ("cold 10 scattered", list(range(100, 20000, 1990)))):
            reload(c, name)
            m0 = metrics(); _, lat, got = run_fetch(c, name, pk, [ids]); time.sleep(0.5); d = delta(m0, metrics())
            res[label] = {"lat_ms": round(lat, 1), "miss_bytes": int(d["internal_cache_cell_access_miss_bytes_total"]),
                          "cell_loads": int(d["internal_cache_load_latency_microseconds_count"]),
                          "storage_ops": int(d["internal_storage_op_count"]), "text_ok": check_texts(got, ids, texts)}
        m0 = metrics(); _, lat, got = run_fetch(c, name, pk, [list(range(100, 110))]); time.sleep(0.5); d = delta(m0, metrics())
        res["warm same 10"] = {"lat_ms": round(lat, 1), "miss_bytes": int(d["internal_cache_cell_access_miss_bytes_total"]),
                               "cell_loads": int(d["internal_cache_load_latency_microseconds_count"])}
        for k, v in res.items():
            log(f"{k}: {v}")
        ok = (res["cold 10 adjacent"]["miss_bytes"] > 0 and res["cold 10 adjacent"]["text_ok"] and res["cold 10 scattered"]["text_ok"]
              and res["warm same 10"]["miss_bytes"] < res["cold 10 adjacent"]["miss_bytes"])
        write_json(os.path.join(out_dir("milvus"), "probe.json"), {"checked_at_kst": now_kst(), "result": res, "pass": ok})
    finally:
        c.release_collection(mine(name)); c.drop_collection(mine(name))
        log("probe collection dropped")
    log("PROBE", "PASS" if ok else "FAIL (콜드 조회에서 메트릭이 움직이지 않음. 결과를 보내주세요)")
    sys.exit(0 if ok else 1)


def cmd_release(a):
    c = client()
    for n in c.list_collections():
        if n.startswith(PREFIX):
            c.release_collection(mine(n)); log("released", DB, n)


def cmd_drop(a):
    if not a.yes:
        sys.exit("삭제하려면 --yes를 붙이세요. 지워지는 것: database jihyeon의 w13c1_로 시작하는 컬렉션만")
    c = client()
    for n in c.list_collections():
        if n.startswith(PREFIX):
            c.drop_collection(mine(n)); log("dropped", DB, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "probe", "load", "m1", "m1p", "m2", "release", "drop"])
    ap.add_argument("--results", default=None, help="m1p: 파이프라인 results.jsonl 경로 (읽기만)")
    ap.add_argument("--nq", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--wait-quiet-min", type=int, default=30)
    a = ap.parse_args()
    if a.nq is None:
        a.nq = {"m1": 100, "m1p": 100, "m2": 500}.get(a.cmd, 100)
    {"check": cmd_check, "probe": cmd_probe, "load": cmd_load, "m1": cmd_m1, "m1p": cmd_m1p, "m2": cmd_m2,
     "release": cmd_release, "drop": cmd_drop}[a.cmd](a)


if __name__ == "__main__":
    main()
