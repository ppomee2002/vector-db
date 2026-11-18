#!/usr/bin/env python3
# load_big.py: 준비된 조각(chunk_vecs.npy, chunks.jsonl)을 Milvus 데이터베이스 junlim에 적재한다.
# 사용: python load_big.py seq|rand [--data DIR] [--coll 이름접두] [--limit N] [--drop] [--cleanup]
# 컬렉션 이름은 <coll>_<layout> 이다. 데이터베이스는 junlim 고정이며 default는 건드리지 않는다.
import os, sys, json, time, argparse
import numpy as np
from pymilvus import MilvusClient, DataType

DB = "junlim"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wait_index(c, name, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = c.describe_index(name, "vec")
            st, pend, tot, idx = d.get("state"), d.get("pending_index_rows"), d.get("total_rows"), d.get("indexed_rows")
            log(f"index state={st} indexed={idx}/{tot} pending={pend}")
            if st == "Finished" or (pend is not None and int(pend) == 0 and tot is not None and int(idx) == int(tot)):
                return True
        except Exception as e:
            log("describe_index:", repr(e))
        time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout", choices=["seq", "rand"])
    ap.add_argument("--data", default="/workspace/junlim/data_nq")
    ap.add_argument("--coll", default="nq")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--index-timeout", dest="index_timeout", type=int, default=3600)
    a = ap.parse_args()
    name = f"{a.coll}_{a.layout}"

    vecs = np.load(f"{a.data}/chunk_vecs.npy", mmap_mode="r")
    n_all, dim = vecs.shape
    n = n_all if a.limit is None else min(a.limit, n_all)
    if a.layout == "seq":
        pk_of_chunk = np.arange(n_all, dtype=np.int64)
    else:
        pk_of_chunk = np.random.default_rng(1).permutation(n_all).astype(np.int64)
    pkfile = f"{a.data}/pk_{a.layout}.npy"
    if a.limit is None:
        if os.path.exists(pkfile) and not np.array_equal(np.load(pkfile), pk_of_chunk):
            sys.exit(f"{pkfile} already exists with different content. Stop.")
        np.save(pkfile, pk_of_chunk)
    else:
        log("limit set: pk file is NOT written")

    c = MilvusClient(uri="http://localhost:19530", db_name=DB)
    log("server", c.get_server_version(), "db", DB, "collection", name, "rows", n, "dim", dim)
    if c.has_collection(name):
        if not a.drop:
            sys.exit(f"collection {name} exists. Use --drop to replace it.")
        c.drop_collection(name)
        log("dropped old", name)

    sch = c.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("pk", DataType.INT64, is_primary=True)
    sch.add_field("doc_idx", DataType.INT64)
    sch.add_field("chunk_idx", DataType.INT32)
    sch.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
    sch.add_field("text", DataType.VARCHAR, max_length=4096)
    c.create_collection(name, schema=sch)

    t0, buf, done = time.time(), [], 0
    with open(f"{a.data}/chunks.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            if i == 0:
                log("first record keys:", sorted(r.keys()), "text preview:", repr(str(r.get("text", ""))[:60]))
            txt = str(r.get("text", ""))
            if i < 100 and not txt:
                sys.exit("empty text in first 100 records: check chunks.jsonl keys")
            b = txt.encode("utf-8")
            if len(b) > 4000:
                txt = b[:4000].decode("utf-8", errors="ignore")
            buf.append({"pk": int(pk_of_chunk[i]), "doc_idx": int(r["doc_idx"]), "chunk_idx": int(r.get("chunk_idx", 0)),
                        "vec": vecs[i].tolist(), "text": txt})
            if len(buf) >= a.batch:
                c.insert(name, buf)
                done += len(buf)
                buf = []
                if (done // a.batch) % 10 == 0:
                    log(f"inserted {done}/{n} ({done / (time.time() - t0):.0f} rows/s)")
    if buf:
        c.insert(name, buf)
        done += len(buf)
    log(f"inserted {done} rows in {time.time() - t0:.0f}s")
    c.flush(name)
    log("flushed")

    ip = c.prepare_index_params()
    ip.add_index(field_name="vec", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
    t1 = time.time()
    c.create_index(name, ip)
    ok = wait_index(c, name, a.index_timeout)
    log(f"index {'finished' if ok else 'NOT confirmed'} after {time.time() - t1:.0f}s")

    try:
        c.release_collection(name)
    except Exception as e:
        log("release:", repr(e))
    c.alter_collection_field(name, field_name="text", field_params={"mmap.enabled": True})
    c.load_collection(name)
    log("load state:", c.get_load_state(name))
    log("stats:", c.get_collection_stats(name))
    for fld in c.describe_collection(name).get("fields", []):
        if fld.get("name") == "text":
            log("text field:", fld)
    if a.cleanup:
        c.drop_collection(name)
        log("cleanup: dropped", name)
    log("DONE")


if __name__ == "__main__":
    main()
