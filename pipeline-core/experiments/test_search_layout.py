"""PR #3(지현) 회귀 테스트: c1.layout이 seq/doc/rand일 때 main.py 실행 경로(검색 → C2 → C1)가 KeyError 없이 끝나는지.

    /workspace/jaeeun/venv/bin/python experiments/test_search_layout.py [--old-main <수정 전 main.py>]

버그: main.py가 ChunkMeta를 layouts=(c1.layout,)로만 로드했는데 검색 어댑터(modules/search_milvus.py)는 항상
chunk_of_pk["seq"]를 써서, c1.layout=doc/rand면 검색 단계에서 KeyError('seq'). 계약 테스트 공통 검사는 가짜 검색이라 못 잡음.

이 테스트는 Milvus를 쓰지 않는다. 바꾸는 것은 서버와 닿는 부분뿐이다.
- pymilvus.AsyncMilvusClient → 가짜 클라이언트 (search: 검색 컬렉션 pk 반환, query: 요청한 pk의 가짜 본문)
- MilvusGuard / scrape / summarize_metrics / server_activity → 가짜 (load·release·메트릭 없음)
- 잠금 파일·결과 폴더 → 임시 폴더, md5 검사 → off (로더 자체는 실제 FileChunkMetaLoader)
그대로 쓰는 것: main.main()(설정 합치기 → 실제 ChunkMeta 로드 → 어댑터 생성 → execute → 결과 검사), 실제
MilvusSearchAdapter(pk → chunk_of_pk["seq"]), 실제 C1 milvus_query_ref(chunk_id → pk_of_chunk[c1.layout]), C2 none_ref, C3 serial_ref.

확인: 종료 코드 0, run_meta에 exception 없음, 결과 max_queries개·오류 0, 검색 pk == 결과 chunk_id(항등),
C1이 Fetch 컬렉션에 보낸 pk == pk_of_chunk[layout][fetched_ids] (rand면 순열이 실제로 적용됐는지).
--old-main을 주면 수정 전 main.py로도 돌려 doc/rand에서 KeyError('seq')로 실패하는지 확인한다(seq는 통과해야 함).
종료 코드: 0 = 기대대로, 1 = 기대와 다름.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.chunk_meta import FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config  # noqa: E402

N_Q = 3
FETCH = {"seq": ("junlim", "nq_seq"), "doc": ("jihyeon", "w13c1_doc"), "rand": ("jihyeon", "w13c1_rand")}
QUERIED: dict[str, list[int]] = {}     # "<db>/<collection>" → 가짜 클라이언트가 받은 query pk
SEARCHED: list[list[int]] = []         # 가짜 검색이 돌려준 pk (질문별)
FAILS: list[str] = []


def check(name: str, cond: bool, msg: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f": {msg}"))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 가짜 Milvus
# ---------------------------------------------------------------------------
class FakeAsyncMilvusClient:
    N = 1_000_000

    def __init__(self, uri=None, db_name=None, **kw):
        self.db = db_name

    async def search(self, collection_name, data, anns_field, limit, output_fields, search_params):
        seed = int(abs(sum(data[0])) * 1e6) % self.N
        pks = [(seed + 7919 * i) % self.N for i in range(limit)]      # 검색 컬렉션 pk (= chunk_id)
        SEARCHED.append(pks)
        return [[{"id": p, "distance": 1.0 - 0.01 * i} for i, p in enumerate(pks)]]

    async def describe_collection(self, name):
        return {"fields": [{"name": f} for f in ("pk", "doc_idx", "vec", "text")]}

    async def query(self, name, filter, output_fields):
        pks = [int(x) for x in filter[filter.index("[") + 1:-1].split(",")]
        QUERIED.setdefault(f"{self.db}/{name}", []).extend(pks)
        return [{"pk": p, "text": f"fake text pk {p}"} for p in pks]

    async def close(self):
        pass


class FakeGuard:
    def __init__(self, uri, metrics_url, allowed, allow_preloaded=False):
        self.allowed = list(allowed)
        self.allow_preloaded = allow_preloaded
        self.log: list[dict] = []
        self.targets: list[str] = []

    def preflight(self, targets, sample_s=0.0):
        self.targets = list(targets)
        return True, {"target_load_state": {t: "NotLoad" for t in targets}, "loaded_collection_num": 0,
                      "swap": None, "busy_reasons": []}

    def load_needed(self):
        self.log += [{"op": "skip_load", "collection": t, "reason": "가짜 Guard"} for t in self.targets]

    def load_state(self, c):
        return "fake"

    def release_safe(self, sample_s=0.0):
        return []


def fake_metrics(_=None):
    return {"collection_num": 0, "cpu_s": 0.0, "rss_bytes": 0, "req_total": {}, "cache_miss_bytes": 0.0,
            "cache_miss_by_loc": {}, "start_time": 0.0}


def install_fake_pymilvus() -> None:
    try:
        import pymilvus
        pymilvus.AsyncMilvusClient = FakeAsyncMilvusClient
    except ImportError:
        mod = types.ModuleType("pymilvus")
        mod.AsyncMilvusClient = FakeAsyncMilvusClient
        sys.modules["pymilvus"] = mod


def load_main(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def patch(m, tmp: Path) -> None:
    class NoMd5Loader(FileChunkMetaLoader):     # 실제 로더, md5 검사만 끔
        def __init__(self, cfg, **kw):
            kw.pop("md5_mode", None)
            super().__init__(cfg, check_md5=False, **kw)

    m.MilvusGuard = FakeGuard
    m.scrape = lambda *a, **k: None
    m.summarize_metrics = fake_metrics
    m.server_activity = lambda *a, **k: {"swap": None, "loaded_collection_num": 0}
    m.LOCK_FILE = tmp / "locks" / "test.lock"
    m.FileChunkMetaLoader = NoMd5Loader


def run_layout(m, tag: str, layout: str, tmp: Path) -> tuple[int, dict, list[dict]]:
    db, coll = FETCH[layout]
    out_dir = tmp / f"{tag}_{layout}"
    argv = ["--set", "c1.impl=milvus_query_ref", "--set", f"c1.layout={layout}",
            "--set", f"milvus.fetch.database={db}", "--set", f"milvus.fetch.collection={coll}",
            "--set", f"milvus.fetch.layout={layout}",
            "--set", "c2.impl=none_ref", "--set", "c2.policy=none", "--set", "c3.impl=serial_ref",
            "--set", "run.purpose=correctness", "--set", f"run.max_queries={N_Q}", "--set", f"run.n_queries={N_Q}",
            "--set", "run.evaluate=false", "--set", "run.preflight_sample_s=0", "--set", "run.release_check_s=0",
            "--set", f"run.milvus_allowed_load=[junlim/nq_seq, {db}/{coll}]",
            "--set", f"run.name=pr3_{tag}_{layout}", "--set", f"run.out_dir={out_dir}"]
    QUERIED.clear()
    SEARCHED.clear()
    rc = m.main(argv)
    runs = sorted(out_dir.glob("*/run_meta.json"))
    meta = json.loads(runs[-1].read_text()) if runs else {}
    rpath = runs[-1].parent / "results.jsonl" if runs else None
    results = [json.loads(ln) for ln in rpath.read_text().splitlines()] if rpath and rpath.exists() else []
    return rc, meta, results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-main", help="수정 전 main.py 경로 (doc/rand에서 실패하는지 확인)")
    args = ap.parse_args(argv)
    install_fake_pymilvus()

    cfg, _ = load_config(None, [])
    pk_of = {"seq": np.arange(1_000_000), "doc": np.arange(1_000_000),
             "rand": np.load(cfg["milvus"]["c1_compare"]["rand"]["pk_map"])}   # chunk_meta.py와 같은 정의

    with tempfile.TemporaryDirectory(prefix="pr3_layout_") as td:
        tmp = Path(td)
        cur = load_main(ROOT / "main.py", "main_current")
        patch(cur, tmp)
        for layout in ("seq", "doc", "rand"):
            rc, meta, results = run_layout(cur, "cur", layout, tmp)
            exc = meta.get("exception")
            check(f"[현재 main.py] c1.layout={layout}: 종료 코드 0, 예외 없음", rc == 0 and not exc,
                  f"rc={rc} exception={exc} aborted={meta.get('aborted')}")
            check(f"[현재 main.py] c1.layout={layout}: ChunkMeta layouts에 seq와 {layout}",
                  set(meta.get("md5_check", {}).get("layouts", [])) == {"seq", layout},
                  str(meta.get("md5_check", {}).get("layouts")))
            errs = [r["error"] for r in results if r.get("error")]
            check(f"[현재 main.py] c1.layout={layout}: 결과 {N_Q}개, 오류 0",
                  len(results) == N_Q and not errs, f"n={len(results)} errors={errs[:2]}")
            if results:
                hit_ids = [[h["chunk_id"] for h in r["hits"]] for r in results]
                check(f"[현재 main.py] c1.layout={layout}: 검색 pk → chunk_id 항등(seq 매핑)",
                      sorted(map(tuple, hit_ids)) == sorted(map(tuple, SEARCHED[-N_Q:])))
                want = sorted({int(pk_of[layout][c]) for r in results for c in r["fetched_ids"]})
                got = sorted(set(QUERIED.get("/".join(FETCH[layout]), [])))
                check(f"[현재 main.py] c1.layout={layout}: C1이 보낸 pk == pk_of_chunk[{layout}][fetched_ids]",
                      want == got and len(got) > 0, f"want {want[:3]}… got {got[:3]}…")

        if args.old_main:
            old = load_main(Path(args.old_main), "main_old")
            patch(old, tmp)
            for layout in ("seq", "doc", "rand"):
                rc, meta, results = run_layout(old, "old", layout, tmp)
                exc = meta.get("exception") or ""
                errs = [r.get("error") or "" for r in results]
                if layout == "seq":
                    check("[수정 전 main.py] c1.layout=seq: 통과(버그는 doc/rand에서만)",
                          rc == 0 and not exc and not any(errs), f"rc={rc} exception={exc} errors={errs[:1]}")
                else:
                    # pipeline.run_one은 예외를 잡아 QueryResult.error에 남긴다 → 질문마다 search 단계 KeyError('seq'),
                    # 결과 검사 위반으로 종료 코드 5
                    bug = lambda e: e.startswith("search: KeyError") and "'seq'" in e  # noqa: E731
                    check(f"[수정 전 main.py] c1.layout={layout}: 검색 단계 KeyError('seq')로 실패 (버그 재현)",
                          rc != 0 and len(errs) == N_Q and all(bug(e) for e in errs),
                          f"rc={rc} exception={exc!r} errors={errs[:1]}")
                    print(f"     → rc={rc}, 질문 {len(errs)}개 모두 error: {errs[0] if errs else exc}")

    print(f"\n# 결과: FAIL {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
