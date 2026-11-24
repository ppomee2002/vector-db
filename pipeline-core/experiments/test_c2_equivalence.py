"""C2 어댑터(bc1_v1, modules/c2_bc1.py)와 원본 pf3_rules.select의 출력이 완전히 같은지 검사한다. Milvus는 쓰지 않는다.

사용법:
    /workspace/jaeeun/venv/bin/python experiments/test_c2_equivalence.py [--impl bc1_v1]

비교 기준(읽기만 함, 이 폴더 밖에 아무것도 쓰지 않음):
- 원본 규칙: /workspace/jaeeun/team_ref 의 `git show origin/w12-c2-prefetching:코드/pf3_rules.py`를 메모리에서 import.
  git을 못 쓰면 /workspace/jaeeun/sel_fetch_c2x/scripts/pf3_rules.py(원본과 같은 파일).
- 저장된 C2 검색 결과: 같은 브랜치 `코드/pf3_results_1116_1051.tgz` 안의 `pf3_out/hits_nq.jsonl`(메모리에서 풀기).
  /workspace/junlim/pf3_out/hits_nq.jsonl(c2x에서 C2 수치를 재현할 때 쓴 파일)이 있으면 둘이 같은지도 확인.
  hits의 ids는 PK → 원본 analyze_pf3.py와 같이 pk_seq.npy로 chunk_id로 바꿈(nq_seq는 항등, 확인만 함).
- doc_idx·flags: 어댑터는 실행기와 같은 FileChunkMetaLoader로 만든 ChunkMeta, 원본 select에는 원본 분석과 같이
  doc_idx.npy와 nq_flags.npy를 직접 넘김.
- hits를 찾을 수 없으면 무엇이 없는지 출력하고 작은 가짜 데이터로 대신 검사.

검사:
1. 원본 코드 블록(RULES ~ select)이 modules/c2_bc1.py 안에 글자 그대로 있는지
2. 질문 3,452개 × none/BC1/all: 어댑터 출력 == pf3_rules.select 출력 (값·순서·타입 list[int])
3. 경계 경우: 문서 첫/마지막 chunk, chunk 하나뿐인 문서, 마지막 문서(doc 27132)의 끝, 기대값과 원본 둘 다 비교
4. 가짜 데이터(원본 _selftest 배열 + 무작위 R)
종료 코드: 0 = 모두 일치, 1 = 불일치 있음.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interfaces import ChunkMeta  # noqa: E402
from modules import registry  # noqa: E402
from modules.chunk_meta import FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config  # noqa: E402

TEAM_REF = "/workspace/jaeeun/team_ref"
BRANCH = "origin/w12-c2-prefetching"
RULES_PATH = "코드/pf3_rules.py"
TGZ_PATH = "코드/pf3_results_1116_1051.tgz"
TGZ_MEMBER = "pf3_out/hits_nq.jsonl"
RULES_FALLBACK = "/workspace/jaeeun/sel_fetch_c2x/scripts/pf3_rules.py"
HITS_C2X = "/workspace/junlim/pf3_out/hits_nq.jsonl"
POLICIES = ("none", "BC1", "all")
N_QUERIES = 3452
LAST_DOC = 27132


class Rep:
    def __init__(self):
        self.n_fail = 0
        self.notes: list[str] = []

    def ok(self, cond: bool, name: str, detail: str = ""):
        tag = "PASS" if cond else "FAIL"
        if not cond:
            self.n_fail += 1
        print(f"{tag} {name}" + (f"  ({detail})" if detail else ""))

    def info(self, msg: str):
        print(f"INFO {msg}")

    def missing(self, msg: str):
        print(f"MISSING {msg}")
        self.notes.append(msg)


def git_show(path: str) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", TEAM_REF, "show", f"{BRANCH}:{path}"],
                              check=True, capture_output=True).stdout
    except Exception:  # noqa: BLE001
        return None


def load_reference(rep: Rep) -> tuple[types.ModuleType, str]:
    src = git_show(RULES_PATH)
    origin = f"{TEAM_REF} {BRANCH}:{RULES_PATH}"
    if src is None:
        rep.missing(f"git show 실패: {origin} → {RULES_FALLBACK} 사용")
        src = Path(RULES_FALLBACK).read_bytes()
        origin = RULES_FALLBACK
    mod = types.ModuleType("pf3_rules_ref")
    exec(compile(src.decode("utf-8"), f"<{origin}>", "exec"), mod.__dict__)  # noqa: S102 - 팀 원본 코드
    return mod, src.decode("utf-8")


def code_block(text: str) -> str:
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("RULES = "))
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "return sorted(out - Rs)")
    return "\n".join(lines[i:j + 1])


def load_hits(rep: Rep) -> list[tuple[int, list[int]]] | None:
    raw = None
    tgz = git_show(TGZ_PATH)
    if tgz is None:
        rep.missing(f"{BRANCH}:{TGZ_PATH} 를 읽을 수 없음")
    else:
        with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tf:
            try:
                raw = tf.extractfile(TGZ_MEMBER).read()
                rep.info(f"hits = {BRANCH}:{TGZ_PATH} 안의 {TGZ_MEMBER} (md5 {hashlib.md5(raw).hexdigest()})")
            except KeyError:
                rep.missing(f"tgz 안에 {TGZ_MEMBER} 없음")
    p2 = Path(HITS_C2X)
    if p2.exists():
        raw2 = p2.read_bytes()
        if raw is None:
            raw = raw2
            rep.info(f"hits = {HITS_C2X} (md5 {hashlib.md5(raw).hexdigest()})")
        else:
            rep.ok(raw == raw2, "tgz hits == c2x 재현에 쓴 hits", HITS_C2X)
    else:
        rep.missing(f"{HITS_C2X} 없음")
    if raw is None:
        return None
    out = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append((int(r["qi"]), [int(p) for p in r["ids"]]))
    return out


def same(a, b) -> bool:
    return type(a) is list and a == b and all(type(x) is int for x in a)


def compare_cases(rep: Rep, name: str, adapter, ref, cases, doc, flags) -> None:
    for pol in POLICIES:
        bad = []
        for R in cases:
            a = adapter.select_neighbors(list(R), policy=pol)
            b = ref.select(pol, list(R), doc, flags)
            if not same(a, b):
                bad.append((R[:6], a[:8], b[:8]))
        rep.ok(not bad, f"{name}/{pol} 어댑터 == pf3_rules.select ({len(cases)}개)",
               f"불일치 {len(bad)}개, 첫 예 R={bad[0][0]} 어댑터={bad[0][1]} 원본={bad[0][2]}" if bad else "")


def fake_meta(doc: np.ndarray, flags: np.ndarray) -> ChunkMeta:
    n = len(doc)
    ident = np.arange(n, dtype=np.int64)
    return ChunkMeta(n_chunks=n, doc_idx=doc, chunk_idx=np.zeros(n, np.int32), flags=flags,
                     nbytes=np.zeros(n, np.int64), pk_of_chunk={"seq": ident}, chunk_of_pk={"seq": ident})


def run_fake(rep: Rep, impl: str, cfg: dict, ref) -> None:
    doc = np.array([0, 0, 0, 1, 1, 1, 1, 2, 3, 3], dtype=np.int64)   # 원본 _selftest 배열
    flags = np.zeros(10, np.uint8)
    flags[1] = 3
    flags[4] = 2
    ad = registry.build("c2", impl, fake_meta(doc, flags), cfg)
    R = [1, 4, 5, 8]
    rep.ok(ad.select_neighbors(R, policy="none") == [], "fake 원본 selftest none == []")
    rep.ok(ad.select_neighbors(R, policy="all") == [0, 2, 3, 6, 9], "fake 원본 selftest all == [0,2,3,6,9]")
    rep.ok(ad.select_neighbors(R, policy="BC1") == [0, 2, 3, 6], "fake 원본 selftest BC1 == [0,2,3,6]")
    rep.ok(ad.select_neighbors([7], policy="BC1") == [], "fake chunk 하나짜리 문서(7) BC1 == []")
    rng = random.Random(0)
    cases = [rng.sample(range(10), rng.randint(1, 6)) for _ in range(500)] + [[], list(range(10))]
    compare_cases(rep, "fake 무작위", ad, ref, cases, doc, flags)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="bc1_v1")
    ap.add_argument("--run", default=None, help="config overlay (기본 없음: config.yaml)")
    args = ap.parse_args(argv)
    rep = Rep()

    ref, ref_src = load_reference(rep)
    adapter_src = (ROOT / "modules" / "c2_bc1.py").read_text(encoding="utf-8")
    rep.ok(code_block(ref_src) in adapter_src, "원본 RULES~select 코드 블록이 c2_bc1.py에 글자 그대로 있음")
    rep.ok(ref._selftest() is True, "원본 pf3_rules._selftest()")

    cfg, _ = load_config(args.run, [])
    run_fake(rep, args.impl, cfg, ref)

    hits = load_hits(rep)
    if hits is None:
        rep.info("저장된 hits가 없어 가짜 데이터 검사로 대신함 (위 MISSING 참고)")
        print(f"\n# 결과: FAIL {rep.n_fail}" + (f", 없는 파일 {len(rep.notes)}개" if rep.notes else ""))
        return 1 if rep.n_fail else 0

    files = cfg["dataset"]["files"]
    meta = FileChunkMetaLoader(cfg, check_md5=False).load(layouts=("seq",))
    adapter = registry.build("c2", args.impl, meta, cfg)
    doc = np.load(files["doc_idx"]["path"])      # 원본 analyze_pf3와 같은 입력
    flags = np.load(files["flags"]["path"])
    N = len(doc)
    rep.ok(np.array_equal(meta.doc_idx, doc) and np.array_equal(meta.flags, flags),
           "ChunkMeta.doc_idx/flags == doc_idx.npy/nq_flags.npy")

    pk_of_chunk = np.load(files["pk_seq"]["path"])
    chunk_of_pk = np.empty(len(pk_of_chunk), np.int64)
    chunk_of_pk[pk_of_chunk] = np.arange(len(pk_of_chunk))
    rep.info(f"pk_seq 항등 여부: {bool((pk_of_chunk == np.arange(N)).all())} (원본과 같이 chunk_of_pk로 변환)")
    Rs = [(qi, [int(chunk_of_pk[p]) for p in ids]) for qi, ids in hits]
    qis = [qi for qi, _ in Rs]
    rep.ok(len(Rs) == N_QUERIES and sorted(qis) == list(range(N_QUERIES)),
           f"질문 수 {len(Rs)} == {N_QUERIES}, qi 0~{N_QUERIES - 1} 빠짐·중복 없음")
    lens = sorted({len(R) for _, R in Rs})
    rep.info(f"질문별 R 길이 {lens}, R 안 중복 있는 질문 {sum(len(set(R)) != len(R) for _, R in Rs)}개")

    mean_nb = {}
    for pol in POLICIES:
        bad, n_nb = [], 0
        for qi, R in Rs:
            a = adapter.select_neighbors(R, policy=pol)
            b = ref.select(pol, R, doc, flags)
            n_nb += len(a)
            if not same(a, b):
                bad.append(qi)
        mean_nb[pol] = n_nb / len(Rs)
        rep.ok(not bad, f"hits {len(Rs)}개 /{pol}: 어댑터 == pf3_rules.select",
               f"불일치 qi {bad[:10]} 외 {max(0, len(bad) - 10)}개" if bad else f"이웃 평균 {mean_nb[pol]:.4f}개")

    # 원본 결과표(pf3_quality_nq.csv, subset=all)의 extra_pks(소수 셋째 자리)와 어댑터 이웃 평균 대조
    tgz = git_show(TGZ_PATH)
    if tgz is not None:
        with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tf:
            rows = tf.extractfile("pf3_out/pf3_quality_nq.csv").read().decode("utf-8").splitlines()
        head = rows[0].split(",")
        tab = {}
        for ln in rows[1:]:
            r = dict(zip(head, ln.split(",")))
            if r["subset"] == "all":
                tab.setdefault(r["rule"], float(r["extra_pks"]))
        for pol in POLICIES:
            rep.ok(round(mean_nb[pol], 3) == tab[pol],
                   f"원본 pf3_quality_nq.csv {pol} extra_pks {tab[pol]} == 어댑터 이웃 평균 {mean_nb[pol]:.3f}")

    # 경계 경우 (실제 doc_idx)
    starts = np.r_[0, np.flatnonzero(np.diff(doc)) + 1]
    ends = np.r_[starts[1:] - 1, N - 1]
    single = starts[starts == ends]
    multi = np.flatnonzero(ends > starts)
    rng = np.random.default_rng(0)
    pick = rng.choice(multi, 5, replace=False)
    rep.info(f"문서 {len(starts)}개, chunk 하나뿐인 문서 {len(single)}개, 마지막 doc {int(doc[-1])} = chunk "
             f"{int(starts[-1])}~{N - 1}")
    rep.ok(int(doc[-1]) == LAST_DOC, f"마지막 문서 번호 == {LAST_DOC}")
    exp = []  # (이름, R, policy, 기대값)
    for d in pick:
        s, e = int(starts[d]), int(ends[d])
        exp += [(f"doc{d} 첫 chunk {s}", [s], "all", [s + 1]),
                (f"doc{d} 첫 chunk {s}", [s], "BC1", [s + 1]),
                (f"doc{d} 마지막 chunk {e}", [e], "all", [e - 1]),
                (f"doc{d} 마지막 chunk {e}", [e], "BC1", [e - 1]),
                (f"doc{d} 첫+마지막", [e, s], "all", sorted({s + 1, e - 1} - {s, e}))]
    for c in single[:5].tolist() + ([int(single[-1])] if len(single) else []):
        exp += [(f"chunk 하나뿐인 문서 chunk {c}", [c], p, []) for p in ("BC1", "all")]
        exp += [(f"chunk 하나뿐인 문서 chunk {c} + 다른 결과", [c, (c + 5000) % N], "BC1", [])]
    ls = int(starts[-1])
    exp += [("마지막 문서 끝 N-1", [N - 1], "all", [N - 2] if ls < N - 1 else []),
            ("마지막 문서 끝 N-1", [N - 1], "BC1", [N - 2] if ls < N - 1 else []),
            ("마지막 문서 끝 N-1, N-2", [N - 1, N - 2], "all", [N - 3] if ls < N - 2 else []),
            ("마지막 문서 첫 chunk", [ls], "all", [ls + 1] if ls < N - 1 else []),
            ("마지막 문서 첫 chunk + 앞 문서 끝", [ls, ls - 1], "all", sorted({ls - 2, ls + 1} - {ls, ls - 1})),
            ("chunk 0", [0], "all", [1] if doc[1] == doc[0] else []),
            ("빈 R", [], "BC1", []), ("빈 R", [], "all", []), ("빈 R", [], "none", [])]
    bad = []
    for name, R, pol, e in exp:
        a = adapter.select_neighbors(R, policy=pol)
        b = ref.select(pol, R, doc, flags)
        if not (same(a, b) and a == e):
            bad.append(f"{name} {pol}: 어댑터 {a} 원본 {b} 기대 {e}")
    rep.ok(not bad, f"경계 경우 {len(exp)}개 (기대값 == 어댑터 == 원본)", "; ".join(bad[:3]))
    for name, R, pol, e in exp:
        if "마지막 문서" in name or "하나뿐" in name and pol == "all":
            rep.info(f"  {name} R={R} {pol} → {adapter.select_neighbors(R, policy=pol)}")

    try:
        adapter.select_neighbors([N], policy="all")
        rep.ok(False, "범위 밖 chunk_id(N) → ValueError")
    except ValueError:
        rep.ok(True, "범위 밖 chunk_id(N) → ValueError")

    print(f"\n# 결과: FAIL {rep.n_fail}" + (f", 없는 파일 {len(rep.notes)}개" if rep.notes else ""))
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
