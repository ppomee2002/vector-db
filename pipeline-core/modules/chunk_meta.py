"""ChunkMeta 로더 (공통 코어). interfaces.ChunkMetaLoader 구현.

출처(config.yaml dataset.files / file_store):
- doc_idx      : files.doc_idx (junlim, int64)       → 기준값
- chunk_idx    : files.chunk_meta npz (jaeeun)
- nbytes       : files.chunk_meta npz (jaeeun)
- flags        : files.flags (junlim pf3_out, uint8, 64단어용)
- pk seq       : files.pk_seq (junlim) — 항등이어야 함(검사)
- offset/length: file_store.seq.offset (jaeeun, 길이 n+1) → offset = off[:n], length = diff(off)
- doc / rand   : 요청할 때만 로드 (doc pk 항등, rand pk = milvus.c1_compare.rand.pk_map). 1차 B0에서는 쓰지 않음.

검사(실패 시 예외로 실행 중단):
- 모든 배열 길이 == dataset.chunks (1,000,000)
- doc_idx 단조 증가, 문서 수 == dataset.docs
- npz의 doc_idx == files.doc_idx, nbytes == diff(offset)
- md5 (C3 검토 반영, 2026-11-23). 방식은 md5_mode로 고르고 결과는 report → 실행 메타 md5_check에 남는다.
  · "full"  : 모든 파일(대용량 chunk_vecs 포함) 전체 md5. 별도 명령 verify_md5.py로 하루 한 번만 한다(측정 실행과 분리).
              결과를 캐시 파일(dataset.integrity.cache, 내 폴더)에 (경로, 크기, 수정 시각 ns, md5)로 기록.
  · "quick" : 일반 실행(main.py). 크기·수정 시각이 캐시와 같고 캐시 md5 == config md5면 통과(파일을 읽지 않음).
              다르거나 캐시가 없으면 그 파일만 md5를 다시 계산해 config와 대조하고 캐시를 갱신.
              cold / cache_constrained 실행에서는 다시 계산하지 않고 중단(측정 직전 대용량 읽기 금지).
              dataset.integrity.quick_exclude 파일(파이프라인이 쓰지 않는 chunk_vecs 등)은 검사하지 않음.
  · "off"   : 검사 안 함(계약 테스트).
  config md5가 null인 파일은 계산해서 기록("computed")만 한다(불일치 판단 없음).
원본 파일은 읽기만 한다(D2).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from interfaces import ChunkMeta, Layout


class ChunkMetaError(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parent.parent
KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def load_md5_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"files": {}, "last_full_verify_kst": None}


def save_md5_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def md5_file(path: str, count_lines: bool = False, bufsize: int = 1 << 24) -> tuple[str, int | None]:
    h = hashlib.md5()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
            if count_lines:
                n += b.count(b"\n")
    return h.hexdigest(), (n if count_lines else None)


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _inverse(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


class FileChunkMetaLoader:
    """config 경로에서 ChunkMeta를 만든다. report에 검사 결과가 쌓인다(실행 메타 md5_check로 저장)."""

    def __init__(self, cfg: dict, *, check_md5: bool = True, md5_mode: str | None = None,
                 cache_condition: str = "warm"):
        self.cfg = cfg
        self.md5_mode = "off" if not check_md5 else (md5_mode or "quick")
        if self.md5_mode not in ("full", "quick", "off"):
            raise ValueError(f"md5_mode는 full / quick / off: {md5_mode!r}")
        self.cache_condition = cache_condition
        integ = cfg["dataset"].get("integrity") or {}
        self.cache_path = ROOT / integ.get("cache", "cache/md5_cache.json")
        self.quick_exclude = set(integ.get("quick_exclude") or [])
        self.full_max_age_h = float(integ.get("full_max_age_h", 24))
        self.report: dict = {"md5": {}, "length": {}, "consistency": {}, "mode": self.md5_mode}

    # -- 검사 도우미 --
    def _len(self, name: str, arr: np.ndarray, n: int) -> None:
        ok = len(arr) == n
        self.report["length"][name] = {"len": int(len(arr)), "expected": n, "ok": ok}
        if not ok:
            raise ChunkMetaError(f"{name} 길이 {len(arr)} != {n}")

    def _hash(self, name: str, spec: dict, n: int, how: str, cache: dict) -> dict:
        path = spec["path"]
        want = spec.get("md5")
        st = os.stat(path)
        t = time.perf_counter()
        got, lines = md5_file(path, count_lines=(name == "chunks"))
        rec = {"path": path, "how": how, "got": got, "size": st.st_size, "sec": round(time.perf_counter() - t, 2)}
        if lines is not None:
            rec["lines"] = lines
            rec["lines_ok"] = lines == n
        cache["files"][path] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "md5": got, "lines": lines,
                                "verified_at_kst": _now_kst(), "how": how}
        if not want:
            rec["status"] = "computed (config md5: null)"
            return rec
        rec.update(status="checked", expected=want, ok=got == want)
        if got != want:
            self.report["md5"][name] = rec
            raise ChunkMetaError(f"md5 불일치 {name}: {got} != {want} ({path})")
        if lines is not None and lines != n:
            self.report["md5"][name] = rec
            raise ChunkMetaError(f"{name} 줄 수 {lines} != {n}")
        return rec

    def _md5(self) -> None:
        files = self.cfg["dataset"]["files"]
        n = self.cfg["dataset"]["chunks"]
        mode = self.md5_mode
        if mode == "off":
            for name, spec in files.items():
                self.report["md5"][name] = {"path": spec["path"], "status": "skipped (md5_mode=off)"}
            return
        cache = load_md5_cache(self.cache_path)
        self.report["cache"] = str(self.cache_path)
        all_ok = False
        try:
            for name, spec in files.items():
                if mode == "full":
                    self.report["md5"][name] = self._hash(name, spec, n, "full", cache)
                    continue
                path, want = spec["path"], spec.get("md5")
                ent = cache["files"].get(path)
                if name in self.quick_exclude:
                    self.report["md5"][name] = {"path": path, "status": "excluded (일반 실행 제외, 전체 검사에만 포함)",
                                                "last_verified_kst": ent and ent.get("verified_at_kst")}
                    continue
                st = os.stat(path)
                same_stat = bool(ent) and ent["size"] == st.st_size and ent["mtime_ns"] == st.st_mtime_ns
                if same_stat and want and ent["md5"] == want and (name != "chunks" or ent.get("lines") == n):
                    self.report["md5"][name] = {"path": path, "status": "quick_ok (크기·수정 시각이 캐시와 같음, 캐시 md5 == config)",
                                                "size": st.st_size, "md5_cached": ent["md5"],
                                                "verified_at_kst": ent.get("verified_at_kst")}
                    continue
                why = ("캐시 없음" if not ent else "크기·수정 시각 변경" if not same_stat
                       else "캐시 md5 != config md5" if want else "config md5 null")
                if self.cache_condition in ("cold", "cache_constrained"):
                    raise ChunkMetaError(
                        f"{name}: {why} → md5를 다시 계산해야 하지만 {self.cache_condition} 측정 직전이라 하지 않음. "
                        f"측정과 분리해서 verify_md5.py를 먼저 실행할 것 ({path})")
                rec = self._hash(name, spec, n, f"quick→recomputed ({why})", cache)
                self.report["md5"][name] = rec
            all_ok = True
        finally:
            if mode == "full" and all_ok:   # 하나라도 불일치면 전체 검사 시각을 갱신하지 않는다
                cache["last_full_verify_kst"] = _now_kst()
            save_md5_cache(self.cache_path, cache)
        last = cache.get("last_full_verify_kst")
        fv = {"last_full_verify_kst": last, "max_age_h": self.full_max_age_h}
        if last:
            age = (datetime.now(KST) - datetime.fromisoformat(last)).total_seconds() / 3600
            fv["age_h"] = round(age, 2)
            fv["stale"] = age > self.full_max_age_h
        else:
            fv["stale"] = True
        if fv["stale"] and mode == "quick":
            fv["warning"] = "전체 md5 검사가 없거나 오래됨 → 측정과 분리해서 verify_md5.py 실행 (cold 직전에는 하지 말 것)"
        self.report["full_verify"] = fv

    def load(self, *, layouts: Sequence[Layout] = ("seq",)) -> ChunkMeta:
        ds = self.cfg["dataset"]
        n = int(ds["chunks"])
        files = ds["files"]
        self._md5()

        doc_idx = np.load(files["doc_idx"]["path"]).astype(np.int64, copy=False)
        self._len("doc_idx", doc_idx, n)
        if not (np.diff(doc_idx) >= 0).all():
            raise ChunkMetaError("doc_idx가 단조 증가가 아님")
        n_docs = int(doc_idx[-1] - doc_idx[0] + 1)
        n_docs_uniq = int(len(np.unique(doc_idx)))
        self.report["consistency"]["n_docs"] = {"uniq": n_docs_uniq, "expected": ds["docs"],
                                               "ok": n_docs_uniq == ds["docs"] == n_docs}
        if n_docs_uniq != ds["docs"]:
            raise ChunkMetaError(f"문서 수 {n_docs_uniq} != {ds['docs']}")

        z = np.load(files["chunk_meta"]["path"])
        chunk_idx = z["chunk_idx"].astype(np.int32, copy=False)
        nbytes = z["nbytes"].astype(np.int64, copy=False)
        self._len("chunk_idx", chunk_idx, n)
        self._len("nbytes", nbytes, n)
        same_doc = bool(np.array_equal(z["doc_idx"], doc_idx))
        self.report["consistency"]["chunk_meta.doc_idx == doc_idx"] = same_doc
        if not same_doc:
            raise ChunkMetaError("chunk_meta.npz의 doc_idx가 files.doc_idx와 다름")

        flags = np.load(files["flags"]["path"]).astype(np.uint8, copy=False)
        self._len("flags", flags, n)

        pk_of: dict[str, np.ndarray] = {}
        chunk_of: dict[str, np.ndarray] = {}
        offset: dict[str, np.ndarray] = {}
        length = None

        if "seq" in layouts:
            pk_seq = np.load(files["pk_seq"]["path"]).astype(np.int64, copy=False)
            self._len("pk_seq", pk_seq, n)
            ident = bool(np.array_equal(pk_seq, np.arange(n, dtype=np.int64)))
            self.report["consistency"]["pk_seq identity"] = ident
            if not ident:
                raise ChunkMetaError("pk_seq가 항등이 아님 (계약: seq pk == chunk_id)")
            pk_of["seq"] = pk_seq
            chunk_of["seq"] = pk_seq  # 항등의 역 = 자기 자신
            off = np.load(ds["file_store"]["seq"]["offset"]).astype(np.int64, copy=False)
            self._len("file_store.seq.offset(n+1)", off, n + 1)
            length = np.diff(off)
            nb_ok = bool(np.array_equal(length, nbytes))
            self.report["consistency"]["seq diff(offset) == nbytes"] = nb_ok
            if not nb_ok:
                raise ChunkMetaError("file_store.seq offset 차이와 nbytes가 다름")
            offset["seq"] = off[:n]

        if "doc" in layouts:  # 1차 B0 미사용. C1 담당이 검증해서 쓸 것.
            pk_of["doc"] = np.arange(n, dtype=np.int64)
            chunk_of["doc"] = pk_of["doc"]
            offset["doc"] = np.load(ds["file_store"]["doc"]["offset"]).astype(np.int64, copy=False)[:n]
        if "rand" in layouts:
            perm = np.load(self.cfg["milvus"]["c1_compare"]["rand"]["pk_map"]).astype(np.int64, copy=False)
            self._len("pk_rand", perm, n)
            if not np.array_equal(np.sort(perm), np.arange(n)):
                raise ChunkMetaError("pk_rand가 순열이 아님")
            pk_of["rand"] = perm
            chunk_of["rand"] = _inverse(perm)
            offset["rand"] = np.load(ds["file_store"]["rand"]["offset"]).astype(np.int64, copy=False)[:n]
        if ("doc" in layouts or "rand" in layouts) and length is None:
            length = np.load(ds["file_store"]["length"]).astype(np.int64, copy=False)
            self._len("file_store.length", length, n)

        self.report["layouts"] = list(layouts)
        return ChunkMeta(
            n_chunks=n,
            doc_idx=_ro(doc_idx), chunk_idx=_ro(chunk_idx), flags=_ro(flags), nbytes=_ro(nbytes),
            pk_of_chunk={k: _ro(v) for k, v in pk_of.items()},
            chunk_of_pk={k: _ro(v) for k, v in chunk_of.items()},
            offset={k: _ro(v) for k, v in offset.items()} or None,
            length=_ro(length) if length is not None else None,
        )
