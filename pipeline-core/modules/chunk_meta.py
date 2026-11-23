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
- md5: config에 md5가 적힌 파일은 검사(불일치면 중단), md5: null 파일은 계산해서 기록("computed")한다.
  계산한 값은 config.yaml에 채워 고정한다(ChunkMeta 계약). 결과는 report → 실행 메타 md5_check.
원본 파일은 읽기만 한다(D2).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Sequence

import numpy as np

from interfaces import ChunkMeta, Layout


class ChunkMetaError(RuntimeError):
    pass


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

    def __init__(self, cfg: dict, *, check_md5: bool = True):
        self.cfg = cfg
        self.check_md5 = check_md5
        self.report: dict = {"md5": {}, "length": {}, "consistency": {}}

    # -- 검사 도우미 --
    def _len(self, name: str, arr: np.ndarray, n: int) -> None:
        ok = len(arr) == n
        self.report["length"][name] = {"len": int(len(arr)), "expected": n, "ok": ok}
        if not ok:
            raise ChunkMetaError(f"{name} 길이 {len(arr)} != {n}")

    def _md5(self) -> None:
        files = self.cfg["dataset"]["files"]
        n = self.cfg["dataset"]["chunks"]
        for name, spec in files.items():
            want = spec.get("md5")
            if not self.check_md5:
                self.report["md5"][name] = {"path": spec["path"], "status": "skipped (check_md5=False)"}
                continue
            t = time.perf_counter()
            got, lines = md5_file(spec["path"], count_lines=(name == "chunks"))
            sec = round(time.perf_counter() - t, 2)
            if not want:
                self.report["md5"][name] = {"path": spec["path"], "status": "computed (config md5: null)",
                                            "got": got, "size": os.path.getsize(spec["path"]), "sec": sec}
                continue
            rec = {"path": spec["path"], "status": "checked", "expected": want, "got": got, "ok": got == want,
                   "sec": sec}
            if lines is not None:
                rec["lines"] = lines
                rec["lines_ok"] = lines == n
            self.report["md5"][name] = rec
            if got != want:
                raise ChunkMetaError(f"md5 불일치 {name}: {got} != {want} ({spec['path']})")
            if lines is not None and lines != n:
                raise ChunkMetaError(f"{name} 줄 수 {lines} != {n}")

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
