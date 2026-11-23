"""어댑터 등록소. config(또는 실행 overlay)의 c1.impl / c2.impl / c3.impl 이름으로 구현을 고른다.

담당자 사용법:
    # modules/c2_<내이름>.py
    from modules.registry import register

    @register("c2", "bc1_kim")
    class MyC2:
        def __init__(self, meta, cfg): ...          # 생성자 규약 (interfaces.py 공통 규칙 5): (meta: ChunkMeta, cfg: dict)
        def select_neighbors(self, ranked_chunk_ids, *, policy): ...

discover()가 modules/ 안의 모든 .py를 import하므로 파일을 추가하고 이름만 등록하면 된다.
선택적 훅(있으면 실행기가 호출):
    async def aclose(self)                   # 종료 시 자원 정리 (Milvus 클라이언트 등)
    def describe(self) -> dict               # 실행 메타에 남길 설명 (경로, 컬렉션 등)
    def milvus_collections(self) -> list[str]  # 이 어댑터가 쓰는 "<db>/<collection>" (load 대상)
C1은 계약상 fetch_source() -> "file:<경로>" | "milvus:<db>/<collection>"을 구현해야 한다(실행 메타에 기록).
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback
from pathlib import Path

KINDS = ("c1", "c2", "c3")
_REG: dict[str, dict[str, type]] = {k: {} for k in KINDS}
_IMPORT_ERRORS: dict[str, str] = {}
_DISCOVERED = False

# 공통 코어 파일은 어댑터가 아니므로 discover 대상에서 뺀다(import 순환 방지).
_CORE = {"__init__", "registry", "config_loader", "chunk_meta", "queries", "search_milvus",
         "pipeline", "evaluate", "run_meta", "result_checks"}


def register(kind: str, name: str):
    if kind not in KINDS:
        raise ValueError(f"kind는 {KINDS} 중 하나여야 함: {kind!r}")

    def deco(cls):
        prev = _REG[kind].get(name)
        if prev is not None and prev is not cls:
            raise ValueError(f"{kind} 이름 {name!r}이 이미 등록됨: {prev.__module__}.{prev.__name__}")
        _REG[kind][name] = cls
        cls._registered_as = (kind, name)
        return cls
    return deco


def discover() -> dict[str, str]:
    """modules/*.py를 모두 import. import에 실패한 파일은 기록만 하고 계속 진행."""
    global _DISCOVERED
    if _DISCOVERED:
        return _IMPORT_ERRORS
    pkg_dir = Path(__file__).resolve().parent
    for m in pkgutil.iter_modules([str(pkg_dir)]):
        if m.name in _CORE or m.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"modules.{m.name}")
        except Exception:  # noqa: BLE001 - 담당자 파일 하나가 깨져도 나머지는 쓸 수 있게
            _IMPORT_ERRORS[m.name] = traceback.format_exc(limit=3)
    _DISCOVERED = True
    return _IMPORT_ERRORS


def available() -> dict[str, list[str]]:
    discover()
    return {k: sorted(v) for k, v in _REG.items()}


def get(kind: str, name: str) -> type:
    discover()
    if name not in _REG[kind]:
        hint = "; ".join(f"{f}: {e.strip().splitlines()[-1]}" for f, e in _IMPORT_ERRORS.items())
        raise KeyError(f"{kind} 구현 {name!r}이 등록되어 있지 않음. 등록된 것: {sorted(_REG[kind])}"
                       + (f". import 실패 파일: {hint}" if hint else ""))
    return _REG[kind][name]


def build(kind: str, name: str, meta, cfg: dict):
    return get(kind, name)(meta, cfg)


def impl_names(cfg: dict) -> dict[str, str]:
    """cfg의 c1.impl / c2.impl / c3.impl (config.yaml에 기본값 = reference 어댑터 이름)."""
    out = {}
    for k in KINDS:
        name = (cfg.get(k) or {}).get("impl")
        if not name:
            raise KeyError(f"config에 {k}.impl이 없음 (config.yaml 기본값을 확인)")
        out[k] = name
    return out
