"""config.yaml(계약, 수정 금지) + 실행 overlay(runs/*.yaml) + --set key=value 를 합친다.

config.yaml은 바꾸지 않는다. 실행마다 달라지는 값(검색 컬렉션 임시 변경, 어댑터 이름, 질문 수 등)은
overlay에 적는다. 합친 결과와 overlay 원문은 실행 메타에 그대로 남긴다.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from interfaces import Settings

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_set(item: str) -> dict:
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"--set 형식은 a.b.c=값: {item!r}")
    val = yaml.safe_load(raw)
    d: dict = {}
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val
    return d


def load_config(overlay: str | Path | None = None, sets: list[str] | None = None,
                base: str | Path = BASE_CONFIG) -> tuple[dict, dict]:
    """(합친 cfg, overlay 기록) 반환."""
    cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    over: dict = {}
    if overlay:
        over = yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {}
        cfg = deep_merge(cfg, over)
    for s in sets or []:
        d = _parse_set(s)
        over = deep_merge(over, d)
        cfg = deep_merge(cfg, d)
    validate(cfg)
    return cfg, over


def validate(cfg: dict) -> None:
    ann = cfg["ann"]
    if ann["ef"] < ann["search_topk"]:
        raise ValueError(f"ann.ef({ann['ef']}) < ann.search_topk({ann['search_topk']}): Milvus HNSW 규칙 위반 (D9)")
    if ann["context_topk"] > ann["search_topk"]:
        raise ValueError("ann.context_topk > ann.search_topk")
    if cfg["c1"]["fetch_mode"] == "range":
        raise ValueError("c1.fetch_mode=range는 1차 제외(D7)")
    if cfg["c2"]["policy"] not in ("none", "BC1", "all"):
        raise ValueError(f"c2.policy 값 오류: {cfg['c2']['policy']}")
    if cfg["c3"]["mode"] not in ("m3", "m7"):
        raise ValueError(f"c3.mode 값 오류: {cfg['c3']['mode']}")


def settings_from(cfg: dict) -> Settings:
    run = cfg.get("run", {})
    s, f = cfg["milvus"]["search"], cfg["milvus"]["fetch"]
    return Settings(
        dataset=f"{cfg['dataset']['name']}_{cfg['dataset']['chunks'] // 1_000_000}M_{cfg['dataset']['chunking']}",
        query_set=run.get("query_set", "nq_test_ans"),
        search_collection=f"{s['database']}/{s['collection']}",
        fetch_collection=f"{f['database']}/{f['collection']}",
        layout=cfg["c1"]["layout"],
        search_topk=cfg["ann"]["search_topk"],
        context_topk=cfg["ann"]["context_topk"],
        ann_ef=cfg["ann"]["ef"],
        neighbor_policy=cfg["c2"]["policy"],
        fetch_mode=cfg["c1"]["fetch_mode"],
        batch_impl=cfg["c1"]["batch_impl"],
        scheduler_mode=cfg["c3"]["mode"],
        clients=cfg["c3"]["clients"],
        concurrency_cap=cfg["c3"]["cap"],
    )
