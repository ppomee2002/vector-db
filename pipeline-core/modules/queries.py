"""질문 로더 (공통 코어). 질문 키 = (query_set, q_idx) (D3).

- nq_test_real: data_nq/queries.jsonl의 real 0~3451. 벡터 = query_vecs[q_idx]. 정답 = nq_gold(없을 수 있음).
- nq_test_ans : nq_test_real 중 정규화 후 비어 있지 않은 정답이 있는 질문(2,254개, config count와 대조).
                q_idx 목록은 data/nq_test_ans_qidx.txt로 고정하고 md5를 기록. 이미 있으면 다시 만든 목록과 대조.
- c2x_test    : sel_fetch_c2x test_queries.jsonl(2,000개, q_idx 0부터 별도 체계). 벡터 = q_train_minilm[train_row].
pseudo 질문(3452~)은 어떤 query_set에도 넣지 않는다.
고정 seed 순서: np.random.default_rng(seed).permutation 으로 섞은 키 목록.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from interfaces import QueryKey
from modules.evaluate import norm_answers

ROOT = Path(__file__).resolve().parent.parent
QIDX_FILE = ROOT / "data" / "nq_test_ans_qidx.txt"
N_REAL = 3452


@dataclass
class QuerySetData:
    name: str
    q_idx: list[int]
    vecs: np.ndarray                     # 행 = 이 query_set의 벡터 행 (row_of로 접근)
    row_of: dict[int, int]
    answers: dict[int, list[str] | None]
    texts: dict[int, str]
    info: dict = field(default_factory=dict)

    def vec(self, q_idx: int) -> np.ndarray:
        return self.vecs[self.row_of[q_idx]]

    def keys(self) -> list[QueryKey]:
        return [(self.name, q) for q in self.q_idx]


class QueryLoader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._cache: dict[str, QuerySetData] = {}

    def load(self, name: str) -> QuerySetData:
        if name in self._cache:
            return self._cache[name]
        qs = self.cfg["query_sets"][name]
        if name in ("nq_test_real", "nq_test_ans"):
            d = self._load_nq(name, qs)
        elif name == "c2x_test":
            d = self._load_c2x(qs)
        else:
            raise KeyError(f"알 수 없는 query_set: {name}")
        self._cache[name] = d
        return d

    def _load_nq(self, name: str, qs: dict) -> QuerySetData:
        texts, kinds = {}, {}
        with open(qs["queries"], encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= N_REAL:
                    break
                r = json.loads(line)
                texts[i], kinds[i] = r["text"], r["kind"]
        if any(k != "real" for k in kinds.values()) or len(texts) != N_REAL:
            raise RuntimeError("queries.jsonl 0~3451이 모두 real이 아님")
        gold_path = self.cfg["query_sets"]["nq_test_ans"]["gold"]
        answers: dict[int, list[str] | None] = {}
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                answers[int(r["qi"])] = r.get("ans")
                if r["q"] != texts[int(r["qi"])]:
                    raise RuntimeError(f"gold q 불일치 qi={r['qi']}")
        vecs = np.load(qs["vectors"], mmap_mode="r")
        info: dict = {"vectors_shape": list(vecs.shape)}
        if name == "nq_test_real":
            lo, hi = qs["q_idx"]
            q_idx = list(range(lo, hi + 1))
        else:
            q_idx = [q for q in range(N_REAL) if norm_answers(answers.get(q))]
            if len(q_idx) != qs["count"]:
                raise RuntimeError(f"nq_test_ans 개수 {len(q_idx)} != config count {qs['count']}")
            body = "".join(f"{q}\n" for q in q_idx).encode()
            md5 = hashlib.md5(body).hexdigest()
            if QIDX_FILE.exists():
                if QIDX_FILE.read_bytes() != body:
                    raise RuntimeError(f"{QIDX_FILE}의 q_idx 목록이 다시 만든 목록과 다름")
                info["qidx_file"] = {"path": str(QIDX_FILE), "md5": md5, "status": "verified"}
            else:
                QIDX_FILE.parent.mkdir(parents=True, exist_ok=True)
                QIDX_FILE.write_bytes(body)
                info["qidx_file"] = {"path": str(QIDX_FILE), "md5": md5, "status": "created"}
        return QuerySetData(name, q_idx, vecs, {q: q for q in q_idx},
                            {q: answers.get(q) for q in q_idx}, {q: texts[q] for q in q_idx}, info)

    def _load_c2x(self, qs: dict) -> QuerySetData:
        rows, answers, texts, q_idx = {}, {}, {}, []
        with open(qs["queries"], encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                q = int(r["q_idx"])
                q_idx.append(q)
                rows[q], answers[q], texts[q] = int(r["train_row"]), r.get("ans"), r["text"]
        if len(q_idx) != qs["count"]:
            raise RuntimeError(f"c2x_test 개수 {len(q_idx)} != {qs['count']}")
        vecs = np.load(qs["vectors"], mmap_mode="r")
        return QuerySetData("c2x_test", q_idx, vecs, rows, answers, texts,
                            {"vectors_shape": list(vecs.shape)})

    def ordered_keys(self, name: str, seed: int, n: int | None = None) -> list[QueryKey]:
        d = self.load(name)
        perm = np.random.default_rng(seed).permutation(len(d.q_idx))
        keys = [(name, int(d.q_idx[i])) for i in perm]
        return keys[:n] if n else keys
