"""두 실행의 results.jsonl에서 같은 질문의 상위 10개 hits를 비교한다 (예: nq_seq B0 vs nq1m_sel B0).

    /workspace/jaeeun/venv/bin/python experiments/compare_top10.py results/<nq_seq B0>/results.jsonl results/b0_smoke_20261123_112701/results.jsonl

출력: 공통 질문 수, 질문별 상위 10개 집합 겹침(|A∩B|/10) 평균, 순서까지 같은 질문 수, 1위 일치 수.
"""
from __future__ import annotations

import json
import sys


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                out[(r["query_set"], r["q_idx"])] = [h["chunk_id"] for h in sorted(r["hits"], key=lambda h: h["rank"])[:10]]
    return out


def main(a, b) -> int:
    A, B = load(a), load(b)
    keys = sorted(set(A) & set(B))
    if not keys:
        print("공통 질문 없음")
        return 1
    ov = [len(set(A[k]) & set(B[k])) / 10 for k in keys]
    same_order = sum(A[k] == B[k] for k in keys)
    top1 = sum(A[k][:1] == B[k][:1] for k in keys)
    print(json.dumps({"n_common": len(keys), "top10_set_overlap_mean": sum(ov) / len(ov),
                      "top10_set_overlap_min": min(ov), "n_identical_order": same_order, "n_top1_same": top1,
                      "per_query": {f"{k[0]}:{k[1]}": o for k, o in zip(keys, ov)}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
