"""pf3_rules.py: 이웃 조각을 고르는 규칙 (bench_pf3.py, analyze_pf3.py가 같이 쓴다).

모든 규칙은 정답(qrels)을 보지 않는다. 쓰는 정보는 세 가지뿐이다.
  - 검색 결과 top-k의 순위
  - 조각의 문서 번호 (같은 문서의 앞뒤 조각만 이웃)
  - 적재 때 미리 계산해 둔 조각 경계 표시 flags (gold_map.py가 만든다)
      bit0 = 이 조각이 문장 중간에서 시작함 (앞 조각과 문장이 이어짐)
      bit1 = 이 조각이 문장 중간에서 끝남 (뒤 조각과 문장이 이어짐)

규칙 (R = top-k 조각, 순위 순)
  none  이웃을 읽지 않음
  A     잘린 문장 잇기: 결과 조각이 문장 중간에서 시작하면 앞 조각, 중간에서 끝나면 뒤 조각
  A3    A를 상위 3개 결과에만
  B     같은 문서 모으기: top-k 안에 같은 문서 조각이 2개 이상이면, 그 문서의 결과 조각들의 이웃
  C1    1위 결과의 이웃
  C3    상위 3개 결과의 이웃
  BC1   B와 C1의 합
  all   모든 결과의 이웃 (기존 "한 번에 읽기")
이웃은 expand=1 (앞뒤 한 조각)이다. 반환값은 R에 없는 이웃 조각 번호(정렬)다.
"""

RULES = ("none", "A", "A3", "B", "C1", "C3", "BC1", "all")


def nbrs(c, doc):
    out = []
    for kk in (c - 1, c + 1):
        if 0 <= kk < len(doc) and doc[kk] == doc[c]:
            out.append(kk)
    return out


def _cut_nbrs(c, doc, flags):
    out = []
    f = int(flags[c])
    if f & 1 and c - 1 >= 0 and doc[c - 1] == doc[c]:
        out.append(c - 1)
    if f & 2 and c + 1 < len(doc) and doc[c + 1] == doc[c]:
        out.append(c + 1)
    return out


def select(rule, R, doc, flags=None):
    """R: 결과 조각 번호(순위 순). doc: 조각별 문서 번호 배열. flags: 조각 경계 표시 배열(규칙 A만 사용)."""
    Rs = set(int(c) for c in R)
    out = set()
    if rule == "none":
        return []
    if rule in ("A", "A3"):
        if flags is None:
            raise ValueError("rule A needs flags")
        src = R if rule == "A" else R[:3]
        for c in src:
            out.update(_cut_nbrs(int(c), doc, flags))
    elif rule in ("B", "BC1"):
        cnt = {}
        for c in R:
            cnt[int(doc[c])] = cnt.get(int(doc[c]), 0) + 1
        for c in R:
            if cnt[int(doc[c])] >= 2:
                out.update(nbrs(int(c), doc))
        if rule == "BC1" and len(R):
            out.update(nbrs(int(R[0]), doc))
    elif rule in ("C1", "C3"):
        for c in R[:1 if rule == "C1" else 3]:
            out.update(nbrs(int(c), doc))
    elif rule == "all":
        for c in R:
            out.update(nbrs(int(c), doc))
    else:
        raise ValueError(f"unknown rule {rule}")
    return sorted(out - Rs)


def _selftest():
    import numpy as np
    doc = np.array([0, 0, 0, 1, 1, 1, 1, 2, 3, 3])
    flags = np.zeros(10, np.uint8)
    flags[1] = 1 | 2  # 4개 조각 문서 0의 가운데: 앞뒤 둘 다 잘림
    flags[4] = 2      # 뒤만 잘림
    R = [1, 4, 5, 8]
    assert select("none", R, doc) == []
    assert select("all", R, doc) == [0, 2, 3, 6, 9], select("all", R, doc)  # 8의 뒤 9 포함
    assert select("A", R, doc, flags) == [0, 2], select("A", R, doc, flags)  # 4의 뒤(5)는 이미 결과
    assert select("A3", [8, 4, 1], doc, flags) == [0, 2, 5]  # 4의 뒤 5는 여기선 결과가 아님
    assert select("B", R, doc) == [3, 6]          # 문서 1에 결과 2개(4, 5)
    assert select("C1", R, doc) == [0, 2]
    assert select("C3", R, doc) == [0, 2, 3, 6]
    assert select("BC1", R, doc) == [0, 2, 3, 6]
    assert select("C1", [7], doc) == []           # 조각 하나짜리 문서
    return True


if __name__ == "__main__":
    print("pf3_rules selftest", "PASS" if _selftest() else "FAIL")
