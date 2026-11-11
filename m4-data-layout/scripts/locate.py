"""읽기 전용 확인: 같은 문서의 연속 chunk들이 실제 Qdrant payload 파일(page_*.dat)의 어디에 저장됐는지 찾음.
문서 순서로 넣었는데 연속 chunk가 서로 다른 4KB 페이지/세그먼트에 흩어져 있으면 -> Qdrant가 삽입 순서를 유지하지 않음.
사용: python locate.py doc   /   python locate.py oracle"""
import glob, mmap, os, sys
import numpy as np
from common import load_data
D = os.path.expanduser("~/m4-exp"); data = os.environ.get("M4_DATA", D + "/data")
L = sys.argv[1] if len(sys.argv) > 1 else "doc"
chunks, topk, train, test, bounds = load_data(data)
ST = os.environ.get("M4_STORAGE", D + "/qdrant_storage")
files = sorted(glob.glob(f"{ST}/collections/m4_{L}/0/segments/*/payload_storage/page_*.dat"))
maps = []
for f in files:
    fd = os.open(f, os.O_RDONLY); maps.append((f, mmap.mmap(fd, 0, prot=mmap.PROT_READ))); os.close(fd)
def key(t):
    t = t[:80]
    for ch in ['"', "\\"]:
        if ch in t: t = t[:t.index(ch)]
    return t[:50].encode()
print(f"== {L}: segments={len(files)} ==")
print(f"{'chunk':>7} {'pos':>4} {'segment':>9} {'offset':>10} {'page':>7}")
for q in test[:3]:
    s, e = bounds[topk[q][0]]
    pages = set(); nf = 0
    for i in range(s, min(e, s + 12)):
        k = key(chunks[i]["text"]); hit = None
        for f, m in maps:
            o = m.find(k)
            if o >= 0: hit = (f.split("/segments/")[1][:8], o); break
        if hit: pages.add((hit[0], hit[1] // 4096)); print(f"{i:7d} {chunks[i]['pos']:4d} {hit[0]:>9} {hit[1]:10d} {hit[1]//4096:7d}")
        else: nf += 1; print(f"{i:7d} {chunks[i]['pos']:4d}   NOT FOUND (압축 저장일 수 있음)")
    print(f"-> 연속 chunk {min(e, s+12)-s}개가 4KB 페이지 {len(pages)}개에 저장됨 (못 찾음 {nf}개)\n")
