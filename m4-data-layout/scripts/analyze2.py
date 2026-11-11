"""추가 실험 요약: (1) 배치별 쿼리당 페이지/읽은 양 (2) 배치별 QPS (반복 3회 중앙값)"""
import json, os, sys
from statistics import median
R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/m4-exp/results2")
L = ["interleaved", "stream8", "doc", "doc_1seg", "doc_b1", "doc_1seg_b1", "doc_1seg_b16", "coaccess", "oracle"]
out = []
cold = [json.loads(l) for l in open(f"{R}/summary.jsonl")]
out.append("=== (1) 쿼리당 I/O (cold_each) ===")
out.append(f"{'layout':12s}{'w':>3s}{'pages':>8s}{'read_KB':>9s}{'need_KB':>9s}{'amp':>6s}{'vs_interleaved':>16s}{'data':>7s}{'tracker':>8s}")
for w in sorted({r['window'] for r in cold}):
    base = next((r['pages_payload'] for r in cold if r['layout'] == 'interleaved' and r['window'] == w), None)
    for l in L:
        r = next((r for r in cold if r['layout'] == l and r['window'] == w), None)
        if not r: continue
        d = f"{(r['pages_payload']/base-1)*100:+.1f}%" if base else ""
        out.append(f"{l:12s}{w:3d}{r['pages_payload']:8.1f}{r['read_bytes']/1024:9.1f}{r['need_bytes']/1024:9.1f}{r['read_bytes']/r['need_bytes']:6.1f}{d:>16s}{r.get('pages_pdata', float('nan')):7.1f}{r.get('pages_ptracker', float('nan')):8.1f}")
if os.path.exists(f"{R}/qps.jsonl"):
    q = [json.loads(l) for l in open(f"{R}/qps.jsonl")]
    out.append("\n=== (2) 처리량 (3회 중앙값, 캐시 비운 상태) ===")
    out.append(f"{'layout':12s}{'w':>3s}{'C':>4s}{'QPS':>8s}{'p50_ms':>8s}{'p95_ms':>8s}{'read_MB':>9s}{'QPS_min-max':>16s}")
    for w in sorted({r['window'] for r in q}):
        for c in sorted({r['conc'] for r in q}):
            for l in L:
                g = [r for r in q if r['layout'] == l and r['window'] == w and r['conc'] == c]
                if not g: continue
                qs = [r['qps'] for r in g]
                out.append(f"{l:12s}{w:3d}{c:4d}{median(qs):8.1f}{median(r['p50_ms'] for r in g):8.2f}{median(r['p95_ms'] for r in g):8.2f}"
                           f"{median(r['total_read_mb'] for r in g):9.1f}{f'{min(qs):.0f}-{max(qs):.0f}':>16s}")
txt = "\n".join(out); print(txt); open(f"{R}/summary2.txt", "w").write(txt + "\n")
