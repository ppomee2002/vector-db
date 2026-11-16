# io_check.py: ③ 마지막 단계 - 실제 원문 Fetch I/O 측정 (Top-K vs 선별 M1-noPos). 저장 방식 고정, chunk별 개별 읽기, 단일 스레드
import os, time, json, shutil, subprocess, numpy as np, pandas as pd
src = open('/workspace/jaeeun/sel_fetch/scripts/q2_predict.py', encoding='utf-8').read()
tf0 = time.time(); exec(src.split('gkf = GroupKFold')[0]); feat_sec = time.time() - tf0
from sklearn.model_selection import GroupKFold
NQ = rr.q_idx.nunique()

# 1) 선별 점수 (위치 특징 제외 M1-noPos, 쿼리 단위 5-fold)
POS = ['f_chunk_pos', 'f_rel_pos', 'f_first_chunk', 'f_doc_len']
COLS = [c for c in F0 + FDOC + FEMB + ['f_nbytes'] if c not in POS]
def hgb(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=0)
p = np.zeros(len(rr))
for tr, te in GroupKFold(n_splits=5).split(rr, rr.y, rr.q_idx):
    p[te] = hgb().fit(rr.iloc[tr][COLS], rr.y.iloc[tr]).predict_proba(rr.iloc[te][COLS])[:, 1]
rr['p_sel'] = p
mfull = hgb().fit(rr[COLS], rr.y); ti = time.time(); mfull.predict_proba(rr[COLS]); infer_ms = (time.time() - ti) / NQ * 1000
log('selection scores ready')

# 2) 예산별 보존율 + 같은 품질 예산 B*
def select(score, B):
    return rr.assign(_s=score).sort_values(['q_idx', '_s'], ascending=[True, False]).groupby('q_idx').head(B)
SC = {'TopK': -rr['rank'].values, 'Select': rr.p_sel.values}
sweep = [dict(policy=nm, B=B, retained=select(sc, B).y.sum() / (10 * NQ)) for B in range(10, 42, 2) for nm, sc in SC.items()]
sweep = pd.DataFrame(sweep); sweep.to_csv(f'{R}/io_sweep.csv', index=False)
target = float(sweep[(sweep.policy == 'TopK') & (sweep.B == 30)].retained.iloc[0])
ok = sweep[(sweep.policy == 'Select') & (sweep.retained >= target)]
BSTAR = int(ok.B.min()) if len(ok) else 30
log(f'TopK@30 retained {target:.4f} -> Select needs B*={BSTAR}')

# 3) 본문 저장 파일 (기존 chunk 순서 그대로, 내 폴더)
STORE = f'{D}/chunk_store_seq.bin'; OFFP = f'{D}/chunk_store_off.npy'
if not (os.path.exists(STORE) and os.path.exists(OFFP)):
    offs = [0]
    with open(f'{J}/chunks.jsonl', encoding='utf-8') as fi, open(STORE, 'wb') as fo:
        for l in fi:
            b = json.loads(l)['text'].encode('utf-8'); fo.write(b); offs.append(offs[-1] + len(b))
    np.save(OFFP, np.array(offs, np.int64))
off = np.load(OFFP); log('store ready', os.path.getsize(STORE) // 2**20, 'MB')

# 4) 디스크 통계 (/proc/diskstats = iostat 원천) + iostat 로그
dev = os.path.basename(subprocess.check_output(['df', '--output=source', STORE]).decode().split()[1])
def ds():
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2] == dev: return int(f[3]), int(f[5]), int(f[6])
    return 0, 0, 0
iolog = '/workspace/jaeeun/sel_fetch/logs/iostat_io.log'; iop = None
if shutil.which('iostat'):
    iop = subprocess.Popen(['iostat', '-dxk', dev, '1'], stdout=open(iolog, 'w'), stderr=subprocess.STDOUT)
log('device', dev, '| iostat', 'ON -> ' + iolog if iop else 'not installed (diskstats only)')
a = ds(); time.sleep(5); b = ds()
idle = dict(policy='IDLE(5s)', B=0, dev_reads=b[0] - a[0], dev_read_MB=(b[1] - a[1]) * 512 / 1e6)

# 5) Fetch 측정
fd = os.open(STORE, os.O_RDONLY)
def run(name, B):
    sel = select(SC[name], B)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); time.sleep(1)
    a = ds(); t_start = time.time(); rows = []
    for q, g in sel.groupby('q_idx', sort=True):
        t = time.perf_counter(); nb = 0
        for c in g.chunk_id.values:
            nb += len(os.pread(fd, int(off[c + 1] - off[c]), int(off[c])))
        rows.append((name, B, q, len(g), nb, (time.perf_counter() - t) * 1000))
    b = ds(); t_end = time.time()
    pq = pd.DataFrame(rows, columns=['policy', 'B', 'q_idx', 'chunks', 'bytes', 'fetch_ms'])
    s = dict(policy=name, B=B, retained=sel.y.sum() / (10 * NQ), chunks_total=int(pq.chunks.sum()), app_MB=pq.bytes.sum() / 1e6,
             mean_ms=pq.fetch_ms.mean(), p50_ms=pq.fetch_ms.quantile(.5), p95_ms=pq.fetch_ms.quantile(.95), p99_ms=pq.fetch_ms.quantile(.99),
             dev_reads=b[0] - a[0], dev_read_MB=(b[1] - a[1]) * 512 / 1e6, dev_read_ms=b[2] - a[2],
             t_start=time.strftime('%H:%M:%S', time.localtime(t_start)), t_end=time.strftime('%H:%M:%S', time.localtime(t_end)))
    log('run', name, B, f"mean {s['mean_ms']:.3f} ms, dev_reads {s['dev_reads']}")
    return pq, s
per, summ = [], [idle]
for name, B in [('TopK', 10), ('Select', 10), ('TopK', 20), ('Select', 20), ('TopK', 30), ('Select', 30), ('Select', BSTAR), ('TopK', 100)]:
    pq, s = run(name, B); per.append(pq); summ.append(s)
os.close(fd)
if iop: iop.terminate()
pd.concat(per).to_csv(f'{R}/io_per_query.csv', index=False)
summ = pd.DataFrame(summ); summ.to_csv(f'{R}/io_summary.csv', index=False)

pd.set_option('display.width', 220)
print('\n================ ③ I/O 검증 결과 ================')
print(f'저장 방식 고정(기존 chunk 순서), chunk별 개별 pread, 단일 스레드, 매 실행 전 콜드 캐시 | device={dev}')
print(f'선별 계산 비용(쿼리당): 특징 계산 <= {feat_sec/NQ*1000:.2f} ms (CSV 로딩 포함 상한), 모델 예측 {infer_ms:.3f} ms')
print(summ.round(4).to_string(index=False))
ts = summ[(summ.policy == 'TopK') & (summ.B == 30)].iloc[0]; ss = summ[(summ.policy == 'Select') & (summ.B == BSTAR)].iloc[0]
print(f'\n[같은 품질 비교] TopK@30 retained {ts.retained:.4f}, mean {ts.mean_ms:.3f} ms, {ts.app_MB:.1f} MB, dev_reads {ts.dev_reads}')
print(f'               Select@{BSTAR} retained {ss.retained:.4f}, mean {ss.mean_ms:.3f} ms, {ss.app_MB:.1f} MB, dev_reads {ss.dev_reads}')
print(f'               -> Fetch 개수 {1 - BSTAR/30:.1%} 감소, 읽은 바이트 {1 - ss.app_MB/ts.app_MB:.1%} 감소')
print(f'\nsaved: {R}/io_per_query.csv, io_summary.csv, io_sweep.csv' + (f', {iolog}' if iop else ''))
log('ALL DONE')
