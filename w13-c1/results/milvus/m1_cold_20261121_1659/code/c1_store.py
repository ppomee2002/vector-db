# c1_store.py: C1 본문 저장소 (문서 순 배치 + 연속 블록 묶어 읽기)  김지현 2024134018
#
# 파이프라인에서 쓰는 법
#   from c1_store import Store
#   st = Store("/workspace/jihyeon/c1_w13/store", layout="doc")     # 또는 "rand"
#   texts, stats = st.read(chunk_ids)                              # chunk_id = Milvus pk = chunks.jsonl 줄 번호
#   texts, stats = st.read(chunk_ids, merge=False)                 # 기준선: chunk마다 따로 읽기
#   stats = {"n_chunks", "n_reads", "bytes_read", "useful_bytes", "latency_us"}
#
# 저장 형식 (store 폴더)
#   {layout}.bin        본문을 배치 순서대로 이어 붙인 파일 (4KB 배수로 끝을 채움)
#   {layout}_off.npy    chunk_id별 파일 안 시작 위치 (int64)
#   len.npy             chunk_id별 본문 바이트 수 (int32)
#   doc_idx.npy, chunk_idx.npy   chunk_id별 문서 번호, 문서 안 순서
#   meta.json           만든 시각, 입력 파일 정보(md5), 배치 설명
#
# 배치
#   doc   chunks.jsonl 순서 그대로. 이 파일은 문서 번호, 문서 안 순서대로 정렬되어 있다 (build 때 확인).
#   rand  chunk 순서를 무작위로 섞음 (seed 1). 기준선.
import os, json, time, mmap
import numpy as np
from common import BLOCK, CHUNKS, log, safe_path, open_read, file_info, write_json, now_kst

LAYOUTS = ("doc", "rand")


# ---------------------------------------------------------------- 만들기
def build(store_dir, chunks_path=CHUNKS, seed=1, md5=True):
    store_dir = safe_path(store_dir)
    os.makedirs(store_dir, exist_ok=True)
    t0 = time.time()
    texts, doc, cix = [], [], []
    with open_read(chunks_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"].encode("utf-8"))
            doc.append(r["doc_idx"])
            cix.append(r["chunk_idx"])
    n = len(texts)
    doc = np.array(doc, np.int64)
    cix = np.array(cix, np.int32)
    ln = np.array([len(t) for t in texts], np.int32)
    log(f"chunks {n}, docs {int(doc.max()) + 1}, mean bytes {ln.mean():.1f}")

    # chunks.jsonl이 정말 문서 순인지 확인 (문서 번호가 줄지 않고, 문서 안 순서가 0부터 1씩 증가)
    same = doc[1:] == doc[:-1]
    ok = bool(np.all(np.diff(doc) >= 0) and np.all(cix[1:][same] == cix[:-1][same] + 1) and np.all(cix[1:][~same] == 0) and cix[0] == 0)
    if not ok:
        raise SystemExit("chunks.jsonl이 문서 순서가 아닙니다. doc 배치를 만들기 전에 확인이 필요합니다.")
    log("chunks.jsonl is in document order: OK")

    orders = {"doc": np.arange(n, dtype=np.int64),
              "rand": np.random.default_rng(seed).permutation(n).astype(np.int64)}
    for lay, order in orders.items():
        off = np.zeros(n, np.int64)
        pos = 0
        path = os.path.join(store_dir, f"{lay}.bin")
        with open(safe_path(path), "wb") as fo:
            for cid in order:
                off[cid] = pos
                fo.write(texts[cid])
                pos += int(ln[cid])
            pad = (-pos) % BLOCK
            fo.write(b"\0" * pad)
        np.save(safe_path(os.path.join(store_dir, f"{lay}_off.npy")), off)
        log(f"{lay}.bin written: {(pos + pad) / 2**20:.1f} MB")
    np.save(safe_path(os.path.join(store_dir, "len.npy")), ln)
    np.save(safe_path(os.path.join(store_dir, "doc_idx.npy")), doc)
    np.save(safe_path(os.path.join(store_dir, "chunk_idx.npy")), cix)
    write_json(os.path.join(store_dir, "meta.json"), {
        "built_at_kst": now_kst(), "n_chunks": n, "n_docs": int(doc.max()) + 1, "block": BLOCK,
        "layouts": {"doc": "chunks.jsonl order (doc_idx, chunk_idx)", "rand": f"random permutation, numpy default_rng({seed})"},
        "input": file_info(chunks_path, md5=md5), "build_sec": round(time.time() - t0, 1)})
    log(f"store built in {time.time() - t0:.0f}s -> {store_dir}")


# ---------------------------------------------------------------- 읽기
class Store:
    def __init__(self, store_dir, layout="doc", direct=True, buf_mb=64):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        self.layout = layout
        self.off = np.load(os.path.join(store_dir, f"{layout}_off.npy"))
        self.len = np.load(os.path.join(store_dir, "len.npy")).astype(np.int64)
        self.doc_idx = np.load(os.path.join(store_dir, "doc_idx.npy"))
        self.chunk_idx = np.load(os.path.join(store_dir, "chunk_idx.npy"))
        self.path = os.path.join(store_dir, f"{layout}.bin")
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        self.fd = os.open(self.path, flags)
        self.direct = direct
        self.buf = mmap.mmap(-1, buf_mb << 20)          # 페이지 정렬된 버퍼 (O_DIRECT 조건)
        self.mv = memoryview(self.buf)
        self.n = len(self.off)

    def close(self):
        os.close(self.fd)

    def _blocks(self, cid):
        a = int(self.off[cid])
        return a // BLOCK, (a + int(self.len[cid]) - 1) // BLOCK

    def plan(self, chunk_ids, merge=True, gap_blocks=0):
        """읽기 계획: [(첫 블록, 마지막 블록), ...]. merge=False면 chunk마다 따로 읽는다."""
        if not merge:
            return [self._blocks(c) for c in chunk_ids]
        bs = set()
        for c in chunk_ids:
            s, e = self._blocks(c)
            bs.update(range(s, e + 1))
        bl = sorted(bs)
        if not bl:
            return []
        ext, s, pv = [], bl[0], bl[0]
        for b in bl[1:]:
            if b - pv - 1 <= gap_blocks:      # 붙어 있거나 빈 블록이 gap 이하이면 이어서 읽음
                pv = b
                continue
            ext.append((s, pv))
            s = pv = b
        ext.append((s, pv))
        return ext

    def read(self, chunk_ids, merge=True, gap=0, want_text=True):
        """chunk_ids의 본문을 읽는다. gap은 바이트 (예: 16384), 4KB 단위로 내림."""
        chunk_ids = [int(c) for c in chunk_ids]
        ext = self.plan(chunk_ids, merge, gap // BLOCK)
        need = sum(e - s + 1 for s, e in ext) * BLOCK
        if need > len(self.buf):
            raise RuntimeError(f"buffer too small: need {need} bytes")
        t = time.perf_counter()
        pos, where = 0, []
        for s, e in ext:
            nb = (e - s + 1) * BLOCK
            os.preadv(self.fd, [self.mv[pos:pos + nb]], s * BLOCK)
            where.append((s, e, pos))
            pos += nb
        lat = (time.perf_counter() - t) * 1e6
        texts = None
        if want_text:
            texts = []
            for i, c in enumerate(chunk_ids):
                a, ln = int(self.off[c]), int(self.len[c])
                if merge:
                    b = a // BLOCK
                    s, e, p = next(w for w in where if w[0] <= b <= w[1])
                else:
                    s, e, p = where[i]
                st = p + a - s * BLOCK
                texts.append(bytes(self.mv[st:st + ln]).decode("utf-8"))
        stats = {"n_chunks": len(chunk_ids), "n_reads": len(ext), "bytes_read": pos,
                 "useful_bytes": int(self.len[chunk_ids].sum()) if chunk_ids else 0, "latency_us": round(lat, 1)}
        return texts, stats
