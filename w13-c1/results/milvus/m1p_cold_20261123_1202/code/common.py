# common.py: 경로 설정과 안전장치 (김지현 2024134018, 캡스톤 VectorDB팀 C1)
#
# 규칙
#   - 다른 팀원 폴더(/workspace/junlim, /workspace/jaeeun 등)의 파일은 읽기만 한다.
#   - 이 코드가 쓰는 파일은 전부 OUT_ROOT(/workspace/jihyeon 아래) 안에만 만든다.
#     safe_path()를 거치지 않은 경로에는 쓰지 않는다.
import os, sys, time, json, hashlib

# 쓰기 허용 범위. 이 밖으로는 절대 쓰지 않는다.
OUT_ROOT = os.path.realpath(os.environ.get("C1_OUT_ROOT", "/workspace/jihyeon"))
WORK = os.environ.get("C1_WORK", os.path.join(OUT_ROOT, "c1_w13"))

# 입력 (읽기만)
CHUNKS = os.environ.get("C1_CHUNKS", "/workspace/junlim/data_nq/chunks.jsonl")
ANN_CSV = os.environ.get("C1_ANN", "/workspace/jaeeun/sel_fetch/data/ann_top100_real.csv")
RERANK_CSV = os.environ.get("C1_RERANK", "/workspace/jaeeun/sel_fetch/data/rerank_top100_real.csv")
QRELS = os.environ.get("C1_QRELS", "/workspace/jaeeun/sel_fetch/data/qrels_chunk.tsv")

BLOCK = 4096


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def now_kst():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def safe_path(p):
    """쓰기 전에 반드시 거친다. OUT_ROOT 밖이면 바로 멈춘다."""
    rp = os.path.realpath(p)
    if rp != OUT_ROOT and not rp.startswith(OUT_ROOT + os.sep):
        sys.exit(f"[중단] 쓰기 경로가 내 폴더({OUT_ROOT}) 밖입니다: {rp}")
    return rp


def out_dir(*parts):
    d = safe_path(os.path.join(WORK, *parts))
    os.makedirs(d, exist_ok=True)
    return d


def open_read(p, mode="r", **kw):
    """입력 파일은 이 함수로만 연다 (읽기 모드만 허용)."""
    if any(ch in mode for ch in "wax+"):
        raise ValueError("입력 파일은 읽기 모드로만 열 수 있습니다")
    return open(p, mode, **kw)


def file_info(p, md5=True):
    """입력 파일이 나중에 바뀌었는지 확인할 수 있게 크기, 수정 시각, md5를 기록한다."""
    st = os.stat(p)
    info = {"path": p, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
    if md5:
        h = hashlib.md5()
        with open_read(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        info["md5"] = h.hexdigest()
    return info


def write_json(path, obj):
    with open(safe_path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
