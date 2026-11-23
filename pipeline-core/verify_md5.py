"""원본 데이터 전체 md5 검사 (별도 명령, 하루 한 번). 측정 실행(main.py)과 분리한다.

    cd /workspace/jaeeun/pipeline-core
    nohup /workspace/jaeeun/venv/bin/python verify_md5.py > logs/verify_md5.log 2>&1 &

- config.yaml dataset.files의 모든 파일(파이프라인이 쓰지 않는 대용량 chunk_vecs 포함)을 전부 읽어 md5를 계산하고
  config 값과 대조한다. 결과를 캐시(dataset.integrity.cache, 기본 cache/md5_cache.json)에 (경로, 크기, 수정 시각, md5)로 쓴다.
- main.py는 이 캐시로 크기·수정 시각만 확인한다(바뀐 파일만 다시 계산).
- cold / cache_constrained 실험 직전에는 실행하지 말 것(수 GB를 읽어 페이지 캐시가 바뀜). 그 조건의 overlay를 주면 거부한다.
- 내 실행 잠금(locks/milvus_run.lock)을 잡고 돌므로 내 측정 실행과 겹치지 않는다.
원본 파일은 읽기만 한다(D2). 종료 코드: 0 전부 일치 / 1 불일치·오류 / 7 거부(잠금, cold 조건).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.chunk_meta import ChunkMetaError, FileChunkMetaLoader  # noqa: E402
from modules.config_loader import load_config  # noqa: E402
from modules.run_meta import RunLock, now_kst  # noqa: E402

LOCK_FILE = ROOT / "locks" / "milvus_run.lock"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="config overlay (경로를 바꿀 때만)")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.run, args.set)
    cc = cfg["experiment"]["cache_condition"]
    if cc in ("cold", "cache_constrained"):
        print(f"[거부] experiment.cache_condition={cc}: cold/cache_constrained 실험 직전에는 전체 md5를 돌리지 않는다")
        return 7
    lock = RunLock(LOCK_FILE)
    try:
        lock.acquire({"run": "verify_md5"})
    except BlockingIOError as e:
        print("[거부]", e)
        return 7
    try:
        print(f"[{now_kst()}] 전체 md5 검사 시작")
        loader = FileChunkMetaLoader(cfg, md5_mode="full")
        try:
            loader._md5()
            rc = 0
        except ChunkMetaError as e:
            print("[불일치]", e)
            rc = 1
        for name, r in loader.report["md5"].items():
            print(f"  {name:12s} {r.get('status'):10s} ok={r.get('ok')} {r.get('got')} {r.get('size')}B {r.get('sec')}s")
        print(json.dumps(loader.report.get("full_verify"), ensure_ascii=False))
        print(f"[{now_kst()}] 끝 → 캐시 {loader.cache_path}")
        return rc
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
