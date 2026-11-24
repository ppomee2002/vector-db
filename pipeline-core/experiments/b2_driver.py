"""B2 드라이버 (PLAN_integ.md B2): main.py를 조건 순서 교차로 15번 호출한다.

    /workspace/jaeeun/venv/bin/python experiments/b2_driver.py [--dry-run]

- 실행마다 main.py가 MilvusGuard 사용 중 검사 → load → 측정 → release 직전 확인 후 내가 load한 것만 release.
- 종료 코드가 0이 아니면(사용 중 3, 시간대 밖 7 등) 그 자리에서 멈추고 남은 조건은 실행하지 않는다.
- 예약 시간대 끝까지 MAX_RUN_S가 남지 않으면 새 실행을 시작하지 않는다.
- 결과 목록: logs/b2_manifest.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.run_meta import KST, parse_window  # noqa: E402

PY = "/workspace/jaeeun/venv/bin/python"
RUN = "runs/b2_integ.yaml"
WINDOW = "2026-11-24 17:34~2026-11-24 21:34"
MAX_RUN_S = 240
COND = {
    "1_B0": ["c2.impl=none_ref", "c2.policy=none", "c1.impl=milvus_query_ref", "c3.mode=m3"],
    "2_ALL": [],
    "3_noC2": ["c2.policy=none"],
    "4_noC3": ["c3.mode=m3"],
    "5_C1rand": ["c1.layout=rand"],
}
ORDER = [["1_B0", "2_ALL", "3_noC2", "4_noC3", "5_C1rand"],
         ["5_C1rand", "4_noC3", "3_noC2", "2_ALL", "1_B0"],
         ["2_ALL", "4_noC3", "1_B0", "5_C1rand", "3_noC2"]]


def main() -> int:
    dry = "--dry-run" in sys.argv
    _, end = parse_window(WINDOW, datetime.now(KST))
    man = ROOT / "logs" / "b2_manifest.json"
    done = json.loads(man.read_text()) if man.exists() and not dry else []
    finished = {(d["rep"], d["cond"]) for d in done if d["rc"] == 0}
    for rep, order in enumerate(ORDER, 1):
        for cond in order:
            if (rep, cond) in finished:
                continue
            now = datetime.now(KST)
            if now + timedelta(seconds=MAX_RUN_S) > end:
                print(f"[stop] 예약 시간대 끝({end:%H:%M})까지 {MAX_RUN_S}s 남지 않음 → rep{rep} {cond}부터 실행 안 함")
                return 7
            name = f"b2_r{rep}_{cond}"
            cmd = [PY, "main.py", "--run", RUN, "--set", f"run.name={name}"]
            for s in COND[cond]:
                cmd += ["--set", s]
            print(f"[{now:%H:%M:%S}] rep{rep} {cond}: {' '.join(cmd[2:])}", flush=True)
            if dry:
                continue
            log = ROOT / "logs" / f"{name}.log"
            with open(log, "w") as f:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            txt = log.read_text()
            m = re.search(r"out=(\S+)", txt)
            done.append({"rep": rep, "cond": cond, "rc": rc, "out": m.group(1) if m else None, "log": str(log)})
            man.write_text(json.dumps(done, ensure_ascii=False, indent=1))
            for line in txt.splitlines():
                if any(k in line for k in ("preflight", "load:", "release:", "abort", "n_results", "checks:", "error")):
                    print("   ", line[:300], flush=True)
            if rc != 0:
                print(f"[stop] rep{rep} {cond} 종료 코드 {rc} → 남은 조건 실행 안 함")
                return rc
    print("[done] 15회 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
