"""측정 중 기록기 (14주차 A 실험용, rec_c3p2.py 2d381784 복사본 + diskstats·cpu.stat 열). 아무것도 바꾸지 않고 읽기만 한다.

interval초마다 한 줄: 시각, 실행 이름(label 파일), Milvus RSS·누적 CPU초(localhost:9091/metrics),
main.py 프로세스의 pid·VmRSS·VmHWM·누적 CPU초(/proc). main.py는 실행마다 새로 뜨므로 pid를 매번 다시 찾는다.
    python rec_c3p2.py --out rec.csv --label-file label --interval 2
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def milvus(url: str):
    rss = cpu = None
    try:
        txt = urllib.request.urlopen(url, timeout=1.5).read().decode("utf-8", "replace")
        for line in txt.splitlines():
            if line.startswith("process_resident_memory_bytes "):
                rss = float(line.split()[1])
            elif line.startswith("process_cpu_seconds_total "):
                cpu = float(line.split()[1])
    except Exception:  # noqa: BLE001 - 기록기는 멈추지 않는다
        pass
    return rss, cpu


def find_main_pid():
    """내 계정의 main.py 프로세스 중 가장 최근 것."""
    me, best = os.getuid(), None
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != me:
                continue
            cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            if any(c.endswith(b"main.py") for c in cmd) and b"python" in cmd[0]:
                start = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[19])
                if best is None or start > best[1]:
                    best = (int(d), start)
        except (OSError, ValueError, IndexError):
            continue
    return best[0] if best else None


def proc_info(pid):
    rss = hwm = cpu = None
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) * 1024
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        cpu = (int(f[11]) + int(f[12])) / TICK   # utime + stime
    except (OSError, ValueError, IndexError):
        pass
    return rss, hwm, cpu


def diskstats():
    """장치별 누적 읽은 섹터(512B). 'dev:sectors' 공백 구분. 어느 장치가 Milvus 저장소인지는 분석에서 본다."""
    out = []
    try:
        for line in open("/proc/diskstats"):
            f = line.split()
            if len(f) >= 6 and int(f[5]) > 0:
                out.append(f"{f[2]}:{f[5]}")
    except (OSError, ValueError):
        pass
    return " ".join(out)


def cpu_stat():
    """우리 컨테이너 cgroup cpu.stat의 nr_throttled, throttled_usec (위쪽 cgroup은 안 보임)."""
    nt = tu = None
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            if k == "nr_throttled":
                nt = int(v)
            elif k == "throttled_usec":
                tu = int(v)
    except (OSError, ValueError):
        pass
    return nt, tu


def other_python():
    """다른 계정(root 제외)의 python 프로세스 'uid:pid' 목록. 기록만(판정에 쓰지 않음)."""
    me, out = os.getuid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            uid = os.stat(f"/proc/{d}").st_uid
            if uid in (me, 0):
                continue
            cmd0 = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")[0]
            if b"python" in cmd0:
                out.append(f"{uid}:{d}")
        except (OSError, IndexError):
            continue
    return " ".join(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-file", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--metrics", default="http://localhost:9091/metrics")
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    with open(a.out, "a", buffering=1) as f:
        if new:
            f.write("t_epoch,time,label,milvus_rss_bytes,milvus_cpu_s,client_pid,client_rss_bytes,client_hwm_bytes,client_cpu_s,"
                    "other_python,nr_throttled,throttled_usec,diskstats_read_sectors\n")
        while True:
            t = time.time()
            try:
                label = open(a.label_file).read().strip()
            except OSError:
                label = ""
            mr, mc = milvus(a.metrics)
            pid = find_main_pid()
            cr, ch, cc = proc_info(pid) if pid else (None, None, None)
            nt, tu = cpu_stat()
            row = [f"{t:.2f}", time.strftime("%H:%M:%S"), label, mr, mc, pid, cr, ch, cc, other_python(), nt, tu, diskstats()]
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
