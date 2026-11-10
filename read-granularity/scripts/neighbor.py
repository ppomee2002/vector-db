#!/usr/bin/env python3
"""Co-located I/O tenant: open-loop 4 KiB O_DIRECT random reads at a fixed rate on the same disk.
   neighbor.py <file> <rate_iops> <threads> <out.json> [cgroup_dir]
Latency is measured from the scheduled issue time, so time spent waiting behind other I/O counts.
Writes a summary on SIGTERM."""
import os, sys, time, mmap, random, signal, threading, json
path, rate, nth, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if len(sys.argv) > 5:
    with open(os.path.join(sys.argv[5], "cgroup.procs"), "w") as f: f.write(str(os.getpid()))
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
nblk = os.fstat(fd).st_size // 4096
lats, lock, stop = [], threading.Lock(), threading.Event()
t_start = time.monotonic()
def worker(i):
    buf = mmap.mmap(-1, 4096); rnd = random.Random(i); per = nth / rate
    nxt = time.monotonic() + rnd.random() * per
    while not stop.is_set():
        d = nxt - time.monotonic()
        if d > 0: time.sleep(d)
        os.preadv(fd, [buf], rnd.randrange(nblk) * 4096)
        lats.append(time.monotonic() - nxt); nxt += per  # list.append is atomic under the GIL
ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nth)]
for t in ts: t.start()
def finish(*_):
    stop.set()  # no join: a worker may sit in a throttled read; take what has completed
    el = time.monotonic() - t_start; l = sorted(list(lats)); q = lambda p: l[min(len(l) - 1, int(p * len(l)))] * 1e3 if l else None
    json.dump({"target_iops": rate, "done_iops": len(l) / el, "p50_ms": q(.5), "p99_ms": q(.99), "p999_ms": q(.999), "n": len(l)}, open(out, "w"))
    os._exit(0)
signal.signal(signal.SIGTERM, finish); signal.signal(signal.SIGINT, finish)
while True: time.sleep(1)
