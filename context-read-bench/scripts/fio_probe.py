#!/usr/bin/env python3
"""Read-only tests of an existing fully populated file; never targets a raw device."""
import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--file", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seconds", type=int, default=10)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--concurrency", default="1,4,16,32,64,128")
a = p.parse_args()
file = a.file.resolve(strict=True)
st = file.stat()
assert st.st_size >= 4096 and st.st_size % 4096 == 0
assert st.st_blocks*512 >= st.st_size, "File appears sparse or incompletely allocated"
a.out.mkdir(parents=True, exist_ok=False)
cases = [(rep, engine, c) for rep in range(a.repeats)
         for engine in ("psync", "libaio") for c in map(int, a.concurrency.split(","))]
random.Random(20261103).shuffle(cases)
rows = []
for rep, engine, c in cases:
    output = a.out / f"{engine}-c{c}-r{rep}.json"
    cmd = ["fio", "--readonly", "--name=4k-random-read", f"--filename={file}",
           "--allow_file_create=0", "--rw=randread", "--bs=4096", "--direct=1",
           f"--ioengine={engine}", f"--numjobs={c if engine=='psync' else 1}",
           f"--iodepth={1 if engine=='psync' else c}", "--thread=1",
           "--time_based=1", f"--runtime={a.seconds}", "--ramp_time=2",
           "--randrepeat=1", f"--randseed={20261103+rep}", "--norandommap=1",
           "--group_reporting=1", "--output-format=json", f"--output={output}"]
    print(f"fio {engine} concurrency={c} repeat={rep}", flush=True)
    subprocess.run(cmd, check=True)
    result = json.loads(output.read_text())
    job = result["jobs"][0]
    assert job["error"] == 0
    read = job["read"]
    pct = read["clat_ns"]["percentile"]
    rows.append(dict(engine=engine, concurrency=c, repeat=rep,
                     iops=read["iops"], p95_us=pct["95.000000"]/1000,
                     p99_us=pct["99.000000"]/1000,
                     user_cpu=job["usr_cpu"], system_cpu=job["sys_cpu"]))
    with (a.out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
