#!/usr/bin/env python3
"""Summarize completed matrix cases without treating syscall counts as device I/O."""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
rows = [json.loads(f.read_text()) for f in a.directory.glob('*.json') if f.name != 'manifest.json']
groups = {}
for row in rows:
    groups.setdefault((row['trace'], row['io'], row['concurrency'], row['mode']), []).append(row)

def median(group, key):
    return statistics.median(r[key] for r in group)

lines = [
    '# Shared Read synthetic pilot', '',
    f'Completed cases: {len(rows)}. Values below are medians across repeats.', '',
    'Dataset: 131,072 synthetic vectors (64 dimensions), 1 GiB context file, '
    'top-10 HNSW retrieval, 16,384 measured queries per case after 1,024 warmup queries. '
    'Each chunk is 6 KiB and spans two 4 KiB pages. I/O modes are reported separately; LRU capacity 4,096 pages. '
    'This is an implementation pilot, not a real-corpus or same-paper result.', '',
    '| Trace | I/O | Concurrency | Mode | Repeats | QPS | p95 ms | CPU cores | Logical pages | pread calls | Joined reads | Process read MiB |',
    '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for (trace, io, concurrency, mode), group in sorted(groups.items()):
    lines.append(f'| {trace} | {io} | {concurrency} | {mode} | {len(group)} | '
                 f'{median(group,"qps"):.0f} | {median(group,"p95_us")/1000:.2f} | '
                 f'{median(group,"cpu_cores"):.2f} | {median(group,"logical_pages"):.0f} | '
                 f'{median(group,"pread_calls"):.0f} | {median(group,"joined_reads"):.0f} | '
                 f'{median(group,"process_read_bytes")/1048576:.1f} |')
lines += ['', '## Observed device counters', '',
          'Counters cover the timed interval and all activity on each named device. '
          'The overlay backing-device identity is unconfirmed. These are Linux block-device '
          'completion counters, not SSD-internal NAND read counts. Do not add counters from '
          'parent disks and their partitions. Neither pread reduction nor these counters alone '
          'establishes that an SSD bottleneck was removed.', '',
          '| Trace | I/O | Concurrency | Mode | Device | Reads | Read MiB | Read IOPS | Await ms |',
          '|---|---|---:|---|---|---:|---:|---:|---:|']
for (trace, io, concurrency, mode), group in sorted(groups.items()):
    for device in sorted(group[0].get('observed_devices', {})):
        observations = [r['observed_devices'][device] for r in group]
        lines.append(f'| {trace} | {io} | {concurrency} | {mode} | {device} | '
                     f'{median(observations,"reads"):.0f} | {median(observations,"read_bytes")/1048576:.1f} | '
                     f'{median(observations,"read_iops"):.0f} | {median(observations,"read_await_ms"):.3f} |')
lines += ['', 'Validation: the runner checks logical_pages = pread_calls + cache_hits + joined_reads '
          'and compares every query digest across modes, concurrency levels, and repeats. '
          'See COMPLETE for successful completion of the full matrix.', '']
report = a.directory/'REPORT.md'
report.write_text('\n'.join(lines))
print(report)
