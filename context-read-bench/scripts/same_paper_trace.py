#!/usr/bin/env python3
"""Generate a locality scenario from real query-to-paper metadata, without inventing it."""
import argparse
import collections
import json
import random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--metadata", type=Path, required=True, help="JSONL: query_id (fvecs row), paper_id")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=20261103)
a = p.parse_args()
papers = collections.defaultdict(list)
seen = set()
for line in a.metadata.read_text().splitlines():
    row = json.loads(line)
    q = row["query_id"]
    assert isinstance(q, int) and q >= 0 and q not in seen
    assert isinstance(row["paper_id"], str) and row["paper_id"]
    seen.add(q)
    papers[row["paper_id"]].append(q)
assert papers and any(len(v) > 1 for v in papers.values()), "No same-paper question groups"
rng = random.Random(a.seed)
groups = list(papers.values())
rng.shuffle(groups)
for group in groups:
    rng.shuffle(group)
with a.output.open("x") as f:
    for group in groups:
        f.writelines(f"{q}\n" for q in group)
print(f"{len(seen)} distinct queries across {len(papers)} papers; grouped locality scenario, not an observed production trace")
