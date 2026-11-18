#!/usr/bin/env python3
# usage: python set_mmap.py on|off [--coll nq_seq]
# 우리 컬렉션(junlim DB)의 벡터 인덱스와 벡터 필드를 mmap으로 바꾼다(on) / 되돌린다(off).
# 컬렉션을 내렸다가(release) 설정을 바꾸고 다시 올린다(load). 다른 DB와 컬렉션은 건드리지 않는다.
import time
import argparse
from pymilvus import MilvusClient

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["on", "off"])
p.add_argument("--coll", default="nq_seq")
a = p.parse_args()
v = a.mode == "on"

c = MilvusClient(uri="http://localhost:19530", db_name="junlim")
if not c.has_collection(a.coll):
    raise SystemExit(f"no collection {a.coll} in junlim")
print("index before:", c.describe_index(a.coll, "vec"), flush=True)

t = time.time()
c.release_collection(a.coll)
print(f"released {a.coll} in {time.time() - t:.1f}s", flush=True)
ok = True
try:
    c.alter_index_properties(a.coll, "vec", properties={"mmap.enabled": v})
    print("index mmap.enabled ->", v, flush=True)
    c.alter_collection_field(a.coll, field_name="vec", field_params={"mmap.enabled": v})
    print("vec field mmap.enabled ->", v, flush=True)
except Exception as e:  # 실패해도 컬렉션은 다시 올려 둔다
    ok = False
    print("ALTER FAILED:", type(e).__name__, e, flush=True)
finally:
    t = time.time()
    c.load_collection(a.coll)
    print(f"loaded {a.coll} in {time.time() - t:.1f}s", c.get_load_state(a.coll), flush=True)

print("index after:", c.describe_index(a.coll, "vec"), flush=True)
fields = {f["name"]: f.get("params", {}) for f in c.describe_collection(a.coll)["fields"]}
print("field params:", fields, flush=True)
print("DONE" if ok else "DONE WITH ERRORS")
