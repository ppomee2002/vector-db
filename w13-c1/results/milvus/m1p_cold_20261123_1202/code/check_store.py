# check_store.py: 저장소 검증  김지현 2024134018
#   1) 두 배치의 모든 chunk 본문이 chunks.jsonl과 똑같은지 전수 비교
#   2) Store.read()가 merge 켬/끔, gap 여러 값에서 같은 본문을 돌려주는지 (무작위 쿼리 2,000개)
#   3) 앱이 센 읽기 횟수·바이트와 /proc/diskstats 비교 (비율이 1.0 근처면 계측이 맞음)
# 결과: {WORK}/results/check_store.json
import os, sys, json, argparse, subprocess
import numpy as np
from common import CHUNKS, WORK, log, open_read, write_json, out_dir, now_kst
from c1_store import Store, LAYOUTS


def device_of(path):
    try:
        src = subprocess.check_output(["df", "--output=source", path]).decode().split()[1]
        dev = os.path.basename(src)
        for l in open("/proc/diskstats"):
            if l.split()[2] == dev:
                return dev
    except Exception:
        pass
    return None


def diskstats(dev):
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == dev:
            return int(f[3]), int(f[5]) * 512
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(WORK, "store"))
    ap.add_argument("--no-direct", action="store_true", help="O_DIRECT 없이 (로컬 테스트용)")
    a = ap.parse_args()
    res = {"checked_at_kst": now_kst()}

    texts = [json.loads(l)["text"] for l in open_read(CHUNKS, encoding="utf-8")]
    n = len(texts)
    for lay in LAYOUTS:
        off = np.load(os.path.join(a.store, f"{lay}_off.npy"))
        ln = np.load(os.path.join(a.store, "len.npy"))
        mm = np.memmap(os.path.join(a.store, f"{lay}.bin"), dtype=np.uint8, mode="r")
        bad = [i for i in range(n) if mm[off[i]:off[i] + ln[i]].tobytes().decode("utf-8") != texts[i]]
        res[f"full_compare_{lay}"] = {"chunks": n, "mismatch": len(bad), "first_bad": bad[:5]}
        log(f"[1] {lay}: full compare {n} chunks, mismatch {len(bad)}")
        del mm

    rng = np.random.default_rng(7)
    doc = np.load(os.path.join(a.store, "doc_idx.npy"))
    for lay in LAYOUTS:
        st = Store(a.store, lay, direct=not a.no_direct)
        bad = 0
        for _ in range(2000):
            k = int(rng.integers(1, 40))
            ids = rng.choice(n, k, replace=False)
            if rng.random() < 0.5:                       # 같은 문서 이웃도 섞어서 병합 경로를 시험
                c = int(ids[0])
                ids = np.unique(np.r_[ids, [x for x in (c - 1, c + 1) if 0 <= x < n and doc[x] == doc[c]]])
            ref = [texts[int(i)] for i in ids]
            for merge, gap in ((False, 0), (True, 0), (True, 16384), (True, 65536)):
                got, _ = st.read(ids, merge=merge, gap=gap)
                bad += got != ref
        st.close()
        res[f"read_check_{lay}"] = {"queries": 2000, "modes": 4, "mismatch": bad}
        log(f"[2] {lay}: read() check 2000 queries x 4 modes, mismatch {bad}")

    dev = device_of(os.path.join(a.store, "doc.bin"))
    if dev and not a.no_direct:
        st = Store(a.store, "rand", direct=True)
        d0 = diskstats(dev); ar = ab = 0
        for _ in range(3000):
            _, s = st.read(rng.choice(n, 10, replace=False), merge=False, want_text=False)
            ar += s["n_reads"]; ab += s["bytes_read"]
        d1 = diskstats(dev); st.close()
        dr, db = d1[0] - d0[0], d1[1] - d0[1]
        res["diskstats_check"] = {"device": dev, "app_reads": ar, "disk_reads": dr, "read_ratio": round(dr / ar, 3),
                                  "app_bytes": ab, "disk_bytes": db, "byte_ratio": round(db / ab, 3),
                                  "note": "다른 사람이 같은 장치를 읽으면 비율이 1보다 커진다"}
        log(f"[3] device {dev}: disk/app reads {dr}/{ar} = {dr / ar:.3f}, bytes ratio {db / ab:.3f}")
    else:
        res["diskstats_check"] = {"skipped": f"device={dev}, direct={not a.no_direct}"}
        log("[3] diskstats check skipped")

    ok = all(v.get("mismatch", 0) == 0 for k, v in res.items() if isinstance(v, dict))
    res["all_ok"] = ok
    write_json(os.path.join(out_dir("results"), "check_store.json"), res)
    log("CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
