# Shared Read synthetic pilot

Completed cases: 288. Values below are medians across repeats.

Dataset: 131,072 synthetic vectors (64 dimensions), 1 GiB context file, top-10 HNSW retrieval, 16,384 measured queries per case after 1,024 warmup queries. Each chunk is 6 KiB and spans two 4 KiB pages. I/O modes are reported separately; LRU capacity 4,096 pages. This is an implementation pilot, not a real-corpus or same-paper result.

| Trace | I/O | Concurrency | Mode | Repeats | QPS | p95 ms | CPU cores | Logical pages | pread calls | Joined reads | Process read MiB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unique | buffered | 1 | lru | 3 | 5980 | 0.19 | 1.00 | 327680 | 322238 | 0 | 0.0 |
| unique | buffered | 1 | lru_shared | 3 | 5611 | 0.20 | 1.00 | 327680 | 322238 | 0 | 0.0 |
| unique | buffered | 1 | none | 3 | 6648 | 0.17 | 1.00 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 1 | shared | 3 | 6307 | 0.18 | 1.00 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 4 | lru | 3 | 22995 | 0.20 | 3.99 | 327680 | 322244 | 0 | 0.0 |
| unique | buffered | 4 | lru_shared | 3 | 21377 | 0.22 | 3.98 | 327680 | 322247 | 4 | 0.0 |
| unique | buffered | 4 | none | 3 | 26930 | 0.17 | 3.99 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 4 | shared | 3 | 25012 | 0.18 | 3.99 | 327680 | 327679 | 1 | 0.0 |
| unique | buffered | 16 | lru | 3 | 68075 | 0.27 | 15.64 | 327680 | 322243 | 0 | 0.0 |
| unique | buffered | 16 | lru_shared | 3 | 64650 | 0.29 | 15.47 | 327680 | 322230 | 12 | 0.0 |
| unique | buffered | 16 | none | 3 | 91066 | 0.20 | 15.95 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 16 | shared | 3 | 83698 | 0.21 | 15.86 | 327680 | 327670 | 10 | 0.0 |
| unique | buffered | 32 | lru | 3 | 99744 | 0.37 | 29.86 | 327680 | 322269 | 0 | 0.0 |
| unique | buffered | 32 | lru_shared | 3 | 98163 | 0.38 | 30.03 | 327680 | 322262 | 26 | 0.0 |
| unique | buffered | 32 | none | 3 | 152556 | 0.23 | 31.63 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 32 | shared | 3 | 139561 | 0.26 | 30.99 | 327680 | 327664 | 16 | 0.0 |
| unique | buffered | 64 | lru | 3 | 90922 | 1.52 | 38.65 | 327680 | 322291 | 0 | 0.0 |
| unique | buffered | 64 | lru_shared | 3 | 89841 | 1.48 | 38.39 | 327680 | 322243 | 40 | 0.0 |
| unique | buffered | 64 | none | 3 | 176794 | 0.36 | 41.93 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 64 | shared | 3 | 158980 | 1.01 | 41.69 | 327680 | 327641 | 39 | 0.0 |
| unique | buffered | 128 | lru | 3 | 78701 | 3.72 | 39.03 | 327680 | 322242 | 0 | 0.0 |
| unique | buffered | 128 | lru_shared | 3 | 77294 | 3.61 | 39.60 | 327680 | 322213 | 71 | 0.0 |
| unique | buffered | 128 | none | 3 | 178673 | 3.48 | 43.84 | 327680 | 327680 | 0 | 0.0 |
| unique | buffered | 128 | shared | 3 | 143877 | 2.46 | 43.35 | 327680 | 327627 | 53 | 0.0 |
| unique | direct | 1 | lru | 3 | 763 | 1.42 | 0.22 | 327680 | 322238 | 0 | 1258.7 |
| unique | direct | 1 | lru_shared | 3 | 769 | 1.41 | 0.23 | 327680 | 322238 | 0 | 1258.7 |
| unique | direct | 1 | none | 3 | 767 | 1.39 | 0.21 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 1 | shared | 3 | 758 | 1.41 | 0.22 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 4 | lru | 3 | 2989 | 1.46 | 0.88 | 327680 | 322247 | 0 | 1258.8 |
| unique | direct | 4 | lru_shared | 3 | 2993 | 1.46 | 0.87 | 327680 | 322233 | 2 | 1258.7 |
| unique | direct | 4 | none | 3 | 2989 | 1.44 | 0.81 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 4 | shared | 3 | 2967 | 1.44 | 0.83 | 327680 | 327676 | 4 | 1280.0 |
| unique | direct | 16 | lru | 3 | 11263 | 1.56 | 3.67 | 327680 | 322263 | 0 | 1258.8 |
| unique | direct | 16 | lru_shared | 3 | 11075 | 1.59 | 3.81 | 327680 | 322236 | 24 | 1258.7 |
| unique | direct | 16 | none | 3 | 11330 | 1.54 | 3.34 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 16 | shared | 3 | 11287 | 1.54 | 3.43 | 327680 | 327658 | 22 | 1279.9 |
| unique | direct | 32 | lru | 3 | 20429 | 1.76 | 7.55 | 327680 | 322304 | 0 | 1259.0 |
| unique | direct | 32 | lru_shared | 3 | 20200 | 1.78 | 7.81 | 327680 | 322251 | 43 | 1258.8 |
| unique | direct | 32 | none | 3 | 20789 | 1.70 | 6.76 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 32 | shared | 3 | 20485 | 1.74 | 7.02 | 327680 | 327647 | 33 | 1279.9 |
| unique | direct | 64 | lru | 3 | 33925 | 2.15 | 15.20 | 327680 | 322301 | 0 | 1259.0 |
| unique | direct | 64 | lru_shared | 3 | 32619 | 2.25 | 15.61 | 327680 | 322203 | 108 | 1258.6 |
| unique | direct | 64 | none | 3 | 34461 | 2.09 | 13.46 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 64 | shared | 3 | 33917 | 2.15 | 13.78 | 327680 | 327592 | 88 | 1279.7 |
| unique | direct | 128 | lru | 3 | 48448 | 3.19 | 26.73 | 327680 | 322316 | 0 | 1259.0 |
| unique | direct | 128 | lru_shared | 3 | 48405 | 3.22 | 27.24 | 327680 | 322126 | 235 | 1258.3 |
| unique | direct | 128 | none | 3 | 50754 | 2.91 | 23.25 | 327680 | 327680 | 0 | 1280.0 |
| unique | direct | 128 | shared | 3 | 50205 | 2.97 | 23.90 | 327680 | 327507 | 173 | 1279.3 |
| zipf | buffered | 1 | lru | 3 | 13310 | 0.18 | 1.00 | 327680 | 76462 | 0 | 0.0 |
| zipf | buffered | 1 | lru_shared | 3 | 12636 | 0.19 | 1.00 | 327680 | 76462 | 0 | 0.0 |
| zipf | buffered | 1 | none | 3 | 8010 | 0.17 | 1.00 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 1 | shared | 3 | 7533 | 0.17 | 1.00 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 4 | lru | 3 | 49887 | 0.20 | 3.98 | 327680 | 76527 | 0 | 0.0 |
| zipf | buffered | 4 | lru_shared | 3 | 49060 | 0.20 | 3.98 | 327680 | 76443 | 78 | 0.0 |
| zipf | buffered | 4 | none | 3 | 31430 | 0.17 | 4.00 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 4 | shared | 3 | 29915 | 0.17 | 3.98 | 327680 | 327237 | 443 | 0.0 |
| zipf | buffered | 16 | lru | 3 | 154697 | 0.25 | 15.50 | 327680 | 76859 | 0 | 0.0 |
| zipf | buffered | 16 | lru_shared | 3 | 145265 | 0.27 | 15.29 | 327680 | 76415 | 366 | 0.0 |
| zipf | buffered | 16 | none | 3 | 100621 | 0.19 | 15.93 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 16 | shared | 3 | 88638 | 0.22 | 15.40 | 327680 | 322036 | 5644 | 0.0 |
| zipf | buffered | 32 | lru | 3 | 230884 | 0.32 | 30.28 | 327680 | 77122 | 0 | 0.0 |
| zipf | buffered | 32 | lru_shared | 3 | 224678 | 0.34 | 30.14 | 327680 | 76328 | 755 | 0.0 |
| zipf | buffered | 32 | none | 3 | 168417 | 0.23 | 31.19 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 32 | shared | 3 | 142095 | 0.29 | 29.54 | 327680 | 310756 | 16924 | 0.0 |
| zipf | buffered | 64 | lru | 3 | 221771 | 0.86 | 37.20 | 327680 | 77020 | 0 | 0.0 |
| zipf | buffered | 64 | lru_shared | 3 | 217733 | 0.85 | 38.11 | 327680 | 76263 | 715 | 0.0 |
| zipf | buffered | 64 | none | 3 | 196812 | 0.82 | 40.26 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 64 | shared | 3 | 160375 | 0.89 | 38.63 | 327680 | 296807 | 30873 | 0.0 |
| zipf | buffered | 128 | lru | 3 | 183577 | 2.01 | 39.30 | 327680 | 77044 | 0 | 0.0 |
| zipf | buffered | 128 | lru_shared | 3 | 180756 | 1.96 | 39.70 | 327680 | 76172 | 729 | 0.0 |
| zipf | buffered | 128 | none | 3 | 202871 | 2.84 | 43.35 | 327680 | 327680 | 0 | 0.0 |
| zipf | buffered | 128 | shared | 3 | 142817 | 2.27 | 38.46 | 327680 | 288573 | 39107 | 0.0 |
| zipf | direct | 1 | lru | 3 | 2829 | 1.38 | 0.31 | 327680 | 76462 | 0 | 298.7 |
| zipf | direct | 1 | lru_shared | 3 | 2811 | 1.38 | 0.31 | 327680 | 76462 | 0 | 298.7 |
| zipf | direct | 1 | none | 3 | 760 | 1.44 | 0.19 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 1 | shared | 3 | 751 | 1.47 | 0.20 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 4 | lru | 3 | 11218 | 1.39 | 1.20 | 327680 | 76574 | 0 | 299.1 |
| zipf | direct | 4 | lru_shared | 3 | 11211 | 1.40 | 1.18 | 327680 | 76411 | 186 | 298.5 |
| zipf | direct | 4 | none | 3 | 2935 | 1.54 | 0.73 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 4 | shared | 3 | 2937 | 1.50 | 0.78 | 327680 | 323459 | 4221 | 1263.5 |
| zipf | direct | 16 | lru | 3 | 41247 | 1.49 | 5.11 | 327680 | 77231 | 0 | 301.7 |
| zipf | direct | 16 | lru_shared | 3 | 41268 | 1.49 | 5.14 | 327680 | 76296 | 1001 | 298.0 |
| zipf | direct | 16 | none | 3 | 11290 | 1.58 | 3.03 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 16 | shared | 3 | 11225 | 1.58 | 3.14 | 327680 | 308825 | 18855 | 1206.3 |
| zipf | direct | 32 | lru | 3 | 73738 | 1.64 | 10.72 | 327680 | 77901 | 0 | 304.3 |
| zipf | direct | 32 | lru_shared | 3 | 73484 | 1.65 | 10.85 | 327680 | 76109 | 2067 | 297.3 |
| zipf | direct | 32 | none | 3 | 20336 | 1.81 | 6.20 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 32 | shared | 3 | 20702 | 1.73 | 6.30 | 327680 | 292498 | 35182 | 1142.6 |
| zipf | direct | 64 | lru | 3 | 119241 | 1.99 | 20.92 | 327680 | 78961 | 0 | 308.4 |
| zipf | direct | 64 | lru_shared | 3 | 117965 | 2.00 | 21.53 | 327680 | 75673 | 3876 | 295.6 |
| zipf | direct | 64 | none | 3 | 31343 | 2.71 | 11.46 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 64 | shared | 3 | 35569 | 2.05 | 12.76 | 327680 | 267553 | 60127 | 1045.1 |
| zipf | direct | 128 | lru | 3 | 158288 | 2.88 | 33.91 | 327680 | 80144 | 0 | 313.1 |
| zipf | direct | 128 | lru_shared | 3 | 156949 | 2.84 | 35.20 | 327680 | 74940 | 6493 | 292.7 |
| zipf | direct | 128 | none | 3 | 33383 | 7.73 | 12.18 | 327680 | 327680 | 0 | 1280.0 |
| zipf | direct | 128 | shared | 3 | 56960 | 2.61 | 23.18 | 327680 | 238063 | 89617 | 929.9 |

## Observed device counters

Counters cover the timed interval and all activity on each named device. The overlay backing-device identity is unconfirmed. These are Linux block-device completion counters, not SSD-internal NAND read counts. Do not add counters from parent disks and their partitions. Neither pread reduction nor these counters alone establishes that an SSD bottleneck was removed.

| Trace | I/O | Concurrency | Mode | Device | Reads | Read MiB | Read IOPS | Await ms |
|---|---|---:|---|---|---:|---:|---:|---:|
| unique | buffered | 1 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 1 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 4 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 16 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 32 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 64 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| unique | buffered | 128 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 1 | lru | nvme0n1 | 322238 | 1258.7 | 15014 | 0.052 |
| unique | direct | 1 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 1 | lru_shared | nvme0n1 | 322238 | 1258.7 | 15128 | 0.052 |
| unique | direct | 1 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 1 | none | nvme0n1 | 327680 | 1280.0 | 15348 | 0.052 |
| unique | direct | 1 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 1 | shared | nvme0n1 | 327680 | 1280.0 | 15162 | 0.052 |
| unique | direct | 1 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 4 | lru | nvme0n1 | 322247 | 1258.8 | 58790 | 0.053 |
| unique | direct | 4 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 4 | lru_shared | nvme0n1 | 322233 | 1258.7 | 58856 | 0.052 |
| unique | direct | 4 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 4 | none | nvme0n1 | 327680 | 1280.0 | 59783 | 0.053 |
| unique | direct | 4 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 4 | shared | nvme0n1 | 327676 | 1280.0 | 59335 | 0.052 |
| unique | direct | 4 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 16 | lru | nvme0n1 | 322263 | 1258.8 | 221533 | 0.055 |
| unique | direct | 16 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 16 | lru_shared | nvme0n1 | 322247 | 1258.8 | 217819 | 0.056 |
| unique | direct | 16 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 16 | none | nvme0n1 | 327680 | 1280.0 | 226599 | 0.056 |
| unique | direct | 16 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 16 | shared | nvme0n1 | 327658 | 1279.9 | 225720 | 0.055 |
| unique | direct | 16 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 32 | lru | nvme0n1 | 322304 | 1259.0 | 401867 | 0.060 |
| unique | direct | 32 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 32 | lru_shared | nvme0n1 | 322251 | 1258.8 | 397307 | 0.060 |
| unique | direct | 32 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 32 | none | nvme0n1 | 327680 | 1280.0 | 415785 | 0.060 |
| unique | direct | 32 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 32 | shared | nvme0n1 | 327647 | 1279.9 | 409661 | 0.060 |
| unique | direct | 32 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 64 | lru | nvme0n1 | 322301 | 1259.0 | 667448 | 0.068 |
| unique | direct | 64 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 64 | lru_shared | nvme0n1 | 322203 | 1258.6 | 641465 | 0.068 |
| unique | direct | 64 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 64 | none | nvme0n1 | 327680 | 1280.0 | 689210 | 0.069 |
| unique | direct | 64 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 64 | shared | nvme0n1 | 327592 | 1279.7 | 678168 | 0.069 |
| unique | direct | 64 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 128 | lru | nvme0n1 | 322316 | 1259.0 | 953093 | 0.092 |
| unique | direct | 128 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 128 | lru_shared | nvme0n1 | 322126 | 1258.3 | 951683 | 0.092 |
| unique | direct | 128 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 128 | none | nvme0n1 | 327680 | 1280.0 | 1015089 | 0.096 |
| unique | direct | 128 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| unique | direct | 128 | shared | nvme0n1 | 327507 | 1279.3 | 1003571 | 0.096 |
| unique | direct | 128 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 1 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 4 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 16 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 32 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 64 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | lru | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | lru_shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | none | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | shared | nvme0n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | buffered | 128 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 1 | lru | nvme0n1 | 76462 | 298.7 | 13203 | 0.052 |
| zipf | direct | 1 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 1 | lru_shared | nvme0n1 | 76462 | 298.7 | 13120 | 0.052 |
| zipf | direct | 1 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 1 | none | nvme0n1 | 327680 | 1280.0 | 15204 | 0.053 |
| zipf | direct | 1 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 1 | shared | nvme0n1 | 327681 | 1280.0 | 15025 | 0.053 |
| zipf | direct | 1 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 4 | lru | nvme0n1 | 76574 | 299.1 | 52432 | 0.053 |
| zipf | direct | 4 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 4 | lru_shared | nvme0n1 | 76411 | 298.5 | 52278 | 0.053 |
| zipf | direct | 4 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 4 | none | nvme0n1 | 327680 | 1280.0 | 58694 | 0.055 |
| zipf | direct | 4 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 4 | shared | nvme0n1 | 323459 | 1263.5 | 57943 | 0.054 |
| zipf | direct | 4 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 16 | lru | nvme0n1 | 77231 | 301.7 | 194430 | 0.055 |
| zipf | direct | 16 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 16 | lru_shared | nvme0n1 | 76296 | 298.0 | 192183 | 0.054 |
| zipf | direct | 16 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 16 | none | nvme0n1 | 327680 | 1280.0 | 225803 | 0.057 |
| zipf | direct | 16 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 16 | shared | nvme0n1 | 308825 | 1206.3 | 211682 | 0.056 |
| zipf | direct | 16 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 32 | lru | nvme0n1 | 77901 | 304.3 | 350602 | 0.059 |
| zipf | direct | 32 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 32 | lru_shared | nvme0n1 | 76109 | 297.3 | 341157 | 0.058 |
| zipf | direct | 32 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 32 | none | nvme0n1 | 327680 | 1280.0 | 406718 | 0.063 |
| zipf | direct | 32 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 32 | shared | nvme0n1 | 292498 | 1142.6 | 369781 | 0.060 |
| zipf | direct | 32 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 64 | lru | nvme0n1 | 78961 | 308.4 | 574649 | 0.067 |
| zipf | direct | 64 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 64 | lru_shared | nvme0n1 | 75673 | 295.6 | 545047 | 0.065 |
| zipf | direct | 64 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 64 | none | nvme0n1 | 327680 | 1280.0 | 626869 | 0.079 |
| zipf | direct | 64 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 64 | shared | nvme0n1 | 267553 | 1045.1 | 580409 | 0.066 |
| zipf | direct | 64 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 128 | lru | nvme0n1 | 80144 | 313.1 | 775608 | 0.082 |
| zipf | direct | 128 | lru | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 128 | lru_shared | nvme0n1 | 74940 | 292.7 | 717881 | 0.076 |
| zipf | direct | 128 | lru_shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 128 | none | nvme0n1 | 327680 | 1280.0 | 667663 | 0.169 |
| zipf | direct | 128 | none | nvme1n1 | 0 | 0.0 | 0 | 0.000 |
| zipf | direct | 128 | shared | nvme0n1 | 238063 | 929.9 | 827847 | 0.081 |
| zipf | direct | 128 | shared | nvme1n1 | 0 | 0.0 | 0 | 0.000 |

Validation: the runner checks logical_pages = pread_calls + cache_hits + joined_reads and compares every query digest across modes, concurrency levels, and repeats. See COMPLETE for successful completion of the full matrix.
