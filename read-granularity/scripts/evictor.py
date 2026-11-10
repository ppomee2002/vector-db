#!/usr/bin/env python3
"""Keep payload files out of the page cache: posix_fadvise(DONTNEED) every --ms until killed."""
import os, sys, time, signal
root, ms = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 20
fds = []
for r, _, fs in os.walk(root):
    if "payload_storage" in r:
        for f in fs:
            if os.environ.get("PAGES_ONLY") == "1" and not f.startswith("page_"): continue
            try: fds.append(os.open(os.path.join(r, f), os.O_RDONLY))
            except OSError: pass
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
while True:
    for fd in fds: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(ms / 1e3)
