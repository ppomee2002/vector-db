#!/usr/bin/env bash
cd ~/gran
sudo -n kill -KILL $(pgrep -f "[g]ran/neighbor.py") 2>/dev/null
sudo -n kill $(pgrep -f "[g]ran/evictor.py") 2>/dev/null
sleep 1
./qgran_ctl.sh stop
rm -f res-C/nq-g262144-n1000*
sudo -n python3 neighbor.py qdrant/target/perf/qdrant 300 4 /tmp/nt2.json & sleep 3
sudo -n kill -TERM $(pgrep -f "^python3 neighbor.py qdrant"); sleep 1; cat /tmp/nt2.json; echo
(setsid nohup env DSS=nq REQ=1500 WARM=200 NEIGHS=1000 ./qgran_sweep.sh C ~/gran/res-C > sweepC2.log 2>&1 < /dev/null &)
echo restarted
