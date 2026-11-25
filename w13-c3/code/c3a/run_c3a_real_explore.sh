#!/bin/bash
# 실제 NQ 질문만(QSET=real, 3,452개)으로 탐색 세션을 돌리는 한 줄짜리 실행 파일 (11/22 추가, run_c3a.sh는 고치지 않음).
# wait_run_pf.sh는 실행할 스크립트에 인자·환경 변수를 넘기지 못해서 여기서 정한다. 결과는 새 폴더 c3a_out_real에만 쓴다.
# usage: cd /workspace/junlim && tmux new -d -s c3rwait 'RUN=run_c3a_real_explore.sh LOG=c3a_real_run.log bash wait_run_pf.sh > c3a_real_wait.log 2>&1'
# 판정: python judge_c3a_real.py --outdir /workspace/junlim/c3a_out_real (규칙은 그 파일 맨 위, 측정 전에 고정)
export OUT=/workspace/junlim/c3a_out_real QSET=real
exec bash run_c3a.sh explore
