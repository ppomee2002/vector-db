#!/bin/bash
# wait_run_pf.sh는 실행할 스크립트에 인자를 넘기지 못해서 만든 한 줄짜리 실행 파일.
# usage: cd /workspace/junlim && tmux new -d -s c3await 'RUN=run_c3a_confirm.sh LOG=c3a_confirm_run.log bash wait_run_pf.sh > c3a_wait.log 2>&1'
exec bash run_c3a.sh confirm
