#!/bin/bash
# Stage-1 screening: PROBA-V (A,B) + MuS2 (A,B,C,D), 2 concurrent
set -u
PY=/home/czy/miniconda3/envs/emssm/bin/python
ROOT=/mnt/e/论文2
LOGD=$ROOT/runs_misr/logs
mkdir -p $LOGD
run() {
  local data=$1 arm=$2 iters=$3 crop=$4
  local name="${data}_arm${arm}"
  echo "START $name"
  $PY -u $ROOT/misr/train.py --data $data --arm $arm --iters $iters --batch 8 --T 15 --c 32 --crop $crop \
      --eval-every 500 --out $name > $LOGD/$name.log 2>&1
  echo "END $name rc=$?"
}
export -f run
echo "PROBAV_A 2000" > $LOGD/jobs.txt
cat > $LOGD/jobs.txt << "JEOF"
probav A 2000 192
probav B 2000 192
mus2 A 1200 192
mus2 B 1200 192
mus2 C 1200 192
mus2 D 1200 192
JEOF
cat $LOGD/jobs.txt | xargs -n 3 -P 2 bash -c 'run "$@"' _
echo ALL_JOBS_DONE
