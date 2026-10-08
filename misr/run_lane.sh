#!/bin/bash
# usage: run_lane.sh <lane 1..6>
set -u
LANE=$1
PY=/home/czy/miniconda3/envs/emssm/bin/python
ROOT=/mnt/e/论文2
LOGD=$ROOT/runs_misr/logs
mkdir -p $LOGD

case "$LANE" in
  1) JOBS="probav A 2000 192|mus2 A 1200 192|mus2 C 1200 192" ;;
  2) JOBS="probav B 2000 192|mus2 B 1200 192|mus2 D 1200 192" ;;
  3) JOBS="mus2 A 1200 192|mus2 C 1200 192" ;;
  4) JOBS="mus2 B 1200 192|mus2 D 1200 192" ;;
  7) JOBS="probav A 8000 192 2027|probav A 8000 192 2028" ;;
  8) JOBS="probav B 8000 192 2027|probav B 8000 192 2028" ;;
  5) while pgrep -f "train.py --data mus2" > /dev/null; do sleep 60; done
     echo "queue: mus2 finished, starting long PROBA-V lane5 $(date +%H:%M:%S)"
     JOBS="probav A 8000 192" ;;
  6) while pgrep -f "train.py --data mus2" > /dev/null; do sleep 60; done
     echo "queue: mus2 finished, starting long PROBA-V lane6 $(date +%H:%M:%S)"
     JOBS="probav B 8000 192" ;;
  *) echo "unknown lane $LANE"; exit 1 ;;
esac

OLDIFS=$IFS
IFS="|"
for j in $JOBS; do
  IFS=" "
  SEED=2026
  read -r DATA ARM ITERS CROP SEED <<< "$j"
  EV=$(( ITERS / 8 )); if [ $EV -lt 500 ]; then EV=500; fi
  NAME=${DATA}_arm${ARM}_s${SEED}
  echo "START $NAME iters=$ITERS $(date +%H:%M:%S)"
  $PY -u $ROOT/misr/train.py --data $DATA --arm $ARM --iters $ITERS --batch 8 --T 15 --c 32 --crop $CROP \
      --eval-every $EV --seed $SEED --out $NAME > $LOGD/$NAME.log 2>&1
  echo "END $NAME rc=$? $(date +%H:%M:%S)"
  IFS="|"
done
echo "LANE$LANE DONE"
