#!/bin/bash
# Live progress dashboard (refresh every 15 s). Ctrl-C to quit.
PY=/home/czy/miniconda3/envs/emssm/bin/python
exec $PY -u /mnt/e/论文2/dataset/watch.py --interval "${1:-15}"
