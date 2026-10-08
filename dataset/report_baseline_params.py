# -*- coding: utf-8 -*-
"""One-off: record the parameter count of every model that enters the comparison table."""
import sys
sys.path.insert(0, "/mnt/e/论文2")
from misr.models import build_model, n_params
from misr.models_e1 import build_model_e1, n_params as np1
from misr.baselines_s2ds import build_baseline, n_params as np2

KW = dict(att_mode="pixel", dt_mode="gate", head_init=0.0)
print("%-14s %8s" % ("model", "params"))
for arm, kw in [("A", dict(use_dt=False, use_q=False)),
                ("B", dict(use_dt=False, use_q=True)),
                ("C", dict(use_dt=True, use_q=False)),
                ("D", dict(use_dt=True, use_q=True))]:
    print("%-14s %8d" % ("arm" + arm, n_params(build_model(cin=4, c=32, scale=4, **kw, **KW))))
for arm in ["B", "B1", "B2"]:
    print("%-14s %8d" % ("e1:" + arm, np1(build_model_e1(cin=4, c=32, scale=4, arm=arm, **KW))))
print("%-14s %8d" % ("highresnet c36", np2(build_baseline("highresnet", cin=4, c=36, scale=4))))
print("%-14s %8d" % ("rams c32", np2(build_baseline("rams", cin=4, c=32, scale=4))))
