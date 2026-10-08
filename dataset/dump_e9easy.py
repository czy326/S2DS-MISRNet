import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("/mnt/e/论文2/dataset/analysis_e9easy.json"))
for s in ["s2026", "s2027", "s2028"]:
    print(s, {k: round(v, 3) for k, v in d["per_seed"][s]["means"].items()})
for k in ["dt_effect", "q_effect", "C_minus_A", "D_minus_B",
          "B_minus_A", "D_minus_C", "interaction"]:
    for s in ["s2026", "s2027", "s2028"]:
        v = d["per_seed"][s][k]
        print("%-12s %s mean=%+.3f p=%.4g win=%.3f ci=[%+.3f,%+.3f] n=%d"
              % (k, s, v["mean"], v["p"], v["win"], v["ci95"][0], v["ci95"][1], v["n"]))
