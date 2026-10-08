import json

d = json.load(open("/mnt/e/论文2/dataset/e8_result.json"))
for mask in ("hard", "valid"):
    print("===", mask)
    for k, v in d[mask].items():
        sl = v["seed_level"]
        print("  %-5s %-42s vsB mean=%+.3f sd=%.3f p=%.3f seeds=%s -> %s"
              % (k, v["desc"], sl["mean"], sl["sd"], sl["p"],
                 [round(s, 3) for s in sl["per_seed"]], sl["verdict"]))
        ma = sl["minus_A"]
        print("        vsA mean=%+.3f sd=%.3f p=%.3f seeds=%s conf=%s"
              % (ma["mean"], ma["sd"], ma["p"],
                 [round(s, 3) for s in ma["per_seed"]], ma["confounded"]))
