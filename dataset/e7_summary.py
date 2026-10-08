import json

d = json.load(open("/mnt/e/论文2/dataset/e7_result.json"))
print("protocol:", d["protocol"])
print("models:", d["models"])
for mask in ("hard", "valid"):
    print("\n======= mask =", mask)
    for r in d[mask]["rows"]:
        print("  seed %d  breizhsr %.3f" % (r["seed"], r["psnr"]))
        for nm in ("A", "C", "D", "nope", "highresnet"):
            k = "minus_" + nm
            if k in r:
                v = r[k]
                print("      -%-11s %+0.3f  CI[%+0.3f,%+0.3f] p=%.3g win=%.0f%% "
                      "dz=%+0.2f n=%d med=%+0.3f"
                      % (nm, v["delta"], v["ci"][0], v["ci"][1], v["p"],
                         100 * v["win"], v["dz"], v["n"], v["median"]))
    print("  -- seed level --")
    for nm, v in d[mask]["seed_level"].items():
        print("   -%-11s mean %+0.3f sd %0.3f p=%.3f seeds=%s %s"
              % (nm, v["mean"], v["sd"], v["p"],
                 [round(x, 3) for x in v["per_seed"]],
                 "same-sign" if v["all_same_sign"] else "SIGN FLIPS"))
