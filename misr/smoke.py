import sys, time
sys.path.insert(0, "/mnt/e/论文2")
import torch, numpy as np
from misr.data import ProbavDataset, MuS2Dataset
from misr.models import build_model, n_params

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
torch.manual_seed(0)

print("== PROBA-V ==")
ds = ProbavDataset(split="val", T=15)
print("scenes:", len(ds))
s = ds[0]
for k, v in s.items():
    print(" ", k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
xb = s["lr"][None].to(dev)
hrb = s["hr"][None].to(dev)
mask = s["sm"][None].to(dev)
q = s["q"][None].to(dev)
qm = s["qm"][None].to(dev)
for arm, (ud, uq) in {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}.items():
    m = build_model(cin=1, c=32, use_dt=ud, use_q=uq).to(dev)
    kw = {"lr": xb}
    if uq: kw.update(q=q, cld=qm)
    if ud: kw.update(dt=torch.zeros(1, 15, device=dev))
    t0 = time.time(); out = m(**kw)
    loss = ((out - hrb).abs() * mask).mean(); loss.backward()
    print(" arm", arm, "params", n_params(m), "out", tuple(out.shape), "loss %.4f" % float(loss), "%.2fs" % (time.time() - t0))

print("== MuS2 ==")
ds2 = MuS2Dataset(split="val", T=15, crop=192)
print("scenes:", len(ds2))
s2 = ds2[0]
for k, v in s2.items():
    if torch.is_tensor(v):
        print(" ", k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
    else:
        print(" ", k, v)
m = build_model(cin=3, c=32, use_dt=True, use_q=True).to(dev)
xb = s2["lr"][None].to(dev); hrb = s2["hr"][None].to(dev)
out = m(lr=xb, q=s2["q"][None].to(dev), cld=s2["cld"][None].to(dev), dt=s2["dt"][None].to(dev))
print(" out", tuple(out.shape), "hr", tuple(hrb.shape), "params", n_params(m))
print("SMOKE_OK")
