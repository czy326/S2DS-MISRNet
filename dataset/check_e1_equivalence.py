# -*- coding: utf-8 -*-
"""Guard for E1: is build_model_e1(arm="B") the SAME model as the main 2x2 arm B?

E1 reuses the already-trained arm B (s2dsV2_armB_*) as its reference.  That is only
legitimate if the copy in misr/models_e1.py, with all three q routes enabled, is
functionally identical to misr/models.py with use_dt=False, use_q=True.  This script
checks three things:

  1. parameter count and the exact state_dict keys
  2. numerical equality of every parameter under the same RNG seed
  3. equality of a forward pass on the same random input

It also prints the parameter count of B1 and B2, which is the caveat to report with the
ablation: dropping a route drops parameters, so E1 is a route ablation and not a
parameter-matched one.
"""
import sys
import torch

sys.path.insert(0, "/mnt/e/论文2")
sys.path.insert(0, "/mnt/e/论文2/dataset")
from misr.models import build_model, n_params                       # noqa: E402
from misr.models_e1 import build_model_e1, n_params as np_e1, Q_ROUTES  # noqa: E402

KW = dict(cin=4, c=32, scale=4, att_mode="pixel", dt_mode="gate", head_init=0.0)


def fresh(builder, **kw):
    torch.manual_seed(0)
    return builder(**kw)


def main():
    m_main = fresh(lambda **k: build_model(use_dt=False, use_q=True, **k), **KW)
    m_e1 = fresh(lambda **k: build_model_e1(arm="B", **k), **KW)

    ok = True
    print("arm B  main params=%d   e1 params=%d" % (n_params(m_main), np_e1(m_e1)))
    if n_params(m_main) != np_e1(m_e1):
        ok = False
        print("  !! parameter count differs")

    k1, k2 = list(m_main.state_dict()), list(m_e1.state_dict())
    print("state_dict keys identical: %s (%d keys)" % (k1 == k2, len(k1)))
    if k1 != k2:
        ok = False
        print("  only in main:", sorted(set(k1) - set(k2)))
        print("  only in e1  :", sorted(set(k2) - set(k1)))
    else:
        worst = 0.0
        for k in k1:
            worst = max(worst, float((m_main.state_dict()[k] - m_e1.state_dict()[k])
                                     .abs().max()))
        print("max |param difference| = %.3e" % worst)
        if worst > 0:
            ok = False

    torch.manual_seed(123)
    B, T, C, H, W = 2, 12, 4, 48, 48
    lr = torch.rand(B, T, C, H, W)
    cld = torch.rand(B, T, 1, H, W)
    dt = torch.randn(B, T) * 30.0
    q = torch.rand(B, T, 1)
    with torch.no_grad():
        o1 = m_main(lr=lr, q=q, cld=cld, dt=dt)
        o2 = m_e1(lr=lr, q=q, cld=cld, dt=dt)
    d = float((o1 - o2).abs().max())
    print("max |output difference| = %.3e" % d)
    if d > 1e-6:
        ok = False

    print("\nparameter count per E1 arm (route ablation is NOT parameter-matched):")
    for arm in sorted(Q_ROUTES):
        m = fresh(lambda **k: build_model_e1(arm=arm, **k), **KW)
        print("  %-3s %-52s params=%d" % (arm, str(Q_ROUTES[arm]), np_e1(m)))

    print("\nEQUIVALENCE %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
