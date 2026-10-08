# -*- coding: utf-8 -*-
"""Single dispatch point so that E1 (q-route ablation) and E2/E7 (published baselines) can
share one train / select / evaluate triple.

`spec` is the argparse namespace, or the `args` dict that was saved into a checkpoint.
"""
import sys

sys.path.insert(0, "/mnt/e/论文2")

from misr.models_e1 import build_model_e1      # noqa: E402
from misr.baselines_s2ds import build_baseline  # noqa: E402
from misr.breizhsr_s2ds import BreizhSR         # noqa: E402

MODELS = ["e1", "highresnet", "highresnet_cld", "rams", "breizhsr", "breizhsr_nope"]


def build_from_spec(spec, cin=4, scale=4, c_default=32):
    if not isinstance(spec, dict):
        spec = vars(spec)
    model = spec.get("model", "e1")
    c = int(spec.get("c", c_default))
    if model == "e1":
        return build_model_e1(cin=cin, c=c, scale=scale,
                              arm=spec.get("arm", "B"),
                              att_mode=spec.get("att_mode", "pixel"),
                              dt_mode=spec.get("dt_mode", "gate"),
                              gamma0=float(spec.get("gamma0", 0.0) or 0.0))
    if model in ("highresnet", "highresnet_cld", "rams"):
        return build_baseline(model, cin=cin, c=c, scale=scale)
    if model in ("breizhsr", "breizhsr_nope"):
        # c is the encoder width; the official value is 64 and n_head=16 requires
        # c % 16 == 0, so pass the default through unchanged unless it is compatible.
        cc = c if c % 16 == 0 else 64
        return BreizhSR(cin=cin, c=cc, scale=scale,
                        position_days=(model == "breizhsr"))
    raise ValueError("unknown model %r" % model)
