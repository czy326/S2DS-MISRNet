"""Pre-decode MuS2 scenes to npz cache (central 384 LR crop + 1152 HR) to remove jp2 bottleneck."""
import os, sys, glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/mnt/e/论文2")
from misr.data import parse_s2_date, parse_wv2_date

ROOT = "/home/czy/data/mus2/x/image_data"
CACHE = "/home/czy/data/mus2/cache"
BANDS = {"b2": "mul_band_1", "b3": "mul_band_2", "b4": "mul_band_4"}
CLOUD_SCL = [3, 8, 9, 10]
T = 15
CROP = 384


def one(scene):
    import rasterio, cv2
    from PIL import Image
    out = os.path.join(CACHE, os.path.basename(scene) + ".npz")
    if os.path.exists(out):
        return "skip " + out
    files = sorted(f for f in os.listdir(os.path.join(scene, "b4", "lrs")) if f.endswith(".jp2"))
    dates = [parse_s2_date(f) for f in files]
    tgt = sorted(dates)[len(dates) // 2]
    order = np.argsort([abs((d - tgt).days) for d in dates])
    sel = list(order[:T])
    while len(sel) < T:
        sel.append(sel[-1])
    lr, cld = [], []
    for k in sel:
        ch = []
        for b in BANDS:
            with rasterio.open(os.path.join(scene, b, "lrs", files[k])) as s:
                ch.append(s.read(1))
        lr.append(np.stack(ch))
        sp = os.path.join(scene, "SCL", "lrs", files[k])
        with rasterio.open(sp) as s:
            a = s.read(1)
        cld.append(np.isin(a, CLOUD_SCL).astype(np.uint8))
    lr = np.stack(lr)          # T,3,H,W uint16
    H, W = lr.shape[-2:]
    cld = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST) for c in cld])
    hr = np.stack([np.asarray(Image.open(os.path.join(scene, "hr_resized", BANDS[b] + ".tiff"))) for b in BANDS])
    h0 = H // 2 - CROP // 2
    w0 = W // 2 - CROP // 2
    lr_c = lr[:, :, h0:h0 + CROP, w0:w0 + CROP]
    cld_c = cld[:, h0:h0 + CROP, w0:w0 + CROP]
    hr_c = hr[:, h0 * 3:h0 * 3 + CROP * 3, w0 * 3:w0 * 3 + CROP * 3]
    dt = np.asarray([abs((dates[k] - tgt).days) / 365.0 for k in sel], dtype=np.float32)
    q = 1.0 - cld_c.reshape(len(sel), -1).mean(1).astype(np.float32)
    np.savez(out, lr=lr_c, cld=cld_c, hr=hr_c, dt=dt, q=q, target=str(tgt))
    return "done " + os.path.basename(scene)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    scenes = sorted(glob.glob(os.path.join(ROOT, "*-2AS_*")))
    with Pool(6) as p:
        for i, r in enumerate(p.imap_unordered(one, scenes)):
            if i % 10 == 0:
                print(i, r, flush=True)
    print("CACHE DONE", len(scenes))
