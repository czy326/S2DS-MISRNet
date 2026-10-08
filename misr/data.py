"""MISR data pipelines: PROBA-V (reliability masks) and MuS2 (real revisit dates + cloud masks)."""
import os, glob, re, json, datetime
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REFL = 10000.0


def _load_png(p):
    return np.asarray(Image.open(p)).astype(np.float32)


class ProbavDataset(Dataset):
    """PROBA-V Kelvin SR dataset. Each scene: T LR frames (128x128) + per-pixel QM, HR 384x384 + SM."""

    def __init__(self, root="/home/czy/data/probav", band="RED", split="train", T=15,
                 test_ratio=0.2, val_ratio=0.1, seed=2026, stats=None):
        # official test/ has no HR -> all our splits are carved from the HR-bearing train dir
        self.root = os.path.join(root, "train", band)
        self.T = T
        self.band = band
        scenes = sorted(glob.glob(os.path.join(self.root, "imgset*")))
        scenes = [s for s in scenes if os.path.exists(os.path.join(s, "HR.png"))]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(scenes))
        n_test = int(len(scenes) * test_ratio)
        n_val = int(len(scenes) * val_ratio)
        if split == "test":
            keep = idx[:n_test]
        elif split == "val":
            keep = idx[n_test:n_test + n_val]
        else:
            keep = idx[n_test + n_val:]
        self.scenes = [scenes[i] for i in sorted(keep)]
        self.stats = stats

    def __len__(self):
        return len(self.scenes)

    def _frames(self, d):
        lr = sorted(glob.glob(os.path.join(d, "LR*.png")))
        qm = sorted(glob.glob(os.path.join(d, "QM*.png")))
        return lr, qm

    def __getitem__(self, i):
        d = self.scenes[i]
        lr_p, qm_p = self._frames(d)
        clear = []
        for f in qm_p:
            m = _load_png(f)
            clear.append(float((m > 0).mean()))
        clear = np.asarray(clear)
        order = np.argsort(-clear)  # clearest first
        sel = list(order[:self.T])
        while len(sel) < self.T:
            sel.append(sel[-1])
        lrs, qms = [], []
        for k in sel:
            a = _load_png(lr_p[k]) / REFL
            m = (_load_png(qm_p[k]) > 0).astype(np.float32)
            lrs.append(a)
            qms.append(m)
        lr = np.stack(lrs)[:, None]          # T,1,128,128
        qm = np.stack(qms)[:, None]          # T,1,128,128
        hr = (_load_png(os.path.join(d, "HR.png")) / REFL)[None]
        sm = (_load_png(os.path.join(d, "SM.png")) > 0).astype(np.float32)[None]
        q = clear[sel][:, None].astype(np.float32)   # T,1 scalar clearance
        if self.stats:
            lr = (lr - self.stats["lr_mean"]) / self.stats["lr_std"]
            hr = (hr - self.stats["hr_mean"]) / self.stats["hr_std"]
            qm = np.ones_like(qm)  # placeholder (unused)
        out = {"scene": os.path.basename(d),
               "lr": torch.from_numpy(lr), "q": torch.from_numpy(q),
               "qm": torch.from_numpy(qm), "hr": torch.from_numpy(hr),
               "sm": torch.from_numpy(sm)}
        return out


def parse_s2_date(fn):
    m = re.search(r"S2[AB]_MSIL2A_(\d{4})(\d{2})(\d{2})T", fn)
    return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_wv2_date(scene):
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{2})\d{4}-2AS_", scene)
    if not m:
        return None
    mons = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7,
            "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    return datetime.date(2000 + int(m.group(3)), mons[m.group(2)], int(m.group(1)))


class MuS2Dataset(Dataset):
    """MuS2 benchmark: S2 time series (real dates, cloud masks) -> WV-2 HR (3x)."""

    BANDS = {"b2": "mul_band_1", "b3": "mul_band_2", "b4": "mul_band_4", "b8": "mul_band_6"}
    CLOUD_SCL = [3, 8, 9, 10]

    def __init__(self, root="/home/czy/data/mus2/x/image_data", bands=("b2", "b3", "b4"),
                 split="train", T=15, crop=256, test_ratio=0.2, val_ratio=0.1, seed=2026,
                 target="median", stats=None):
        self.root = root
        self.bands = bands
        self.T = T
        self.crop = crop
        self.target = target
        self.stats = stats
        scenes = sorted(glob.glob(os.path.join(root, "*-2AS_*")))
        scenes = [s for s in scenes if os.path.exists(os.path.join(s, "hr_resized"))]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(scenes))
        n_test = int(len(scenes) * test_ratio)
        n_val = int(len(scenes) * val_ratio)
        if split == "test":
            keep = idx[:n_test]
        elif split == "val":
            keep = idx[n_test:n_test + n_val]
        else:
            keep = idx[n_test + n_val:]
        self.scenes = [scenes[i] for i in sorted(keep)]
        self.split_name = split

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        cpath = os.path.join("/home/czy/data/mus2/cache", os.path.basename(self.scenes[i]) + ".npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            C = self.crop
            lr0, cld0, hr0 = z["lr"], z["cld"], z["hr"]
            H, W = lr0.shape[-2:]
            if H < C or W < C:
                ph, pw = max(0, C - H), max(0, C - W)
                lr0 = np.pad(lr0, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
                cld0 = np.pad(cld0, ((0, 0), (0, ph), (0, pw)), mode="edge")
                hr0 = np.pad(hr0, ((0, 0), (0, ph * 3), (0, pw * 3)), mode="edge")
                H, W = lr0.shape[-2:]
            if self.split_name == "train":
                h0 = int(np.random.randint(0, max(H - C, 0) + 1))
                w0 = int(np.random.randint(0, max(W - C, 0) + 1))
            else:
                h0 = max(0, min(H // 2 - C // 2, max(H - C, 0)))
                w0 = max(0, min(W // 2 - C // 2, max(W - C, 0)))
            lr = lr0[:, :, h0:h0 + C, w0:w0 + C].astype(np.float32) / REFL
            cm = cld0[:, h0:h0 + C, w0:w0 + C][:, None].astype(np.float32)
            hr = hr0[:, h0 * 3:h0 * 3 + C * 3, w0 * 3:w0 * 3 + C * 3].astype(np.float32) / 255.0
            dt = z["dt"].astype(np.float32)
            q = z["q"].astype(np.float32)
            if self.stats:
                lr = (lr - self.stats["lr_mean"]) / self.stats["lr_std"]
            return {"lr": torch.tensor(np.ascontiguousarray(lr)), "q": torch.tensor(np.ascontiguousarray(q[:, None])),
                    "cld": torch.tensor(np.ascontiguousarray(cm)), "dt": torch.tensor(np.ascontiguousarray(dt)),
                    "hr": torch.tensor(np.ascontiguousarray(hr)), "scene": os.path.basename(self.scenes[i])}
        import rasterio
        d = self.scenes[i]
        ref_band = "b4"
        files = sorted(f for f in os.listdir(os.path.join(d, ref_band, "lrs")) if f.endswith(".jp2"))
        dates = [parse_s2_date(f) for f in files]
        hr_date = parse_wv2_date(os.path.basename(d))
        if self.target == "median" and dates:
            tgt = sorted(dates)[len(dates) // 2]
        else:
            tgt = hr_date
        # order by |dt| to target, keep T
        order = np.argsort([abs((x - tgt).days) if x and tgt else 0 for x in dates])
        sel = list(order[:self.T])
        while len(sel) < self.T:
            sel.append(sel[-1])
        lrs, cld = [], []
        for k in sel:
            chans = []
            for b in self.bands:
                with rasterio.open(os.path.join(d, b, "lrs", files[k])) as s:
                    chans.append(s.read(1).astype(np.float32))
            lrs.append(np.stack(chans))
            sp = os.path.join(d, "SCL", "lrs", files[k])
            if os.path.exists(sp):
                with rasterio.open(sp) as s:
                    a = s.read(1)
                cld.append(np.isin(a, self.CLOUD_SCL).astype(np.float32))
            else:
                cld.append(np.zeros_like(lrs[-1][0]))
        lr = np.stack(lrs) / REFL                 # T,C,H,W
        import cv2
        H0, W0 = lr.shape[-2:]
        cld = [cv2.resize(c, (W0, H0), interpolation=cv2.INTER_NEAREST) for c in cld]
        cm = np.stack(cld)[:, None]               # T,1,H,W
        H, W = lr.shape[-2:]
        h0, w0 = H // 2 - self.crop // 2, W // 2 - self.crop // 2
        lr = lr[:, :, h0:h0 + self.crop, w0:w0 + self.crop]
        cm = cm[:, :, h0:h0 + self.crop, w0:w0 + self.crop]
        hr_ch = []
        for b in self.bands:
            hr_ch.append(_load_png(os.path.join(d, "hr_resized", self.BANDS[b] + ".tiff")))
        hr = np.stack(hr_ch) / 255.0
        c = self.crop * 3
        hr = hr[:, h0 * 3:h0 * 3 + c, w0 * 3:w0 * 3 + c]
        dt = np.asarray([abs((dates[k] - tgt).days) / 365.0 for k in sel], dtype=np.float32)
        q = 1.0 - np.asarray([cm[i].mean() for i in range(len(sel))], dtype=np.float32)
        if self.stats:
            lr = (lr - self.stats["lr_mean"]) / self.stats["lr_std"]
        return {"lr": torch.from_numpy(lr), "q": torch.from_numpy(q[:, None]),
                "cld": torch.from_numpy(cm), "dt": torch.from_numpy(dt),
                "hr": torch.from_numpy(hr), "scene": os.path.basename(d)}
