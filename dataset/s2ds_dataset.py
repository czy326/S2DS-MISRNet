"""PyTorch dataset over the built S2 multi-temporal SR shards.

Each sample: T=12 real-date LR frames (40 m equivalent), the target date's 10 m image,
per-frame signed dt (days), per-frame per-pixel cloud fraction, per-frame clear fraction.
"""
import os
import glob
import io
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

DST = "/home/czy/data/s2ds_built"
BANDS = ["B02", "B03", "B04", "B08"]


def _raw_member(zi, blob):
    """Decompress one zip member straight from the raw bytes (bypasses zip sanity checks)."""
    off = zi.header_offset
    fnl = int.from_bytes(blob[off + 26:off + 28], "little")
    efl = int.from_bytes(blob[off + 28:off + 30], "little")
    start = off + 30 + fnl + efl
    data = blob[start:start + zi.compress_size]
    if zi.compress_type == 8:
        import zlib
        data = zlib.decompress(data, -15)
    return np.load(io.BytesIO(data))


def read_npz(path, tries=3):
    """Materialise an npz into a plain {name: ndarray} dict.

    Materialising matters: an np.load() NpzFile keeps an open file handle, and PyTorch's
    DataLoader forks workers -- the forked children then share that handle and the zip
    reader state gets corrupted (symptom: BadZipFile("Overlapped entries")).  Reading
    everything up-front into plain arrays makes the dataset safe to fork, and is faster.
    """
    last = None
    for _ in range(tries):
        try:
            z = np.load(path)
            return {k: z[k] for k in z.files}
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(0.5)
    try:
        zz = zipfile.ZipFile(path)
        return {n[:-4]: np.load(io.BytesIO(zz.read(n))) for n in zz.namelist()}
    except Exception as e:                            # noqa: BLE001
        last = e
    blob = open(path, "rb").read()
    zz = zipfile.ZipFile(io.BytesIO(blob))
    try:
        return {i.filename[:-4]: _raw_member(i, blob) for i in zz.infolist()}
    except Exception as e:                            # noqa: BLE001
        raise RuntimeError("cannot read %s: %s | %s" % (path, str(last)[:90], str(e)[:90]))


def block_permute_cld(cld, block=8, rng=None):
    """Scramble the per-pixel cloud pattern within each frame, keeping frame-level stats.

    The T-frame cloud cube (T,H,W) is cut into `block`x`block` tiles and the tiles are
    permuted independently for every frame.  Two things are preserved on purpose:
      * the frame-level clear fraction (hence the frame-level reliability embedding stays
        meaningful);
      * the marginal distribution of cloud values.
    What is destroyed is the spatial correspondence between the cloud mask and the image
    content -- i.e. exactly the signal that a per-pixel cloud gate is supposed to use.
    This is the negative control for "does q actually exploit per-pixel structure?".
    """
    T, H, W = cld.shape
    nh, nw = H // block, W // block
    rng = rng or np.random.default_rng(0)
    out = cld.copy()
    x = cld[:, :nh * block, :nw * block]
    x = x.reshape(T, nh, block, nw, block).transpose(0, 1, 3, 2, 4)
    x = x.reshape(T, nh * nw, block, block).copy()
    for t in range(T):
        x[t] = x[t][rng.permutation(nh * nw)]
    x = x.reshape(T, nh, nw, block, block).transpose(0, 1, 3, 2, 4)
    out[:, :nh * block, :nw * block] = x.reshape(T, nh * block, nw * block)
    return out


class S2DS(Dataset):
    """S2DS loader.

    shuffle      : permute the T frame axis per sample with a fixed, reproducible
                   permutation.  This removes the ordinal time rank that the canonical
                   |dt|-ascending frame order leaks to every arm (see paper 3.1), so it
                   turns the dt factor into a genuine from-scratch test.
    permute_cld  : block size for the cloud-mask negative control (0 = off).
    """

    def __init__(self, split, aois=None, max_samples=0, shuffle=False, seed=0,
                 permute_cld=0, flip_cld=0.0):
        self.files = sorted(glob.glob(os.path.join(DST, "%s_*.npz" % split)))
        if aois:
            keep = set(aois)
            self.files = [f for f in self.files
                          if os.path.basename(f).split("_", 1)[1].rsplit(".", 1)[0] in keep]
        if not self.files:
            raise RuntimeError("no shards for split=%s under %s" % (split, DST))
        self.arrs = [read_npz(f) for f in self.files]
        self.index = [(i, j) for i, a in enumerate(self.arrs) for j in range(a["hr"].shape[0])]
        if max_samples:
            self.index = self.index[:max_samples]
        self.shuffle = shuffle
        self.permute_cld = int(permute_cld)
        self.flip_cld = float(flip_cld)
        # fixed per-sample permutation of the frame axis; reproducible across runs
        rng = np.random.default_rng(seed)
        n = len(self.index)
        T = self.arrs[0]["lr"].shape[1]
        self.perms = np.stack([rng.permutation(T) for _ in range(n)]) if shuffle else None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, j = self.index[k]
        a = self.arrs[i]
        lr = a["lr"][j].astype(np.float32) / 10000.0            # (T,4,48,48)
        hr = a["hr"][j].astype(np.float32) / 10000.0            # (4,192,192)
        cld = a["cld"][j].astype(np.float32) / 255.0            # (T,48,48) in [0,1]
        dt = a["dt"][j].astype(np.float32)                      # (T,) signed days
        if self.shuffle:
            p = self.perms[k]
            lr, cld, dt = lr[p], cld[p], dt[p]
        if self.permute_cld:
            cld = block_permute_cld(cld, self.permute_cld,
                                    np.random.default_rng(self.permute_cld * 1000 + k))
        if self.flip_cld > 0:
            # 模拟云检测算法本身的出错（漏检 + 虚警）：随机翻转该比例的像素标签。
            # 与 8×8 块置换的区别——块置换保留帧级统计量只破坏空间对应，
            # 翻转噪声同时污染帧级晴空率 q，是更严苛也更真实的退化。
            rng = np.random.default_rng(int(self.flip_cld * 1000) * 100000 + k)
            sel = rng.random(cld.shape) < self.flip_cld
            cld = np.where(sel, 1.0 - cld, cld)
        valid = 1.0 - a["hr_cloud"][j].astype(np.float32)       # (192,192)
        q = 1.0 - cld.reshape(cld.shape[0], -1).mean(1)         # (T,) clear fraction
        return {
            "lr": torch.from_numpy(lr),
            "dt": torch.from_numpy(dt / 30.0),
            "cld": torch.from_numpy(cld),
            "q": torch.from_numpy(q),
            "hr": torch.from_numpy(hr),
            "valid": torch.from_numpy(valid),
            "scene": str(a["scene"][j]),
            "aoi": str(a["aoi"][j]),
            "date": str(a["date"][j]),
        }


def dataset_stats(split="train"):
    ds = S2DS(split)
    return dict(n=len(ds), files=[os.path.basename(f) for f in ds.files])
