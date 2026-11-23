"""
Put the separate groups of one survey into one image, from the finished outputs.

    mosaic2d assemble <output folder>        (or "Assemble groups..." in the window)

A series that broke into groups (photos in between that match nothing: a turn,
blur, a jump) gives <name>.tif, <name>_block02.tif, ... This step does not match
or render photos again. It takes the groups in shooting order and continues the
path of the camera across each gap: from the last photos of a group it measures
the step between photos and how fast the path is turning, carries that on over
the photos that are missing (leaving their area empty), and puts the next group
where the path arrives, turned so that its own first steps follow the path.
Inside each group nothing changes; where the groups sit relative to each other
is an estimate - straight on if the diver went straight, a continuing curve if
the diver was turning.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------- inputs
def _frame_size(photos: Path | None, names: list):
    """(w, h) of the photos (native pixels), read from the first readable one."""
    if photos is None:
        return None
    for nm in names[:5]:
        f = photos / nm
        if not f.is_file():
            continue
        try:
            import tifffile
            with tifffile.TiffFile(str(f)) as t:
                p = t.pages[0]
                return int(p.imagewidth), int(p.imagelength)
        except Exception:                               # noqa: BLE001
            pass
        try:
            from . import rawio
            a = rawio.read_image(f)
            return a.shape[1], a.shape[0]
        except Exception:                               # noqa: BLE001
            pass
    return None


def _read_tfw(path):
    try:
        v = [float(x) for x in Path(path).read_text().split()]
        return v[0]
    except Exception:                                   # noqa: BLE001
        return None


def load_groups(out_dir, name=None, photos=None):
    """Groups of a finished run: [{tif, centres {frame: (x, y)}, first, last, px}], in shooting order."""
    out_dir = Path(out_dir)
    if name is None:
        cands = sorted(out_dir.glob("*_blocks.csv"))
        if not cands:
            raise FileNotFoundError(f"no *_blocks.csv in {out_dir}: run mosaic2d first")
        name = cands[0].name[: -len("_blocks.csv")]
    images = [r["image"] for r in csv.DictReader(open(out_dir / f"{name}_images.csv", newline=""))]
    if photos is None:
        photos = out_dir.parent
    size = _frame_size(Path(photos), images)
    groups = []
    for r in csv.DictReader(open(out_dir / f"{name}_blocks.csv", newline="")):
        tif = out_dir / r["output"]
        if not r["output"].endswith(".tif") or not tif.is_file():
            continue
        tr = tif.with_name(tif.stem + "_transforms.npy")
        if not tr.is_file():
            continue
        H = np.load(tr, allow_pickle=True).item()
        if size is None:          # no photo at hand: the transform's origin stands in for the centre
            w = h = 0.0
        else:
            w, h = size
        cen = {int(k): cv2.perspectiveTransform(np.array([[[w / 2.0, h / 2.0]]]), np.asarray(v, np.float64))[0, 0]
               for k, v in H.items()}
        groups.append({"tif": tif, "centres": cen, "first": min(cen), "last": max(cen),
                       "px": _read_tfw(tif.with_suffix(".tfw")), "images": len(cen)})
    groups.sort(key=lambda g: g["first"])
    return name, images, groups


# ---------------------------------------------------------------------- path model
def path_motion(points: np.ndarray, t: np.ndarray, at_end=True, k=12):
    """
    Net step (px per photo), heading (rad) and turn rate (rad per photo) of a
    camera path at its end (or start), from the last (first) k photos.

    The path of a hand-held camera wiggles and single photos can sit off it, so
    nothing is taken from individual steps: the k photos are cut into thirds by
    time, and the medians of the thirds give two chords - their directions give
    heading and turn rate, the distance covered gives the net step.
    points: N x 2 centres, t: photo numbers.
    """
    order = np.argsort(t)
    P, T = points[order].astype(np.float64), t[order].astype(np.float64)
    P, T = (P[-k:], T[-k:]) if at_end else (P[:k], T[:k])
    n = len(P)
    if n < 2:
        return 0.0, 0.0, 0.0
    if n < 6:                                            # too few for thirds: one chord
        d = P[-1] - P[0]
        return float(np.hypot(*d) / max(T[-1] - T[0], 1.0)), float(np.arctan2(d[1], d[0])), 0.0
    parts = np.array_split(np.arange(n), 3)
    q = [np.median(P[i], axis=0) for i in parts]
    tm = [float(np.median(T[i])) for i in parts]
    h1 = np.arctan2(*(q[1] - q[0])[::-1])
    h2 = np.arctan2(*(q[2] - q[1])[::-1])
    dh = (h2 - h1 + np.pi) % (2 * np.pi) - np.pi
    rate = dh / max((tm[2] - tm[0]) / 2.0, 1.0)
    rate = float(np.clip(rate, -np.radians(6), np.radians(6)))   # at most 6 deg per photo
    step = float(np.hypot(*(q[2] - q[0])) / max(tm[2] - tm[0], 1.0))
    if at_end:
        heading = h2 + rate * (T[-1] - (tm[1] + tm[2]) / 2.0)
    else:
        heading = h1 - rate * ((tm[0] + tm[1]) / 2.0 - T[0])
    return step, float(heading), rate


def continue_path(p0, heading, step, rate, n):
    """Position and heading after n more photos of constant step and turn rate."""
    p = np.array(p0, np.float64)
    th = heading
    for _ in range(int(n)):
        th += rate
        p = p + step * np.array([np.cos(th), np.sin(th)])
    return p, th


def _similarity(theta, scale, src, dst):
    """3x3 matrix: rotate by theta and scale about src, then move src to dst."""
    c, s = np.cos(theta) * scale, np.sin(theta) * scale
    R = np.array([[c, -s], [s, c]])
    t = np.asarray(dst, np.float64) - R @ np.asarray(src, np.float64)
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = R, t
    return M


def place_groups(groups, k=12, straight=False, log=print):
    """Similarity (group pixels -> assembled pixels) for every group; the first stays as it is."""
    px0 = groups[0]["px"] or 1.0
    groups[0]["M"] = np.eye(3)
    groups[0]["gap"] = 0
    for a, b in zip(groups, groups[1:]):
        ta = np.array(sorted(a["centres"]))
        Pa = np.array([a["centres"][i] for i in ta], np.float64)
        Pa = cv2.perspectiveTransform(Pa[None], a["M"])[0]           # in assembled pixels
        step, heading, rate = path_motion(Pa, ta, at_end=True, k=k)
        if straight:
            rate = 0.0
        gap = b["first"] - a["last"]
        p_pred, th_pred = continue_path(Pa[np.argmax(ta)], heading, step, rate, gap)
        tb = np.array(sorted(b["centres"]))
        Pb = np.array([b["centres"][i] for i in tb], np.float64)
        _, hb, _ = path_motion(Pb, tb, at_end=False, k=k)
        scale = (b["px"] or px0) / px0
        b["M"] = _similarity(th_pred - hb, scale, Pb[0], p_pred)
        b["gap"] = int(gap - 1)
        b["turn_deg"] = float(np.degrees(rate * gap))
        log(f"  {b['tif'].name}: {gap - 1} photo(s) missing before it; path continued "
            f"{'straight' if abs(np.degrees(rate)) < 0.5 else f'turning {np.degrees(rate):+.1f} deg per photo'}, "
            f"step {step:.0f} px; group turned {np.degrees(th_pred - hb):+.1f} deg")
    return groups


# ---------------------------------------------------------------------- composition
class AssembledCanvas:
    """Lazy H x W x B image: every read draws the groups into the requested rows."""

    def __init__(self, groups, size, origin, dtype, bands, tile=512):
        from .viewer import TiffMosaic
        self.groups, self.tile = groups, tile
        self.W, self.H = size
        self.ox, self.oy = origin
        self.dtype = np.dtype(dtype)
        self.B = bands
        self.shape = (self.H, self.W, self.B)
        self.ndim = 3
        self.src = [TiffMosaic(g["tif"], cache_mb=256) for g in groups]
        self.inv = []
        self.masks = []
        for g, m in zip(groups, self.src):
            T = np.array([[1, 0, -self.ox], [0, 1, -self.oy], [0, 0, 1.0]]) @ g["M"]
            self.inv.append(np.linalg.inv(T))
            ov, f = m.overview()                       # coarse coverage, to skip empty tiles fast
            cov = (ov > 0).any(axis=2).astype(np.uint8)
            cov = cv2.dilate(cov, np.ones((5, 5), np.uint8))
            self.masks.append((cov, f))

    def _covers(self, k, x0, y0, x1, y1):
        cov, f = self.masks[k]
        xs, ys = np.meshgrid(np.linspace(x0, x1, 6), np.linspace(y0, y1, 6))
        q = cv2.perspectiveTransform(np.dstack([xs, ys]).reshape(1, -1, 2).astype(np.float64), self.inv[k])[0] / f
        qx, qy = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (qx >= 0) & (qy >= 0) & (qx < cov.shape[1]) & (qy < cov.shape[0])
        if ok.any() and cov[qy[ok], qx[ok]].any():
            return True
        # tile larger than the coverage features: also test the group's corners inside the tile
        m = self.src[k]
        corners = np.array([[[0, 0], [m.W, 0], [m.W, m.H], [0, m.H]]], np.float64)
        c = cv2.perspectiveTransform(corners, np.linalg.inv(self.inv[k]))[0]
        return bool(((c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)).any())

    def _tile(self, x0, y0, x1, y1):
        out = np.zeros((y1 - y0, x1 - x0, self.B), self.dtype)
        filled = np.zeros((y1 - y0, x1 - x0), bool)
        for k, m in enumerate(self.src):
            if not self._covers(k, x0, y0, x1, y1):
                continue
            X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
            q = cv2.perspectiveTransform(np.dstack([X, Y]).reshape(1, -1, 2), self.inv[k])[0]
            mx = q[:, 0].reshape(X.shape).astype(np.float32)
            my = q[:, 1].reshape(X.shape).astype(np.float32)
            bx0, by0 = int(np.floor(mx.min())) - 1, int(np.floor(my.min())) - 1
            bx1, by1 = int(np.ceil(mx.max())) + 2, int(np.ceil(my.max())) + 2
            if bx1 <= 0 or by1 <= 0 or bx0 >= m.W or by0 >= m.H:
                continue
            region = m.read(0, bx0, by0, bx1, by1, list(range(self.B)))
            mx -= bx0
            my -= by0
            vals = np.empty((y1 - y0, x1 - x0, self.B), self.dtype)
            for c0 in range(0, self.B, 4):
                c1 = min(self.B, c0 + 4)
                r = cv2.remap(np.ascontiguousarray(region[..., c0:c1]), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                vals[..., c0:c1] = r.reshape(y1 - y0, x1 - x0, c1 - c0)
            # nearest-neighbour validity: no blending of data with the empty border
            valid = cv2.remap((region > 0).any(axis=2).astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            take = valid & ~filled                     # earlier groups keep what they cover
            out[take] = vals[take]
            filled |= take
        return out

    def __getitem__(self, key):
        ys = key if isinstance(key, slice) else key[0]
        y0, y1 = max(0, ys.start or 0), min(self.H, ys.stop if ys.stop is not None else self.H)
        rows = np.zeros((max(0, y1 - y0), self.W, self.B), self.dtype)
        for x0 in range(0, self.W, self.tile):
            x1 = min(self.W, x0 + self.tile)
            rows[:, x0:x1] = self._tile(x0, y0, x1, y1)
        return rows

    def close(self):
        for m in self.src:
            m.close()


def assemble(out_dir, name=None, photos=None, out=None, k=12, straight=False, log=print):
    from .quicklook import make_preview
    from .rawio import tiff_band_info
    from .tiffwriter import write_tiff
    out_dir = Path(out_dir)
    name, images, groups = load_groups(out_dir, name, photos)
    if len(groups) < 2:
        log("  nothing to assemble: only one group was rendered")
        return None
    log(f"assembling {len(groups)} groups of {name} ({sum(g['images'] for g in groups)} photos) "
        f"along the camera path:")
    place_groups(groups, k=k, straight=straight, log=log)
    # bounds of the assembled image
    pts = []
    from .viewer import TiffMosaic
    first = TiffMosaic(groups[0]["tif"])
    dtype, B = first.dtype, first.S
    first.close()
    for g in groups:
        m = TiffMosaic(g["tif"])
        c = np.array([[[0, 0], [m.W, 0], [m.W, m.H], [0, m.H]]], np.float64)
        m.close()
        pts.append(cv2.perspectiveTransform(c, g["M"])[0])
    pts = np.concatenate(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    W, H = int(x1 - x0), int(y1 - y0)
    canvas = AssembledCanvas(groups, (W, H), (x0, y0), dtype, B)
    info = tiff_band_info(groups[0]["tif"])
    names = [info.get(b, (f"band {b + 1}", None))[0] for b in range(B)] if isinstance(info, dict) else None
    wls = [info.get(b, (None, None))[1] for b in range(B)] if isinstance(info, dict) else None
    px = groups[0]["px"]
    out = Path(out) if out else out_dir / f"{name}_assembled.tif"
    log(f"  writing {out.name}: {W} x {H} px, {B} band(s)"
        + (f", {W * px:.2f} x {H * px:.2f} m" if px else ""))
    step = {"next": 0.1}

    def prog(fr):
        if fr >= step["next"]:
            log(f"  written {min(fr, 1.0) * 100:.0f} %")
            step["next"] = 2.0 if fr >= 1.0 else min(1.0, fr + 0.1)
    try:
        write_tiff(out, canvas, band_names=names, wavelengths=wls if wls and any(wls) else None, nodata=0,
                   description=json.dumps({"mosaic2d": "assembled", "groups": [g["tif"].name for g in groups]}),
                   pixel_size=px, origin=(0.0, H * px if px else 0.0), progress=prog)
    finally:
        canvas.close()
    with open(out.with_name(out.stem + "_groups.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "first", "last", "photos", "missing_before", "rotation_deg", "x_px", "y_px"])
        for g in groups:
            rot = np.degrees(np.arctan2(g["M"][1, 0], g["M"][0, 0]))
            w.writerow([g["tif"].name, images[g["first"]] if g["first"] < len(images) else g["first"],
                        images[g["last"]] if g["last"] < len(images) else g["last"], g["images"],
                        g.get("gap", 0), round(float(rot), 2), round(float(g["M"][0, 2] - x0), 1),
                        round(float(g["M"][1, 2] - y0), 1)])
    make_preview(out, log=log)
    log(f"  done: {out}")
    return out
