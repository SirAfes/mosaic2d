"""
render.py

Out-of-core mosaic rendering.

The canvas lives in a disk-backed NumPy memmap, so the mosaic can be much
larger than memory. Frames are decoded one at a time (frame-major order) and
written block by block; blocks a frame neither wins nor feathers are skipped
after a coarse check.

Which frame supplies a pixel (the *seam* layout) is decided analytically: the
frame in which the pixel lies closest to the image centre wins (elliptical
distance, normalised by the valid image size) -- the rule Metashape uses when
blending is disabled; it keeps every pixel from its least distorted, best lit
view and needs no global label image. The distance is multiplied by a
per-frame sharpness penalty, so a blurred frame (motion, focus) gives way to
sharper overlapping frames and only fills areas nobody else covers.

Blending modes (names follow Metashape):

disabled  every pixel comes unmodified from the winning frame. Radiometry is
          preserved exactly -- preferred for quantitative multispectral work.
mosaic    (default) two-band blending: high frequencies from the winning frame,
          low frequencies harmonised across overlaps. Implemented as a smooth
          per-frame gain field  g_f = L / L_f  where L is the feather-weighted
          low-resolution mosaic and L_f the frame's own low-resolution image.
          Removes brightness steps at seams (vignetting, strobe falloff,
          exposure changes) without ghosting.
average   feather-weighted average of all frames at every pixel (smooth, but
          blurs/ghosts where the scene has relief or moved).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RenderFrame:
    index: int
    path: str
    H: np.ndarray                 # ref-lens px (undistorted) -> output px
    ref_size: tuple               # (w, h) of the reference lens image, native px
    valid: tuple                  # (x0, y0, x1, y1) valid rect in ref px (all lenses)
    lens_A: dict                  # lens -> 3x3, lens px -> ref px
    thumb: np.ndarray | None = None   # h_t x w_t x B (in ref px * thumb_scale)
    thumb_scale: float = 1.0
    penalty: float = 1.0          # seam score multiplier (>1 = blurrier than its neighbours)
    seam_img: np.ndarray | None = None   # h x w x 3 image for seamline optimisation (ref px * seam_scale)
    seam_scale: float = 0.0
    Hinv: np.ndarray = field(default=None)
    bbox: tuple = (0, 0, 0, 0)    # x0, y0, x1, y1 in output px

    def __post_init__(self):
        self.Hinv = np.linalg.inv(self.H)


@dataclass
class Distortion:
    k: np.ndarray
    center: np.ndarray
    radius: float

    @property
    def active(self) -> bool:
        return bool(np.any(np.abs(self.k) > 1e-9))

    def distort(self, u, v):
        """Undistorted -> distorted coordinates (inverse of bundle.undistort_points)."""
        if not self.active:
            return u, v
        cx, cy = self.center
        du, dv = u - cx, v - cy
        x, y = du.copy(), dv.copy()
        for _ in range(5):
            r2 = (x * x + y * y) / (self.radius ** 2)
            f = 1 + self.k[0] * r2 + self.k[1] * r2 * r2
            x, y = du / f, dv / f
        return x + cx, y + cy

    def undistort(self, pts):
        if not self.active:
            return pts
        d = (pts - self.center) / self.radius
        r2 = (d ** 2).sum(1, keepdims=True)
        return self.center + (pts - self.center) * (1 + self.k[0] * r2 + self.k[1] * r2 * r2)


def frame_size_px(fr: RenderFrame) -> float:
    """Typical frame size in output pixels: sqrt of the mapped valid area."""
    x0, y0, x1, y1 = fr.valid
    det = abs(np.linalg.det(fr.H[:2, :2] / fr.H[2, 2]))
    return float(np.sqrt(max(det * (x1 - x0) * (y1 - y0), 1.0)))


def footprint(fr: RenderFrame, dist: Distortion, n=12) -> np.ndarray:
    x0, y0, x1, y1 = fr.valid
    t = np.linspace(0, 1, n, endpoint=False)
    pts = np.concatenate([
        np.stack([x0 + (x1 - x0) * t, np.full(n, y0)], 1),
        np.stack([np.full(n, x1), y0 + (y1 - y0) * t], 1),
        np.stack([x1 - (x1 - x0) * t, np.full(n, y1)], 1),
        np.stack([np.full(n, x0), y1 - (y1 - y0) * t], 1)])
    pts = dist.undistort(pts.astype(np.float64))
    return cv2.perspectiveTransform(pts[None], fr.H)[0]


def _map(fr: RenderFrame, X, Y):
    """Output px -> undistorted ref px."""
    Hi = fr.Hinv
    w = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
    u = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / w
    v = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / w
    return u, v


def _score(fr: RenderFrame, u, v, dist: Distortion):
    """(distance score, valid mask, distorted u, distorted v)."""
    ud, vd = dist.distort(u, v)
    x0, y0, x1, y1 = fr.valid
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    s = (((ud - cx) / hw) ** 2 + ((vd - cy) / hh) ** 2) * fr.penalty
    valid = (ud >= x0) & (ud <= x1) & (vd >= y0) & (vd <= y1)
    if dist.active:
        # the inverse distortion is only meaningful near the image; far outside
        # the fixed-point iteration can land back inside the frame
        valid &= (np.abs(u - cx) < 1.4 * hw) & (np.abs(v - cy) < 1.4 * hh)
    return s, valid, ud, vd


def _cheb_weight(fr, ud, vd):
    x0, y0, x1, y1 = fr.valid
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d = np.maximum(np.abs(ud - cx) / ((x1 - x0) / 2), np.abs(vd - cy) / ((y1 - y0) / 2))
    return np.clip(1 - d, 0, 1) ** 1.5


def _remap_multi(img, mx, my, interp=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT):
    """cv2.remap for any channel count (OpenCV handles at most 4 at once)."""
    if img.ndim == 2:
        return cv2.remap(img, mx, my, interp, borderMode=border, borderValue=0)[..., None]
    C = img.shape[2]
    parts = []
    for c0 in range(0, C, 4):
        sub = np.ascontiguousarray(img[..., c0:c0 + 4])
        r = cv2.remap(sub, mx, my, interp, borderMode=border, borderValue=0)
        parts.append(r if r.ndim == 3 else r[..., None])
    return np.concatenate(parts, axis=2)


class _SeamAccumulator:
    """
    Sparse float accumulator for the narrow feathering band along seamlines.
    Only tiles the band touches are allocated, so memory stays proportional to
    the seam length, not to the canvas.
    """

    def __init__(self, n_bands: int, tile: int = 32):
        self.B, self.t = n_bands, tile
        self.tiles: dict = {}

    def add(self, oy: int, ox: int, mask: np.ndarray, vals: np.ndarray, w: np.ndarray):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return
        gy, gx = ys + oy, xs + ox
        ty, tx = gy // self.t, gx // self.t
        key = ty * 10_000_000 + tx
        order = np.argsort(key, kind="stable")
        key, ys, xs, gy, gx = key[order], ys[order], xs[order], gy[order], gx[order]
        cuts = np.flatnonzero(np.diff(key)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, key.size]):
            k = (int(gy[a] // self.t), int(gx[a] // self.t))
            if k not in self.tiles:
                self.tiles[k] = (np.zeros((self.t, self.t, self.B), np.float32),
                                 np.zeros((self.t, self.t), np.float32))
            acc, ws = self.tiles[k]
            ly, lx = gy[a:b] - k[0] * self.t, gx[a:b] - k[1] * self.t
            ww = w[ys[a:b], xs[a:b]]
            acc[ly, lx] += vals[ys[a:b], xs[a:b]] * ww[:, None]
            ws[ly, lx] += ww

    def flush(self, out, maxv, dtype):
        H, W = out.shape[:2]
        for (ty, tx), (acc, ws) in self.tiles.items():
            y0, x0 = ty * self.t, tx * self.t
            h, w = min(self.t, H - y0), min(self.t, W - x0)
            m = ws[:h, :w] > 1e-3
            if not m.any():
                continue
            blk = np.array(out[y0:y0 + h, x0:x0 + w])
            # The frame that owns a pixel outright (weight >= 0.998) wrote it
            # directly and is not in the accumulator: it gets the weight the
            # band frames leave over, so a lone minor contribution cannot take
            # the pixel over.
            a = acc[:h, :w][m]
            s = ws[:h, :w][m]
            cur = blk[m].astype(np.float32)
            has = cur.max(axis=1) > 0
            rest = np.where(has, np.maximum(1.0 - s, 0.0), 0.0)
            v = (a + rest[:, None] * cur) / np.maximum(s + rest, 1e-6)[:, None]
            blk[m] = np.clip(v, 1, maxv).astype(dtype)
            out[y0:y0 + h, x0:x0 + w] = blk
        n = len(self.tiles)
        self.tiles = {}
        return n


def cleanup(renderer) -> None:
    """Remove the renderer's private temporary folder (canvas and accumulators)."""
    import shutil
    shutil.rmtree(renderer.work_dir, ignore_errors=True)
    try:                                   # parent folder too, if nothing else uses it
        Path(renderer.work_dir).parent.rmdir()
    except OSError:
        pass


class Renderer:
    def __init__(self, frames: list[RenderFrame], dist: Distortion, canvas_size: tuple,
                 band_sources: list, n_bands: int, dtype=np.uint16, blend="mosaic",
                 work_dir=None, strip=512, log=print, progress=None, flat=None,
                 seams="graphcut", vis_bands=(0, 1, 2), deghost=True, feather=True):
        """
        band_sources: per output band (lens, channel).
        flat: optional flat field, h_t x w_t x B (thumbnail geometry, reference-lens px * thumb_scale)
        seams: "graphcut" (optimised seamlines) or "voronoi" (closest frame centre)
        """
        self.flat = flat
        self.seam_mode = seams
        self.deghost = deghost
        self.feather = feather
        self.vis_bands = tuple(vis_bands)
        self.seams = None
        self.frames = frames
        self.dist = dist
        self.W, self.H = canvas_size
        self.band_sources = band_sources
        self.B = n_bands
        self.dtype = np.dtype(dtype)
        self.blend = blend
        self.strip = strip
        self.log = log
        self.progress = progress
        # private sub-folder: runs sharing a work dir must never share temporary files
        import uuid
        base = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir = base / f"mosaic2d_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.canvas_path = None
        for fr in frames:
            fp = footprint(fr, dist)
            x0, y0 = np.floor(fp.min(0)).astype(int)
            x1, y1 = np.ceil(fp.max(0)).astype(int) + 1
            fr.bbox = (max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1))
        self.maxv = np.iinfo(self.dtype).max
        self.gain = {}

    # ---------------------------------------------------------- low-res pass
    def low_frequency_pass(self):
        """Feather-weighted low-resolution mosaic and per-frame gain fields."""
        # "low frequency" is relative to the frame size: ~70 cells across a frame
        fw = float(np.median([frame_size_px(f) for f in self.frames])) if self.frames else max(self.W, self.H)
        ls = min(1.0, 1600.0 / max(self.W, self.H), 70.0 / max(fw, 1.0))
        Wl, Hl = max(1, int(self.W * ls)), max(1, int(self.H * ls))
        acc = np.zeros((Hl, Wl, self.B), np.float64)
        wsum = np.zeros((Hl, Wl), np.float64)
        own = {}
        sig = 2.0
        for fr in self.frames:
            if fr.thumb is None:
                continue
            bx0, by0, bx1, by1 = [int(v * ls) for v in fr.bbox]
            bx1, by1 = min(Wl, bx1 + 2), min(Hl, by1 + 2)
            if bx1 <= bx0 or by1 <= by0:
                continue
            X, Y = np.meshgrid((np.arange(bx0, bx1) + 0.5) / ls, (np.arange(by0, by1) + 0.5) / ls)
            u, v = _map(fr, X, Y)
            s, valid, ud, vd = _score(fr, u, v, self.dist)
            w = _cheb_weight(fr, ud, vd) * valid
            mx = (ud * fr.thumb_scale).astype(np.float32)
            my = (vd * fr.thumb_scale).astype(np.float32)
            t = _remap_multi(self._flat_thumb(fr), mx, my)
            t = np.where(valid[..., None], t, 0)
            acc[by0:by1, bx0:bx1] += t * w[..., None]
            wsum[by0:by1, bx0:bx1] += w
            own[fr.index] = (bx0, by0, t, valid)
        L = acc / np.maximum(wsum, 1e-9)[..., None]
        # Objects seen by one frame only (a rope or fish close to the camera) must
        # not enter the gain fields: that frame would be darkened (or brightened)
        # all around the object, and the step stays after the object is removed.
        # Cells where a frame's brightness ratio to the consensus departs from
        # the frame's local ratio are left out of the consensus and of the frame.
        inl = self._gain_inliers(own, L)
        acc2 = np.zeros_like(acc)
        wsum2 = np.zeros_like(wsum)
        for fr in self.frames:
            if fr.index not in own:
                continue
            bx0, by0, t, valid = own[fr.index]
            h, w = valid.shape
            u, v = _map(fr, *np.meshgrid((np.arange(bx0, bx0 + w) + 0.5) / ls,
                                         (np.arange(by0, by0 + h) + 0.5) / ls))
            _, _, ud, vd = _score(fr, u, v, self.dist)
            wt = _cheb_weight(fr, ud, vd) * inl[fr.index]
            acc2[by0:by0 + h, bx0:bx0 + w] += t * wt[..., None]
            wsum2[by0:by0 + h, bx0:bx0 + w] += wt
        L2 = np.where((wsum2 > 1e-6)[..., None], acc2 / np.maximum(wsum2, 1e-9)[..., None], L)
        # Outlier exclusion is only meant for thin or small objects. Where most
        # cells disagree (relief and parallax on rocks), the reduced estimate is
        # unstable, so it fades back to the plain all-cell estimate.
        wide = 4.0 * sig
        q = (cv2.GaussianBlur(wsum2.astype(np.float32), (0, 0), wide)
             / np.maximum(cv2.GaussianBlur(wsum.astype(np.float32), (0, 0), wide), 1e-9))
        kL = np.clip((q - 0.15) / 0.3, 0, 1)[..., None]
        Lb = cv2.GaussianBlur((kL * L2 + (1 - kL) * L).astype(np.float32), (0, 0), sig)
        Lb_all = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sig)

        def masked_blur(t, m, h, w):
            tb = cv2.GaussianBlur(t.astype(np.float32) * m[..., None], (0, 0), sig)
            nb = cv2.GaussianBlur(m, (0, 0), sig)
            return tb.reshape(h, w, -1) / np.maximum(nb, 1e-6)[..., None]

        for fr in self.frames:
            if fr.index not in own:
                continue
            bx0, by0, t, valid = own[fr.index]
            h, w = valid.shape
            va = valid.astype(np.float32)
            vm = (valid & inl[fr.index]).astype(np.float32)
            f = (cv2.GaussianBlur(vm, (0, 0), wide)
                 / np.maximum(cv2.GaussianBlur(va, (0, 0), wide), 1e-6))
            k = np.clip((f - 0.15) / 0.3, 0, 1).reshape(h, w, 1)
            if vm.sum() < 0.2 * max(valid.sum(), 1):
                k = np.zeros_like(k)
            # normalised (masked) blur of the frame's own low-res image
            tb_all = masked_blur(t, va, h, w)
            tb = k * masked_blur(t, vm, h, w) + (1 - k) * tb_all
            Lf = (k * Lb[by0:by0 + h, bx0:bx0 + w].reshape(h, w, -1)
                  + (1 - k) * Lb_all[by0:by0 + h, bx0:bx0 + w].reshape(h, w, -1))
            g = Lf / np.maximum(tb, 1e-3 * max(float(tb.mean()), 1e-6))
            g = np.clip(np.where(valid[..., None], g, 1.0), 0.33, 3.0).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 1.0).reshape(h, w, -1)
            self.gain[fr.index] = (bx0, by0, ls, g)
        self.low_mosaic = L

    @staticmethod
    def _gain_inliers(own, L, thr=0.12, sig=4.0):
        """Per frame: low-res cells whose brightness agrees with the consensus up
        to the frame's smooth gain (|log ratio - local ratio| < thr)."""
        lumL = L.mean(axis=2)
        Hl, Wl = lumL.shape
        eps = 0.02 * float(np.median(lumL[lumL > 0])) if (lumL > 0).any() else 1e-6
        # per-cell median of log brightness over the frames that see the cell
        # (sparse: memory proportional to the frame footprints, not frames x canvas)
        cells, vals, logs = [], [], {}
        for fi, (bx0, by0, t, valid) in own.items():
            h, w = valid.shape
            lg = np.log(t.reshape(h, w, -1).mean(axis=2) + eps)
            logs[fi] = lg
            yy, xx = np.nonzero(valid)
            cells.append((yy + by0) * Wl + (xx + bx0))
            vals.append(lg[yy, xx])
        med = np.log(lumL + eps)
        if cells:
            c = np.concatenate(cells)
            v = np.concatenate(vals)
            o = np.lexsort((v, c))
            c, v = c[o], v[o]
            starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
            counts = np.diff(np.r_[starts, c.size])
            ok = counts >= 3                          # with fewer views keep the mean
            mflat = med.reshape(-1)
            mflat[c[starts[ok]]] = v[starts[ok] + counts[ok] // 2]
            med = mflat.reshape(Hl, Wl)
        out = {}
        for fi, (bx0, by0, t, valid) in own.items():
            h, w = valid.shape
            r = logs[fi] - med[by0:by0 + h, bx0:bx0 + w]
            m = valid.copy()
            for _ in range(3):                       # robust local ratio (outliers left out)
                mf = m.astype(np.float32)
                ref = (cv2.GaussianBlur((r * mf).astype(np.float32), (0, 0), sig)
                       / np.maximum(cv2.GaussianBlur(mf, (0, 0), sig), 1e-6))
                m = valid & (np.abs(r - ref) < thr)
            out[fi] = m
        # A frame is left out of a cell only as a minority: where most views of
        # the cell disagree with the median (relief / parallax), none is left out.
        agree = np.zeros((Hl, Wl), np.float32)
        views = np.zeros((Hl, Wl), np.float32)
        for fi, (bx0, by0, t, valid) in own.items():
            h, w = valid.shape
            agree[by0:by0 + h, bx0:bx0 + w] += out[fi]
            views[by0:by0 + h, bx0:bx0 + w] += valid
        frac = agree / np.maximum(views, 1)
        for fi, (bx0, by0, t, valid) in own.items():
            h, w = valid.shape
            cons = (frac[by0:by0 + h, bx0:bx0 + w] >= 0.6) & (views[by0:by0 + h, bx0:bx0 + w] >= 3)
            bad = valid & ~out[fi] & cons
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            out[fi] = valid & ~bad
        return out

    def _flat_thumb(self, fr):
        t = fr.thumb.astype(np.float32)
        if self.flat is not None and self.flat.shape[:2] == t.shape[:2]:
            t = t / self.flat
        return t

    def _flat_at(self, fr, ud, vd):
        """Flat field at distorted reference-lens coordinates (h x w x B)."""
        ts = fr.thumb_scale
        return _remap_multi(self.flat, (ud * ts).astype(np.float32), (vd * ts).astype(np.float32),
                            cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

    def prepare(self):
        """Low-frequency gains (always: also used to compare frames for seams) and seamlines."""
        if any(fr.thumb is not None for fr in self.frames):
            self.low_frequency_pass()
        if self.seam_mode == "graphcut":
            imgs = {}
            for fr in self.frames:
                if fr.seam_img is not None:
                    img = fr.seam_img.astype(np.float32)
                    if self.flat is not None:
                        fv = self.flat[..., [b for b in self.vis_bands if b < self.flat.shape[2]]]
                        fl = cv2.resize(fv, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                        img = img / np.maximum(fl.reshape(img.shape[0], img.shape[1], -1), 0.05)
                    imgs[fr.index] = (img, fr.seam_scale)
            if len(imgs) >= 2:
                from .seams import optimise_seams
                vb = list(self.vis_bands)

                def gain_fn(fr, X, Y):
                    if fr.index not in self.gain:
                        return np.ones(X.shape + (3,), np.float32)
                    g = self._gain_at(fr, X, Y)
                    return g[..., vb] if g.shape[2] > max(vb) else g
                self.seams = optimise_seams(self.frames, self.dist, (self.W, self.H), imgs,
                                            _score, _map, gain_fn, deghost=self.deghost,
                                            log=self.log)

    def _gain_at(self, fr, X, Y):
        bx0, by0, ls, g = self.gain[fr.index]
        mx = (X * ls - bx0 - 0.5).astype(np.float32)
        my = (Y * ls - by0 - 0.5).astype(np.float32)
        out = _remap_multi(g, mx, my)
        return np.where(out > 0, out, 1.0)

    # ---------------------------------------------------------- full pass
    def _others(self, fr):
        x0, y0, x1, y1 = fr.bbox
        return [g for g in self.frames if g is not fr and g.bbox[0] < x1 and g.bbox[2] > x0
                and g.bbox[1] < y1 and g.bbox[3] > y0]

    def _lose_coarse(self, fr, others, xs, xe, ys, ye, step=4):
        """
        Coarse seam decision for the block [ys:ye, xs:xe] on every `step`-th
        pixel: (valid, lose, Xc, Yc), with one grid cell of margin on every side.

        `lose` marks cells that another frame wins. Seam positions only need to
        be approximately right, and this is ~step^2 times cheaper with many
        overlapping frames. The mask is eroded by one cell (the margin makes this
        independent of block boundaries) so that, near any seam, the current frame
        keeps the pixel -- coarse decisions can create small overlaps (later
        frames overwrite) but never holes.
        """
        Xc, Yc = np.meshgrid(np.arange(xs - step, xe + step, step, dtype=np.float64),
                             np.arange(ys - step, ye + step, step, dtype=np.float64))
        uc, vc = _map(fr, Xc, Yc)
        sc, valid, _, _ = _score(fr, uc, vc, self.dist)
        lose = np.zeros(Xc.shape, bool)
        if self.blend == "average" or not valid.any():
            return valid, lose, Xc, Yc
        cand = [g for g in others if g.bbox[0] < xe and g.bbox[2] > xs
                and g.bbox[1] < ye and g.bbox[3] > ys]
        if not cand:
            return valid, lose, Xc, Yc
        if self.seams is not None:
            # optimised seamlines decide, as long as the chosen frame really covers the pixel
            lab = self.seams.sample(Xc, Yc)
            decided = lab == fr.index
            for g in cand:
                m = lab == g.index
                if m.any():
                    ug, vg = _map(g, Xc, Yc)
                    _, vgm, _, _ = _score(g, ug, vg, self.dist)
                    lose |= m & vgm
                    decided |= m & vgm
            open_ = ~decided
        else:
            open_ = np.ones(Xc.shape, bool)
        if open_.any():
            for g in cand:
                ug, vg = _map(g, Xc, Yc)
                sg, vgm, _, _ = _score(g, ug, vg, self.dist)
                lose |= open_ & vgm & ((sg < sc) | ((sg == sc) & (g.index < fr.index)))
        if lose.any():
            lose = cv2.erode(lose.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        return valid, lose, Xc, Yc

    @staticmethod
    def _upsample(mask, step, h, w):
        """Coarse cell mask (with one cell of margin) -> h x w pixel mask."""
        return np.repeat(np.repeat(mask[1:-1, 1:-1], step, 0), step, 1)[:h, :w]

    def render(self, load_lenses, out_name="mosaic"):
        """load_lenses(frame) -> {lens: H x W x C array}. Returns the memmap."""
        shape = (self.H, self.W, self.B)
        need = shape[0] * shape[1] * shape[2] * self.dtype.itemsize
        self.log(f"  canvas {self.W} x {self.H} px, {self.B} band(s), "
                 f"{need / 1e9:.2f} GB ({self.blend} blending)")
        canvas_path = self.work_dir / f"{out_name}.canvas"
        self.canvas_path = canvas_path
        out = sparse_memmap(canvas_path, self.dtype, shape)
        acc = wsum = None
        if self.blend == "average":
            acc = sparse_memmap(self.work_dir / "acc.f32", np.float32, shape)
            wsum = sparse_memmap(self.work_dir / "wsum.f32", np.float32, shape[:2])
        if not self.gain and self.seams is None:
            self.prepare()
        # narrow feathering across optimised seamlines (two-band blending only;
        # 'disabled' keeps every pixel value exactly as in one source frame)
        feathering = (self.feather and self.blend == "mosaic" and self.seams is not None
                      and bool(self.seams.soft))
        seam_acc = _SeamAccumulator(self.B) if feathering else None

        step, block_w = 4, 512
        for n, fr in enumerate(self.frames):
            x0, y0, x1, y1 = fr.bbox
            if x1 <= x0 or y1 <= y0:
                continue
            lenses = None
            others = self._others(fr)
            Ainv = {ln: np.linalg.inv(fr.lens_A[ln]) for ln in fr.lens_A}
            # blocks of strip x block_w px; a coarse pass first skips blocks this frame
            # neither wins nor feathers (most of a frame with 5-10x overlap)
            for ys in range(y0, y1, self.strip):
                ye = min(y1, ys + self.strip)
                for xs in range(x0, x1, block_w):
                    xe = min(x1, xs + block_w)
                    valid_c, lose_c, Xc, Yc = self._lose_coarse(fr, others, xs, xe, ys, ye, step)
                    if not valid_c.any():
                        continue
                    need = valid_c & ~lose_c
                    wf_c = self.seams.weight(fr.index, Xc, Yc) if feathering else None
                    if wf_c is not None:
                        need |= valid_c & (wf_c > 0.002) & (wf_c < 0.998)
                    if not (cv2.dilate(need.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[1:-1, 1:-1].any():
                        continue
                    X, Y = np.meshgrid(np.arange(xs, xe, dtype=np.float64),
                                       np.arange(ys, ye, dtype=np.float64))
                    u, v = _map(fr, X, Y)
                    s, valid, ud, vd = _score(fr, u, v, self.dist)
                    if not valid.any():
                        continue
                    if self.blend == "average":
                        sel = valid
                    else:
                        sel = valid & ~self._upsample(lose_c, step, *X.shape)
                    band = None
                    if wf_c is not None:
                        wf = self.seams.weight(fr.index, X, Y)
                        band = valid & (wf > 0.002) & (wf < 0.998)
                        if not band.any():
                            band = None
                    region = sel if band is None else (sel | band)
                    if not region.any():
                        continue
                    if lenses is None:
                        lenses = load_lenses(fr)
                    rows = np.where(region.any(1))[0]
                    cols = np.where(region.any(0))[0]
                    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
                    sel = sel[r0:r1, c0:c1]
                    udc, vdc = ud[r0:r1, c0:c1], vd[r0:r1, c0:c1]
                    vals = np.zeros(sel.shape + (self.B,), np.float32)
                    for ln, A in Ainv.items():
                        bands = [b for b, (bl, _) in enumerate(self.band_sources) if bl == ln]
                        if not bands:
                            continue
                        mx = (A[0, 0] * udc + A[0, 1] * vdc + A[0, 2]).astype(np.float32)
                        my = (A[1, 0] * udc + A[1, 1] * vdc + A[1, 2]).astype(np.float32)
                        img = lenses[ln]
                        chans = [self.band_sources[b][1] for b in bands]
                        sub = img[..., chans] if img.ndim == 3 else img[..., None]
                        r = _remap_multi(np.ascontiguousarray(sub), mx, my)
                        vals[..., bands] = r
                    Xb, Yb = X[r0:r1, c0:c1], Y[r0:r1, c0:c1]
                    if self.flat is not None:
                        vals /= np.maximum(self._flat_at(fr, udc, vdc), 0.05)
                    if self.blend == "mosaic" and fr.index in self.gain:  # two-band blending
                        vals *= self._gain_at(fr, Xb, Yb)
                    oy, ox = ys + r0, xs + c0
                    if self.blend == "average":
                        w = _cheb_weight(fr, udc, vdc) * sel + 1e-6 * sel
                        acc[oy:oy + sel.shape[0], ox:ox + sel.shape[1]] += vals * w[..., None]
                        wsum[oy:oy + sel.shape[0], ox:ox + sel.shape[1]] += w
                    else:
                        tgt = out[oy:oy + sel.shape[0], ox:ox + sel.shape[1]]
                        v = np.clip(vals, 1, self.maxv).astype(self.dtype)
                        tgt[sel] = v[sel]
                        if band is not None:
                            seam_acc.add(oy, ox, band[r0:r1, c0:c1], vals, wf[r0:r1, c0:c1])
            del lenses
            if self.progress:
                self.progress(n + 1, len(self.frames))
        if seam_acc is not None:
            n = seam_acc.flush(out, self.maxv, self.dtype)
            self.log(f"  seamlines feathered ({n} tiles of 32 px)")
        if self.blend == "average":
            for ys in range(0, self.H, self.strip):
                w = np.asarray(wsum[ys:ys + self.strip])
                a = np.asarray(acc[ys:ys + self.strip]) / np.maximum(w, 1e-9)[..., None]
                o = np.clip(a, 1, self.maxv).astype(self.dtype)
                o[w <= 0] = 0
                out[ys:ys + self.strip] = o
            del acc, wsum
            for f in ("acc.f32", "wsum.f32"):
                try:
                    os.remove(self.work_dir / f)
                except OSError:
                    pass
        out.flush()
        return out

    # ---------------------------------------------------------- previews
    def seam_map(self, max_side=1600):
        """Label image (frame index per pixel, -1 = empty) at preview scale."""
        s = min(1.0, max_side / max(self.W, self.H))
        Wp, Hp = max(1, int(self.W * s)), max(1, int(self.H * s))
        X, Y = np.meshgrid((np.arange(Wp) + 0.5) / s, (np.arange(Hp) + 0.5) / s)
        if self.seams is not None:
            return self.seams.sample(X, Y).astype(np.int32), s
        best = np.full((Hp, Wp), np.inf)
        label = np.full((Hp, Wp), -1, np.int32)
        for fr in self.frames:
            u, v = _map(fr, X, Y)
            sc, valid, _, _ = _score(fr, u, v, self.dist)
            m = valid & (sc < best)
            best[m] = sc[m]
            label[m] = fr.index
        return label, s


def _set_sparse(path) -> bool:
    """Mark a file sparse on NTFS (Windows). Without this, writing into a large
    memory-mapped canvas far from its start makes Windows zero-fill everything
    before that point first: rendering slowed down from minutes to hours per 100
    frames as it moved through a 50-160 GB canvas."""
    if os.name != "nt":
        return True                       # Linux / macOS: unwritten parts are holes already
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        FSCTL_SET_SPARSE = 0x000900C4
        with open(path, "r+b") as fh:
            h = msvcrt.get_osfhandle(fh.fileno())
            ret = wintypes.DWORD()
            ok = ctypes.windll.kernel32.DeviceIoControl(
                wintypes.HANDLE(h), FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(ret), None)
        return bool(ok)
    except Exception:                     # noqa: BLE001 (FAT/exFAT, network drives)
        return False


def sparse_memmap(path, dtype, shape):
    """Zero-initialised disk array that only occupies the parts written to."""
    dtype = np.dtype(dtype)
    size = int(np.prod(shape)) * dtype.itemsize
    with open(path, "wb"):
        pass
    _set_sparse(path)
    with open(path, "r+b") as fh:
        fh.truncate(size)
    return np.memmap(path, dtype=dtype, mode="r+", shape=tuple(shape))


def downsample(arr, max_side=3000):
    """Area-downsample a (memmap) H x W x C array in row chunks."""
    H, W = arr.shape[:2]
    f = max(1, int(np.ceil(max(H, W) / max_side)))
    Ho, Wo = H // f, W // f
    out = np.zeros((Ho, Wo, arr.shape[2]), np.float32)
    # row chunks of about 256 MB as float32 (a 4096-row chunk of a 90 000 px wide
    # mosaic was 4.5 GB and made machines with little memory swap)
    C = arr.shape[2]
    rows = f * max(1, int(256e6 / (Wo * f * C * 4) / f))
    for y in range(0, Ho * f, rows):
        blk = np.asarray(arr[y:min(y + rows, Ho * f), :Wo * f]).astype(np.float32)
        h = blk.shape[0] // f
        blk = blk[:h * f].reshape(h, f, Wo, f, -1).mean(axis=(1, 3))
        out[y // f:y // f + h] = blk
    return out


def visualise(img: np.ndarray, channels=(0, 1, 2), gamma=0.5) -> np.ndarray:
    """Float H x W x C -> uint8 BGR for JPEG, per-channel 1-99.5 % stretch."""
    rgb = img[..., list(channels)].astype(np.float32)
    valid = (img > 0).any(axis=2)
    out = np.zeros(rgb.shape, np.float32)
    for c in range(3):
        ch = rgb[..., c]
        vals = ch[valid]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [1, 99.5])
        out[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    out[~valid] = 0
    return (out[..., ::-1] * 255).astype(np.uint8)
