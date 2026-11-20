"""
pipeline.py

End-to-end 2D mosaicking, modelled on the photo-alignment and orthomosaic
steps of photogrammetry software such as Agisoft Metashape, reduced to a
planar (2D) scene model:

  1. Ingest      decode frames; for multi-lens cameras register every lens to
                 the reference lens (bands.py); choose the reference band.
  2. Features    one or several detector/descriptor methods (features.py).
  3. Pairs       pair preselection: sequential neighbours + "generic"
                 (all-pairs matching on small thumbnails) and, after a first
                 solution, "estimated" (pairs whose footprints overlap).
  4. Matching    per pair a method cascade (first method that passes all
                 checks wins), a pooled union of methods, or a single method;
                 phase correlation as a last resort for low-texture neighbours.
  5. Adjustment  global bundle adjustment of all transforms, optional radial
                 distortion self-calibration, iterative outlier removal.
  6. Render      out-of-core mosaic with seam selection and blending.
  7. Report      JSON / Markdown / CSV, seam map and preview.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from . import bands as BD
from . import bundle as BA
from . import camera as CAM
from . import features as FT
from . import matching as MT
from . import rawio
from . import render as RD
from . import sources as SR
from .tiffwriter import write_tiff, write_tiled

ACCURACY = {"highest": 6000, "high": 3600, "medium": 2400, "low": 1200, "lowest": 700}


@dataclass
class Settings:
    profile: str = "auto"            # auto | rgb | agrowing6 | multiband | profile.json
    methods: str = "auto"            # auto | comma list (see `mosaic2d methods`)
    strategy: str = "cascade"        # cascade | union | single
    cascade_limit: int = 4           # max. methods tried per pair (cascade)
    model: str = "camera"            # camera (photos over a plane) | homography | affine | similarity | rigid | translation
    focal_px: float | None = None    # nominal focal length in native px (default: EXIF / camera profile)
    height_m: float | None = None    # typical camera-to-ground distance (m) -> metric pixel size
    height_sigma: float = 0.1        # camera model: heights within ~ +-10 % unless the tie points say otherwise
    tilt_sigma: float = 15.0         # camera model: tilt from nadir (deg), prior width
    accuracy: str = "medium"         # working resolution for matching (max side px)
    n_features: int = 8000
    preselection: str = "auto"       # auto | sequential | generic | exhaustive
    window: int = 3                  # sequential neighbours
    sequence: bool = False           # photos in shooting order: neighbours matched harder, gaps bridged
    lens_from: str | None = None     # report.json / project.pkl of an earlier run: reuse its lens calibration
    distortion: bool = True          # self-calibrate radial distortion k1, k2
    ref_band: str | None = None      # e.g. "560 nm"; None = automatic
    blend: str = "mosaic"            # mosaic | average | disabled
    resolution: float = 1.0          # output px per native px of the reference lens
    orient: str = "pca"              # pca (long axis horizontal) | anchor
    demosaic: str | None = None      # superpixel | bilinear (RAW); None = profile default
    phasecorr: bool = True
    lens_affine: bool = True
    jobs: int = 0
    max_images: int = 0
    work_dir: str | None = None
    keep_canvas: bool = False
    tile_size: int = 0               # >0: write tiles of this size + a VRT instead of one TIFF
    min_block: int = 5               # also render other connected groups of >= this many images (0: largest only)
    seams: str = "graphcut"          # graphcut (optimised seamlines) | voronoi (closest centre)
    cache: bool = True               # reuse decoded/prepared frames between runs (<out>/_cache)
    deghost: bool = True             # keep moving objects (rope, fish) out of the mosaic where possible
    feather: bool = True             # soften seamlines over a narrow band (mosaic blending)
    gsd: float | None = None         # metres per pixel at resolution 1.0 (scale only, not georeferencing)
    flatfield: str = "auto"          # auto (estimate vignetting from the series) | none
    ratio: float = 0.8
    ransac_px: float = 3.0
    min_inliers: int = 25
    guided: bool = True

    def match_params(self) -> MT.MatchParams:
        return MT.MatchParams(ratio=self.ratio, ransac_px=self.ransac_px,
                              min_inliers=self.min_inliers, guided=self.guided)


@dataclass
class Frame:
    index: int
    path: Path
    exif: dict
    ref_size: tuple = (0, 0)          # native px of the reference lens image
    work_scale: float = 1.0
    reg: np.ndarray | None = None     # uint8 working image of the reference band
    lens: BD.FrameLensAlignment | None = None
    valid: tuple = (0, 0, 0, 0)
    thumb: np.ndarray | None = None
    thumb_scale: float = 1.0
    seam_img: np.ndarray | None = None  # visual bands, reference-lens px * seam_scale (uint16)
    seam_scale: float = 0.0
    qimg: np.ndarray | None = None    # log-intensity image (half working res, float32) for quality
    sharpness: float = 0.0            # Laplacian energy of qimg
    quality: float = 1.0              # sharpness relative to the series median


def quality_image(g: np.ndarray) -> np.ndarray:
    """
    Image for sharpness / contrast measurements: log intensity at half size.

    No CLAHE or local normalisation (those restore the contrast a veil removes,
    and amplify noise, so a hazy frame would look *sharper*). In the log domain
    multiplicative differences (exposure, vignetting) do not change the
    Laplacian, additive veils (scattered light) and blur reduce it.
    """
    g = cv2.resize(g.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    med = float(np.median(g[::4, ::4])) or 1.0
    return np.log(np.maximum(g, 0) + 0.02 * med + 1e-6).astype(np.float32)


def image_sharpness(img: np.ndarray) -> float:
    """Mean squared Laplacian of an image (quality_image), central 80 %."""
    img_u8 = img
    h, w = img_u8.shape[:2]
    c = img_u8[h // 10: h - h // 10, w // 10: w - w // 10]
    g = cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.0)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


QCELLS = 3                       # overlap split into QCELLS x QCELLS cells per image


def _overlap_log_ratio(a: np.ndarray, b: np.ndarray, H: np.ndarray, s: float = 0.5,
                       cells: int = 0):
    """log(sharpness of a / sharpness of b) on their common area (H: a -> b).
    ``s``: internal down-scaling of a and b (1.0 when they are already small).

    With ``cells`` > 0 the common area is split by where it lies in each image
    (a cells x cells grid in both) and a list [(mean basis(a) - mean basis(b),
    log ratio, weight)] is returned instead, the basis being a quadratic in the
    image position (u, v, u^2, v^2, uv): sharpness varies inside a frame (a camera tilted
    forward sees its far edge through more water), and consecutive frames overlap
    with the far edge of one and the near edge of the next."""
    S = np.diag([s, s, 1.0])
    a, b = a.astype(np.float32), b.astype(np.float32)
    a2 = a if s == 1.0 else cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b2 = b if s == 1.0 else cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    Hs = S @ H @ np.linalg.inv(S)
    ones_a = np.full(a2.shape, 255, np.uint8)
    ones_b = np.full(b2.shape, 255, np.uint8)
    a_in_b = cv2.warpPerspective(a2, Hs, b2.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    m_b = cv2.warpPerspective(ones_a, Hs, b2.shape[::-1], flags=cv2.INTER_NEAREST)
    b_in_a = cv2.warpPerspective(b2, np.linalg.inv(Hs), a2.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    m_a = cv2.warpPerspective(ones_b, np.linalg.inv(Hs), a2.shape[::-1], flags=cv2.INTER_NEAREST)

    def lap2(img):
        g = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.Laplacian(g, cv2.CV_32F) ** 2

    k = np.ones((15, 15), np.uint8)                      # > blur + Laplacian support
    mb = cv2.erode(m_b, k) > 0
    ma = cv2.erode(m_a, k) > 0
    if not cells:
        if mb.sum() <= 2000 or ma.sum() <= 2000:
            return None
        e = [lap2(a_in_b)[mb].mean(), lap2(b2)[mb].mean(), lap2(a2)[ma].mean(), lap2(b_in_a)[ma].mean()]
        if min(e) <= 0:
            return None
        # (warped a vs native b) and (native a vs warped b): interpolation bias cancels
        return 0.5 * (np.log(e[0] / e[1]) + np.log(e[2] / e[3]))

    def cell_of(x, y, shape):
        h, w = shape
        return (np.clip((y * cells / h).astype(int), 0, cells - 1) * cells
                + np.clip((x * cells / w).astype(int), 0, cells - 1))

    def basis(x, y, shape):
        h, w = shape
        u, v = 2 * x / w - 1, 2 * y / h - 1
        return np.stack([u, v, u * u, v * v, u * v])

    nk = cells ** 4

    def grouped(mask, e_num, e_den, own_shape, other_shape, Hm, own_is_a):
        ys, xs = np.nonzero(mask)
        if len(xs) <= 2000:
            return {}
        p = cv2.perspectiveTransform(np.column_stack([xs, ys]).astype(np.float64)[None], Hm)[0]
        c_own = cell_of(xs, ys, own_shape)
        c_oth = cell_of(p[:, 0], p[:, 1], other_shape)
        key = c_own * cells * cells + c_oth if own_is_a else c_oth * cells * cells + c_own
        nn = np.bincount(key, minlength=nk).astype(np.float64)
        en = np.bincount(key, e_num[ys, xs], minlength=nk)
        ed = np.bincount(key, e_den[ys, xs], minlength=nk)
        b_own = basis(xs.astype(np.float64), ys.astype(np.float64), own_shape)
        b_oth = basis(p[:, 0], p[:, 1], other_shape)
        b_a, b_b = (b_own, b_oth) if own_is_a else (b_oth, b_own)
        diff = np.stack([np.bincount(key, b_a[t] - b_b[t], minlength=nk) for t in range(5)], 1)
        return {kk: (en[kk], ed[kk], nn[kk], diff[kk] / nn[kk]) for kk in np.nonzero(nn > 800)[0]}

    g1 = grouped(mb, lap2(a_in_b), lap2(b2), b2.shape, a2.shape, np.linalg.inv(Hs), False)
    g2 = grouped(ma, lap2(a2), lap2(b_in_a), a2.shape, b2.shape, Hs, True)
    out = []
    for kk in set(g1) & set(g2):
        (n1, d1, w1, f1), (n2, d2, w2, f2) = g1[kk], g2[kk]
        if min(n1, d1, n2, d2) <= 0:
            continue
        r = 0.5 * (np.log(n1 / d1) + np.log(n2 / d2))
        out.append((0.5 * (f1 + f2), float(r), float(np.sqrt(min(w1, w2)))))
    return out or None


def short_label(stem: str) -> str:
    """Short frame label for seam maps: the image number if there is one
    (DSC06795_ARW_aligned_multiband -> 6795), else the last characters."""
    import re
    nums = re.findall(r"\d{3,}", stem)
    return nums[0][-4:] if nums else stem[-4:]


def local_scale(H: np.ndarray, x: float, y: float) -> float:
    """Linear scale of homography H at (x, y): sqrt |det Jacobian|."""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    p = (H[:2, :2] @ [x, y] + H[:2, 2]) / w
    J = (H[:2, :2] - np.outer(p, H[2, :2])) / w
    return float(np.sqrt(abs(np.linalg.det(J))))


def seam_penalty(quality: float) -> float:
    """Seam score multiplier: blurred frames (quality < 1) lose overlaps to sharper ones."""
    return float(np.clip(1.0 / max(quality, 1e-3) ** 2, 0.5, 25.0))


class Log:
    def __init__(self, path: Path | None = None, quiet=False):
        self.t0 = time.time()
        self.fh = open(path, "w", encoding="utf-8") if path else None
        self.quiet = quiet

    def __call__(self, *msg):
        line = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(m) for m in msg)
        if not self.quiet:
            print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


class Project:
    def __init__(self, inputs, out_dir, settings: Settings, log: Log | None = None):
        self.s = settings
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = log or Log(self.out / "mosaic2d.log")
        self.files = SR.list_images(inputs)
        if settings.max_images:
            self.files = self.files[: settings.max_images]
        if len(self.files) < 2:
            raise ValueError("Need at least two images")
        prof_name = settings.profile
        if prof_name == "auto":
            prof_name = SR.detect_profile(self.files[0])
        self.prof = SR.get_profile(prof_name, self.files[0])
        self.registry = FT.build_registry(settings.n_features)
        self.methods = FT.resolve_methods(settings.methods, self.registry)
        if not self.methods:
            raise ValueError("None of the requested feature methods is available")
        self.feats: dict = {}
        self.frames: list[Frame] = []
        self.pairs: dict = {}            # (i, j) -> PairResult (accepted or not)
        self.ref_band: SR.Band | None = None
        self.band_scores: dict = {}
        self.jobs = settings.jobs or min(4, os.cpu_count() or 1)
        self.result: BA.BundleResult | None = None
        self.timing: dict = {}

    # ------------------------------------------------------------------ ingest
    def _work_scale(self, w, h):
        return min(1.0, ACCURACY[self.s.accuracy] / max(w, h))

    def _check_forced_band(self, band):
        """A reference band given by hand: warn if it is clipped or black in this data set,
        and name the band that would be better (saturation only, no matching)."""
        try:
            n = len(self.files)
            idx = sorted({int(n * q) for q in (0.25, 0.5, 0.75)} & set(range(n)))
            sample = [SR.load_frame(self.files[i], self.prof, self.s.demosaic) for i in idx]
            stats = {}
            for b in self.prof.bands:
                valid = [None] * len(sample)
                for k, fd in enumerate(sample):
                    for bb in self.prof.bands:
                        nz = BD.band_image(fd, bb) > 0
                        valid[k] = nz if valid[k] is None else (valid[k] | nz)
                sd = np.array([BD.band_saturation(BD.band_image(fd, b), v) for fd, v in zip(sample, valid)])
                stats[b.name] = (float(np.median(sd[:, 0])), float(np.median(sd[:, 1])))
        except Exception:                        # noqa: BLE001
            return
        sat, dark = stats[band.name]
        if sat > 0.05 or dark > 0.3:
            ok = [b for b in self.prof.bands if stats[b.name][0] < 0.01 and stats[b.name][1] < 0.1]
            alt = min(ok, key=lambda b: abs((b.wavelength_nm or 550) - 550)).name if ok else None
            self.log(f"  WARNING: reference band {band.name} (set by hand) is {sat * 100:.0f} % clipped "
                     f"and {dark * 100:.0f} % black in this data set: few features, weak alignment."
                     + (f" {alt} is not clipped; leave the field empty to choose automatically."
                        if alt else ""))

    def _select_reference_band(self):
        prof = self.prof
        if len(prof.bands) == 1:
            self.ref_band = prof.bands[0]
            return
        if self.s.ref_band:
            want = self.s.ref_band.strip().lower().replace("nm", "").strip()
            for b in prof.bands:
                if b.name.lower().replace("nm", "").strip() == want:
                    self.ref_band = b
                    self._check_forced_band(b)
                    return
            raise ValueError(f"Reference band '{self.s.ref_band}' not in profile: "
                             + ", ".join(b.name for b in prof.bands))
        if not prof.multilens and prof.name == "rgb":
            self.ref_band = None  # luminance of all channels
            return
        # runs of consecutive photos at 1/4, 1/2 and 3/4 of the series (one run of the
        # middle photos could be blank water, a slate or a turn)
        n = len(self.files)
        run_len = 3 if n >= 9 else min(n, 6)
        starts = sorted({max(0, min(n - run_len, int(n * q) - run_len // 2))
                         for q in ((0.25, 0.5, 0.75) if n >= 9 else (0.5,))})
        idx, groups = [], []
        for st in starts:
            g = [i for i in range(st, st + run_len) if i not in idx]
            groups.append(list(range(len(idx), len(idx) + len(g))))
            idx += g
        sample = [SR.load_frame(self.files[i], prof, self.s.demosaic) for i in idx]
        any_lens = sample[0].lenses[prof.bands[0].lens]
        # band ranking does not need the full working resolution
        ws = min(self._work_scale(any_lens.shape[1], any_lens.shape[0]), 1400 / max(any_lens.shape[:2]))
        meth = self.registry[self.methods[0]].build
        p = self.s.match_params()
        p.guided = False
        best, scores = BD.choose_reference_band(
            sample, prof, lambda g: meth(g), lambda a, b: MT.match_pair(a, b, 0, 1, p), ws, self.log,
            groups=groups)
        self.ref_band, self.band_scores = best, scores
        self.log(f"  reference band: {best.name} (lens {best.lens})")

    def _reg_image(self, fd: SR.FrameData):
        if self.ref_band is None:
            g = SR.to_gray_float(fd.lenses[1])
        else:
            g = SR.band_image(fd, self.ref_band).astype(np.float32)
        return g

    def _ingest_one(self, idx: int, path: Path, priors=None):
        prof = self.prof
        fd = SR.load_frame(path, prof, self.s.demosaic)
        ref_lens = self.ref_band.lens if self.ref_band else 1
        ref_img = fd.lenses[ref_lens]
        h, w = ref_img.shape[:2]
        fr = Frame(idx, path, fd.exif, (w, h))
        if prof.multilens:
            fr.lens = BD.align_frame_lenses(fd, prof, ref_lens, scale=min(1.0, 1400 / max(w, h)),
                                            priors=priors, max_dev=40 if priors else None,
                                            affine=self.s.lens_affine)
        self._finish_frame(fr, fd)
        return fr

    def _finish_frame(self, fr: Frame, fd: SR.FrameData):
        prof = self.prof
        w, h = fr.ref_size
        g = self._reg_image(fd)
        fr.work_scale = self._work_scale(w, h)
        if fr.work_scale != 1.0:
            g = cv2.resize(g, None, fx=fr.work_scale, fy=fr.work_scale, interpolation=cv2.INTER_AREA)
        fr.reg = SR.normalise_for_features(g)
        fr.qimg = quality_image(g)
        fr.sharpness = image_sharpness(fr.qimg)
        # valid rectangle (in ref px) shared by all lenses
        m = 2.0
        x0, y0, x1, y1 = m, m, w - m, h - m
        if fr.lens is not None:
            for ln in range(1, prof.n_lenses + 1):
                if ln == fr.lens.ref_lens:
                    continue
                lh, lw = fd.lenses[ln].shape[:2]
                c = np.array([[0, 0], [lw, 0], [lw, lh], [0, lh]], np.float64)
                pc = cv2.perspectiveTransform(c[None], fr.lens.A(ln))[0]
                x0 = max(x0, pc[0, 0], pc[3, 0]) + 0
                x1 = min(x1, pc[1, 0], pc[2, 0])
                y0 = max(y0, pc[0, 1], pc[1, 1])
                y1 = min(y1, pc[2, 1], pc[3, 1])
            x0, y0, x1, y1 = x0 + m, y0 + m, x1 - m, y1 - m
        fr.valid = (float(x0), float(y0), float(x1), float(y1))
        # thumbnail of all output bands in reference-lens coordinates
        ts = min(1.0, 400.0 / max(w, h))
        tw, th = max(1, int(round(w * ts))), max(1, int(round(h * ts)))
        S = np.diag([ts, ts, 1.0])
        chans = []
        for b in prof.output_bands():
            img = SR.band_image(fd, b).astype(np.float32)
            lh, lw = img.shape[:2]
            small = cv2.resize(img, (max(1, int(round(lw * ts))), max(1, int(round(lh * ts)))),
                               interpolation=cv2.INTER_AREA)
            A = np.eye(3) if fr.lens is None else fr.lens.A(b.lens)
            M = S @ A @ np.linalg.inv(S)
            chans.append(cv2.warpPerspective(small, M, (tw, th), flags=cv2.INTER_LINEAR))
        fr.thumb = np.clip(np.dstack(chans), 0, 65535).astype(np.uint16)
        fr.thumb_scale = ts
        # medium-resolution visual image for seamline optimisation
        ss = min(1.0, 600.0 / max(w, h))   # seam map uses ~370 px per frame
        sw, sh = max(1, int(round(w * ss))), max(1, int(round(h * ss)))
        S2 = np.diag([ss, ss, 1.0])
        bands = prof.output_bands()
        vis = []
        for k in self.vis_channels():
            b = bands[k]
            img = SR.band_image(fd, b).astype(np.float32)
            lh, lw = img.shape[:2]
            small = cv2.resize(img, (max(1, int(round(lw * ss))), max(1, int(round(lh * ss)))),
                               interpolation=cv2.INTER_AREA)
            A = np.eye(3) if fr.lens is None else fr.lens.A(b.lens)
            vis.append(cv2.warpPerspective(small, S2 @ A @ np.linalg.inv(S2), (sw, sh),
                                           flags=cv2.INTER_LINEAR))
        fr.seam_img = np.clip(np.dstack(vis), 0, 65535).astype(np.uint16)
        fr.seam_scale = ss

    # ------------------------------------------------------------------ cache
    def _cache_dir(self) -> Path:
        return Path(self.s.work_dir or self.out) / "_cache"

    def _cache_key(self, path: Path, extra="") -> str:
        """Depends on the file (name, size, time) and every setting that changes ingest."""
        import hashlib
        st = path.stat()
        parts = [__version__.rsplit(".", 1)[0], path.name, st.st_size, int(st.st_mtime), self.prof.name,
                 self.s.demosaic, self.s.accuracy, self.s.lens_affine,
                 self.ref_band.name if self.ref_band else "lum", extra]
        return hashlib.sha1(repr(parts).encode()).hexdigest()[:20]

    def _cache_load(self, key):
        import pickle
        f = self._cache_dir() / f"{key}.pkl"
        if not (self.s.cache and f.is_file()):
            return None
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def _cache_save(self, key, obj):
        import pickle
        if not self.s.cache:
            return
        d = self._cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{key}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        os.replace(tmp, d / f"{key}.pkl")

    def _cached_reference_band(self):
        """Reference band selection is expensive for multispectral series: cache it."""
        trivial = len(self.prof.bands) == 1 or (not self.prof.multilens and self.prof.name == "rgb")
        if not self.s.cache or self.s.ref_band or trivial:
            self._select_reference_band()
            return
        import hashlib
        key = "refband2_" + hashlib.sha1(repr([self.prof.name, self.methods[0], self.s.accuracy] + [
            (f.name, f.stat().st_size) for f in self.files]).encode()).hexdigest()[:16]
        hit = self._cache_load(key)
        if hit is not None:
            name, scores = hit
            band = None if name is None else next((b for b in self.prof.bands if b.name == name), None)
            if name is None or band is not None:
                self.ref_band, self.band_scores = band, scores
                self.log(f"  reference band (cached): {name}")
                return
        self._select_reference_band()
        self._cache_save(key, (self.ref_band.name if self.ref_band else None, self.band_scores))

    def ingest(self):
        t = time.time()
        self.log(f"mosaic2d {__version__}: {len(self.files)} images, profile '{self.prof.name}', "
                 f"methods {','.join(self.methods)}, strategy {self.s.strategy}, model {self.s.model}")
        self._cached_reference_band()
        frames = [None] * len(self.files)
        done = [0]
        keys = [self._cache_key(f) for f in self.files]
        hits = 0

        def work(i):
            fr = self._cache_load(keys[i])
            if fr is not None:
                fr.index, fr.path = i, self.files[i]
                return fr, True
            fr = self._ingest_one(i, self.files[i])
            done[0] += 1
            if done[0] % 10 == 0:
                self.log(f"  decoded {done[0]} images")
            return fr, False
        with ThreadPoolExecutor(self.jobs) as ex:
            for fr, hit in ex.map(work, range(len(self.files))):
                frames[fr.index] = fr
                hits += hit
        self.frames = frames
        if hits:
            self.log(f"  {hits}/{len(self.files)} images from the cache, {len(self.files) - hits} decoded")
        if self.prof.multilens and hits < len(self.files):
            self._lens_consensus()
        for k, fr in enumerate(self.frames):
            self._cache_save(keys[k], fr)
        self.estimate_flatfield()
        self.timing["ingest_s"] = time.time() - t

    def estimate_flatfield(self):
        """Vignetting / fixed illumination pattern from the whole series (seams.py)."""
        self.flat, self.flat_stats = None, {}
        if self.s.flatfield == "none":
            return
        from .seams import estimate_flatfield
        thumbs, valids = [], []
        for fr in self.frames:
            if fr.thumb is None:
                continue
            ts = fr.thumb_scale
            h, w = fr.thumb.shape[:2]
            x0, y0, x1, y1 = [v * ts for v in fr.valid]
            yy, xx = np.mgrid[0:h, 0:w]
            valids.append((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1) & (fr.thumb.min(2) > 0))
            thumbs.append(fr.thumb)
        flat, stats = estimate_flatfield(thumbs, valids)
        if flat is None:
            self.log(f"  flat field not estimated: {stats}")
            return
        bands = self.prof.output_bands()
        self.flat = flat
        self.flat_stats = {bands[b].name: v for b, v in stats.items()}
        worst = min(self.flat_stats.items(), key=lambda kv: kv[1])
        self.log(f"  flat field: corner/centre brightness {np.mean(list(stats.values())):.2f} on average "
                 f"(strongest falloff {worst[0]}: {worst[1]:.2f}); corrected in the mosaic")

    def _image_quality(self):
        """
        Relative sharpness per frame, estimated from overlaps so that scene
        content (smooth rock vs. textured sand) does not count as blur.

        For every accepted pair the two working images are compared on their
        common area only, in both directions (each image warped onto the other,
        so interpolation blur affects both sides equally), separately for each
        combination of image cells (QCELLS x QCELLS grid) the common area falls in.
        One sharpness per frame and one in-frame sharpness field shared by all
        frames are solved together by robust least squares (median frame = 1).
        Without the field, a camera tilted forward (far edge blurrier) made every
        frame look sharper than the one before it, and along a 280-frame strip
        the "quality" drifted from 0.00 to 3000.
        Frames without accepted pairs keep the plain per-image measure.
        """
        idx = {f.index: k for k, f in enumerate(self.frames)}
        n = len(self.frames)
        nb = 5                                          # quadratic field: u, v, u^2, v^2, uv
        rows, rhs, wts, fdiff = [], [], [], []
        for (i, j), r in self.pairs.items():
            if not r.ok or r.H is None or r.method == "phasecorr":
                continue
            fa, fb = self.frames[i], self.frames[j]
            a, b = getattr(fa, "qimg", None), getattr(fb, "qimg", None)
            if a is not None and b is not None:   # quality images: half the working resolution
                Hh = np.diag([0.5, 0.5, 1.0]) @ r.H @ np.diag([2.0, 2.0, 1.0])
                res = _overlap_log_ratio(a, b, Hh, s=1.0, cells=QCELLS)
            elif fa.reg is not None and fb.reg is not None:   # projects from older versions
                res = _overlap_log_ratio(fa.reg, fb.reg, r.H, cells=QCELLS)
            else:
                continue
            for fd, ra, w in res or ():
                rows.append((idx[i], idx[j]))
                fdiff.append(fd)
                rhs.append(ra)
                wts.append(w)
        x = np.log(np.array([max(f.sharpness, 1e-9) for f in self.frames]))
        x -= np.median(x)
        coef = np.zeros(nb)
        if rows:
            import scipy.sparse as sp
            from scipy.sparse.linalg import lsqr
            m = len(rows)
            R = np.array(rows)
            Fd = np.asarray(fdiff)
            rr = np.r_[np.repeat(np.arange(m), 2), np.repeat(np.arange(m), nb)]
            cc = np.r_[R.ravel(), np.tile(n + np.arange(nb), m)]
            vv = np.r_[np.tile([1.0, -1.0], m), Fd.ravel()]
            # every frame weakly pulled to 1. On a single strip a steady trend from
            # frame to frame and a difference between the leading and trailing edge
            # explain the same ratios; the pull lets the field take it. A blurred
            # frame contradicts its neighbours in dozens of cells and stays low.
            gr = m + np.arange(n)
            A = sp.csr_matrix((np.r_[vv, np.ones(n)], (np.r_[rr, gr], np.r_[cc, np.arange(n)])),
                              shape=(m + n, n + nb))
            y = np.r_[rhs, np.zeros(n)]
            w0 = np.asarray(wts) / np.median(wts)
            prior_w = np.full(n, 0.3)
            w = np.r_[w0, prior_w]
            for _ in range(5):                          # Cauchy IRLS: occlusions, moving objects
                sol = lsqr(sp.diags(w) @ A, w * y, atol=1e-10, btol=1e-10)[0]
                res_ = (A @ sol - y)[:m]
                sc = 1.4826 * np.median(np.abs(res_)) + 1e-3
                w = np.r_[w0 / np.sqrt(1 + (res_ / (2 * sc)) ** 2), prior_w]
            connected = np.zeros(n, bool)
            connected[R[:, 0]] = connected[R[:, 1]] = True
            q = sol[:n]
            x = np.where(connected, q - np.median(q[connected]), x)
            coef = sol[n:]
        gy, gx = np.mgrid[-1:1.01:0.5, -1:1.01:0.5]
        field = np.einsum("k,kij->ij", coef, np.stack([gx, gy, gx * gx, gy * gy, gx * gy]))
        self.sharpness_field = np.exp(field - np.median(field))          # 5 x 5 samples
        for f, v in zip(self.frames, x):
            f.quality = float(np.exp(v))
        if np.ptp(field) > np.log(1.5):
            self.log(f"  sharpness inside the frame varies {np.exp(np.ptp(field)):.1f}x "
                     f"across the frame (accounted for)")
        # photos placed only from the shooting order: any matched photo is better there
        real = set()
        for (i, j), r in self.pairs.items():
            if r.ok:
                real |= {i, j}
        for i, j in getattr(self, "sequence_links", []):
            for k in (i, j):
                if k not in real:
                    self.frames[k].quality = min(self.frames[k].quality, 0.3)
        low = [f.path.name for f in self.frames if f.quality < 0.5]
        if low:
            self.log(f"  {len(low)} blurred frame(s) (sharpness < 0.5 x median, from overlaps): "
                     f"{', '.join(low)} -- used only where no sharper frame overlaps")

    def _lens_consensus(self):
        res = [f.lens for f in self.frames]
        med, bad = BD.consensus_check(res)
        lenses_txt = ", ".join(
            f"L{ln}: ({A[0, 2]:+.1f}, {A[1, 2]:+.1f}) px" for ln, A in sorted(med.items()))
        self.log(f"  lens offsets to reference lens {res[0].ref_lens} (series median): {lenses_txt}")
        if not bad:
            return
        frames_bad = sorted({k for k, _ in bad})
        self.log(f"  {len(bad)} lens alignment(s) disagree with the series in "
                 f"{len(frames_bad)} frame(s); re-estimating around the series median")
        for k in frames_bad:
            fr = self.frames[k]
            new = self._ingest_one(k, fr.path, priors=med)
            for ln in new.lens.lenses:
                old = fr.lens.lenses[ln]
                if (k, ln) in bad and new.lens.lenses[ln].ncc < 0.5 * np.median(
                        [f.lens.lenses[ln].ncc for f in self.frames]):
                    new.lens.lenses[ln] = BD.LensResult(med[ln].copy(), "series_median", old.ncc)
            if any(new.lens.lenses[ln].method == "series_median" for ln in new.lens.lenses):
                fd = SR.load_frame(fr.path, self.prof, self.s.demosaic)
                self._finish_frame(new, fd)
            self.frames[k] = new

    # ---------------------------------------------------------------- features
    def features(self, method: str, i: int) -> FT.Features:
        key = (method, i)
        if key not in self.feats:
            self.feats[key] = self.registry[method].build(self.frames[i].reg)
        return self.feats[key]

    def compute_features(self, methods):
        t = time.time()
        todo = [(m, i) for m in methods for i in range(len(self.frames)) if (m, i) not in self.feats]
        with ThreadPoolExecutor(self.jobs) as ex:
            for (m, i), f in zip(todo, ex.map(lambda mi: self.registry[mi[0]].build(self.frames[mi[1]].reg), todo)):
                self.feats[(m, i)] = f
        self.timing["features_s"] = self.timing.get("features_s", 0) + time.time() - t

    # ------------------------------------------------------------- preselection
    def generic_candidates(self, min_inliers=12) -> dict:
        """All-pairs matching on small thumbnails ('generic preselection')."""
        t = time.time()
        meth = "sift" if "sift" in self.registry and self.registry["sift"].available else self.methods[0]
        small, sc = [], []
        for fr in self.frames:
            s = min(1.0, 900.0 / max(fr.reg.shape))
            small.append(cv2.resize(fr.reg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            sc.append(s)
        reg = FT.build_registry(1500)
        feats = list(ThreadPoolExecutor(self.jobs).map(reg[meth].build, small))
        p = MT.MatchParams(min_inliers=min_inliers, guided=False, ransac_px=3.0, min_coverage=0.003)
        n = len(self.frames)
        allp = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def m(ij):
            i, j = ij
            r = MT.match_pair(feats[i], feats[j], i, j, p)
            return ij, (r.n_inliers if r.ok else 0)
        with ThreadPoolExecutor(self.jobs) as ex:
            scores = dict(ex.map(m, allp))
        self.timing["preselection_s"] = time.time() - t
        return {k: v for k, v in scores.items() if v > 0}

    def candidate_pairs(self) -> list:
        n = len(self.frames)
        mode = self.s.preselection
        cand = set()
        seq = getattr(self.s, "sequence", False)
        if mode in ("auto", "sequential") or seq:
            window = max(self.s.window, 6) if seq else self.s.window
            for i in range(n):
                for d in range(1, window + 1):
                    if i + d < n:
                        cand.add((i, i + d))
        if mode == "exhaustive":
            cand |= {(i, j) for i in range(n) for j in range(i + 1, n)}
        self.generic_scores = {}
        if mode in ("auto", "generic") and n <= 600:
            self.generic_scores = self.generic_candidates()
            cand |= set(self.generic_scores)
            self.log(f"  generic preselection: {len(self.generic_scores)} overlapping pairs found "
                     f"among {n * (n - 1) // 2}")
        return sorted(cand)

    # ------------------------------------------------------------------ matching
    def match(self, pairs, methods=None, strategy=None, trusted=False):
        methods = methods or self.methods
        strategy = strategy or self.s.strategy
        p = self.s.match_params()
        seq = getattr(self.s, "sequence", False)
        # sequence: direct neighbours in the shooting order are known to overlap, so
        # fewer matches are enough and every method is tried before giving up
        import dataclasses
        p_seq = dataclasses.replace(p, min_inliers=max(10, p.min_inliers // 2), min_inlier_ratio=0.05,
                                    min_coverage=0.005)
        t = time.time()
        if strategy == "single":
            methods = methods[:1]
        need = methods if strategy != "cascade" else methods[:1]
        self.compute_features(need)

        def one(ij):
            i, j = ij
            scale_i, scale_j = self.frames[i].work_scale, self.frames[j].work_scale
            tried = []
            res = None
            if strategy == "union":
                res = MT.match_pair_union([self.features(m, i) for m in methods],
                                          [self.features(m, j) for m in methods], i, j, p)
                tried = list(methods)
            else:
                limit = self.s.cascade_limit if strategy == "cascade" else 1
                near = seq and j - i <= 2
                if near and strategy == "cascade":
                    limit = len(methods)
                # far pairs that the thumbnail check found no overlap for: primary method only
                far = (not trusted and not near and (j - i > 1) and bool(self.generic_scores)
                       and (ij not in self.generic_scores))
                for m in methods[: (1 if far else limit)]:
                    r = MT.match_pair(self.features(m, i), self.features(m, j), i, j, p_seq if near else p)
                    tried.append(m)
                    if res is None or r.n_inliers > res.n_inliers or r.ok:
                        res = r
                    if r.ok:
                        break
            if (not res.ok and self.s.phasecorr and (j - i == 1 or (seq and j - i == 2))
                    and strategy != "single"):
                r = MT.phase_correlation_pair(self.frames[i].reg, self.frames[j].reg, i, j)
                tried.append("phasecorr")
                if r.ok:
                    res = r
            res.tried = tried
            if res.ok and res.method != "phasecorr" and self.frames[i].reg is not None:
                h_, w_ = self.frames[i].reg.shape[:2]
                if BD._static(res, (w_, h_), 0.002):
                    # every match at the same image position: a pattern fixed to the camera
                    # (sensor noise, dust, a housing part, an empty or clipped band), or the
                    # same photo twice - not a measurement of the camera's movement
                    res.ok = False
                    res.reason = "matches at the same image position (fixed pattern or duplicate photo)"
            return ij, res
        with ThreadPoolExecutor(self.jobs) as ex:
            for ij, r in ex.map(one, pairs):
                self.pairs[ij] = r
        self.timing["matching_s"] = self.timing.get("matching_s", 0) + time.time() - t
        ok = sum(1 for ij in pairs if self.pairs[ij].ok)
        self.log(f"  matched {len(pairs)} pairs: {ok} accepted")
        stat = sum(1 for ij in pairs if "same image position" in (getattr(self.pairs[ij], "reason", "") or ""))
        if stat:
            self.log(f"  {stat} pair(s) rejected: all matches at the same image position (a pattern "
                     f"fixed to the camera or duplicate photos)"
                     + (" - most pairs: check the reference band" if stat > ok else ""))
        return ok

    # ---------------------------------------------------------------- adjustment
    def _corr(self):
        corr, pairH, sizes = [], {}, {}
        for fr in self.frames:
            sizes[fr.index] = fr.ref_size
        for (i, j), r in self.pairs.items():
            if not r.ok:
                continue
            si, sj = self.frames[i].work_scale, self.frames[j].work_scale
            p = r.pts_i / si
            q = r.pts_j / sj
            Hn = np.diag([1 / sj, 1 / sj, 1]) @ r.H @ np.diag([si, si, 1])
            w = 0.3 if r.method == "phasecorr" else 1.0
            corr.append(BA.Correspondences(i, j, p, q, w, r.method))
            pairH[(i, j)] = Hn
        self.sequence_links = []
        if getattr(self.s, "sequence", False):
            corr, pairH = self._sequence_links(corr, pairH, sizes)
        return corr, pairH, sizes

    def _sequence_links(self, corr, pairH, sizes):
        """
        Sequence mode: where neighbours in the shooting order are not connected by
        any matched pair (featureless sand, blur, a turn), add a weak link that
        assumes the typical step between consecutive photos (median shift of the
        matched neighbours, no rotation or scale change). Only links that join two
        separate groups are added, so matched geometry is not touched; photos placed
        this way are listed in the log and lose overlaps to matched photos.
        """
        n = len(self.frames)
        steps = []
        for (i, j), H in pairH.items():
            if j == i + 1:
                w, h = sizes[i]
                c = np.array([[[w / 2.0, h / 2.0]]])
                steps.append(cv2.perspectiveTransform(c, H)[0, 0] - c[0, 0])
        if not steps:
            return corr, pairH
        step = np.median(np.array(steps), axis=0)
        parent = list(range(n))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for c in corr:
            parent[root(c.i)] = root(c.j)
        corr, pairH = list(corr), dict(pairH)
        for i in range(n - 1):
            j = i + 1
            if root(i) == root(j):
                continue
            w, h = sizes[i]
            g = np.array([[x, y] for x in np.linspace(0.1, 0.9, 5) * w for y in np.linspace(0.1, 0.9, 4) * h])
            q = g + step
            wj, hj = sizes[j]
            ok = (q[:, 0] > 0) & (q[:, 0] < wj) & (q[:, 1] > 0) & (q[:, 1] < hj)
            if ok.sum() < 6:                        # typical step leaves no overlap: assume half a frame
                q = g + step * (0.5 * min(w, h) / max(1e-6, float(np.hypot(*step))))
                ok = (q[:, 0] > 0) & (q[:, 0] < wj) & (q[:, 1] > 0) & (q[:, 1] < hj)
            corr.append(BA.Correspondences(i, j, g[ok], q[ok], 0.05, "sequence"))
            T = np.array([[1, 0, step[0]], [0, 1, step[1]], [0, 0, 1]], np.float64)
            pairH[(i, j)] = T
            parent[root(i)] = root(j)
            self.sequence_links.append((i, j))
        if self.sequence_links:
            names = ", ".join(f"{self.frames[i].path.stem}-{self.frames[j].path.stem}"
                              for i, j in self.sequence_links[:10])
            more = " ..." if len(self.sequence_links) > 10 else ""
            self.log(f"  sequence: {len(self.sequence_links)} gap(s) between consecutive photos bridged "
                     f"by the typical step ({step[0]:+.0f}, {step[1]:+.0f}) px: {names}{more}")
        return corr, pairH

    def adjust(self):
        t = time.time()
        corr, pairH, sizes = self._corr()
        nodes = [f.index for f in self.frames]
        if not corr:
            raise RuntimeError("No image pair could be matched")
        self.result, self.s.model = self._solve(nodes, corr, pairH, sizes, self.s.model,
                                                self.s.distortion)
        r = self.result
        for rem in r.removed_pairs:
            key = (rem["i"], rem["j"])
            if key in self.pairs:
                self.pairs[key].ok = False
                self.pairs[key].reason = f"rejected by adjustment (median {rem['median_px']} px)"
        n_main = len(r.components[0])
        self.log(f"  adjustment: {n_main}/{len(nodes)} images in the main block, "
                 f"RMSE {r.rmse_px:.2f} px, median {r.median_px:.2f} px, "
                 f"k1={r.k[0]:+.4f} k2={r.k[1]:+.4f}, anchor {self.frames[r.anchor].path.name}")
        self.timing["adjustment_s"] = self.timing.get("adjustment_s", 0) + time.time() - t

    def _check_camera(self, r):
        """Reject a camera adjustment that ran into a non-physical corner."""
        cam = r.camera or {}
        f, f0 = cam.get("focal_px"), cam.get("focal_nominal_px")
        if f and f0 and not (1 / 3 < f / f0 < 3):
            raise ValueError(f"focal length {f:.0f} px is {f / f0:.2f} x the expected one")
        tilts = list((cam.get("tilt_deg") or {}).values())
        if tilts and max(tilts) > 60:
            raise ValueError(f"a photo tilted {max(tilts):.0f} deg")
        if not np.isfinite(r.median_px):
            raise ValueError("no finite residuals")

    def _solve(self, nodes, corr, pairH, sizes, model, distortion, lens=None):
        """
        Bundle adjustment with a geometry check. Returns (result, model used).

        Frames with degenerate transforms (folded, strongly sheared, scale far
        from the median) are dropped. The main block is then the largest group
        still connected through good pairs - a piece that was attached only
        through dropped frames is split off instead of being drawn somewhere
        beside it - and the main block is adjusted once more on its own.
        result.components = [main] + the other groups (for render_blocks).
        """
        if lens is None and model == "camera":
            lens = self.fixed_lens()

        def solve(nds, mdl):
            cs = set(nds)
            c_sub = [c for c in corr if c.i in cs and c.j in cs]
            p_sub = {e: H for e, H in pairH.items() if e[0] in cs and e[1] in cs}
            if mdl == "camera":
                f0, fsig, src = self.nominal_focal()
                kw = {}
                if lens is not None:                # a separate group: same lens as the main block
                    f0, fsig = lens["focal_px"], 0.02
                    kw = {"k_init": lens["k"], "k_sigma": 0.005}
                return CAM.run_camera_bundle(nds, c_sub, p_sub, sizes, f0, focal_sigma=fsig,
                                             distortion=distortion, height_sigma=self.s.height_sigma,
                                             tilt_sigma_deg=self.s.tilt_sigma, log=self.log, **kw)
            return BA.run_bundle(nds, c_sub, p_sub, sizes, model=mdl, distortion=distortion, log=self.log)

        if model == "camera" and lens is None:
            f0, fsig, src = self.nominal_focal()
            self.log(f"  camera model: photos over one plane, shared lens f = {f0:.0f} px ({src}), "
                     f"heights +-{self.s.height_sigma:.0%}, tilt prior {self.s.tilt_sigma:g} deg")
        # never leave the user without a mosaic: simpler models if one fails
        chain = {"camera": ["camera", "homography", "similarity"],
                 "homography": ["homography", "similarity"],
                 "affine": ["affine", "similarity"]}.get(model, [model])
        r, err = None, None
        for k, mdl in enumerate(chain):
            try:
                r = solve(nodes, mdl)
                if mdl == "camera":
                    self._check_camera(r)
                model = mdl
                break
            except Exception as e:              # noqa: BLE001
                err = e
                if k + 1 < len(chain):
                    self.log(f"  {mdl} adjustment failed ({e}); trying the {chain[k + 1]} model")
        if r is None:
            raise RuntimeError(f"adjustment failed with every model: {err}")
        bad = self.degenerate_frames(r)
        if bad and model == "homography" and any(self._folds(r, i) for i in bad):
            # long chains: perspective terms are weakly constrained and can drift
            # until far frames fold over; the affine model cannot fold. Used only
            # if it is clearly better.
            try:
                ra = solve(nodes, "affine")
                bad_a = self.degenerate_frames(ra)
            except Exception:                   # noqa: BLE001
                ra, bad_a = None, bad
            if ra is not None and len(bad_a) < len(bad) and ra.median_px <= 2.0 * r.median_px:
                self.log(f"  {len(bad)} frame(s) degenerate under the homography model; "
                         f"the affine model is better here ({len(bad_a)} degenerate, "
                         f"median {ra.median_px:.2f} px)")
                r, bad, model = ra, bad_a, "affine"
        for _ in range(2):
            if not bad:
                break
            names = ", ".join(self.frames[i].path.name for i in bad[:12]) + (" ..." if len(bad) > 12 else "")
            self.log(f"  {len(bad)} frame(s) left out of the main block (degenerate geometry): {names}")
            removed = {(d["i"], d["j"]) for d in r.removed_pairs}
            kept = [n for n in r.components[0] if n not in set(bad)]
            ks = set(kept)
            edges = [(c.i, c.j) for c in corr if c.i in ks and c.j in ks and (c.i, c.j) not in removed]
            main = BA.connected_components(kept, edges)[0] if kept else []
            if len(main) < 2:
                break
            if len(main) < len(kept):
                self.log(f"  without them the main block splits; keeping its largest piece "
                         f"({len(main)} images), the others become separate groups")
            try:
                r2 = solve(main, model)
                if model == "camera":
                    self._check_camera(r2)
            except Exception as e:              # noqa: BLE001
                self.log(f"  adjustment of the reduced block failed ({e}); keeping the previous one")
                break
            r = r2
            bad = self.degenerate_frames(r)
        if bad:
            for i in bad:
                r.H.pop(i, None)
            r.components[0] = [n for n in r.components[0] if n in r.H]
        main = set(r.components[0])
        rest = [n for n in nodes if n not in main]
        rs = set(rest)
        r.components = [sorted(main)] + BA.connected_components(
            rest, [(c.i, c.j) for c in corr if c.i in rs and c.j in rs])
        return r, model

    def _folds(self, r, i) -> bool:
        """True if frame i's transform puts a corner on or behind the horizon."""
        H = r.H.get(i)
        if H is None:
            return False
        if not np.all(np.isfinite(H)):
            return True
        w, h = self.frames[i].ref_size
        den = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [w / 2, h / 2, 1]], np.float64) @ H[2]
        return bool(np.any(den * np.sign(den[-1]) <= 1e-6 * abs(den[-1])))

    def render_blocks(self, name="mosaic"):
        """
        Groups of images that share no overlap with the largest group (gaps in
        the coverage) are adjusted and rendered on their own as
        <name>_blockNN; <name>_blocks.csv lists all groups.
        """
        main_result = self.result
        main_out = (getattr(self, "canvas_size", None), getattr(self, "T_out", None))
        comps = [c for c in main_result.components]
        results = [(comps[0], f"{name}.tif")]
        extra = [c for c in comps[1:] if len(c) >= self.s.min_block]
        small = sum(len(c) for c in comps[1:] if len(c) < self.s.min_block)
        if not extra:
            if len(comps) > 1:
                self.log(f"  {len(comps) - 1} smaller group(s) ({small} images) below --min-block "
                         f"{self.s.min_block}; not rendered")
            return []
        self.log(f"  {len(extra)} more group(s) of >= {self.s.min_block} images without overlap with "
                 f"the main block; rendering each on its own ({small} images in smaller groups skipped)")
        corr, pairH, sizes = self._corr()
        model0 = self.s.model
        try:
            for k, comp in enumerate(extra, start=2):
                bname = f"{name}_block{k:02d}"
                cs = set(comp)
                c_sub = [c for c in corr if c.i in cs and c.j in cs]
                p_sub = {e: H for e, H in pairH.items() if e[0] in cs and e[1] in cs}
                self.log(f"block {k}: {len(comp)} images ({self.frames[comp[0]].path.name} ...)")
                try:
                    # few images cannot constrain lens distortion reliably
                    cam = getattr(main_result, "camera", None)
                    lens = {"focal_px": cam["focal_px"], "k": main_result.k} if cam else None
                    self.result, _ = self._solve(comp, c_sub, p_sub, sizes, model0,
                                                 self.s.distortion and (len(comp) >= 15 or lens is not None),
                                                 lens=lens)
                    self.render(bname)
                    results.append((comp, f"{bname}.tif"))
                except Exception as e:            # one bad group must not stop the others
                    self.log(f"  block {k} not rendered: {e}")
                    results.append((comp, f"not rendered: {e}"))
        finally:
            self.result = main_result
            self.canvas_size, self.T_out = main_out
        for c in comps[1:]:
            if len(c) < self.s.min_block:
                results.append((c, "below --min-block"))
        path = _block_report_rows(self, name, results)
        self.log(f"  groups listed in {path.name}")
        return results

    def ground_pixel(self):
        """Metres per native pixel of the median frame: --gsd, or --height / focal length."""
        if getattr(self.s, "gsd", None):
            return float(self.s.gsd)
        hm = getattr(self.s, "height_m", None)
        if hm:
            cam = getattr(self.result, "camera", None) if getattr(self, "result", None) else None
            f = cam["focal_px"] if cam else self.nominal_focal()[0]
            return float(hm) / f
        return None

    def fixed_lens(self):
        """Lens calibration (focal length, k1, k2) from an earlier run (--lens-from), or None."""
        src = getattr(self.s, "lens_from", None)
        if not src:
            return None
        if getattr(self, "_fixed_lens", None) is not None:
            return self._fixed_lens
        src = Path(src)
        if src.is_dir():
            cand = sorted(src.glob("*_report.json"))
            if not cand:
                raise FileNotFoundError(f"--lens-from: no *_report.json in {src}")
            src = cand[0]
        if src.suffix.lower() == ".json":
            d = json.loads(src.read_text(encoding="utf-8"))
            cam = d.get("camera") or {}
            f, k = cam.get("focal_px"), d.get("distortion_k1_k2")
            w0 = max(d["lens_reference_size"]) if d.get("lens_reference_size") else None
        else:
            other = Project.load(src, out_dir=self.out)
            cam = (other.result.camera or {}) if other.result is not None else {}
            f, k = cam.get("focal_px"), list(other.result.k)
            w0 = max(other.frames[0].ref_size)
        if not f or k is None:
            raise ValueError(f"--lens-from: {src} has no camera-model lens (focal length, k1, k2)")
        w, h = self.frames[0].ref_size
        if w0 and abs(max(w, h) / w0 - 1) > 0.01:     # same camera, other working size
            f = f * max(w, h) / w0
        self._fixed_lens = {"focal_px": float(f), "k": np.asarray(k, np.float64)}
        self.log(f"  lens from {src.name}: f = {f:.0f} px, k1 = {k[0]:+.4f}, k2 = {k[1]:+.4f} (kept fixed)")
        return self._fixed_lens

    def nominal_focal(self):
        """(focal length in native px of the reference image, prior width, source)."""
        if self.s.focal_px:
            return float(self.s.focal_px), 0.05, "given"
        fr = self.frames[0]
        w, h = fr.ref_size
        prof = self.prof
        if prof.multilens and prof.focal_mm and prof.pixel_um and prof.cell_size:
            cell = max(w, h) / max(prof.crop_frac, 1e-6)       # whole cell at the working resolution
            return prof.focal_mm / (prof.pixel_um * 1e-3) * cell / prof.cell_size, 0.3, \
                f"{prof.name} lens {prof.focal_mm:g} mm"
        if fr.path.suffix.lower() in (".tif", ".tiff"):
            cam = rawio.tiff_camera_info(fr.path) if Path(str(fr.path)).exists() else {}
            fmm = cam.get("focal_length_mm")
            pum = cam.get("pixel_size_um_output") or cam.get("pixel_size_um")
            if fmm and pum:
                return float(fmm) / (float(pum) * 1e-3), 0.3, f"{fmm:g} mm / {pum:g} um (file metadata)"
        ex = fr.exif or {}
        f35 = ex.get("focal_length_35mm")
        if f35 and not prof.multilens:
            return float(f35) / 36.0 * max(w, h), 0.3, f"EXIF {f35:g} mm (35 mm equiv.)"
        return 1.2 * max(w, h), 0.6, "unknown, estimated"

    def degenerate_frames(self, r=None, max_scale_ratio=3.0, max_aniso=3.0) -> list:
        """Frames whose adjusted transform folds (a corner on or behind the
        horizon), is strongly stretched, or whose scale is far from the median."""
        r = r if r is not None else self.result
        scales, bad = {}, []
        for fr in self.frames:
            H = r.H.get(fr.index)
            if H is None:
                continue
            w, h = fr.ref_size
            cx, cy = w / 2.0, h / 2.0
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [cx, cy, 1]], np.float64)
            den = corners @ H[2]
            if not np.all(np.isfinite(H)) or np.any(den * np.sign(den[-1]) <= 1e-6 * abs(den[-1])):
                bad.append(fr.index)
                continue
            p = (H[:2, :2] @ [cx, cy] + H[:2, 2]) / den[-1]
            J = (H[:2, :2] - np.outer(p, H[2, :2])) / den[-1]
            sv = np.linalg.svd(J, compute_uv=False)
            if sv[-1] <= 0 or sv[0] / sv[-1] > max_aniso:
                bad.append(fr.index)
                continue
            scales[fr.index] = float(np.sqrt(sv[0] * sv[1]))
        if scales:
            med = float(np.median(list(scales.values())))
            bad += [i for i, sc in scales.items() if sc > max_scale_ratio * med or sc < med / max_scale_ratio]
        return sorted(bad)

    def estimated_pairs(self, min_overlap=0.12) -> list:
        """Pairs whose adjusted footprints overlap but were never matched."""
        r = self.result
        dist = RD.Distortion(r.k, r.center, r.radius)
        polys = {}
        for fr in self.frames:
            if fr.index in r.H:
                rf = RD.RenderFrame(fr.index, str(fr.path), r.H[fr.index], fr.ref_size, fr.valid, {})
                polys[fr.index] = cv2.convexHull(RD.footprint(rf, dist, 4).astype(np.float32))
        new = []
        keys = sorted(polys)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                i, j = keys[a], keys[b]
                if (i, j) in self.pairs:
                    continue
                inter, _ = cv2.intersectConvexConvex(polys[i], polys[j])
                if inter <= 0:
                    continue
                frac = inter / min(cv2.contourArea(polys[i]), cv2.contourArea(polys[j]))
                if frac >= min_overlap:
                    new.append((i, j))
        return new

    def align(self):
        """Steps 2-5 (features, pairs, matching, adjustment)."""
        pairs = self.candidate_pairs()
        self.log(f"  {len(pairs)} candidate pairs")
        self.match(pairs)
        self.adjust()
        if self.s.preselection == "auto":
            extra = self.estimated_pairs()
            if extra:
                self.log(f"  estimated preselection: {len(extra)} additional overlapping pairs")
                if self.match(extra, trusted=True):
                    self.adjust()
        # images outside the main block: try them against everything once more
        main = set(self.result.components[0])
        lost = [f.index for f in self.frames if f.index not in main]
        if lost and self.s.preselection != "exhaustive":
            extra = [(min(i, j), max(i, j)) for i in lost for j in main
                     if (min(i, j), max(i, j)) not in self.pairs]
            extra = [ij for ij in extra if ij in self.generic_scores or abs(ij[0] - ij[1]) <= 6]
            if extra:
                self.log(f"  {len(lost)} image(s) not connected; trying {len(extra)} more pairs")
                if self.match(extra):          # nothing new: the adjustment would be the same
                    self.adjust()

    def refresh_quality(self):
        """Estimate the frame sharpness again (after an update of mosaic2d) from the
        quality images in the ingest cache; frames not in the cache keep their value."""
        import pickle
        dirs = [self._cache_dir()]
        if getattr(self, "_project_dir", None) is not None:
            dirs.append(Path(self._project_dir) / "_cache")
        hits = 0
        for k, fr in enumerate(self.frames):
            c = None
            try:
                key = self._cache_key(self.files[k])
            except OSError:                        # photo moved or deleted
                key = None
            for d_ in dirs if key else ():
                f = d_ / f"{key}.pkl"
                if f.is_file():
                    try:
                        with open(f, "rb") as fh:
                            c = pickle.load(fh)
                        break
                    except Exception:
                        c = None
            if c is not None and getattr(c, "qimg", None) is not None:
                fr.qimg = c.qimg
                hits += 1
        if hits < max(2, len(self.frames) // 2):
            self.log(f"  sharpness not re-estimated: only {hits}/{len(self.frames)} frames in the cache")
            for fr in self.frames:
                fr.qimg = None
            return
        self.log(f"  sharpness re-estimated from the cache ({hits}/{len(self.frames)} frames)")
        self._image_quality()
        for fr in self.frames:
            fr.qimg = None

    # ------------------------------------------------------------------ project file
    def save(self, path):
        """Save the aligned project (no features, no log handle) for later rendering."""
        import pickle
        state = {k: v for k, v in self.__dict__.items() if k not in ("feats", "log", "registry")}
        for fr in state["frames"]:
            fr.reg = None  # working images are not needed for rendering
            fr.qimg = None
        with open(path, "wb") as fh:
            pickle.dump({"mosaic2d": __version__, "state": state}, fh, protocol=4)

    @classmethod
    def load(cls, path, out_dir=None, **overrides):
        import pathlib
        import pickle

        class _Unpickler(pickle.Unpickler):
            # a project saved on Windows holds WindowsPath objects, which cannot
            # be created on Linux/macOS (and PosixPath not on Windows)
            def find_class(self, module, name):
                if module.startswith("pathlib") and name in ("WindowsPath", "PosixPath"):
                    native = type(pathlib.Path())
                    if name == native.__name__:
                        return native
                    return pathlib.PureWindowsPath if name == "WindowsPath" else pathlib.PurePosixPath
                return super().find_class(module, name)

        with open(path, "rb") as fh:
            d = _Unpickler(fh).load()
        proj = cls.__new__(cls)
        proj.__dict__.update(d["state"])
        proj._project_dir = Path(path).resolve().parent
        from dataclasses import fields as _fields
        for f in _fields(Settings):            # projects saved by older versions
            if not hasattr(proj.s, f.name):
                setattr(proj.s, f.name, f.default)
        for k, v in overrides.items():
            if v is not None:
                setattr(proj.s, k, v)
        if out_dir:
            proj.out = Path(out_dir)
        proj.out.mkdir(parents=True, exist_ok=True)
        proj.log = Log(proj.out / "mosaic2d_render.log")
        proj.feats = {}
        proj.registry = FT.build_registry(proj.s.n_features)
        if proj.prof.name in ("rgb", "agrowing6"):  # fields added after the project was saved
            cur = SR.get_profile(proj.prof.name)
            for fld in ("focal_mm", "pixel_um"):
                if getattr(proj.prof, fld, None) is None and getattr(cur, fld, None) is not None:
                    setattr(proj.prof, fld, getattr(cur, fld))
        if proj.prof.name == "multiband" and all(b.name.startswith("band ") for b in proj.prof.bands):
            try:                                   # projects from before band metadata was read
                prof = SR.get_profile("multiband", proj.files[0])
                if len(prof.bands) == len(proj.prof.bands) and prof.bands != proj.prof.bands:
                    proj.prof = prof
                    if proj.ref_band is not None:
                        proj.ref_band = next((b for b in prof.bands if b.channel == proj.ref_band.channel),
                                             proj.ref_band)
                    proj.log(f"  band names from the image metadata: {', '.join(b.name for b in prof.bands)}")
            except Exception:
                pass
        return proj

    # ------------------------------------------------------------------ output
    def median_frame_scale(self) -> float:
        """Median linear scale (mosaic px per native px) at the frame centres."""
        r = self.result
        sc = []
        for fr in self.frames:
            H = r.H.get(fr.index)
            if H is None:
                continue
            d = local_scale(H, fr.ref_size[0] / 2.0, fr.ref_size[1] / 2.0)
            if np.isfinite(d) and d > 0:
                sc.append(d)
        return float(np.median(sc)) if sc else 1.0

    def output_transform(self, resolution=None, orient=None):
        """Similarity T (mosaic px -> output px) and canvas size."""
        r = self.result
        resolution = resolution or self.s.resolution
        orient = orient or self.s.orient
        dist = RD.Distortion(r.k, r.center, r.radius)
        pts, centres = [], []
        for fr in self.frames:
            if fr.index not in r.H:
                continue
            rf = RD.RenderFrame(fr.index, "", r.H[fr.index], fr.ref_size, fr.valid, {})
            fp = RD.footprint(rf, dist, 6)
            pts.append(fp)
            centres.append(fp.mean(0))
        pts = np.concatenate(pts)
        R = np.eye(3)
        if orient == "pca" and len(centres) >= 3:
            c = np.array(centres)
            ev, evec = np.linalg.eigh(np.cov((c - c.mean(0)).T))
            ax = evec[:, np.argmax(ev)]
            ang = np.arctan2(ax[1], ax[0])
            ca, sa = np.cos(-ang), np.sin(-ang)
            R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        # output px relative to the *median* frame's native px (not the anchor's:
        # with varying camera height the anchor can be far from typical)
        k = resolution / self.median_frame_scale()
        S = np.diag([k, k, 1.0])
        T = S @ R
        q = cv2.perspectiveTransform(pts[None], T)[0]
        mn = np.floor(q.min(0)) - 2
        mx = np.ceil(q.max(0)) + 2
        T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]]) @ T
        W, H = int(mx[0] - mn[0]), int(mx[1] - mn[1])
        return T, (W, H)

    def render_frames(self, T):
        r = self.result
        out = []
        for fr in self.frames:
            if fr.index not in r.H:
                continue
            A = {1: np.eye(3)} if fr.lens is None else {
                ln: fr.lens.A(ln) for ln in range(1, self.prof.n_lenses + 1)}
            out.append(RD.RenderFrame(fr.index, str(fr.path), T @ r.H[fr.index], fr.ref_size,
                                      fr.valid, A, fr.thumb, fr.thumb_scale,
                                      penalty=seam_penalty(fr.quality),
                                      seam_img=getattr(fr, "seam_img", None),
                                      seam_scale=getattr(fr, "seam_scale", 0.0)))
        return out

    def band_sources(self):
        return [(b.lens, b.channel) for b in self.prof.output_bands()]

    def preview_mosaic(self, max_side=2500, blend="mosaic"):
        """Fast low-resolution mosaic from the thumbnails (no full decode)."""
        r = self.result
        T, (W, H) = self.output_transform(resolution=1.0)
        f = min(1.0, max_side / max(W, H))
        T = np.diag([f, f, 1.0]) @ T
        T, (W, H) = T, (max(1, int(W * f)), max(1, int(H * f)))
        frames = []
        for fr in self.frames:
            if fr.index not in r.H:
                continue
            ts = fr.thumb_scale
            Si = np.diag([1 / ts, 1 / ts, 1.0])
            Hs = T @ r.H[fr.index] @ Si
            x0, y0, x1, y1 = fr.valid
            frames.append(RD.RenderFrame(fr.index, "", Hs,
                                         (fr.thumb.shape[1], fr.thumb.shape[0]),
                                         (x0 * ts, y0 * ts, x1 * ts, y1 * ts), {0: np.eye(3)},
                                         fr.thumb, 1.0, penalty=seam_penalty(fr.quality)))
        ts0 = self.frames[0].thumb_scale
        dist = RD.Distortion(r.k, r.center * ts0, r.radius * ts0)
        nb = self.frames[0].thumb.shape[2]
        rr = RD.Renderer(frames, dist, (W, H), [(0, b) for b in range(nb)], nb, np.uint16,
                         blend=blend, strip=4096, log=lambda *a: None, seams="voronoi",
                         flat=getattr(self, "flat", None),
                         work_dir=Path(self.s.work_dir or self.out) / "_tmp_preview")
        thumbs = {fr.index: fr.thumb for fr in frames}
        canvas = rr.render(lambda fr: {0: thumbs[fr.index]}, out_name="preview")
        img = np.array(canvas)
        del canvas
        RD.cleanup(rr)
        return img, rr

    def vis_channels(self):
        bands = self.prof.output_bands()
        if self.prof.name == "rgb" or len(bands) == 3:
            return (0, 1, 2)
        wl = [b.wavelength_nm or 0 for b in bands]
        if not any(wl):                          # no wavelengths: spread over the band order
            n = len(bands)
            return (min(n - 1, int(round(0.7 * (n - 1)))), int(round(0.45 * (n - 1))),
                    int(round(0.2 * (n - 1))))

        def near(x):
            return int(np.argmin([abs(w - x) for w in wl]))
        return (near(650), near(550), near(450))

    def render(self, name="mosaic"):
        t = time.time()
        T, (W, H) = self.output_transform()
        if max(W, H) > 300000:
            raise RuntimeError(f"Output would be {W} x {H} px - registration is probably wrong "
                               "(try --model affine), or lower --resolution.")
        frames = self.render_frames(T)
        r = self.result
        dist = RD.Distortion(r.k, r.center, r.radius)
        bands = self.prof.output_bands()
        sample = SR.load_frame(self.files[0], self.prof, self.s.demosaic)
        dtype = next(iter(sample.lenses.values())).dtype
        del sample
        work = Path(self.s.work_dir or self.out)
        rr = RD.Renderer(frames, dist, (W, H), self.band_sources(), len(bands), dtype,
                         blend=self.s.blend, work_dir=work / "_tmp_canvas", log=self.log,
                         flat=getattr(self, "flat", None), seams=self.s.seams,
                         deghost=self.s.deghost, feather=getattr(self.s, "feather", True),
                         vis_bands=self.vis_channels(),
                         progress=lambda k, n: (k % 5 == 0 or k == n) and self.log(f"  rendered {k}/{n} frames"))
        prof = self.prof

        def load(fr):
            return SR.load_frame(fr.path, prof, self.s.demosaic).lenses
        canvas = rr.render(load, out_name=name)
        meta = {"SOFTWARE": f"mosaic2d {__version__}", "PROFILE": prof.name,
                "BLEND": self.s.blend, "MODEL": self.s.model,
                "RESOLUTION_REL_NATIVE": self.s.resolution,
                "REFERENCE_BAND": self.ref_band.name if self.ref_band else "luminance"}
        names_ = [b.name for b in bands]
        wls = [b.wavelength_nm for b in bands]
        ps = (self.ground_pixel() / self.s.resolution) if self.ground_pixel() else None
        if ps:
            meta["PIXEL_SIZE_M"] = ps
            self.log(f"  output pixel size {ps * 1000:.3f} mm ({W * ps:.2f} x {H * ps:.2f} m)")
        if self.s.tile_size and max(W, H) > self.s.tile_size:
            tif = write_tiled(self.out, name, canvas, names_, wls, 0, self.s.tile_size, meta, self.log,
                              pixel_size=ps)
            self.log(f"  wrote {tif.name} + tiles")
        else:
            tif = self.out / f"{name}.tif"
            self.log(f"  writing {tif.name}")
            step = {"next": 0.1}

            def wprog(fr_):
                if fr_ >= step["next"]:
                    self.log(f"  written {min(fr_, 1.0) * 100:.0f} %")
                    step["next"] = 2.0 if fr_ >= 1.0 else min(1.0, fr_ + 0.1)
            write_tiff(tif, canvas, band_names=names_, wavelengths=wls, nodata=0,
                       description=json.dumps({"mosaic2d": __version__, "images": len(frames)}),
                       metadata=meta, pixel_size=ps, origin=(0.0, H * ps if ps else 0.0),
                       progress=wprog)
        small = RD.downsample(canvas, 3000)
        cv2.imwrite(str(self.out / f"{name}_preview.jpg"), RD.visualise(small, self.vis_channels()),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        label, s = rr.seam_map(1600)
        self._save_seams(label, small, self.out / f"{name}_seams.jpg")
        np.save(self.out / f"{name}_transforms.npy", {fr.index: fr.H for fr in frames}, allow_pickle=True)
        del canvas
        if not self.s.keep_canvas:
            RD.cleanup(rr)
        else:
            self.log(f"  canvas kept: {rr.canvas_path}")
        self.seam_changed = None if rr.seams is None else round(rr.seams.changed_frac, 4)
        self.timing["render_s"] = time.time() - t
        self.canvas_size = (W, H)
        self.T_out = T
        return tif

    def _save_seams(self, label, small, path):
        rng = np.random.default_rng(3)
        colors = rng.integers(60, 255, (len(self.frames) + 1, 3)).astype(np.uint8)
        vis = colors[np.clip(label, 0, None)]
        vis[label < 0] = 0
        edges = np.zeros(label.shape, bool)
        edges[:, 1:] |= label[:, 1:] != label[:, :-1]
        edges[1:, :] |= label[1:, :] != label[:-1, :]
        if small is not None:
            base = cv2.resize(RD.visualise(small, self.vis_channels()), label.shape[::-1])
            vis = cv2.addWeighted(base, 0.65, vis, 0.35, 0)
        vis[edges] = (255, 255, 255)
        for fr in self.frames:
            m = label == fr.index
            if m.any():
                ys, xs = np.nonzero(m)
                tag = short_label(fr.path.stem)
                cv2.putText(vis, tag, (int(xs.mean()) - 15, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, tag, (int(xs.mean()) - 15, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ------------------------------------------------------------------ reports
    def report(self, name="mosaic"):
        r = self.result
        main = set(r.components[0]) if r else set()
        method_count = {}
        for p in self.pairs.values():
            if p.ok:
                method_count[p.method] = method_count.get(p.method, 0) + 1
        rep = {
            "mosaic2d": __version__,
            "settings": asdict(self.s),
            "profile": self.prof.name,
            "images": len(self.frames),
            "aligned_images": len(main),
            "not_aligned": [self.frames[i].path.name for i in range(len(self.frames)) if i not in main],
            "reference_band": self.ref_band.name if self.ref_band else "luminance",
            "band_scores": self.band_scores,
            "methods_available": self.methods,
            "pairs_tried": len(self.pairs),
            "pairs_accepted": sum(1 for p in self.pairs.values() if p.ok),
            "pairs_by_method": method_count,
            "reprojection_rmse_px": r.rmse_px if r else None,
            "reprojection_median_px": r.median_px if r else None,
            "distortion_k1_k2": r.k.tolist() if r else None,
            "lens_reference_size": list(self.frames[0].ref_size) if self.frames else None,
            "anchor": self.frames[r.anchor].path.name if r else None,
            "canvas_px": getattr(self, "canvas_size", None),
            "flat_field_corner_over_centre": getattr(self, "flat_stats", {}),
            "seams": self.s.seams,
            "seam_area_moved_by_graphcut": getattr(self, "seam_changed", None),
            "image_quality": {f.path.name: round(f.quality, 3) for f in self.frames},
            "sequence_links": [[self.frames[i].path.name, self.frames[j].path.name]
                               for i, j in getattr(self, "sequence_links", [])],
            "timing_s": {k: round(v, 1) for k, v in self.timing.items()},
        }
        cam = getattr(r, "camera", None) if r else None
        if cam:
            hr = np.array(list(cam["height_rel"].values()))
            tl = np.array(list(cam["tilt_deg"].values()))
            rep["camera"] = {"focal_px": round(cam["focal_px"], 1),
                             "focal_nominal_px": round(cam["focal_nominal_px"], 1),
                             "height_rel_min_median_max": [round(float(x), 3) for x in
                                                           (hr.min(), np.median(hr), hr.max())],
                             "tilt_deg_median_max": [round(float(np.median(tl)), 2), round(float(tl.max()), 2)],
                             "ground_pixel_m": self.ground_pixel()}
        (self.out / f"{name}_report.json").write_text(json.dumps(rep, indent=2, default=str))
        with open(self.out / f"{name}_pairs.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_i", "image_j", "accepted", "method", "matches", "inliers",
                        "inlier_ratio", "coverage", "pair_rmse_px", "adjusted_rmse_px", "tried", "note"])
            adj = {(d["i"], d["j"]): d["rmse_px"] for d in (r.per_pair if r else [])}
            for (i, j), p in sorted(self.pairs.items()):
                w.writerow([self.frames[i].path.name, self.frames[j].path.name, int(p.ok), p.method,
                            p.n_matches, p.n_inliers, f"{p.inlier_ratio:.3f}", f"{p.coverage:.3f}",
                            f"{p.rmse:.3f}" if np.isfinite(p.rmse) else "",
                            f"{adj[(i, j)]:.3f}" if (i, j) in adj else "",
                            "|".join(p.tried), p.reason])
        with open(self.out / f"{name}_images.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            hdr = ["image", "aligned", "pairs", "rmse_px", "quality", "time", "height_rel", "tilt_deg"]
            if self.prof.multilens:
                hdr += [f"lens{ln}_{k}" for ln in range(1, self.prof.n_lenses + 1) for k in ("dx", "dy", "ncc", "method")]
            w.writerow(hdr)
            for fr in self.frames:
                pi = r.per_image.get(fr.index, {}) if r else {}
                row = [fr.path.name, int(fr.index in main), pi.get("pairs", 0),
                       f"{pi['rmse_px']:.3f}" if pi.get("rmse_px") is not None else "",
                       f"{fr.quality:.2f}", fr.exif.get("datetime_original", "")]
                row += [f"{cam['height_rel'][fr.index]:.4f}", f"{cam['tilt_deg'][fr.index]:.2f}"] \
                    if cam and fr.index in cam["height_rel"] else ["", ""]
                if self.prof.multilens:
                    for ln in range(1, self.prof.n_lenses + 1):
                        if ln == fr.lens.ref_lens:
                            row += [0, 0, 1, "reference"]
                        else:
                            lr = fr.lens.lenses[ln]
                            row += [f"{lr.A[0, 2]:.2f}", f"{lr.A[1, 2]:.2f}", f"{lr.ncc:.3f}", lr.method]
                w.writerow(row)
        self._markdown(rep, name)
        return rep

    def _markdown(self, rep, name):
        lines = [f"# mosaic2d report — {name}", "",
                 f"- Images: {rep['images']}, aligned: {rep['aligned_images']}",
                 f"- Profile: {rep['profile']}, reference band: {rep['reference_band']}",
                 f"- Feature methods: {', '.join(rep['methods_available'])} "
                 f"(strategy {self.s.strategy}, model {self.s.model})",
                 f"- Pairs: {rep['pairs_accepted']} accepted of {rep['pairs_tried']} tried; "
                 f"by method: {rep['pairs_by_method']}",
                 f"- Reprojection error: RMSE {rep['reprojection_rmse_px']:.2f} px, "
                 f"median {rep['reprojection_median_px']:.2f} px (native px of the reference lens)",
                 f"- Radial distortion k1, k2: {np.round(rep['distortion_k1_k2'], 5).tolist()}",
                 *([f"- Camera model: focal {rep['camera']['focal_px']:.0f} px (nominal "
                    f"{rep['camera']['focal_nominal_px']:.0f}), height min/median/max "
                    f"{rep['camera']['height_rel_min_median_max']} x typical, tilt median/max "
                    f"{rep['camera']['tilt_deg_median_max']} deg"
                    + (f", ground pixel {rep['camera']['ground_pixel_m'] * 1000:.3f} mm at resolution 1"
                       if rep['camera']['ground_pixel_m'] else "")] if rep.get("camera") else []),
                 f"- Canvas: {rep['canvas_px']}", f"- Timing (s): {rep['timing_s']}"]
        if rep["not_aligned"]:
            lines.append(f"- **Not aligned:** {', '.join(rep['not_aligned'])}")
        if self.band_scores:
            lines += ["", "## Reference band selection (median verified inliers between neighbours)", "",
                      "| band | inliers |", "|---|---|"]
            lines += [f"| {k} | {v:.0f} |" for k, v in sorted(self.band_scores.items(), key=lambda kv: -kv[1])]
        (self.out / f"{name}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- API

def _block_report_rows(proj, name, results):
    rows = []
    for k, (comp, status) in enumerate(results, start=1):
        files = sorted(proj.frames[i].path.name for i in comp)
        rows.append({"block": k, "images": len(comp), "first": files[0], "last": files[-1],
                     "output": status})
    path = proj.out / f"{name}_blocks.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def run(inputs, out_dir, settings: Settings | None = None, name="mosaic", render=True):
    s = settings or Settings()
    proj = Project(inputs, out_dir, s)
    proj.ingest()
    try:
        proj.align()
    except Exception:
        # keep the (long) matching: the adjustment can be run again from the saved project
        if proj.pairs:
            pk = proj.out / f"{name}_project.pkl"
            try:
                proj.result = None
                proj.save(pk)
                proj.log(f"  adjustment failed; the matches are saved in {pk.name}. After fixing the "
                         f"cause, continue without matching again: mosaic2d render \"{pk}\" --readjust")
            except Exception:                   # noqa: BLE001
                pass
        raise
    proj._image_quality()
    proj.save(proj.out / f"{name}_project.pkl")
    proj.feats.clear()          # features are not needed any more; free memory for rendering
    if render:
        proj.render(name)
    else:
        img, _ = proj.preview_mosaic()
        cv2.imwrite(str(proj.out / f"{name}_preview.jpg"), RD.visualise(img.astype(np.float32), proj.vis_channels()))
    rep = proj.report(name)
    if render and s.min_block > 0:
        proj.render_blocks(name)
    proj.log(f"done: {proj.out}")
    return proj, rep


def benchmark(inputs, out_dir, settings: Settings | None = None, methods="auto"):
    """
    Register the same image set with every method separately and compare:
    accepted pairs, images in the main block, reprojection error, run time.
    Writes benchmark.csv / benchmark.md and one low-resolution preview per method.
    """
    s = settings or Settings()
    proj = Project(inputs, out_dir, s)
    proj.ingest()
    names = FT.resolve_methods(methods, proj.registry)
    pairs = proj.candidate_pairs()
    proj.log(f"benchmark: {len(names)} methods on {len(pairs)} candidate pairs")
    rows = []
    for m in names:
        proj.pairs = {}
        t = time.time()
        proj.compute_features([m])
        tf = time.time() - t
        t = time.time()
        proj.match(pairs, methods=[m], strategy="single")
        tm = time.time() - t
        ok = [p for p in proj.pairs.values() if p.ok]
        row = {"method": m, "family": proj.registry[m].family,
               "keypoints_mean": int(np.mean([len(proj.features(m, i).pts) for i in range(len(proj.frames))])),
               "pairs_accepted": len(ok),
               "inliers_median": int(np.median([p.n_inliers for p in ok])) if ok else 0,
               "inlier_ratio_median": round(float(np.median([p.inlier_ratio for p in ok])), 3) if ok else 0,
               "features_s": round(tf, 1), "matching_s": round(tm, 1)}
        try:
            t = time.time()
            proj.adjust()
            row.update({"aligned_images": len(proj.result.components[0]),
                        "rmse_px": round(proj.result.rmse_px, 3),
                        "median_px": round(proj.result.median_px, 3),
                        "adjust_s": round(time.time() - t, 1)})
            img, _ = proj.preview_mosaic(max_side=1800, blend="disabled")
            cv2.imwrite(str(proj.out / f"benchmark_{m}.jpg"),
                        RD.visualise(img.astype(np.float32), proj.vis_channels()),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as exc:  # pragma: no cover - reported, not raised
            row.update({"aligned_images": 0, "rmse_px": None, "median_px": None, "error": str(exc)})
        rows.append(row)
        proj.log(f"  {m}: {row}")
        for k in [k for k in proj.feats if k[0] == m]:
            del proj.feats[k]
    rows.sort(key=lambda r: (-r["aligned_images"], -r["pairs_accepted"],
                             r["median_px"] if r["median_px"] is not None else 1e9))
    keys = ["method", "family", "aligned_images", "pairs_accepted", "inliers_median",
            "inlier_ratio_median", "median_px", "rmse_px", "keypoints_mean",
            "features_s", "matching_s", "adjust_s"]
    with open(proj.out / "benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["# mosaic2d method benchmark", "",
          f"{len(proj.frames)} images, profile {proj.prof.name}, reference band "
          f"{proj.ref_band.name if proj.ref_band else 'luminance'}, {len(pairs)} candidate pairs, "
          f"model {s.model}, accuracy {s.accuracy}.", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    md += ["", f"Ranking: most aligned images, then most accepted pairs, then lowest median "
               f"reprojection error. Best: **{rows[0]['method']}**."]
    (proj.out / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    proj.log("benchmark written to " + str(proj.out / "benchmark.md"))
    return rows
