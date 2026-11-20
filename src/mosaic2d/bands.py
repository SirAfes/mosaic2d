"""
bands.py

Multispectral band handling.

1. Intra-frame lens alignment (multi-lens cameras). Each lens image of a frame
   is registered to the reference lens of the same frame:
   phase correlation on gradient-magnitude images (works across spectral
   bands) gives the translation, an ECC refinement (Evangelidis & Psarakis
   2008) on the same gradient images adds a small affine part (focal-length
   and mounting differences between lenses). Results are checked against
   the series: lens offsets of a rigid lens array change only with distance,
   so a frame whose offset is far from the series median *and* that
   correlates poorly is re-estimated around the median (or set to it).

2. Reference band selection. The band used to register frames to each other
   is chosen by what actually matters: on a sample of neighbouring frame
   pairs, every band is matched with the primary feature method, and the band
   with the most verified inliers wins (texture, signal-to-noise and water
   attenuation are all reflected in that number).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .sources import Band, CameraProfile, FrameData, band_image, normalise_for_features


def gradient_image(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    g = img.astype(np.float32)
    g = g / (cv2.GaussianBlur(g, (0, 0), max(g.shape) / 20.0) + 1e-3 * (g.mean() + 1e-6))
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    step = max(1, m.size // 200000)
    hi = np.percentile(m.ravel()[::step], 99.5)
    return np.clip(m / max(hi, 1e-6), 0, 1).astype(np.float32)


def lens_gray(fd: FrameData, prof: CameraProfile, lens: int) -> np.ndarray:
    """Mean of the used channels of a lens, each scaled to its own mean."""
    chans = [band_image(fd, b).astype(np.float32) for b in prof.lens_bands(lens)]
    chans = [c / max(float(c.mean()), 1e-6) for c in chans]
    return np.mean(chans, axis=0)


def _ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


@dataclass
class LensResult:
    A: np.ndarray            # 3x3: lens px -> reference-lens px
    method: str
    ncc: float
    response: float = 0.0


@dataclass
class FrameLensAlignment:
    ref_lens: int
    lenses: dict = field(default_factory=dict)   # lens -> LensResult

    def A(self, lens: int) -> np.ndarray:
        return np.eye(3) if lens == self.ref_lens else self.lenses[lens].A


def _warp_ncc(ref_g, mov_g, W):
    """NCC of ref with mov warped by W (2x3, ref -> mov coords)."""
    h, w = ref_g.shape
    warped = cv2.warpAffine(mov_g, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones_like(mov_g), W, (w, h),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0.5
    m = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return _ncc(ref_g, warped, m) if m.sum() > 1000 else 0.0


def align_lens(ref_g: np.ndarray, mov_g: np.ndarray, prior_shift=None, max_dev=None,
               affine=True) -> LensResult:
    """Register gradient image mov_g to ref_g. Returns lens->ref transform."""
    h = min(ref_g.shape[0], mov_g.shape[0])
    w = min(ref_g.shape[1], mov_g.shape[1])
    ref_g, mov_g = ref_g[:h, :w], mov_g[:h, :w]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    if prior_shift is not None:
        # search only around the prior: pre-shift, then measure the residual
        M0 = np.float32([[1, 0, prior_shift[0]], [0, 1, prior_shift[1]]])
        pre = cv2.warpAffine(mov_g, M0, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        (rx, ry), resp = cv2.phaseCorrelate(ref_g, pre, win)
        if max_dev is not None and np.hypot(rx, ry) > max_dev:
            rx = ry = 0.0
            resp = 0.0
        dx, dy = prior_shift[0] + rx, prior_shift[1] + ry
    else:
        (dx, dy), resp = cv2.phaseCorrelate(ref_g, mov_g, win)
    W = np.float32([[1, 0, dx], [0, 1, dy]])
    best_W, best_m = W, "phasecorr"
    best = _warp_ncc(ref_g, mov_g, W)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    modes = [(cv2.MOTION_AFFINE, "ecc_affine")] if affine else []
    modes.append((cv2.MOTION_TRANSLATION, "ecc_translation"))
    for mode, name in modes:
        try:
            _, Wn = cv2.findTransformECC(ref_g, mov_g, W.copy(), mode, crit, None, 5)
        except cv2.error:
            continue
        L = Wn[:, :2]
        sv = np.linalg.svd(L, compute_uv=False)
        rot = np.degrees(np.arctan2(L[1, 0], L[0, 0]))
        if sv.max() > 1.04 or sv.min() < 0.96 or abs(rot) > 3:
            continue
        if np.hypot(Wn[0, 2] - dx, Wn[1, 2] - dy) > 25:
            continue
        s = _warp_ncc(ref_g, mov_g, Wn)
        if s > best + 0.005:
            best_W, best_m, best = Wn, name, s
            break
    A = np.linalg.inv(np.vstack([best_W, [0, 0, 1]]).astype(np.float64))
    return LensResult(A, best_m, best, float(resp))


def align_frame_lenses(fd: FrameData, prof: CameraProfile, ref_lens: int,
                       scale: float = 1.0, priors: dict | None = None, max_dev=None,
                       affine=True) -> FrameLensAlignment:
    """All lenses of one frame -> reference lens. scale < 1 works on reduced images."""
    def prep(lens):
        g = lens_gray(fd, prof, lens)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gradient_image(g)

    ref_g = prep(ref_lens)
    out = FrameLensAlignment(ref_lens)
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    for lens in range(1, prof.n_lenses + 1):
        if lens == ref_lens:
            continue
        prior = None
        if priors and lens in priors:
            # prior given as lens->ref A (full res); search uses ref->lens shift
            Wp = np.linalg.inv(priors[lens])
            prior = (Wp[0, 2] * scale, Wp[1, 2] * scale)
        r = align_lens(ref_g, prep(lens), prior, None if max_dev is None else max_dev * scale, affine)
        r.A = Si @ r.A @ S
        out.lenses[lens] = r
    return out


def consensus_check(results: list[FrameLensAlignment], tol_px: float = 12.0,
                    ncc_frac: float = 0.6):
    """
    Robust per-lens series statistics. Returns (median A per lens, list of
    (frame index, lens) that disagree with the series and correlate poorly).
    """
    lenses = sorted({ln for r in results for ln in r.lenses})
    med, bad = {}, []
    for ln in lenses:
        T = np.array([r.lenses[ln].A[:2, 2] for r in results])
        nc = np.array([r.lenses[ln].ncc for r in results])
        mt = np.median(T, axis=0)
        mad = np.median(np.linalg.norm(T - mt, axis=1)) * 1.4826
        lim = max(tol_px, 4 * mad)
        mn = np.median(nc)
        Am = np.eye(3)
        L = np.median(np.array([r.lenses[ln].A[:2, :2] for r in results]), axis=0)
        Am[:2, :2], Am[:2, 2] = L, mt
        med[ln] = Am
        for k, r in enumerate(results):
            d = np.linalg.norm(T[k] - mt)
            if d > lim and nc[k] < max(ncc_frac * mn, mn - 0.15):
                bad.append((k, ln))
            elif nc[k] < 0.35 * mn:
                bad.append((k, ln))
    return med, bad


def band_saturation(img: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float]:
    """(fraction of clipped pixels, fraction of black pixels) of one band over the
    image area ``valid`` (default: its non-zero pixels; pass the pixels where any
    band has data, so that a band that is empty everywhere counts as black)."""
    v = img.reshape(-1)
    m = (v > 0) if valid is None else valid.reshape(-1)
    step = max(1, v.size // 500000)
    v, m = v[::step], m[::step]
    v = v[m]
    if v.size == 0:
        return 0.0, 1.0
    if np.issubdtype(img.dtype, np.integer):
        top = float(np.iinfo(img.dtype).max)
        if top > 255:                   # 10/12/14-bit data in 16-bit words: its own ceiling
            vmax = float(v.max())
            bits = max(10, int(np.ceil(np.log2(vmax + 1))))
            top = min(top, float(2 ** bits - 1))
    else:
        top = float(v.max()) or 1.0
    sat = float((v >= 0.98 * top).mean())
    dark = float((v <= 0.02 * top).mean())
    return sat, dark


def _static(r, size, frac=0.005) -> bool:
    """True if a matched pair maps the frame onto itself: features that stay at the
    same image position (sensor pattern, dust, a housing part, a band that is all
    black or clipped) rather than scene content."""
    if not r.ok or r.H is None:
        return False
    w, h = size
    c = np.array([[[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]], np.float64)
    d = np.abs(cv2.perspectiveTransform(c, r.H) - c).max()
    return d < frac * np.hypot(w, h)


def choose_reference_band(frames_data: list, prof: CameraProfile, method,
                          match_fn, work_scale: float, log=print, groups=None) -> tuple[Band, dict]:
    """
    Pick the band whose images match best between consecutive photos.

    frames_data: FrameData of a sample of photos; ``groups`` lists runs of
    consecutive indices into it (default: all in one run). Every band of every
    photo is matched with the next photo of its run; the score is the mean
    number of verified matches (times their spread), reduced for clipped (saturated) pixels - clipped
    areas carry no texture and no usable values - and for nearly black images.
    method: callable(img_u8) -> Features ; match_fn(fa, fb) -> PairResult
    Returns (best band, {band name: score}).
    """
    groups = groups or [list(range(len(frames_data)))]
    scores, info = {}, {}
    # image area with data: any band non-zero (borders left by band alignment are 0 everywhere)
    valid = []
    for fd in frames_data:
        m = None
        for b in prof.bands:
            nz = band_image(fd, b) > 0
            m = nz if m is None else (m | nz)
        valid.append(m)
    n_static = {}
    for b in prof.bands:
        raw = [band_image(fd, b) for fd in frames_data]
        sd = np.array([band_saturation(r, v) for r, v in zip(raw, valid)])
        sat, dark = float(np.median(sd[:, 0])), float(np.median(sd[:, 1]))
        feats, sizes = [], []
        for r in raw:
            g = r.astype(np.float32)
            if work_scale != 1.0:
                g = cv2.resize(g, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
            feats.append(method(normalise_for_features(g)))
            sizes.append((g.shape[1], g.shape[0]))
        inl, stat = [], 0
        for run in groups:
            for k in range(len(run) - 1):
                rr = match_fn(feats[run[k]], feats[run[k + 1]])
                if _static(rr, sizes[run[k]]):      # pattern fixed in the image, not the scene
                    stat += 1
                    inl.append(0.0)
                    continue
                inl.append(rr.n_inliers * min(1.0, rr.coverage / 0.2) if rr.ok else 0.0)
        n_static[b.name] = stat
        # mean, not median: with a few runs, one bad stretch of the survey (blank water,
        # a turn) would set every band to zero
        m = float(np.mean(inl)) if inl else 0.0
        penalty = max(0.0, 1.0 - 10.0 * sat) * max(0.0, 1.0 - dark)
        scores[b.name] = round(m * penalty, 1)
        info[b.name] = (m, sat, dark)

    def key(b):
        return (scores[b.name], -abs((b.wavelength_nm or 550) - 550))
    best = max(prof.bands, key=key)
    log("  reference band scores (matches between consecutive photos x coverage, "
        "minus clipped/black pixels):")
    for name, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        m, sat, dark = info[name]
        note = "   (matches at the same image position: fixed pattern)" if n_static[name] else ""
        log(f"    {name:>10s}: {v:7.0f}   matches {m:6.0f}   saturated {sat * 100:5.2f} %   "
            f"black {dark * 100:5.1f} %{note}")
    unusable = [n for n, (m, sat, dark) in info.items() if sat > 0.5 or dark > 0.5]
    if unusable:
        log(f"  WARNING: band(s) {', '.join(unusable)} are mostly black or clipped in the input photos "
            f"(check the export / bit depth); they will be empty or flat in the mosaic too")
    if scores[best.name] <= 0:
        # nothing matched (e.g. a sample of blank photos): green-most band with least clipping
        best = min(prof.bands, key=lambda b: (info[b.name][1] > 0.01 or info[b.name][2] > 0.3,
                                               abs((b.wavelength_nm or 550) - 550)))
        log(f"  no band matched in the sample; using {best.name}")
    return best, scores
