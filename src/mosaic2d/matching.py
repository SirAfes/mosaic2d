"""
matching.py

Pairwise matching and geometric verification.

A pair (i, j) is matched by descriptor nearest neighbours with Lowe's ratio
test and a one-to-one constraint, then verified with a robust homography
(MAGSAC++ when OpenCV provides it, RANSAC otherwise). An optional *guided
matching* pass (as in Metashape) re-matches all keypoints near the position
the first model predicts, which recovers matches the global ratio test
rejected in repetitive texture.

A result is accepted only if it has enough inliers, a reasonable inlier
ratio, inliers spread over the overlap (not one cluster, not one line --
e.g. a transect rope) and a plausible transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import Features

_USAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


@dataclass
class MatchParams:
    ratio: float = 0.8
    ransac_px: float = 3.0           # inlier threshold in working-image pixels
    min_inliers: int = 25
    min_inlier_ratio: float = 0.08
    min_coverage: float = 0.01       # inlier hull area / image area
    guided: bool = True
    guided_radius_px: float = 6.0
    max_scale_change: float = 2.5
    max_perspective: float = 0.35    # max relative scale change across the frame


@dataclass
class PairResult:
    i: int
    j: int
    method: str
    ok: bool
    H: np.ndarray | None = None      # maps working coords of i -> working coords of j
    pts_i: np.ndarray | None = None  # inlier points (working coords of i)
    pts_j: np.ndarray | None = None
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    rmse: float = float("nan")
    reason: str = ""
    tried: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "i": self.i, "j": self.j, "method": self.method, "ok": self.ok,
            "matches": self.n_matches, "inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 3), "coverage": round(self.coverage, 3),
            "rmse_px": round(self.rmse, 3) if np.isfinite(self.rmse) else None,
            "reason": self.reason, "tried": ",".join(self.tried),
        }


def knn_ratio_match(fa: Features, fb: Features, ratio: float) -> np.ndarray:
    """Ratio-test matches (K x 2 index array), one-to-one in the target set."""
    if fa.desc is None or fb.desc is None or len(fa.pts) < 2 or len(fb.pts) < 2:
        return np.zeros((0, 2), np.int32)
    if fa.norm == cv2.NORM_L2:
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
        da, db = fa.desc.astype(np.float32), fb.desc.astype(np.float32)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        da, db = fa.desc, fb.desc
    knn = matcher.knnMatch(da, db, k=2)
    out, dist = [], []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            out.append((m[0].queryIdx, m[0].trainIdx))
            dist.append(m[0].distance)
    if not out:
        return np.zeros((0, 2), np.int32)
    out = np.array(out, np.int32)
    order = np.argsort(np.array(dist))
    out = out[order]
    _, first = np.unique(out[:, 1], return_index=True)
    return out[np.sort(first)]


def guided_match(fa: Features, fb: Features, H: np.ndarray, radius: float,
                 ratio: float = 0.9) -> np.ndarray:
    """Match every keypoint of A with the keypoints of B near H(A)."""
    if fa.desc is None or fb.desc is None:
        return np.zeros((0, 2), np.int32)
    proj = cv2.perspectiveTransform(fa.pts[None].astype(np.float64), H)[0]
    cell = max(radius, 1.0)
    keys = np.floor(fb.pts / cell).astype(np.int64)
    grid: dict = {}
    for idx, (kx, ky) in enumerate(keys):
        grid.setdefault((kx, ky), []).append(idx)
    binary = fa.norm != cv2.NORM_L2
    out, dists = [], []
    r2 = radius * radius
    for ia, (px, py) in enumerate(proj):
        if not np.isfinite(px):
            continue
        kx, ky = int(np.floor(px / cell)), int(np.floor(py / cell))
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(grid.get((kx + dx, ky + dy), ()))
        if not cand:
            continue
        cand = np.array(cand)
        d2 = ((fb.pts[cand] - (px, py)) ** 2).sum(1)
        cand = cand[d2 <= r2]
        if len(cand) == 0:
            continue
        if binary:
            x = np.bitwise_xor(fb.desc[cand], fa.desc[ia])
            dd = np.unpackbits(x, axis=1).sum(1).astype(np.float32)
        else:
            dd = np.linalg.norm(fb.desc[cand].astype(np.float32) - fa.desc[ia], axis=1)
        o = np.argsort(dd)
        if len(o) == 1 or dd[o[0]] < ratio * dd[o[1]]:
            out.append((ia, cand[o[0]]))
            dists.append(dd[o[0]])
    if not out:
        return np.zeros((0, 2), np.int32)
    out = np.array(out, np.int32)
    out = out[np.argsort(np.array(dists))]
    _, first = np.unique(out[:, 1], return_index=True)
    return out[np.sort(first)]


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))


def _min_side_ratio(pts: np.ndarray) -> float:
    """Ratio of the minor to the major spread of points (0 = all on a line)."""
    if len(pts) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T))
    return float(np.sqrt(max(ev[0], 0) / max(ev[1], 1e-9)))


def plausible(H: np.ndarray, size: tuple, p: MatchParams) -> tuple[bool, str]:
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return False, "degenerate model"
    H = H / H[2, 2]
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    pr = cv2.perspectiveTransform(corners[None], H)[0]
    if not cv2.isContourConvex(pr.astype(np.float32)):
        return False, "non-convex footprint"
    area = cv2.contourArea(pr.astype(np.float32))
    s = np.sqrt(abs(area) / (w * h))
    if not (1 / p.max_scale_change <= s <= p.max_scale_change):
        return False, f"scale change {s:.2f}"
    # local scale at the four corners vs centre (perspective strength)
    sides = np.linalg.norm(np.roll(pr, -1, 0) - pr, axis=1)
    ref = np.array([w, h, w, h], np.float64)
    rel = sides / ref
    if rel.max() / max(rel.min(), 1e-9) > 1 + p.max_perspective * 2:
        return False, "strong perspective"
    return True, ""


def verify(fa: Features, fb: Features, idx: np.ndarray, p: MatchParams):
    """Robust homography on matched indices -> (H, inlier_idx, stats) or (None, ..)."""
    stats = {"n_matches": len(idx)}
    if len(idx) < max(8, p.min_inliers // 2):
        return None, None, stats | {"reason": f"only {len(idx)} matches"}
    a = fa.pts[idx[:, 0]].astype(np.float64)
    b = fb.pts[idx[:, 1]].astype(np.float64)
    H, mask = cv2.findHomography(a, b, _USAC, p.ransac_px, maxIters=5000, confidence=0.9999)
    if H is None:
        return None, None, stats | {"reason": "no model"}
    inl = idx[mask.ravel().astype(bool)]
    return H, inl, stats


def match_pair(fa: Features, fb: Features, i: int, j: int, p: MatchParams) -> PairResult:
    idx = knn_ratio_match(fa, fb, p.ratio)
    H, inl, st = verify(fa, fb, idx, p)
    res = PairResult(i, j, fa.method, False, n_matches=st["n_matches"])
    if H is None:
        res.reason = st["reason"]
        return res
    if p.guided and len(inl) >= 8:
        g = guided_match(fa, fb, H, p.guided_radius_px)
        if len(g) > len(inl):
            H2, inl2, _ = verify(fa, fb, g, p)
            if H2 is not None and len(inl2) > len(inl):
                H, inl = H2, inl2
                res.n_matches = max(res.n_matches, len(g))
    return finalize(res, fa, fb, H, inl, p)


def finalize(res: PairResult, fa: Features, fb: Features, H, inl, p: MatchParams) -> PairResult:
    a = fa.pts[inl[:, 0]].astype(np.float64)
    b = fb.pts[inl[:, 1]].astype(np.float64)
    res.H = H / H[2, 2]
    res.pts_i, res.pts_j = a, b
    res.n_inliers = len(inl)
    res.inlier_ratio = len(inl) / max(res.n_matches, 1)
    w, h = fa.size
    res.coverage = _hull_area(a) / float(w * h)
    if len(a):
        pr = cv2.perspectiveTransform(a[None], res.H)[0]
        res.rmse = float(np.sqrt(np.mean(np.sum((pr - b) ** 2, axis=1))))
    ok, why = plausible(res.H, fa.size, p)
    if res.n_inliers < p.min_inliers:
        res.reason = f"{res.n_inliers} inliers < {p.min_inliers}"
    elif res.inlier_ratio < p.min_inlier_ratio:
        res.reason = f"inlier ratio {res.inlier_ratio:.2f}"
    elif res.coverage < p.min_coverage:
        res.reason = f"inliers cover {res.coverage:.3f} of the image"
    elif _min_side_ratio(a) < 0.08:
        res.reason = "inliers lie on a line"
    elif not ok:
        res.reason = why
    else:
        res.ok = True
    return res


def match_pair_union(feats_a: list, feats_b: list, i: int, j: int, p: MatchParams) -> PairResult:
    """Pool ratio-test matches of several methods, verify once."""
    pa, pb, names, n = [], [], [], 0
    for fa, fb in zip(feats_a, feats_b):
        idx = knn_ratio_match(fa, fb, p.ratio)
        if len(idx):
            pa.append(fa.pts[idx[:, 0]])
            pb.append(fb.pts[idx[:, 1]])
            names.append(fa.method)
            n += len(idx)
    res = PairResult(i, j, "+".join(f.method for f in feats_a), False, n_matches=n)
    if n < max(8, p.min_inliers // 2):
        res.reason = f"only {n} matches"
        return res
    A = Features(np.concatenate(pa), None, cv2.NORM_L2, res.method, feats_a[0].size)
    B = Features(np.concatenate(pb), None, cv2.NORM_L2, res.method, feats_b[0].size)
    idx = np.stack([np.arange(n), np.arange(n)], 1).astype(np.int32)
    H, inl, st = verify(A, B, idx, p)
    if H is None:
        res.reason = st["reason"]
        return res
    return finalize(res, A, B, H, inl, p)


def phase_correlation_pair(img_a: np.ndarray, img_b: np.ndarray, i: int, j: int,
                           min_response: float = 0.08) -> PairResult:
    """
    Direct (feature-free) translation estimate for low-texture pairs.
    Returns a PairResult with a synthetic grid of correspondences so that it can
    enter the global adjustment like a feature pair (with low weight).
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    res = PairResult(i, j, "phasecorr", False)
    res.n_matches = 1
    if resp < min_response:
        res.reason = f"phase correlation response {resp:.3f}"
        return res
    ov_w, ov_h = w - abs(dx), h - abs(dy)
    if ov_w < 0.1 * w or ov_h < 0.1 * h:
        res.reason = "overlap too small"
        return res
    # Verify with NCC in the overlap
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    bw = cv2.warpAffine(b, M, (w, h))
    valid = cv2.warpAffine(np.ones_like(b), M, (w, h)) > 0.99
    va, vb = a[valid], bw[valid]
    if va.size <= 100 or va.std() < 1e-3 * (abs(va.mean()) + 1) or vb.std() < 1e-3 * (abs(vb.mean()) + 1):
        res.reason = "no texture for phase correlation"          # blank / empty photo
        return res
    ncc = float(np.corrcoef(va, vb)[0, 1])
    if not np.isfinite(ncc) or ncc < 0.3:
        res.reason = f"NCC {ncc:.2f} at phase-correlation shift"
        return res
    x0, x1 = max(0, -dx), min(w, w - dx)
    y0, y1 = max(0, -dy), min(h, h - dy)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 8)[1:-1], np.linspace(y0, y1, 8)[1:-1])
    pa = np.stack([gx.ravel(), gy.ravel()], 1)
    res.pts_i, res.pts_j = pa, pa + (dx, dy)
    res.H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    res.n_inliers = len(pa)
    res.inlier_ratio = 1.0
    res.coverage = (x1 - x0) * (y1 - y0) / float(w * h)
    res.rmse = 0.0
    res.ok = True
    res.reason = f"response {resp:.3f}, NCC {ncc:.2f}"
    return res
