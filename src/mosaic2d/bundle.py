"""
bundle.py

Global 2D bundle adjustment of a mosaic.

Every image i gets a planar transform H_i (image px -> mosaic px). All
accepted pairs contribute their inlier correspondences (p in image i, q in
image j). The adjustment minimises, over all images at once,

    r = pi(H_j^-1 H_i u(p)) - u(q)        (and the same from j to i)

i.e. the reprojection error measured in pixels of the *target image*, which
-- unlike an error measured in mosaic space -- cannot be reduced by shrinking
the mosaic. u() removes a radial lens distortion (k1, k2) shared by all images
of the same camera, estimated together with the transforms (self-calibration,
as in Metashape's camera optimisation; useful for flat-port underwater housings
and wide lenses).

Models (per image): translation (2 dof), similarity (4), affine (6),
homography (8). Optimisation is Levenberg-Marquardt with a Cauchy robust loss
on dense normal equations, which is exact and fast for a few hundred images.
Gauge: the anchor image keeps H = identity.

After convergence, correspondences and whole pairs whose residuals are
inconsistent with the rest are removed and the adjustment is repeated
(comparable to Metashape's gradual selection on reprojection error).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

MODELS = ("translation", "rigid", "similarity", "affine", "homography")


def _basis(model: str):
    """vec(H) = c + M @ theta (row-major 3x3)."""
    c = np.zeros(9)
    c[8] = 1.0
    if model == "translation":
        c[0] = c[4] = 1.0
        M = np.zeros((9, 2))
        M[2, 0] = M[5, 1] = 1
    elif model in ("similarity", "rigid"):      # rigid = similarity with the scale tied to 1
        M = np.zeros((9, 4))
        M[0, 0] = M[4, 0] = 1     # a
        M[1, 1], M[3, 1] = -1, 1  # b
        M[2, 2] = M[5, 3] = 1
    elif model == "affine":
        M = np.zeros((9, 6))
        for k, idx in enumerate([0, 1, 2, 3, 4, 5]):
            M[idx, k] = 1
    elif model == "homography":
        M = np.zeros((9, 8))
        for k in range(8):
            M[k, k] = 1
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return c, M


def project_to_model(H: np.ndarray, model: str, size) -> np.ndarray:
    """Closest model-type transform to H (least squares on a 5x5 grid)."""
    H = H / H[2, 2]
    if model == "homography":
        return H
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, w, 5), np.linspace(0, h, 5))
    src = np.stack([gx.ravel(), gy.ravel()], 1)
    dst = cv2.perspectiveTransform(src[None], H)[0]
    if model == "affine":
        A, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    elif model in ("similarity", "rigid"):
        A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if model == "rigid":
            sc = float(np.hypot(A[0, 0], A[1, 0])) or 1.0
            A[:, :2] /= sc
    else:
        t = (dst - src).mean(0)
        A = np.array([[1, 0, t[0]], [0, 1, t[1]]], np.float64)
    return np.vstack([A, [0, 0, 1]])


def theta_from_H(H: np.ndarray, model: str) -> np.ndarray:
    c, M = _basis(model)
    v = (H / H[2, 2]).ravel() - c
    th, *_ = np.linalg.lstsq(M, v, rcond=None)
    return th


def H_from_theta(th: np.ndarray, model: str) -> np.ndarray:
    c, M = _basis(model)
    return (c + M @ th).reshape(3, 3)


# Tie points of these methods are assumptions, not measurements (sequence mode:
# "consecutive photos are one typical step apart"), so they are never rejected.
# Their loss is set per set of tie points (Correspondences.robust_px): a link that
# is the only connection of a group is quadratic (inf) - a Cauchy loss would give
# up on it as soon as the group hanging on it starts far off, and the group would
# stay wherever it started - the others have a wide Cauchy loss.
LINK_METHODS = ("sequence",)


@dataclass
class Correspondences:
    i: int
    j: int
    p: np.ndarray       # N x 2, native px of image i
    q: np.ndarray       # N x 2, native px of image j
    weight: float = 1.0
    method: str = ""
    robust_px: float = 0.0      # Cauchy scale (px); 0: the adjustment's, inf: quadratic


@dataclass
class BundleResult:
    H: dict                      # image index -> 3x3 (native px -> mosaic px)
    k: np.ndarray                # radial distortion (k1, k2) in normalised radius
    center: np.ndarray           # distortion centre (px)
    radius: float                # normalising radius (px)
    rmse_px: float
    median_px: float
    per_pair: list = field(default_factory=list)
    per_image: dict = field(default_factory=dict)
    removed_pairs: list = field(default_factory=list)
    iterations: int = 0
    anchor: int = 0
    components: list = field(default_factory=list)


def undistort_points(p: np.ndarray, k: np.ndarray, center, radius) -> np.ndarray:
    d = (p - center) / radius
    r2 = (d ** 2).sum(1, keepdims=True)
    return center + (p - center) * (1 + k[0] * r2 + k[1] * r2 * r2)


def _thin(p, q, n_max, size):
    """Spatially even subsample of correspondences (grid buckets)."""
    if len(p) <= n_max:
        return p, q
    w, h = size
    g = int(np.ceil(np.sqrt(n_max)))
    cx = np.clip((p[:, 0] / max(w, 1) * g).astype(int), 0, g - 1)
    cy = np.clip((p[:, 1] / max(h, 1) * g).astype(int), 0, g - 1)
    cell = cy * g + cx
    rng = np.random.default_rng(0)
    order = rng.permutation(len(p))
    _, first = np.unique(cell[order], return_index=True)
    keep = order[first]
    if len(keep) < n_max:
        rest = np.setdiff1d(order, keep)[: n_max - len(keep)]
        keep = np.concatenate([keep, rest])
    return p[keep], q[keep]


def connected_components(nodes, edges) -> list[list[int]]:
    adj = {n: set() for n in nodes}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        comp, dq = [], deque([n])
        seen.add(n)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def initial_transforms(nodes, pair_H: dict, weights: dict, anchor: int) -> dict:
    """
    Maximum spanning tree (by inlier count) from the anchor; transforms are
    chained along the tree. pair_H[(i, j)] maps native px of i -> native px of j.
    """
    import heapq
    adj = {n: [] for n in nodes}
    for (i, j), H in pair_H.items():
        w = weights[(i, j)]
        adj[i].append((w, i, j, H))
        adj[j].append((w, j, i, np.linalg.inv(H)))
    Hs = {anchor: np.eye(3)}
    heap = [(-w, i, j, id(H), H) for (w, i, j, H) in adj[anchor]]
    heapq.heapify(heap)
    while heap:
        _, i, j, _, Hij = heapq.heappop(heap)
        if j in Hs:
            continue
        Hj = Hs[i] @ np.linalg.inv(Hij)
        Hs[j] = Hj / Hj[2, 2]
        for (w, a, b, H) in adj[j]:
            if b not in Hs:
                heapq.heappush(heap, (-w, a, b, id(H), H))
    return Hs


def choose_anchor(nodes, edges_w: dict) -> int:
    """Most connected image (by summed inliers); ties -> middle of the sequence."""
    score = {n: 0.0 for n in nodes}
    for (i, j), w in edges_w.items():
        score[i] += w
        score[j] += w
    mid = nodes[len(nodes) // 2]
    return max(nodes, key=lambda n: (score[n], -abs(n - mid)))


class BundleAdjuster:
    def __init__(self, model="homography", distortion=True, robust_px=2.0,
                 max_corr_per_pair=250, perspective_prior=1e-6, verbose=None):
        if model not in MODELS:
            raise ValueError(model)
        self.model = model
        self.distortion = distortion
        self.robust_px = robust_px
        self.max_corr = max_corr_per_pair
        self.persp_prior = perspective_prior
        # rigid: same camera at the same height -> every frame keeps the scale of
        # the anchor (strong prior on |(a, b)| = 1; residuals are in normalised units)
        self.scale_prior = 1e3 if model == "rigid" else 0.0
        self.log = verbose or (lambda *a: None)

    # ------------------------------------------------------------ core LM
    def _residuals_and_jac(self, x, corr, idx_of, k_off, sizes, need_jac=True):
        """Return list of (r, Ji, Jj, Jk, i, j) per correspondence set (both directions)."""
        model = self.model
        _, Mb = _basis(model)
        nk = Mb.shape[1]
        Hs = {}
        for img, off in idx_of.items():
            Hs[img] = np.eye(3) if off is None else H_from_theta(x[off:off + nk], model)
        kvec = x[k_off:k_off + 2] if self.distortion else np.zeros(2)
        out = []
        for c in corr:
            for (a, b, P, Q) in ((c.i, c.j, c.p, c.q), (c.j, c.i, c.q, c.p)):
                Pu = undistort_points(P, kvec, self.center, self.radius) * self.s
                Qu = undistort_points(Q, kvec, self.center, self.radius) * self.s
                Ha, Hb = Hs[a], Hs[b]
                Hb_inv = np.linalg.inv(Hb)
                Pt = np.c_[Pu, np.ones(len(Pu))]
                X = Pt @ Ha.T                    # mosaic homogeneous
                Y = X @ Hb_inv.T                 # target homogeneous
                z = Y[:, 2:3]
                proj = Y[:, :2] / z
                r = (proj - Qu).ravel()
                rob = float(getattr(c, "robust_px", 0.0) or 0.0)
                if not need_jac:
                    out.append((r, None, None, None, a, b, c.weight, rob))
                    continue
                n = len(P)
                # d proj / dY : n x 2 x 3
                dP = np.zeros((n, 2, 3))
                dP[:, 0, 0] = 1 / z[:, 0]
                dP[:, 1, 1] = 1 / z[:, 0]
                dP[:, 0, 2] = -Y[:, 0] / z[:, 0] ** 2
                dP[:, 1, 2] = -Y[:, 1] / z[:, 0] ** 2
                # dY/dvec(Ha) = Hb_inv[:, r] * Pt[c]   -> n x 3 x 9
                dYdHa = np.einsum("ir,nc->nirc", Hb_inv, Pt).reshape(n, 3, 9)
                # dY/dvec(Hb) = -Hb_inv[:, r] * Y[c]
                dYdHb = -np.einsum("ir,nc->nirc", Hb_inv, Y).reshape(n, 3, 9)
                Ja = np.einsum("nij,njk->nik", dP, dYdHa) @ Mb   # n x 2 x nk
                Jb = np.einsum("nij,njk->nik", dP, dYdHb) @ Mb
                Jk = None
                if self.distortion:
                    # d u(P)/dk and d u(Q)/dk (in scaled coords)
                    def du(Pts):
                        d = (Pts - self.center) / self.radius
                        r2 = (d ** 2).sum(1)
                        base = (Pts - self.center) * self.s
                        return np.stack([base * r2[:, None], base * (r2 ** 2)[:, None]], 2)
                    dPu = du(P)                                  # n x 2 x 2(k)
                    G = (Hb_inv @ Ha)[:, :2]                     # dY/dPu, 3 x 2
                    Jk = np.einsum("nij,jk,nkl->nil", dP, G, dPu) - du(Q)
                out.append((r, Ja.reshape(-1, nk), Jb.reshape(-1, nk),
                            None if Jk is None else Jk.reshape(-1, 2), a, b, c.weight, rob))
        return out

    def _c2(self, rob):
        """Squared Cauchy scale (normalised units) of a set of tie points; inf: quadratic."""
        return ((rob or self.robust_px) * self.s) ** 2

    def _cost(self, terms):
        tot = 0.0
        for (r, *_rest, w, rob) in terms:
            c2 = self._c2(rob)
            tot += w * 0.5 * ((r * r).sum() if np.isinf(c2) else c2 * np.log1p(r * r / c2).sum())
        return tot

    def solve(self, nodes, corr: list, init_H: dict, anchor: int, sizes: dict,
              iters=40, k_init=None) -> tuple[dict, np.ndarray, int]:
        _, Mb = _basis(self.model)
        nk = Mb.shape[1]
        w0, h0 = sizes[anchor]
        self.center = np.array([w0 / 2.0, h0 / 2.0])
        self.radius = float(np.hypot(w0, h0) / 2.0)
        self.s = 1.0 / max(w0, h0)        # coordinate normalisation
        S = np.diag([self.s, self.s, 1.0])
        Sinv = np.diag([1 / self.s, 1 / self.s, 1.0])

        idx_of, x = {}, []
        off = 0
        for n in nodes:
            if n == anchor:
                idx_of[n] = None
                continue
            Hn = S @ init_H[n] @ Sinv
            idx_of[n] = off
            x.append(theta_from_H(Hn, self.model))
            off += nk
        k_off = off
        x = np.concatenate(x + [np.zeros(2) if k_init is None else np.asarray(k_init, float)]) \
            if x else np.zeros(2)
        npar = len(x)

        mu = 1e-3
        terms = self._residuals_and_jac(x, corr, idx_of, k_off, sizes, need_jac=True)
        cost = self._cost(terms) + self._prior_cost(x, idx_of, nk)
        it = 0
        for it in range(1, iters + 1):
            A = np.zeros((npar, npar))
            g = np.zeros(npar)
            for (r, Ja, Jb, Jk, a, b, w, rob) in terms:
                wr = w / (1 + r * r / self._c2(rob))   # Cauchy IRLS weights (r / inf = 0: quadratic)
                blocks = []
                if idx_of[a] is not None:
                    blocks.append((idx_of[a], Ja))
                if idx_of[b] is not None:
                    blocks.append((idx_of[b], Jb))
                if self.distortion and Jk is not None:
                    blocks.append((k_off, Jk))
                for (oa, JA) in blocks:
                    WJA = JA * wr[:, None]
                    g[oa:oa + JA.shape[1]] += WJA.T @ r
                    for (ob, JB) in blocks:
                        A[oa:oa + JA.shape[1], ob:ob + JB.shape[1]] += WJA.T @ JB
            self._add_prior(A, g, x, idx_of, nk)
            diag = np.diag(A).copy()
            diag[diag <= 0] = 1e-12
            improved = False
            for _ in range(10):
                try:
                    dx = -np.linalg.solve(A + mu * np.diag(diag), g)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                xn = x + dx
                tn = self._residuals_and_jac(xn, corr, idx_of, k_off, sizes, need_jac=True)
                cn = self._cost(tn) + self._prior_cost(xn, idx_of, nk)
                if np.isfinite(cn) and cn < cost:
                    rel = (cost - cn) / max(cost, 1e-30)
                    x, terms, cost = xn, tn, cn
                    mu = max(mu / 3, 1e-9)
                    improved = True
                    break
                mu *= 8
            if not improved or rel < 1e-7:
                break

        Hs = {}
        for n in nodes:
            Hn = np.eye(3) if idx_of[n] is None else H_from_theta(x[idx_of[n]:idx_of[n] + nk], self.model)
            Hs[n] = Sinv @ Hn @ S
        k = x[k_off:k_off + 2] if self.distortion else np.zeros(2)
        return Hs, k, it

    def _prior_cost(self, x, idx_of, nk):
        tot = 0.0
        if self.model == "homography" and self.persp_prior:
            for off in idx_of.values():
                if off is not None:
                    tot += 0.5 * self.persp_prior * (x[off + 6] ** 2 + x[off + 7] ** 2)
        if self.scale_prior:
            for off in idx_of.values():
                if off is not None:
                    tot += 0.5 * self.scale_prior * (np.hypot(x[off], x[off + 1]) - 1.0) ** 2
        return tot

    def _add_prior(self, A, g, x, idx_of, nk):
        if self.model == "homography" and self.persp_prior:
            for off in idx_of.values():
                if off is not None:
                    for t in (6, 7):
                        A[off + t, off + t] += self.persp_prior
                        g[off + t] += self.persp_prior * x[off + t]
        if self.scale_prior:
            lam = self.scale_prior
            for off in idx_of.values():
                if off is None:
                    continue
                a, b = x[off], x[off + 1]
                sc = max(np.hypot(a, b), 1e-9)
                J = np.array([a / sc, b / sc])
                A[off:off + 2, off:off + 2] += lam * np.outer(J, J)
                g[off:off + 2] += lam * (sc - 1.0) * J

    # ------------------------------------------------------------ evaluation
    def residuals_px(self, Hs, k, corr) -> list[np.ndarray]:
        """Per correspondence set: residual magnitudes (px of the target image), both directions."""
        out = []
        for c in corr:
            Pu = undistort_points(c.p, k, self.center, self.radius)
            Qu = undistort_points(c.q, k, self.center, self.radius)
            A = np.linalg.inv(Hs[c.j]) @ Hs[c.i]
            pr = cv2.perspectiveTransform(Pu[None].astype(np.float64), A)[0]
            e1 = np.linalg.norm(pr - Qu, axis=1)
            B = np.linalg.inv(A)
            pr2 = cv2.perspectiveTransform(Qu[None].astype(np.float64), B)[0]
            e2 = np.linalg.norm(pr2 - Pu, axis=1)
            out.append(np.maximum(e1, e2))
        return out


def run_bundle(nodes: list, corr: list, pair_H: dict, sizes: dict, model="homography",
               distortion=True, robust_px=2.0, max_corr_per_pair=250,
               pair_reject_px=None, point_reject_sigma=3.0, rounds=3,
               perspective_prior=1e-6, log=print) -> BundleResult:
    """
    Full adjustment with initialisation and outlier rounds.

    corr     : list[Correspondences] (native px)
    pair_H   : {(i, j): H native i -> native j} of accepted pairs (initialisation)
    sizes    : {i: (w, h)} native image sizes
    """
    edges_w = {(c.i, c.j): float(len(c.p)) * c.weight for c in corr}
    comps = connected_components(nodes, list(edges_w))
    main = comps[0]
    anchor = choose_anchor(main, {e: w for e, w in edges_w.items() if e[0] in main})
    corr = [c for c in corr if c.i in main and c.j in main]
    thin = []
    for c in corr:
        p, q = _thin(c.p, c.q, max_corr_per_pair, sizes[c.i])
        thin.append(Correspondences(c.i, c.j, p, q, c.weight, c.method, getattr(c, "robust_px", 0.0)))
    corr = thin

    # chain the pair transforms in the model itself: chained homographies collect
    # perspective along a long strip, and their similarity part can then be far off
    init = initial_transforms(main, {e: project_to_model(H, model, sizes[e[0]]) for e, H in pair_H.items()
                                     if e in edges_w and e[0] in main}, edges_w, anchor)
    init = {n: project_to_model(init[n], model, sizes[n]) for n in main}

    removed = []
    k = None
    Hs = init
    it_total = 0
    for rnd in range(rounds):
        # Coarse-to-fine: first rounds without distortion for stability.
        use_dist = distortion and rnd > 0
        ba = BundleAdjuster(model, use_dist, robust_px, max_corr_per_pair, perspective_prior)
        Hs, k, its = ba.solve(main, corr, Hs, anchor, sizes, k_init=k if use_dist else None)
        it_total += its
        if not use_dist:
            k = np.zeros(2)
        res = ba.residuals_px(Hs, k, corr)
        allr = np.concatenate(res) if res else np.zeros(1)
        med = float(np.median(allr))
        sigma = 1.4826 * med + 1e-6
        thr_pair = pair_reject_px if pair_reject_px else max(8.0, 6 * med)
        keep = []
        dropped_any = False
        for c, r in zip(corr, res):
            pm = float(np.median(r))
            if c.method in LINK_METHODS:
                keep.append(c)
                continue
            if pm > thr_pair and c.method != "phasecorr":
                removed.append({"i": c.i, "j": c.j, "median_px": round(pm, 2), "round": rnd})
                dropped_any = True
                continue
            m = r <= max(point_reject_sigma * sigma, robust_px * 2)
            if m.sum() >= 6:
                if (~m).any():
                    dropped_any = True
                keep.append(Correspondences(c.i, c.j, c.p[m], c.q[m], c.weight, c.method))
        # keep the graph connected: if dropping pairs split it, keep them
        comps2 = connected_components(main, [(c.i, c.j) for c in keep])
        if len(comps2) > 1:
            kept_edges = {(c.i, c.j) for c in keep}
            for c in corr:
                if (c.i, c.j) not in kept_edges:
                    keep.append(c)
            removed = [r for r in removed if r["round"] != rnd]
        corr = keep
        log(f"  bundle round {rnd + 1}: model={model} distortion={use_dist} "
            f"median={med:.2f}px  pairs={len(corr)}  k={np.round(k, 4).tolist()}")
        if not dropped_any and rnd > 0 and (use_dist or not distortion):
            break

    ba = BundleAdjuster(model, distortion, robust_px, max_corr_per_pair, perspective_prior)
    ba.center = np.array([sizes[anchor][0] / 2.0, sizes[anchor][1] / 2.0])
    ba.radius = float(np.hypot(*sizes[anchor]) / 2.0)
    ba.s = 1.0 / max(sizes[anchor])
    res = ba.residuals_px(Hs, k, corr)
    allr = np.concatenate(res) if res else np.zeros(1)
    per_pair, per_img = [], {n: [] for n in main}
    for c, r in zip(corr, res):
        per_pair.append({"i": c.i, "j": c.j, "method": c.method, "n": len(r),
                         "rmse_px": float(np.sqrt(np.mean(r ** 2))), "median_px": float(np.median(r))})
        per_img[c.i].append(r)
        per_img[c.j].append(r)
    per_image = {n: {"rmse_px": float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if v else None,
                     "pairs": len(v)} for n, v in per_img.items()}
    return BundleResult(Hs, k, ba.center, ba.radius, float(np.sqrt(np.mean(allr ** 2))),
                        float(np.median(allr)), per_pair, per_image, removed, it_total, anchor, comps)
