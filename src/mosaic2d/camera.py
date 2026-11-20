"""
Camera-over-plane adjustment (model "camera").

Every photo is a real pinhole camera looking at one common plane (the
seabed / ground), instead of a free 2D transform:

    pose per photo : x, y (position over the plane), log h (height),
                     tilt ax, ay (rotation vector of the tilt), yaw
    shared lens    : focal length f (from EXIF, refined), radial k1, k2
    priors         : similar height (log h ~ N(ref, height_sigma)),
                     close to nadir (tilt ~ N(0, tilt_sigma))

Each photo's mosaic transform is the homography the plane induces,
H_i = (K R_i [e1 e2 -C_i])^-1, so a photo can be bigger or smaller only by
a real change of height, it cannot shear, and it cannot fold. A free
homography has 8 parameters per photo with no physical meaning; along a
long chain small errors in them pile up as scale drift, shear and folding.
This is the planar special case of what Metashape solves (shared camera
calibration + camera poses); there is no surface model (DEM), so no extra
estimate enters the mosaic.

Residuals: a tie point seen in photo i is cast onto the plane along its ray
and projected into photo j; the residual is the distance (px) to where it was
measured in j, in both directions. Robust (Cauchy) IRLS, Levenberg-Marquardt
on the sparse normal equations, Jacobian by finite differences done per
parameter for all tie points at once (15 evaluations per iteration).
Rounds of outlier rejection as in Metashape's gradual selection: points far
from the fit are dropped, pairs whose matches do not fit at all are removed.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import bundle as BA

NP = 6          # x, y, log h, ax, ay, yaw
NG = 3          # log f, k1, k2


def _rot(ax, ay, yaw):
    """Batch rotation (world -> camera): R = Rodrigues([ax, ay, 0]) @ Rz(yaw). Returns N x 3 x 3."""
    n = np.broadcast(ax, ay, yaw).shape[0] if np.ndim(ax) else 1
    ax = np.broadcast_to(ax, (n,)).astype(np.float64)
    ay = np.broadcast_to(ay, (n,)).astype(np.float64)
    yaw = np.broadcast_to(yaw, (n,)).astype(np.float64)
    th = np.sqrt(ax * ax + ay * ay)
    small = th < 1e-12
    ths = np.where(small, 1.0, th)
    kx, ky = ax / ths, ay / ths
    s, c = np.sin(th), np.cos(th)
    v = 1 - c
    T = np.empty((n, 3, 3))
    T[:, 0, 0] = c + kx * kx * v
    T[:, 0, 1] = kx * ky * v
    T[:, 0, 2] = ky * s
    T[:, 1, 0] = kx * ky * v
    T[:, 1, 1] = c + ky * ky * v
    T[:, 1, 2] = -kx * s
    T[:, 2, 0] = -ky * s
    T[:, 2, 1] = kx * s
    T[:, 2, 2] = c
    T[small] = np.eye(3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Z = np.zeros((n, 3, 3))
    Z[:, 0, 0], Z[:, 0, 1] = cy, -sy
    Z[:, 1, 0], Z[:, 1, 1] = sy, cy
    Z[:, 2, 2] = 1.0
    return T @ Z


def plane_to_image(P, f, c):
    """3x3 homography plane (X, Y, 1) -> undistorted image px for pose P = (x, y, logh, ax, ay, yaw)."""
    x, y, lh, ax, ay, yaw = P
    R = _rot(np.array([ax]), np.array([ay]), np.array([yaw]))[0]
    C = np.array([x, y, -np.exp(lh)])
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    M = np.column_stack([R[:, 0], R[:, 1], -R @ C])
    return K @ M


def image_to_plane(P, f, c):
    G = plane_to_image(P, f, c)
    H = np.linalg.inv(G)
    return H / H[2, 2]


class _Problem:
    def __init__(self, rows_i, rows_j, p, q, cen_i, cen_j, center, radius, wt=None):
        self.ri, self.rj = rows_i, rows_j
        self.wt = np.ones(len(p)) if wt is None else wt      # per tie point (pair weight)
        self.p, self.q = p, q
        self.ci, self.cj = cen_i, cen_j
        self.center, self.radius = center, radius

    def residuals(self, Pi, Pj, g):
        """
        N x 2 residuals (px). Pi, Pj: F x 6 photo parameters used in the source and
        target role (the same array, or one with a perturbed column for the Jacobian).
        """
        f = np.exp(g[0])
        k = g[1:3]
        pu = BA.undistort_points(self.p, k, self.center, self.radius)
        qu = BA.undistort_points(self.q, k, self.center, self.radius)
        Rf_i = _rot(Pi[:, 3], Pi[:, 4], Pi[:, 5])                 # one rotation per photo ...
        Rf_j = Rf_i if Pj is Pi else _rot(Pj[:, 3], Pj[:, 4], Pj[:, 5])
        Ri, Rj = Rf_i[self.ri], Rf_j[self.rj]                     # ... indexed per tie point
        a = (pu[:, 0] - self.ci[:, 0]) / f
        b = (pu[:, 1] - self.ci[:, 1]) / f
        # world ray d = R^T (a, b, 1)
        dx = Ri[:, 0, 0] * a + Ri[:, 1, 0] * b + Ri[:, 2, 0]
        dy = Ri[:, 0, 1] * a + Ri[:, 1, 1] * b + Ri[:, 2, 1]
        dz = Ri[:, 0, 2] * a + Ri[:, 1, 2] * b + Ri[:, 2, 2]
        Pis, Pjs = Pi[self.ri], Pj[self.rj]
        bad = dz <= 1e-9
        s = np.exp(Pis[:, 2]) / np.where(bad, 1.0, dz)
        X = Pis[:, 0] + s * dx - Pjs[:, 0]
        Y = Pis[:, 1] + s * dy - Pjs[:, 1]
        Z = np.exp(Pjs[:, 2])                                     # plane point minus camera j (z: 0 - (-h))
        vx = Rj[:, 0, 0] * X + Rj[:, 0, 1] * Y + Rj[:, 0, 2] * Z
        vy = Rj[:, 1, 0] * X + Rj[:, 1, 1] * Y + Rj[:, 1, 2] * Z
        vz = Rj[:, 2, 0] * X + Rj[:, 2, 1] * Y + Rj[:, 2, 2] * Z
        bad |= vz <= 1e-9
        vz = np.where(bad, 1.0, vz)
        r = np.column_stack([f * vx / vz + self.cj[:, 0] - qu[:, 0], f * vy / vz + self.cj[:, 1] - qu[:, 1]])
        r[bad] = 1e4
        return r


def _robust_cost(r, R):
    s = (r * r).sum(1)
    return 0.5 * R * R * np.log1p(s / (R * R))


def run_camera_bundle(nodes, corr, pair_H, sizes, focal_px, focal_sigma=0.3, distortion=True,
                      height_sigma=0.1, tilt_sigma_deg=15.0, robust_px=3.0, max_corr_per_pair=40,
                      rounds=3, max_iter=60, k_init=None, k_sigma=0.2, log=print) -> BA.BundleResult:
    """
    focal_px     : nominal focal length in px of the native image (EXIF); refined with a
                   log-normal prior of width focal_sigma (underwater housings change it)
    height_sigma : prior width of log(height / typical height); 0.1 ~ heights within +-10 %
    tilt_sigma   : prior width of the tilt from nadir (deg)
    k_init, k_sigma : starting radial distortion and its prior width (e.g. the main block's
                   lens for a small separate group: same lens, k_sigma small)
    Priors are scaled so that one sigma costs as much as all of a photo's tie points
    being off by 1 px: they decide only what the tie points cannot.
    """
    # 1. robust start: similarity adjustment (stable, no folding), pair outliers removed
    sim = BA.run_bundle(nodes, corr, pair_H, sizes, model="similarity", distortion=False,
                        max_corr_per_pair=max_corr_per_pair, rounds=2, log=lambda *a: None)
    main = sim.components[0]
    ms = set(main)
    removed_sim = {(d["i"], d["j"]) for d in sim.removed_pairs}
    use = [c for c in corr if c.i in ms and c.j in ms and (c.i, c.j) not in removed_sim]
    center, radius = sim.center, sim.radius
    rows = {n: k for k, n in enumerate(main)}
    F = len(main)

    # initial poses from the similarity transforms (nadir, height = scale * f)
    P = np.zeros((F, NP))
    for n in main:
        H = sim.H[n] / sim.H[n][2, 2]
        a, b = H[0, 0], H[1, 0]
        sig = float(np.hypot(a, b))
        phi = float(np.arctan2(b, a))
        cen = np.array([sizes[n][0] / 2.0, sizes[n][1] / 2.0, 1.0])
        m = H @ cen
        P[rows[n]] = [m[0] / m[2], m[1] / m[2], np.log(sig * focal_px), 0.0, 0.0, -phi]
    k0 = np.zeros(2) if k_init is None else np.asarray(k_init, np.float64)
    g = np.array([np.log(focal_px), k0[0], k0[1]], np.float64)
    logf0 = float(np.log(focal_px))
    href = float(np.median(P[:, 2]))
    anchor = sim.anchor if sim.anchor in rows else main[0]
    a_row = rows[anchor]
    anchor_fix = P[a_row, [0, 1, 5]].copy()

    # 2. tie points (thinned, both directions)
    def build(cs):
        I, J, p, q, W = [], [], [], [], []
        for c in cs:
            pp, qq = BA._thin(c.p, c.q, max_corr_per_pair, sizes[c.i])
            for a_, b_, x_, y_ in ((c.i, c.j, pp, qq), (c.j, c.i, qq, pp)):
                W.append(np.full(len(x_), float(c.weight)))
                I.append(np.full(len(x_), rows[a_]))
                J.append(np.full(len(x_), rows[b_]))
                p.append(x_)
                q.append(y_)
        I = np.concatenate(I)
        J = np.concatenate(J)
        cen = np.array([[sizes[n][0] / 2.0, sizes[n][1] / 2.0] for n in main])
        return _Problem(I, J, np.concatenate(p).astype(np.float64), np.concatenate(q).astype(np.float64),
                        cen[I], cen[J], center, radius, np.concatenate(W))

    cs = list(use)
    prob = build(cs)
    tilt_sig = np.radians(tilt_sigma_deg)
    removed = list(sim.removed_pairs)
    its_total = 0

    for rnd in range(rounds):
        P, g, its = _lm(prob, P, g, F, a_row, anchor_fix, href, logf0, focal_sigma, height_sigma,
                        tilt_sig, robust_px, distortion, max_iter, k0, k_sigma)
        its_total += its
        r = prob.residuals(P, P, g)
        e = np.hypot(r[:, 0], r[:, 1])
        med = float(np.median(e))
        # per pair (both directions) medians -> remove pairs that do not fit at all
        keep_c, dropped, cut = [], False, []
        off = 0
        thr_pair = max(10.0, 6.0 * med)
        thr_pt = max(3.0 * robust_px, 5.0 * 1.4826 * med)
        for c in cs:
            pp, qq = BA._thin(c.p, c.q, max_corr_per_pair, sizes[c.i])
            n = len(pp)
            e_ij = e[off:off + n]
            e_ji = e[off + n:off + 2 * n]
            off += 2 * n
            ep = np.maximum(e_ij, e_ji)
            pm = float(np.median(ep))
            if pm > thr_pair and c.method != "phasecorr":
                removed.append({"i": c.i, "j": c.j, "median_px": round(pm, 2), "round": 10 + rnd})
                cut.append((pm, c))
                dropped = True
                continue
            m = ep <= thr_pt
            if m.sum() >= 6:
                dropped |= bool((~m).any())
                keep_c.append(BA.Correspondences(c.i, c.j, pp[m], qq[m], c.weight, c.method))
            else:
                cut.append((pm, c))
        # removals must not split the block here: put back the best-fitting removed
        # pairs until it is connected again (the others stay out)
        parent = {n: n for n in main}

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for c in keep_c:
            parent[root(c.i)] = root(c.j)
        for pm, c in sorted(cut, key=lambda t: t[0]):
            if root(c.i) != root(c.j):               # this pair is the only link: keep it
                parent[root(c.i)] = root(c.j)
                keep_c.append(c)
                removed = [d for d in removed if not (d.get("round") == 10 + rnd and d["i"] == c.i
                                                      and d["j"] == c.j)]
        cs = keep_c
        prob = build(cs)
        f_now = float(np.exp(g[0]))
        h = np.exp(P[:, 2] - href)
        tilt = np.degrees(np.hypot(P[:, 3], P[:, 4]))
        log(f"  camera adjustment round {rnd + 1}: median={med:.2f}px pairs={len(cs)} "
            f"f={f_now:.0f}px ({f_now / focal_px:.2f} x nominal) k={np.round(g[1:], 4).tolist()} "
            f"height {h.min():.2f}-{h.max():.2f} x typical, tilt median {np.median(tilt):.1f} deg "
            f"(max {tilt.max():.1f})")
        if not dropped:
            break

    # 3. results in the BundleResult form used by the rest of the pipeline
    f = float(np.exp(g[0]))
    Hs = {}
    for n in main:
        Hs[n] = image_to_plane(P[rows[n]], f, (sizes[n][0] / 2.0, sizes[n][1] / 2.0))
    r = prob.residuals(P, P, g)
    e = np.hypot(r[:, 0], r[:, 1])
    per_pair, per_img = [], {n: [] for n in main}
    off = 0
    for c in cs:
        n = len(BA._thin(c.p, c.q, max_corr_per_pair, sizes[c.i])[0])   # as in build()
        ec = np.concatenate([e[off:off + n], e[off + n:off + 2 * n]])
        off += 2 * n
        per_pair.append({"i": c.i, "j": c.j, "method": c.method, "n": n,
                         "rmse_px": float(np.sqrt(np.mean(ec ** 2))), "median_px": float(np.median(ec))})
        per_img[c.i].append(ec)
        per_img[c.j].append(ec)
    per_image = {n: {"rmse_px": float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if v else None,
                     "pairs": len(v)} for n, v in per_img.items()}
    res = BA.BundleResult(Hs, g[1:3].copy(), center, radius, float(np.sqrt(np.mean(e ** 2))),
                          float(np.median(e)), per_pair, per_image, removed, its_total + sim.iterations,
                          anchor, [list(main)] + [list(c) for c in sim.components[1:]])
    res.camera = {"focal_px": f, "focal_nominal_px": float(focal_px),
                  "height_rel": {n: float(np.exp(P[rows[n], 2] - href)) for n in main},
                  "tilt_deg": {n: float(np.degrees(np.hypot(P[rows[n], 3], P[rows[n], 4]))) for n in main},
                  "poses": {n: P[rows[n]].tolist() for n in main}}
    return res


MAX_TILT = np.radians(60.0)


def _lm(prob, P, g, F, a_row, anchor_fix, href, logf0, focal_sigma, height_sigma, tilt_sig,
        robust_px, distortion, max_iter, k0=(0.0, 0.0), k_sigma=0.2):
    """Levenberg-Marquardt with Cauchy IRLS and per-photo priors."""
    N = len(prob.p)
    nvar = F * NP + NG
    ri, rj = prob.ri, prob.rj
    # tie points per photo -> prior weights ("one sigma = all its tie points off by 1 px")
    cnt = np.bincount(ri, minlength=F) + np.bincount(rj, minlength=F)
    w_prior = np.maximum(cnt, 1).astype(np.float64)
    R2 = robust_px * robust_px
    steps = np.array([1e-3, 1e-3, 1e-6, 1e-7, 1e-7, 1e-7])
    gsteps = np.array([1e-6, 1e-6, 1e-6])
    big = 1e10

    def prior_terms(P, g):
        """Return (cost, list of (var index, value - reference, sigma, weight))."""
        items = []
        for k in range(F):
            items.append((k * NP + 2, P[k, 2] - href, height_sigma, w_prior[k]))
            items.append((k * NP + 3, P[k, 3], tilt_sig, w_prior[k]))
            items.append((k * NP + 4, P[k, 4], tilt_sig, w_prior[k]))
        items.append((F * NP, g[0] - logf0, focal_sigma, 0.1 * N))
        items.append((F * NP + 1, g[1] - k0[0], k_sigma, 0.1 * N))
        items.append((F * NP + 2, g[2] - k0[1], k_sigma, 0.1 * N))
        cost = 0.5 * sum(w * (d / sg) ** 2 for _, d, sg, w in items)
        return cost, items

    def total_cost(P, g):
        r = prob.residuals(P, P, g)
        c, _ = prior_terms(P, g)
        return float((_robust_cost(r, robust_px) * prob.wt).sum()) + c

    cost = total_cost(P, g)
    lam = 1e-3
    its = 0
    for it in range(max_iter):
        its = it + 1
        r0 = prob.residuals(P, P, g)
        s = (r0 * r0).sum(1)
        w = prob.wt / (1.0 + s / R2)                   # Cauchy IRLS weights x pair weight
        rows_l, cols_l, vals_l = [], [], []
        rid = np.arange(N)
        for k in range(NP):
            Pp = P.copy()
            Pp[:, k] += steps[k]
            for which, idx in ((0, ri), (1, rj)):
                # perturb every photo's parameter k in one role only: d r / d theta_(role, k)
                d = (prob.residuals(Pp, P, g) if which == 0 else prob.residuals(P, Pp, g)) - r0
                d /= steps[k]
                for comp in (0, 1):
                    rows_l.append(2 * rid + comp)
                    cols_l.append(idx * NP + k)
                    vals_l.append(d[:, comp])
        nglob = NG if distortion else 1
        for k in range(nglob):
            gp = g.copy()
            gp[k] += gsteps[k]
            d = (prob.residuals(P, P, gp) - r0) / gsteps[k]
            for comp in (0, 1):
                rows_l.append(2 * rid + comp)
                cols_l.append(np.full(N, F * NP + k))
                vals_l.append(d[:, comp])
        J = sp.csr_matrix((np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
                          shape=(2 * N, nvar))
        W = np.repeat(w, 2)
        rv = r0.reshape(-1)
        JW = J.multiply(W[:, None]).tocsr()
        A = (J.T @ JW).tocsc()
        b = J.T @ (W * rv)
        # priors (cost 0.5 w ((v - ref) / sigma)^2) and gauge (anchor position and heading)
        _, items = prior_terms(P, g)
        diag_add = np.zeros(nvar)
        b = np.asarray(b).ravel().copy()
        for idx, d, sg, wt in items:
            diag_add[idx] += wt / sg ** 2
            b[idx] += wt * d / sg ** 2
        if not distortion:
            diag_add[F * NP + 1:] += big
        for t, col in enumerate((0, 1, 5)):
            idx = a_row * NP + col
            diag_add[idx] += big
            b[idx] += big * (P[a_row, col] - anchor_fix[t])
        A = A + sp.diags(diag_add)
        dA = A.diagonal()
        improved = False
        for _ in range(8):
            Al = A + sp.diags(lam * np.maximum(dA, 1e-9))
            try:
                delta = spla.spsolve(Al.tocsc(), -b)
            except Exception:
                lam *= 10
                continue
            if not np.all(np.isfinite(delta)):
                lam *= 10
                continue
            Pn = P + delta[:F * NP].reshape(F, NP)
            gn = g + delta[F * NP:]
            # stay physical: no photo tilted beyond 60 deg (a fold / camera flipped over),
            # focal length within 4x of the nominal one
            if (np.hypot(Pn[:, 3], Pn[:, 4]).max() > MAX_TILT or abs(gn[0] - logf0) > np.log(4.0)):
                lam *= 5
                continue
            cn = total_cost(Pn, gn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                P, g, cost = Pn, gn, cn
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 5
        if not improved or rel < 1e-7:
            break
    return P, g, its
