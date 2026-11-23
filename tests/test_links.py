"""Weak sequence links must hold a group that hangs on them, wherever it starts."""
import numpy as np
import pytest

from mosaic2d import bundle as BA
from mosaic2d import camera as CAM

W, H_ = 1000, 800
STEP = 300.0


def _setup(offset=0.0, persp=0.0):
    """Frames 0-9 and 10-19 on a straight line, two groups joined only by a
    sequence link 9-10 whose initial transform is `offset` px off; `persp`:
    perspective noise in the matched pairs' transforms (as estimated pair
    homographies have), which chained along the strip spoils the start."""
    rng = np.random.default_rng(3)
    pos = {k: np.array([STEP * k, 0.0]) for k in range(20)}
    g = np.array([[x, y] for x in np.linspace(50, 950, 8) for y in np.linspace(50, 750, 6)])
    corr, pairH = [], {}
    for a, b in [(k, k + 1) for k in range(19) if k != 9] + [(k, k + 2) for k in range(18) if k not in (8, 9)]:
        d = pos[a] - pos[b]
        q = g + d
        ok = (q[:, 0] > 0) & (q[:, 0] < W) & (q[:, 1] > 0) & (q[:, 1] < H_)
        corr.append(BA.Correspondences(a, b, g[ok], q[ok], 1.0, "sift"))
        Hp = np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]], float)
        if persp:                            # same centre mapping, tilted plane around it
            c = np.array([W / 2, H_ / 2])
            G = np.eye(3)
            G[2, :2] = rng.normal(0, persp, 2)
            G[2, 2] = 1 - G[2, :2] @ c
            Tc = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
            Hp = Hp @ np.linalg.inv(Tc) @ np.diag([1, 1, 1.0]) @ Tc @ G
        pairH[(a, b)] = Hp
    # the link: typical step, 20 points; its initial transform is off
    gl = np.array([[x, y] for x in np.linspace(100, 900, 5) for y in np.linspace(80, 720, 4)])
    d = pos[9] - pos[10]
    ql = gl + d
    ok = (ql[:, 0] > 0) & (ql[:, 0] < W)
    corr.append(BA.Correspondences(9, 10, gl[ok], ql[ok], 0.05, "sequence", float("inf")))
    pairH[(9, 10)] = np.array([[1, 0, d[0] + offset], [0, 1, offset], [0, 0, 1]], float)
    sizes = {k: (W, H_) for k in range(20)}
    return corr, pairH, sizes


def _step(res, i, j):
    c = np.array([[W / 2, H_ / 2, 1.0]])
    a = (res.H[i] @ c.T).ravel()
    b = (res.H[j] @ c.T).ravel()
    return float(np.hypot(*(a[:2] / a[2] - b[:2] / b[2])))


@pytest.mark.parametrize("offset,persp", [(5000.0, 0.0), (0.0, 3e-4)])
def test_similarity_bundle_follows_link(offset, persp):
    corr, pairH, sizes = _setup(offset, persp)
    res = BA.run_bundle(list(range(20)), corr, pairH, sizes, model="similarity", distortion=False,
                        log=lambda *a: None)
    assert len(res.components[0]) == 20
    assert abs(_step(res, 9, 10) - STEP) < 10
    assert not any((d["i"], d["j"]) == (9, 10) for d in res.removed_pairs)


@pytest.mark.parametrize("offset,persp", [(5000.0, 0.0), (0.0, 3e-4)])
def test_camera_bundle_follows_link(offset, persp):
    corr, pairH, sizes = _setup(offset, persp)
    res = CAM.run_camera_bundle(list(range(20)), corr, pairH, sizes, 1000.0, distortion=False,
                                log=lambda *a: None)
    s = _step(res, 0, 1)                     # mosaic px per native px may differ
    assert abs(_step(res, 9, 10) - s) < 0.05 * s
    assert res.rmse_px < 5
