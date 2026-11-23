"""Registration of small previews for consecutive photos whose features did not match."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from mosaic2d import pipeline as PL
from mosaic2d.matching import register_thumbnails


def _seabed(seed, size=1400):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    for _ in range(900):
        x, y = rng.integers(0, size, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 40)), float(rng.uniform(0.2, 1)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return np.dstack([img * 30000 + 2000] * 3).astype(np.uint16)


def _crop(big, cx, cy, deg, w=600, h=580):
    R = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    R[:, 2] += (w / 2 - cx, h / 2 - cy)
    return cv2.warpAffine(big, R, (w, h))


def test_register_recovers_shift_and_rotation():
    big = _seabed(1)
    a = _crop(big, 600, 700, 0)
    b = _crop(big, 860, 650, 6)                 # 260 px further, turned 6 deg
    c, M, ov = register_thumbnails(a, b)
    assert c > 0.6 and ov > 0.3
    centre_b_in_a = M @ [300, 290, 1]
    assert np.hypot(centre_b_in_a[0] - 560, centre_b_in_a[1] - 240) < 12
    other = _crop(_seabed(2), 700, 700, 0)
    assert register_thumbnails(a, other)[0] < 0.3


def test_sequence_register_adds_pair_in_working_coordinates():
    big = _seabed(3)
    imgs = [_crop(big, 500 + 240 * k, 700, 0) for k in range(3)]
    p = PL.Project.__new__(PL.Project)
    p.s = SimpleNamespace(sequence=True)
    p.log = lambda *a: None
    p.pairs = {}
    p.frames = [SimpleNamespace(index=k, path=Path(f"P{k}.tif"), seam_img=im, seam_scale=0.25,
                                thumb=None, thumb_scale=0.1, work_scale=0.5) for k, im in enumerate(imgs)]
    assert p.sequence_register() == 2
    r = p.pairs[(0, 1)]
    assert r.ok and r.method == "thumbnail"
    # preview px / 0.25 = native px, * 0.5 = working px: 240 preview px = 480 working px
    c = cv2.perspectiveTransform(np.array([[[600.0, 580.0]]]), r.H)[0, 0]
    assert abs(c[0] - (600 - 480)) < 15 and abs(c[1] - 580) < 15
    assert p.sequence_register() == 0          # each pair is tried once


def test_verify_by_previews_rejects_matches_between_unlike_photos():
    from mosaic2d import matching as MT
    big = _seabed(4)
    a = _crop(big, 600, 700, 0)
    b = _crop(big, 800, 700, 0)                  # 200 preview px further: overlaps a
    c = _crop(_seabed(5), 700, 700, 0)           # other seabed
    p = PL.Project.__new__(PL.Project)
    p.s = SimpleNamespace(sequence=False, verify_previews=True)
    p.log = lambda *a: None
    p.frames = [SimpleNamespace(index=k, path=Path(f"P{k}.tif"), seam_img=im, seam_scale=0.25,
                                thumb=None, thumb_scale=0.1, work_scale=0.5) for k, im in enumerate([a, b, a, a, c])]
    # working px = preview px * 2; i -> j: x_j = x_i - 400
    T = np.array([[1, 0, -400.0], [0, 1, 0], [0, 0, 1]])
    p.pairs = {(0, 1): MT.PairResult(0, 1, "sift", True, H=T),
               (0, 4): MT.PairResult(0, 4, "sift", True, H=T),          # other seabed
               (1, 3): MT.PairResult(1, 3, "sift", True, H=np.linalg.inv(T))}
    assert p.verify_by_previews() == 1
    assert p.pairs[(0, 1)].ok and p.pairs[(1, 3)].ok and not p.pairs[(0, 4)].ok
    assert p.verify_by_previews() == 0           # checked once
