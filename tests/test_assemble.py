"""Assembling separate groups along the continued camera path."""

import csv

import numpy as np

from mosaic2d.assemble import assemble, path_motion, place_groups
from mosaic2d.tiffwriter import write_tiff


def _group(name, frames, centres):
    return {"tif": __import__("pathlib").Path(name), "centres": dict(zip(frames, map(np.array, centres))),
            "first": min(frames), "last": max(frames), "px": 0.001, "images": len(frames)}


def test_straight_path_continues_over_the_gap():
    a = _group("a.tif", list(range(10)), [(100.0 * i, 50.0) for i in range(10)])
    # group b in its own pixels: moving along +y, starting at (500, 500)
    b = _group("b.tif", list(range(16, 26)), [(500.0, 500.0 + 100 * i) for i in range(10)])
    place_groups([a, b], log=lambda *x: None)
    M = b["M"]
    p = M @ np.array([500.0, 500.0, 1.0])
    assert np.allclose(p[:2], [900.0 + 7 * 100.0, 50.0], atol=1e-6)        # 6 missing photos
    q = M @ np.array([500.0, 600.0, 1.0])
    assert np.allclose(q[:2] - p[:2], [100.0, 0.0], atol=1e-6)            # turned to +x


def test_turning_path_keeps_turning():
    th = np.radians(5) * np.arange(12)
    pts = np.cumsum(np.column_stack([np.cos(th), np.sin(th)]) * 100, axis=0)
    step, heading, rate = path_motion(pts, np.arange(12), at_end=True)
    assert abs(np.degrees(rate) - 5) < 0.6
    assert abs(np.degrees(heading) - 55) < 5


def test_assemble_writes_one_image(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for k, (stem, frames, cen) in enumerate((
            ("m", list(range(0, 8)), [(150 + 60 * i, 200) for i in range(8)]),
            ("m_block02", list(range(12, 20)), [(150 + 60 * i, 200) for i in range(8)]))):
        img = (rng.random((400, 700, 5)) * 200 + 20).astype(np.uint8)
        write_tiff(out / f"{stem}.tif", img, band_names=[f"{w} nm" for w in (450, 550, 650, 730, 850)],
                   pixel_size=0.001)
        H = {f: np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1.0]]) for f, c in zip(frames, cen)}
        np.save(out / f"{stem}_transforms.npy", H, allow_pickle=True)
        rows.append([k + 1, len(frames), f"P{frames[0]}.tif", f"P{frames[-1]}.tif", f"{stem}.tif"])
    with open(out / "m_blocks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "images", "first", "last", "output"])
        w.writerows(rows)
    with open(out / "m_images.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "aligned"])
        w.writerows([[f"P{i}.tif", 1] for i in range(20)])
    res = assemble(out, log=lambda *x: None)
    import tifffile
    a = tifffile.imread(res)
    # second group continues to the right after 3 missing photos: wider than both
    assert a.shape[2] == 5 and a.shape[1] > 1100
    assert (a > 0).any(axis=2).mean() > 0.4
    assert (out / "m_assembled_preview_rgb.jpg").is_file()
    assert (out / "m_assembled_groups.csv").is_file()
