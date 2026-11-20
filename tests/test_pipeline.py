"""End-to-end test on a synthetic survey with known geometry."""

import json

import cv2
import numpy as np
import pytest

from mosaic2d import rawio
from mosaic2d import camera as CAM
from mosaic2d.pipeline import Settings, local_scale, run


def _texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (2, 6, 18, 50):
        n = rng.normal(0, 1, (h // s + 2, w // s + 2)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * s ** 0.5
    img = (img - img.min()) / (img.max() - img.min())
    rgb = np.dstack([img, img ** 1.3, 1 - img * 0.5])
    return (rgb * 50000 + 2000).astype(np.uint16)


def _survey(tmp_path, n_cols=4, n_rows=2, w=900, h=600, seed=1, step=600, tape_y=None):
    rng = np.random.default_rng(seed)
    world = _texture(3600, 1800, seed)
    truth = {}
    k = 0
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else reversed(range(n_cols))  # zig-zag
        for c in cols:
            x, y = 250 + c * step, 250 + r * 420
            a = np.deg2rad(rng.normal(0, 3))
            s = 1 + rng.normal(0, 0.03)
            # a real camera over the textured plane (focal 1100 px, height s, small tilt);
            # image -> world: the homography the plane induces
            f = 1100.0
            tilt = np.deg2rad(rng.normal(0, 1.0, 2))
            Hc = CAM.image_to_plane([0.0, 0.0, np.log(s * f), tilt[0], tilt[1], -a], f, (w / 2, h / 2))
            H = np.array([[1, 0, x + s * w / 2], [0, 1, y + s * h / 2], [0, 0, 1]]) @ Hc
            src_world = world
            if tape_y is not None:
                # a dashed tape held above the bottom: parallax moves it between views
                src_world = world.copy()
                dy = int(rng.integers(-25, 26))
                for x0 in range(0, world.shape[1], 60):
                    cv2.line(src_world, (x0, tape_y + dy), (x0 + 42, tape_y + dy), (65535,) * 3, 7)
            img = cv2.warpPerspective(src_world, np.linalg.inv(H), (w, h), flags=cv2.INTER_LINEAR)
            img = img.astype(np.float32) * np.float32(1 - 0.25 * ((np.linspace(-1, 1, w)[None] ** 2
                                                                + np.linspace(-1, 1, h)[:, None] ** 2) / 2))[..., None]
            img = np.clip(img + rng.normal(0, 300, img.shape), 0, 65535).astype(np.uint16)
            fn = tmp_path / f"IMG_{k:03d}.tif"
            cv2.imwrite(str(fn), img[..., ::-1])
            truth[fn.name] = H
            k += 1
    return truth


@pytest.mark.parametrize("blend", ["mosaic", "disabled"])
def test_synthetic_survey(tmp_path, blend):
    src = tmp_path / "src"
    src.mkdir()
    truth = _survey(src)
    s = Settings(methods="sift,orb,fast", accuracy="high", blend=blend, resolution=0.5,
                 distortion=False, n_features=3000, orient="anchor", jobs=2)
    proj, rep = run(src, tmp_path / "out", s)
    assert rep["aligned_images"] == len(truth)
    assert rep["reprojection_median_px"] < 1.0
    # geometry: compare every image's centre relative to the anchor with the truth
    names = [f.path.name for f in proj.frames]
    a = proj.result.anchor
    Ha_true = truth[names[a]]
    errs = []
    for f in proj.frames:
        rel_true = np.linalg.inv(Ha_true) @ truth[f.path.name]   # image -> anchor image
        rel_est = np.linalg.inv(proj.result.H[a]) @ proj.result.H[f.index]
        c = np.array([[[450.0, 300.0], [0, 0], [900, 600]]])
        errs.append(np.abs(cv2.perspectiveTransform(c, rel_true) - cv2.perspectiveTransform(c, rel_est)).max())
    assert max(errs) < 3.0, errs
    # output scale refers to the median frame, whichever frame is the anchor
    s_true = np.median([local_scale(np.linalg.inv(Ha_true) @ truth[n], 450, 300) for n in names])
    s_anchor = local_scale(proj.result.H[a], 450, 300)
    assert abs(proj.median_frame_scale() / s_anchor / s_true - 1) < 0.01
    out = tmp_path / "out" / "mosaic.tif"
    assert out.is_file()
    img = rawio.read_image(out)
    assert img.dtype == np.uint16 and img.shape[2] == 3
    assert (img > 0).mean() > 0.4
    rep2 = json.loads((tmp_path / "out" / "mosaic_report.json").read_text())
    assert rep2["pairs_accepted"] >= len(truth) - 1


def test_project_rerender(tmp_path):
    from mosaic2d.pipeline import Project
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)
    s = Settings(methods="orb", accuracy="high", resolution=0.3, distortion=False, n_features=2000, jobs=2)
    run(src, tmp_path / "out", s)
    proj = Project.load(tmp_path / "out" / "mosaic_project.pkl", tmp_path / "out2",
                        blend="disabled", resolution=0.2)
    proj.render("again")
    a = rawio.read_image(tmp_path / "out" / "mosaic.tif")
    b = rawio.read_image(tmp_path / "out2" / "again.tif")
    assert abs(b.shape[1] / a.shape[1] - 2 / 3) < 0.02


def test_blurred_frame_detected_from_overlaps(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=4, n_rows=1)
    blurred = src / "IMG_002.tif"
    img = cv2.imread(str(blurred), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(blurred), cv2.GaussianBlur(img, (0, 0), 2.5))
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2)
    proj, rep = run(src, tmp_path / "out", s)
    q = {f.path.name: f.quality for f in proj.frames}
    assert q["IMG_002.tif"] < 0.5, q
    assert all(v > 0.7 for k, v in q.items() if k != "IMG_002.tif"), q


def test_ingest_cache_reused(tmp_path):
    from mosaic2d.pipeline import Project
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)
    s = Settings(methods="orb", accuracy="high", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    p1 = Project(src, tmp_path / "out", s)
    p1.ingest()
    p2 = Project(src, tmp_path / "out", s)
    p2.ingest()
    assert len(list((tmp_path / "out" / "_cache").glob("*.pkl"))) == 3
    assert all(np.array_equal(a.reg, b.reg) for a, b in zip(p1.frames, p2.frames))
    # a changed setting that affects ingest must not hit the cache
    s2 = Settings(methods="orb", accuracy="lowest", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    p3 = Project(src, tmp_path / "out", s2)
    p3.ingest()
    assert p3.frames[0].reg.shape != p1.frames[0].reg.shape


def test_degenerate_frames_detected():
    """Folded (corner behind the horizon), stretched and wrongly scaled frames."""
    from pathlib import Path
    from types import SimpleNamespace
    from mosaic2d.pipeline import Project
    proj = Project.__new__(Project)
    proj.frames = [SimpleNamespace(index=i, ref_size=(600, 400), path=Path(f"IMG_{i}.tif")) for i in range(6)]
    T = lambda x: np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
    H = {0: T(0), 1: T(300), 2: T(600)}
    H[3] = T(900) @ np.array([[1, 0, 0], [0, 1, 0], [-0.003, 0, 1]])    # folds inside the frame
    H[4] = T(1200) @ np.diag([4.0, 1.0, 1.0])                            # stretched 4:1
    H[5] = T(1500) @ np.diag([5.0, 5.0, 1.0])                            # 5x the median scale
    proj.result = SimpleNamespace(H=H)
    assert proj.degenerate_frames() == [3, 4, 5]
    proj.result = SimpleNamespace(H={i: H[i] for i in range(3)})
    assert proj.degenerate_frames() == []


def test_separate_groups_rendered_as_blocks(tmp_path):
    """Two strips with no overlap (a gap in the coverage): the second one is
    rendered on its own instead of being dropped."""
    import csv
    import tifffile
    src = tmp_path / "src"
    src.mkdir()
    for prefix, seed in (("A", 1), ("B", 9)):
        sub = tmp_path / prefix
        sub.mkdir()
        _survey(sub, n_cols=3, n_rows=1, seed=seed)
        for f in sub.glob("*.tif"):
            f.rename(src / f"{prefix}_{f.name}")
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2, min_block=3)
    proj, rep = run(src, tmp_path / "out", s)
    assert rep["aligned_images"] == 3
    rows = list(csv.DictReader(open(tmp_path / "out" / "mosaic_blocks.csv")))
    assert [int(r["images"]) for r in rows] == [3, 3]
    assert rows[1]["output"] == "mosaic_block02.tif"
    assert {rows[0]["first"][0], rows[1]["first"][0]} == {"A", "B"}
    out = tifffile.imread(tmp_path / "out" / "mosaic_block02.tif")
    assert (out[..., 0] > 0).mean() > 0.3
    assert proj.result.components[0] == sorted(proj.result.components[0])   # main result restored


def test_floating_tape_is_kept(tmp_path):
    """A long thin object that moves between views (transect tape above the
    bottom) is not removed as a moving object."""
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=8, n_rows=1, step=200, tape_y=560, seed=5)
    s = Settings(methods="sift", accuracy="high", resolution=0.5, distortion=False,
                 n_features=3000, jobs=2, flatfield="none")
    proj, rep = run(src, tmp_path / "out", s)
    assert rep["aligned_images"] == 8
    img = rawio.read_image(tmp_path / "out" / "mosaic.tif").astype(np.float32)
    covered = (img > 0).all(2)
    hi = np.percentile(img[covered], 99.9, axis=0)
    tape = (img > 0.8 * hi).all(2)          # only the tape is bright in all three channels
    cols = np.where(covered.any(0))[0]
    inner = tape[:, cols[0] + 40:cols[-1] - 40]
    # the tape crosses the whole mosaic: tape pixels in (nearly) every 30-px stretch
    # (the dashes are 21 px with 9 px gaps at this resolution)
    runs = inner.any(0)[: inner.shape[1] // 30 * 30].reshape(-1, 30).any(1)
    assert runs.mean() > 0.9, runs.astype(int)


def test_sharpness_falloff_inside_frames_is_not_drift(tmp_path):
    """Every frame sharp on one side and soft on the other (camera tilted forward,
    far edge seen through more water): all frames are equally good, the quality
    must not drift along the strip."""
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=8, n_rows=1, step=200, seed=3)
    for f in sorted(src.glob("*.tif")):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        soft = cv2.GaussianBlur(img, (0, 0), 2.5)
        t = np.linspace(0, 1, img.shape[1], dtype=np.float32)[None, :, None]
        cv2.imwrite(str(f), (img * (1 - t) + soft * t).astype(np.uint16))
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2)
    proj, rep = run(src, tmp_path / "out", s)
    q = np.array([f.quality for f in proj.frames])
    assert q.max() / q.min() < 1.4, q


@pytest.mark.parametrize("sequence", [False, True])
def test_sequence_mode_bridges_featureless_photo(tmp_path, sequence):
    """A photo without usable texture in the middle of a series cuts the strip in
    two. In sequence mode the gap is bridged with the typical step between
    consecutive photos and the series stays one mosaic in the right order."""
    src = tmp_path / "src"
    src.mkdir()
    truth = _survey(src, n_cols=7, n_rows=1, step=450, seed=2)
    blank = src / "IMG_003.tif"
    img = cv2.imread(str(blank), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(blank), np.full_like(img, int(img.mean())))
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2, min_block=0, sequence=sequence)
    proj, rep = run(src, tmp_path / "out", s)
    if not sequence:
        assert rep["aligned_images"] < 7
        return
    assert rep["aligned_images"] == 7
    assert len(rep["sequence_links"]) == 2
    names = [f.path.name for f in proj.frames]
    est = np.array([cv2.perspectiveTransform(np.array([[[450.0, 300.0]]]), proj.result.H[k])[0, 0]
                    for k in range(7)])
    tru = np.array([cv2.perspectiveTransform(np.array([[[450.0, 300.0]]]), truth[n])[0, 0] for n in names])
    from test_camera import _similarity_fit
    err = np.hypot(*(_similarity_fit(est, tru) - tru).T)
    assert err.max() < 0.15 * 900, err
    assert proj.frames[3].quality <= 0.3


def test_lens_from_earlier_run(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=4, n_rows=1, step=450, seed=6)
    s = Settings(methods="sift", accuracy="high", resolution=0.2, distortion=False, n_features=3000,
                 jobs=2, min_block=0)
    run(src, tmp_path / "a", s, render=False)
    rep_a = json.loads((tmp_path / "a" / "mosaic_report.json").read_text())
    s2 = Settings(methods="sift", accuracy="high", resolution=0.2, distortion=False, n_features=3000,
                  jobs=2, min_block=0, lens_from=str(tmp_path / "a"))
    proj, rep = run(src, tmp_path / "b", s2, render=False)
    assert abs(rep["camera"]["focal_px"] / rep_a["camera"]["focal_px"] - 1) < 0.03
    assert "lens from mosaic_report.json" in (tmp_path / "b" / "mosaic2d.log").read_text()


def test_failed_adjustment_keeps_matches_and_resumes(tmp_path, monkeypatch):
    """A crash in the adjustment must not throw away hours of matching."""
    from mosaic2d import cli
    from mosaic2d.pipeline import Project
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)
    s = Settings(methods="sift", accuracy="high", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    orig = Project.adjust

    def boom(self):
        raise np.linalg.LinAlgError("Singular matrix")
    monkeypatch.setattr(Project, "adjust", boom)
    with pytest.raises(np.linalg.LinAlgError):
        run(src, tmp_path / "out", s)
    pk = tmp_path / "out" / "mosaic_project.pkl"
    assert pk.is_file()
    monkeypatch.setattr(Project, "adjust", orig)
    assert cli.main(["render", str(pk), "-o", str(tmp_path / "out2"), "--resolution", "0.2"]) in (0, None)
    assert (tmp_path / "out2" / "mosaic.tif").is_file()


def test_camera_failure_falls_back(tmp_path, monkeypatch):
    from mosaic2d import camera as CAMM
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)

    def boom(*a, **k):
        raise np.linalg.LinAlgError("Singular matrix")
    monkeypatch.setattr(CAMM, "run_camera_bundle", boom)
    s = Settings(methods="sift", accuracy="high", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    proj, rep = run(src, tmp_path / "out", s, render=False)
    assert rep["aligned_images"] == 3


def test_readjust_with_sequence_joins_blocks(tmp_path):
    """A saved project split into groups is joined by `render --readjust --sequence`
    without matching again."""
    from mosaic2d import cli
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=7, n_rows=1, step=450, seed=2)
    blank = src / "IMG_003.tif"
    img = cv2.imread(str(blank), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(blank), np.full_like(img, int(img.mean())))
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2, min_block=0)
    proj, rep = run(src, tmp_path / "out", s, render=False)
    assert rep["aligned_images"] < 7
    pk = tmp_path / "out" / "mosaic_project.pkl"
    assert cli.main(["render", str(pk), "--readjust", "--sequence", "-o", str(tmp_path / "out2"),
                     "--resolution", "0.25", "--min-block", "0"]) in (0, None)
    rep2 = json.loads((tmp_path / "out2" / "mosaic_report.json").read_text())
    assert rep2["aligned_images"] == 7 and len(rep2["sequence_links"]) == 2
