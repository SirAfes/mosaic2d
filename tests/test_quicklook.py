"""Natural-colour previews of multispectral mosaics."""

import numpy as np

from mosaic2d.bands import natural_colour_bands
from mosaic2d.quicklook import make_preview
from mosaic2d.sources import wavelength_from_name
from mosaic2d.tiffwriter import write_tiff

WL = [405, 430, 450, 490, 525, 550, 560, 570, 630, 650, 685, 710, 735, 850]


def test_nearest_usable_bands():
    rng = np.random.default_rng(0)
    img = (rng.random((50, 60, 14)) * 200 + 20).astype(np.uint8)
    assert natural_colour_bands(WL)[0] == (8, 5, 2)          # 630, 550, 450
    img[..., 9] = 0                                           # 650 nm black
    img[..., 2] = 255                                         # 450 nm clipped
    img[..., 5] = 255                                         # 550 nm clipped
    idx, skipped = natural_colour_bands(WL, img)
    assert idx == (8, 6, 3) and set(skipped) == {2, 5, 9}     # 630, 560, 490


def test_wavelength_from_name():
    assert [wavelength_from_name(n) for n in ("560 nm", "B7_560nm", "Band 6 (560)", "band 3")] == \
        [560.0, 560.0, 560.0, None]


def test_preview_from_tiff(tmp_path):
    rng = np.random.default_rng(1)
    a = (rng.random((900, 1300, 14)) * 200 + 20).astype(np.uint8)
    a[..., 9] = 0
    f = tmp_path / "ms.tif"
    write_tiff(f, a, band_names=[f"{w} nm" for w in WL])      # names only, no wavelength tags
    out, rgb = make_preview(f, max_side=400, log=lambda *x: None)
    assert out.is_file() and rgb == (8, 5, 2)
    import cv2
    j = cv2.imread(str(out))
    assert max(j.shape[:2]) == 400 and j.std() > 5
