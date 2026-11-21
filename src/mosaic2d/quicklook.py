"""
Natural-colour JPEG of a finished mosaic, multispectral included.

    mosaic2d preview mosaic.tif [--max-side 4000] [--bands 630,550,450]

Reads only the overview level that matches the requested size (mosaic2d,
Metashape and GDAL write these inside the TIFF; files without one are read once
in tiles), picks the bands nearest the screen's red, green and blue (skipping
bands that are black or clipped) and stretches each channel to its own range.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .bands import natural_colour_bands
from .render import visualise
from .sources import wavelength_from_name


def _band_index(spec, names_wl):
    """'630' / '630nm' / '9' (1-based band number) -> band index."""
    s = str(spec).lower().replace("nm", "").strip()
    v = float(s)
    wls = [wl for _, wl in names_wl]
    if v >= 300 and any(wls):
        return int(np.argmin([abs((w or 1e9) - v) for w in wls]))
    k = int(v) - 1
    if not 0 <= k < len(names_wl):
        raise ValueError(f"no band {spec}")
    return k


def make_preview(path, out=None, max_side=4000, bands=None, log=print):
    from .viewer import TiffMosaic
    path = Path(path)
    m = TiffMosaic(path)
    try:
        names_wl = [(n, wl if wl else wavelength_from_name(n)) for n, wl in m.band_names]
        # smallest overview that is still at least max_side
        k = 0
        for i, lv in enumerate(m.levels):
            if max(lv["W"], lv["H"]) >= max_side:
                k = i
        lv = m.levels[k]
        if max(lv["W"], lv["H"]) > 4 * max_side:          # no overview: one pass over the tiles
            log(f"  {path.name} has no overviews; reading it once (this can take a few minutes)")
            img = m.build_overview(max_side=max_side,
                                   progress=lambda f: None)
        else:
            img = m.read(k, 0, 0, lv["W"], lv["H"], list(range(m.S)))
        f = max_side / max(img.shape[:2])
        if f < 1:                     # per band: OpenCV resizes at most 4 channels at once
            img = np.dstack([cv2.resize(img[..., b].astype(np.float32), None, fx=f, fy=f,
                                        interpolation=cv2.INTER_AREA) for b in range(img.shape[2])])
        img = img.astype(np.float32)
        if m.S < 3:
            rgb = (0, 0, 0)
            skipped = []
        elif bands:
            rgb = tuple(_band_index(b, names_wl) for b in bands)
            skipped = []
        elif m.S in (3, 4) and not any(wl for _, wl in names_wl):
            rgb, skipped = (0, 1, 2), []
        else:
            rgb, skipped = natural_colour_bands([wl for _, wl in names_wl], img)
        out = Path(out) if out else path.with_name(path.stem + "_preview_rgb.jpg")
        cv2.imwrite(str(out), visualise(img, rgb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        msg = (f"  {out.name}: red {names_wl[rgb[0]][0]}, green {names_wl[rgb[1]][0]}, "
               f"blue {names_wl[rgb[2]][0]}")
        if skipped:
            msg += f" (not used, black or clipped: {', '.join(names_wl[b][0] for b in skipped)})"
        log(msg)
        return out, rgb
    finally:
        m.close()
