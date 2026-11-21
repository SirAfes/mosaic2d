"""
sources.py

Camera profiles and frame loading.

A *frame* is one exposure. It consists of one or more *lens images* (a normal
camera has one; a multi-lens snapshot camera such as the Agrowing Sextuple has
six side by side on one sensor). Each lens image has one or more sensor
channels, and each channel is one *band* of the mosaic.

Profiles
--------
rgb         one lens, channels R, G, B. RAW: bilinear demosaic (full size).
agrowing6   Agrowing Sextuple on a 61 MP Sony body: 3 x 2 lens cells of
            3168 px, 14 bands 405-850 nm. RAW: super-pixel demosaic (each
            2 x 2 Bayer quad -> one pixel, no interpolation), so a lens image
            is 1584 px before cropping its vignetted border.
multiband   pre-aligned multiband TIFFs (for example msalign output): one lens,
            every TIFF channel is a band.

Custom profiles can be given as JSON (see ``profile_from_json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from pathlib import Path

import cv2
import numpy as np

from . import rawio


@dataclass
class Band:
    name: str
    lens: int
    channel: int
    wavelength_nm: float | None = None


@dataclass
class CameraProfile:
    name: str
    grid: tuple = (1, 1)            # lens cells (cols, rows)
    cell_size: int | None = None    # native px of one cell (None = whole image)
    crop_frac: float = 1.0          # centred part of each cell that is kept
    demosaic: str = "bilinear"      # for RAW input
    bands: list = field(default_factory=list)
    focal_mm: float | None = None   # lens focal length (multi-lens cameras without EXIF focal)
    pixel_um: float | None = None   # sensor pixel pitch at cell_size

    @property
    def n_lenses(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def multilens(self) -> bool:
        return self.n_lenses > 1

    def lens_bands(self, lens: int) -> list[Band]:
        return [b for b in self.bands if b.lens == lens]

    def output_bands(self) -> list[Band]:
        def key(b):
            return (b.wavelength_nm is None, b.wavelength_nm or 0, b.lens, b.channel)
        return sorted(self.bands, key=key)


def _agrowing6() -> CameraProfile:
    # (lens, channel R/G/B = 0/1/2, wavelength). Verified against the sensor
    # physics and Agrowing's own software in msalign (TECHNICAL_NOTES.md).
    table = [
        (1, 0, 850),
        (2, 0, 630), (2, 1, 525),
        (3, 0, 710), (3, 1, 570), (3, 2, 405),
        (4, 0, 650), (4, 1, 550), (4, 2, 430),
        (5, 0, 735), (5, 2, 490),
        (6, 0, 685), (6, 1, 560), (6, 2, 450),
    ]
    bands = [Band(f"{w} nm", lens, ch, float(w)) for lens, ch, w in table]
    return CameraProfile("agrowing6", (3, 2), 3168, 0.80, "superpixel", bands,
                         focal_mm=21.8, pixel_um=3.76)


def _rgb() -> CameraProfile:
    return CameraProfile("rgb", (1, 1), None, 1.0, "bilinear",
                         [Band("red", 1, 0), Band("green", 1, 1), Band("blue", 1, 2)])


def profile_from_json(path) -> CameraProfile:
    """
    {"name": "my6", "grid": [3, 2], "cell_size": 3168, "crop_frac": 0.8,
     "demosaic": "superpixel",
     "bands": [{"name": "850 nm", "lens": 1, "channel": 0, "wavelength_nm": 850}, ...]}
    """
    d = json.loads(Path(path).read_text())
    bands = [Band(b["name"], int(b["lens"]), int(b["channel"]), b.get("wavelength_nm"))
             for b in d["bands"]]
    return CameraProfile(d.get("name", Path(path).stem), tuple(d.get("grid", (1, 1))),
                         d.get("cell_size"), float(d.get("crop_frac", 1.0)),
                         d.get("demosaic", "bilinear"), bands)


def wavelength_from_name(name):
    """Centre wavelength (nm) written in a band name ("560 nm", "B7_560nm",
    "Band 6 (560)"), or None."""
    if not name:
        return None
    m = (re.search(r"(\d{3,4}(?:\.\d+)?)\s*nm", name, re.I)
         or re.search(r"(\d{3,4}(?:\.\d+)?)\s*\)?\s*$", name.strip()))
    if m and 300 <= float(m.group(1)) <= 2500:
        return float(m.group(1))
    return None


def get_profile(name_or_path: str, sample_file=None) -> CameraProfile:
    if name_or_path in ("rgb", "single"):
        return _rgb()
    if name_or_path == "agrowing6":
        return _agrowing6()
    if name_or_path == "multiband":
        if sample_file is None:
            raise ValueError("multiband profile needs a sample file")
        arr = rawio.read_image(sample_file)
        c = arr.shape[2] if arr.ndim == 3 else 1
        info = rawio.tiff_band_info(sample_file) if Path(sample_file).suffix.lower() in (".tif", ".tiff") else {}
        bands = []
        for k in range(c):
            name, wl = info.get(k, (None, None))
            if wl is None and name:
                wl = wavelength_from_name(name)
            if not name:
                name = f"{wl:g} nm" if wl else f"band {k + 1}"
            bands.append(Band(name, 1, k, wl))
        return CameraProfile("multiband", (1, 1), None, 1.0, "bilinear", bands)
    if Path(name_or_path).is_file():
        return profile_from_json(name_or_path)
    raise ValueError(f"Unknown camera profile '{name_or_path}' (rgb, agrowing6, multiband or a JSON file)")


def detect_profile(sample_file) -> str:
    """Guess the profile: Agrowing if the six cells are separated by dark gutters."""
    prev = None
    if rawio.is_raw(sample_file):
        prev = rawio.embedded_preview(sample_file)
    if prev is None:
        try:
            img = rawio.read_image(sample_file)
        except Exception:
            return "rgb"
        if img.ndim == 3 and img.shape[2] > 4:
            return "multiband"
        prev = img[..., :3].astype(np.float32)
    g = prev.astype(np.float32).mean(2) if prev.ndim == 3 else prev.astype(np.float32)
    col = g.mean(0)
    row = g.mean(1)
    # Agrowing: dark vertical gutters at 1/3 and 2/3 width, one horizontal at 1/2.
    def dip(profile, frac, limit):
        n = len(profile)
        c = int(frac * n)
        local = profile[max(0, c - n // 12): c + n // 12].min()
        return local < limit * np.median(profile)
    if dip(col, 1 / 3, 0.8) and dip(col, 2 / 3, 0.8) and dip(row, 1 / 2, 0.6):
        return "agrowing6"
    return "rgb"


# --------------------------------------------------------------------------

@dataclass
class FrameData:
    """Decoded frame: lens id -> H x W x C array (uint16 / uint8)."""
    lenses: dict
    exif: dict
    native_shape: tuple


def _split_cells(img, prof: CameraProfile, scale: float):
    cols, rows = prof.grid
    h, w = img.shape[:2]
    out = {}
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            lens = r * cols + c + 1
            x0, y0 = c * cw, r * ch
            side_w = cw * prof.crop_frac
            side_h = ch * prof.crop_frac
            if prof.cell_size:
                side_w = side_h = min(prof.cell_size * scale * prof.crop_frac, side_w, side_h)
            xa = int(round(x0 + (cw - side_w) / 2))
            ya = int(round(y0 + (ch - side_h) / 2))
            out[lens] = np.ascontiguousarray(img[ya:ya + int(side_h), xa:xa + int(side_w)])
    return out


def load_frame(path, prof: CameraProfile, demosaic: str | None = None) -> FrameData:
    path = Path(path)
    mode = demosaic or prof.demosaic
    if rawio.is_raw(path):
        raw = rawio.read_raw(path)
        img = rawio.demosaic(raw, mode)
        scale = 0.5 if mode == "superpixel" else 1.0
        native = raw.cfa.shape
    else:
        img = rawio.read_image(path)
        scale = 1.0
        native = img.shape[:2]
    if prof.multilens:
        lenses = _split_cells(img, prof, scale)
    else:
        lenses = {1: img}
    return FrameData(lenses, rawio.read_exif(path) if rawio.is_raw(path) or path.suffix.lower()
                     in (".tif", ".tiff") else {}, native)


def band_image(fd: FrameData, band: Band) -> np.ndarray:
    a = fd.lenses[band.lens]
    return a[..., band.channel] if a.ndim == 3 else a


def to_gray_float(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    return a.mean(2) if a.ndim == 3 else a


def normalise_for_features(g: np.ndarray, clahe=True) -> np.ndarray:
    """
    Float image -> uint8 for feature detection: divide by a heavy blur (removes
    vignetting and uneven artificial light), robust stretch, CLAHE.
    """
    g = g.astype(np.float32)
    s = max(g.shape) / 25.0
    bg = cv2.GaussianBlur(g, (0, 0), s)
    g = g / (bg + 1e-3 * max(float(bg.mean()), 1e-6))
    step = max(1, g.size // 400000)
    lo, hi = np.percentile(g.ravel()[::step], [0.5, 99.5])
    g = np.clip((g - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    if clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def list_images(folder_or_files) -> list[Path]:
    """Images of a folder (RAW preferred over a JPG twin) or an explicit list."""
    items = folder_or_files if isinstance(folder_or_files, (list, tuple)) else [folder_or_files]
    files = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir()
                                if q.suffix.lower() in rawio.SUPPORTED_EXTENSIONS and q.is_file()))
        elif p.is_file():
            files.append(p)
    by_stem = {}
    for f in files:
        prev = by_stem.get(f.stem)
        if prev is None or (rawio.is_raw(f) and not rawio.is_raw(prev)):
            by_stem[f.stem] = f
    return sorted(by_stem.values(), key=lambda q: q.name)
