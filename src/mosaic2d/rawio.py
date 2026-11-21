"""
rawio.py

Image reading for mosaic2d.

Camera RAW
----------
Sony ARW files are decoded without any third-party RAW library when they are
uncompressed (SonyRawFileType 0) or in Sony's lossy "compressed RAW" format
(SonyRawFileType 2, "ARW 2.x", 1 byte per pixel). The cRAW bitstream is
decoded with NumPy following the public description in dcraw/LibRaw
(``sony_arw2_load_raw``): 32 pixels of one row are stored in two 16-byte
blocks; each block holds 16 same-colour pixels as an 11-bit max, an 11-bit
min, the positions of max and min and fourteen 7-bit deltas, followed by
Sony's tone curve (tag 0x7010) which expands the 11-bit code to 14 bits.

Every other RAW format (and Sony lossless compressed RAW) is decoded with
rawpy / LibRaw if it is installed.

Decoding stays close to the sensor: no white balance, no colour matrix, linear
response, black level subtracted. Two demosaic modes:

- ``superpixel`` -- every 2 x 2 Bayer quad becomes one RGB pixel (half size,
  no interpolation, each channel a pure measurement). Default for
  multispectral cameras.
- ``bilinear`` -- full resolution, each channel interpolated only from
  neighbours of the same colour (OpenCV's bilinear Bayer conversion).

Arrays are returned in R, G, B channel order (not OpenCV's BGR).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RAW_EXTENSIONS = (".arw", ".srf", ".sr2", ".dng", ".nef", ".cr2", ".cr3", ".orf",
                  ".rw2", ".raf", ".pef", ".srw", ".raw")
STANDARD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS + STANDARD_EXTENSIONS

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
               11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 5: "II", 6: "b", 8: "h", 9: "i", 10: "ii",
             11: "f", 12: "d", 13: "I", 16: "Q", 17: "q", 18: "Q"}


def is_raw(path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


# --------------------------------------------------------------------------
# Minimal TIFF / EXIF reader (enough for ARW structure and basic EXIF)
# --------------------------------------------------------------------------

def _read_ifd(fh, offset: int, endian: str) -> tuple[dict, int]:
    fh.seek(offset)
    (n,) = struct.unpack(endian + "H", fh.read(2))
    entries = fh.read(12 * n)
    (next_ifd,) = struct.unpack(endian + "I", fh.read(4))
    tags = {}
    for i in range(n):
        tag, typ, count, raw = struct.unpack(endian + "HHI4s", entries[12 * i:12 * i + 12])
        size = _TYPE_SIZES.get(typ, 1) * count
        if size > 4:
            (ptr,) = struct.unpack(endian + "I", raw)
            pos = fh.tell()
            fh.seek(ptr)
            data = fh.read(size)
            fh.seek(pos)
        else:
            data = raw[:size]
        if typ == 2:
            value = data.split(b"\0", 1)[0].decode("latin-1", "replace").strip()
        elif typ in (7,):
            value = data
        elif typ in _TYPE_FMT:
            fmt = _TYPE_FMT[typ]
            vals = struct.unpack(endian + fmt * count, data[:struct.calcsize(fmt) * count])
            if typ in (5, 10):
                vals = tuple(vals[k] / vals[k + 1] if vals[k + 1] else 0.0
                             for k in range(0, len(vals), 2))
            value = vals[0] if count == 1 else tuple(vals)
        else:
            value = data
        tags[tag] = value
    return tags, next_ifd


@dataclass
class TiffInfo:
    endian: str
    ifds: list = field(default_factory=list)
    subifds: list = field(default_factory=list)
    exif: dict = field(default_factory=dict)


def read_tiff_structure(path) -> TiffInfo | None:
    """IFD chain, SubIFDs and EXIF of a TIFF-based file; None if not TIFF."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if head[:2] == b"II":
            endian = "<"
        elif head[:2] == b"MM":
            endian = ">"
        else:
            return None
        magic, off = struct.unpack(endian + "HI", head[2:8])
        if magic != 42:
            return None
        info = TiffInfo(endian)
        seen = set()
        while off and off not in seen and len(info.ifds) < 16:
            seen.add(off)
            tags, off = _read_ifd(fh, off, endian)
            info.ifds.append(tags)
        for tags in info.ifds[:1]:
            sub = tags.get(330)
            if sub is not None:
                for so in (sub if isinstance(sub, tuple) else (sub,)):
                    info.subifds.append(_read_ifd(fh, so, endian)[0])
            ex = tags.get(34665)
            if ex:
                info.exif = _read_ifd(fh, ex, endian)[0]
        return info


def tiff_band_info(path) -> dict:
    """
    Band names and centre wavelengths (nm) stored in a multiband TIFF:
    GDAL_METADATA items (DESCRIPTION / wavelength / CENTRAL_WAVELENGTH_UM, as
    written by GDAL, msalign and mosaic2d) or an msalign JSON ImageDescription.
    Returns {band index (0-based): (name or None, wavelength nm or None)}.
    """
    import html
    import json
    import re
    try:
        info = read_tiff_structure(path)
    except (OSError, struct.error):
        return {}
    if not info or not info.ifds:
        return {}
    tags = info.ifds[0]
    names, wls = {}, {}
    meta = tags.get(42112)
    if isinstance(meta, str) and "<GDALMetadata" in meta:
        unit = 1.0
        for attrs, val in re.findall(r"<Item\s+([^>]*)>([^<]*)</Item>", meta):
            a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            v = html.unescape(val).strip()
            if a.get("name", "").lower() == "wavelength_units" and "sample" not in a:
                unit = 1000.0 if v.lower() in ("um", "micrometers", "micrometer", "µm") else 1.0
        for attrs, val in re.findall(r"<Item\s+([^>]*)>([^<]*)</Item>", meta):
            a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            if "sample" not in a:
                continue
            k, key, v = int(a["sample"]), a.get("name", ""), html.unescape(val).strip()
            try:
                if key == "DESCRIPTION" and v:
                    names[k] = v
                elif key == "CENTRAL_WAVELENGTH_UM":
                    wls[k] = float(v) * 1000.0
                elif key.upper() in ("WAVELENGTH_NM", "CENTRAL_WAVELENGTH_NM"):
                    wls[k] = float(v)
                elif key.upper() == "BAND_NAME" and v and k not in names:
                    names[k] = v
                elif key.lower() == "wavelength" and k not in wls:
                    wls[k] = float(v) * unit
            except ValueError:
                pass
    desc = tags.get(270)
    if not names and isinstance(desc, str) and desc.lstrip().startswith("{"):
        try:
            for k, b in enumerate(json.loads(desc).get("bands", [])):
                names[k] = b.get("name")
                if b.get("center_wavelength_nm") is not None:
                    wls[k] = float(b["center_wavelength_nm"])
        except (ValueError, AttributeError, TypeError):
            pass
    return {k: (names.get(k), wls.get(k)) for k in sorted(set(names) | set(wls))}


def tiff_camera_info(path) -> dict:
    """Camera block of an msalign JSON ImageDescription (focal length, pixel size), or {}."""
    import json
    try:
        info = read_tiff_structure(path)
        desc = info.ifds[0].get(270) if info and info.ifds else None
        if isinstance(desc, str) and desc.lstrip().startswith("{"):
            cam = json.loads(desc).get("camera") or {}
            return cam if isinstance(cam, dict) else {}
    except (OSError, ValueError, struct.error, AttributeError):
        pass
    return {}


EXIF_NAMES = {
    271: "make", 272: "model", 306: "datetime", 33434: "exposure_time", 33437: "f_number",
    34855: "iso", 36867: "datetime_original", 37386: "focal_length",
    42036: "lens_model", 41989: "focal_length_35mm",
}


def read_exif(path) -> dict:
    """Small set of EXIF fields (make, model, time, exposure, lens) or {}."""
    try:
        info = read_tiff_structure(path)
    except Exception:
        return {}
    if info is None or not info.ifds:
        return {}
    out = {}
    for src in (info.ifds[0], info.exif):
        for tag, name in EXIF_NAMES.items():
            if tag in src and name not in out:
                v = src[tag]
                if isinstance(v, (bytes, bytearray)):
                    continue
                out[name] = v
    return out


def embedded_preview(path) -> np.ndarray | None:
    """The camera JPEG preview stored in IFD0 of an ARW/DNG (BGR) or None."""
    info = read_tiff_structure(path)
    if info is None:
        return None
    for tags in info.ifds:
        off, n = tags.get(513), tags.get(514)
        if off and n:
            with open(path, "rb") as fh:
                fh.seek(off if isinstance(off, int) else off[0])
                buf = fh.read(n if isinstance(n, int) else n[0])
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


# --------------------------------------------------------------------------
# Sony ARW decoding
# --------------------------------------------------------------------------

@dataclass
class RawFrame:
    """Bayer mosaic after black-level subtraction plus its description."""
    cfa: np.ndarray            # uint16, H x W, cropped to the active area
    pattern: str               # e.g. "RGGB" (colour at (0,0),(0,1),(1,0),(1,1))
    white: int                 # white level after black subtraction
    black: int
    source: str                # "numpy-sony" or "rawpy"


def _sony_curve(tag_values) -> np.ndarray:
    curve = np.arange(0x10000, dtype=np.int64)
    pts = [0, 0, 0, 0, 0, 4095]
    for c in range(4):
        pts[c + 1] = (int(tag_values[c]) >> 2) & 0xFFF
    for i in range(5):
        for j in range(pts[i] + 1, pts[i + 1] + 1):
            curve[j] = curve[j - 1] + (1 << i)
    return curve


def decode_sony_craw_rows(data: np.ndarray, width: int, curve: np.ndarray) -> np.ndarray:
    """
    Decode rows of Sony compressed RAW.

    data  : uint8 array (rows, width) -- one byte per pixel.
    Returns uint16 array (rows, width) on the 14-bit scale.
    """
    rows = data.shape[0]
    nblk = width // 16
    blocks = np.ascontiguousarray(data[:, :nblk * 16]).reshape(rows, nblk, 16)
    lo = blocks[..., :8].copy().view("<u8")[..., 0]
    hi = blocks[..., 8:].copy().view("<u8")[..., 0]
    lo32 = (lo & np.uint64(0xFFFFFFFF)).astype(np.int64)
    vmax = lo32 & 0x7FF
    vmin = (lo32 >> 11) & 0x7FF
    imax = (lo32 >> 22) & 0xF
    imin = (lo32 >> 26) & 0xF
    diff = vmax - vmin
    sh = sum(((0x80 << t) <= diff).astype(np.int64) for t in range(4))

    deltas = np.empty(lo.shape + (14,), np.int64)
    for k in range(14):
        start = 30 + 7 * k
        if start + 7 <= 64:
            v = (lo >> np.uint64(start))
        elif start >= 64:
            v = (hi >> np.uint64(start - 64))
        else:
            v = (lo >> np.uint64(start)) | (hi << np.uint64(64 - start))
        deltas[..., k] = (v & np.uint64(0x7F)).astype(np.int64)

    pos = np.arange(16, dtype=np.int64)
    k_index = pos - (imax[..., None] < pos) - (imin[..., None] < pos)
    k_index = np.clip(k_index, 0, 13)
    vals = np.take_along_axis(deltas, k_index, axis=-1)
    vals = np.minimum((vals << sh[..., None]) + vmin[..., None], 0x7FF)
    vals = np.where(pos == imin[..., None], vmin[..., None], vals)
    vals = np.where(pos == imax[..., None], vmax[..., None], vals)
    vals = curve[vals << 1]

    # Two consecutive blocks cover 32 pixels: first even columns, then odd.
    out = np.zeros((rows, width), np.uint16)
    pairs = nblk // 2
    v = vals[:, :pairs * 2].reshape(rows, pairs, 2, 16)
    seg = np.empty((rows, pairs, 16, 2), np.int64)
    seg[..., 0] = v[:, :, 0, :]
    seg[..., 1] = v[:, :, 1, :]
    out[:, :pairs * 32] = np.minimum(seg.reshape(rows, pairs * 32), 0xFFFF)
    return out


def _read_sony_arw(path: Path) -> RawFrame | None:
    info = read_tiff_structure(path)
    if info is None:
        return None
    raw_ifd = None
    for tags in info.subifds + info.ifds:
        if tags.get(259) in (32767, 1) and tags.get(262) == 32803:
            raw_ifd = tags
            break
    if raw_ifd is None:
        return None
    width, height = raw_ifd[256], raw_ifd[257]
    rawtype = raw_ifd.get(0x7000, None)
    offset = raw_ifd[273] if isinstance(raw_ifd[273], int) else raw_ifd[273][0]
    nbytes = raw_ifd[279] if isinstance(raw_ifd[279], int) else sum(raw_ifd[279])
    black_tag = raw_ifd.get(0x7310) or raw_ifd.get(0x7300)
    black = int(np.median(black_tag)) if black_tag is not None else 512
    white = int(raw_ifd.get(50717, 16383))

    if rawtype == 2 or (rawtype is None and nbytes == width * height):
        curve = _sony_curve(raw_ifd.get(0x7010, (8000, 10400, 12900, 14100)))
        data = np.fromfile(path, np.uint8, count=width * height, offset=offset)
        data = data.reshape(height, width)
        cfa = np.empty((height, width), np.uint16)
        step = 512
        for r in range(0, height, step):
            cfa[r:r + step] = decode_sony_craw_rows(data[r:r + step], width, curve)
    elif rawtype == 0 or nbytes >= width * height * 2:
        cfa = np.fromfile(path, "<u2", count=width * height, offset=offset)
        cfa = cfa.reshape(height, width).copy()
    else:
        return None  # lossless compressed (LJPEG) etc. -> rawpy

    x0, y0 = raw_ifd.get(50719, (0, 0))
    cw, ch = raw_ifd.get(50720, (width, height))
    x0, y0, cw, ch = int(x0), int(y0), int(cw), int(ch)
    cfa = cfa[y0:y0 + ch, x0:x0 + cw]

    pat = raw_ifd.get(33422, b"\x00\x01\x01\x02")
    if isinstance(pat, (tuple, list)):
        pat = bytes(pat)
    names = "RGB"
    pattern = "".join(names[b] if b < 3 else "G" for b in pat[:4])
    if x0 % 2:
        pattern = pattern[1] + pattern[0] + pattern[3] + pattern[2]
    if y0 % 2:
        pattern = pattern[2:] + pattern[:2]

    cfa = np.subtract(cfa, black, dtype=np.int32)
    np.clip(cfa, 0, None, out=cfa)
    return RawFrame(cfa.astype(np.uint16), pattern, white - black, black, "numpy-sony")


def _read_rawpy(path: Path) -> RawFrame:
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"{path.name}: this RAW format needs the 'rawpy' package "
            "(pip install rawpy). Sony compressed/uncompressed ARW works without it."
        ) from exc
    with rawpy.imread(str(path)) as raw:
        vis = raw.raw_image_visible.astype(np.int32)
        colors = raw.raw_colors_visible
        black = np.array(raw.black_level_per_channel)
        vis = np.clip(vis - black[colors], 0, None).astype(np.uint16)
        desc = raw.color_desc.decode()
        pat = "".join(desc[c] for c in colors[:2, :2].ravel())
        pat = pat.replace("g", "G")
        white = int(raw.white_level - black.mean())
    return RawFrame(vis, pat, white, int(black.mean()), "rawpy")


def read_raw(path) -> RawFrame:
    path = Path(path)
    frame = None
    if path.suffix.lower() in (".arw", ".srf", ".sr2"):
        try:
            frame = _read_sony_arw(path)
        except Exception:
            frame = None
    if frame is None:
        frame = _read_rawpy(path)
    return frame


_BAYER_TO_RGB = {
    # OpenCV names Bayer patterns by the 2x2 block starting at (1,1).
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}


def demosaic(raw: RawFrame, mode: str = "superpixel") -> np.ndarray:
    """RGB uint16 array from a RawFrame."""
    cfa = raw.cfa
    h, w = cfa.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    cfa = cfa[:h2, :w2]
    if mode == "superpixel":
        q = {k: cfa[i::2, j::2].astype(np.uint32) for k, (i, j) in
             zip(range(4), [(0, 0), (0, 1), (1, 0), (1, 1)])}
        out = np.zeros((h2 // 2, w2 // 2, 3), np.uint32)
        count = np.zeros(3, np.uint32)
        for k, ch in enumerate(raw.pattern):
            c = "RGB".index(ch)
            out[..., c] += q[k]
            count[c] += 1
        out //= np.maximum(count, 1)
        return out.astype(np.uint16)
    if mode == "bilinear":
        code = _BAYER_TO_RGB.get(raw.pattern)
        if code is None:
            raise ValueError(f"Unsupported Bayer pattern {raw.pattern}")
        return cv2.cvtColor(np.ascontiguousarray(cfa), code)
    raise ValueError("demosaic mode must be 'superpixel' or 'bilinear'")


def read_image(path, demosaic_mode: str = "bilinear") -> np.ndarray:
    """
    Read any supported image as an H x W x C array (RGB order for colour
    images, all channels of a multiband TIFF in file order).
    """
    path = Path(path)
    if is_raw(path):
        return demosaic(read_raw(path), demosaic_mode)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(str(path))
            if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 32:
                arr = np.moveaxis(arr, 0, -1)  # planar -> interleaved
            return arr if arr.ndim == 3 else arr[..., None]
        except (ImportError, ValueError):
            pass  # e.g. LZW without imagecodecs -> OpenCV below
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if img.ndim == 2:
        return img[..., None]
    if img.shape[2] in (3, 4):                # OpenCV colour order is BGR(A); multiband stays as stored
        img = img[..., [2, 1, 0] + list(range(3, img.shape[2]))]
    return img
