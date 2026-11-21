"""
Command line interface.

    mosaic2d run FOLDER [-o OUT] [options]        build a mosaic
    mosaic2d benchmark FOLDER [-o OUT] [options]  compare feature methods
    mosaic2d methods                              list methods and availability
    mosaic2d gui                                  window for choosing folders and options
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

from . import __version__
from . import features as FT
from .pipeline import ACCURACY, Settings, benchmark, run


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("inputs", nargs="+", help="image folder(s) or files (RAW, JPG, TIFF)")
    p.add_argument("-o", "--out", help="output folder (default: <first input>/mosaic2d)")
    p.add_argument("--profile", default="auto",
                   help="camera profile: auto, rgb, agrowing6, multiband or a JSON file")
    p.add_argument("--methods", default="auto",
                   help="feature methods in cascade order, e.g. sift,akaze,orb,fast (default: all available)")
    p.add_argument("--quick", action="store_true",
                   help="fewer alternatives: SIFT only (plus phase correlation for neighbours), "
                        "fewer extra pairs; much faster matching, usually the same result")
    p.add_argument("--strategy", default="cascade", choices=["cascade", "union", "single"],
                   help="cascade: per pair, next method only if the previous failed; "
                        "union: pool matches of all methods; single: first method only")
    p.add_argument("--model", default="camera",
                   choices=["camera", "homography", "affine", "similarity", "rigid", "translation"],
                   help="camera: every photo is a camera over one plane with a shared lens "
                        "(position, height, heading, tilt; like Metashape's camera model, no DEM); "
                        "homography: free 2D transform per photo (can drift along long chains)")
    p.add_argument("--focal-px", type=float, default=None,
                   help="focal length in px of the photos (default: EXIF / camera profile)")
    p.add_argument("--height", type=float, default=None, dest="height_m",
                   help="typical camera-to-ground distance in metres: output pixel size in metres")
    p.add_argument("--height-sigma", type=float, default=0.1,
                   help="camera model: how much the height may vary (0.1 = about +-10 %%)")
    p.add_argument("--accuracy", default="medium", choices=list(ACCURACY),
                   help="working resolution for matching (like Metashape's alignment accuracy)")
    p.add_argument("--features", type=int, default=8000, dest="n_features", help="key point limit per image")
    p.add_argument("--preselection", default="auto",
                   choices=["auto", "sequential", "generic", "exhaustive"])
    p.add_argument("--window", type=int, default=3, help="sequential neighbours matched per image")
    p.add_argument("--lens-from", default=None, dest="lens_from",
                   help="report.json, project.pkl or output folder of an earlier run with the same "
                        "camera and housing: reuse its focal length and distortion instead of "
                        "estimating them (useful when the scene is not flat, e.g. a wall or ledge)")
    p.add_argument("--sequence", action="store_true",
                   help="photos taken as a series (file names in shooting order): neighbours are "
                        "matched with relaxed thresholds and every method, and gaps between "
                        "consecutive photos are bridged with the typical step")
    p.add_argument("--no-distortion", dest="distortion", action="store_false",
                   help="do not self-calibrate radial lens distortion")
    p.add_argument("--ref-band", default=None, help='reference band for multispectral, e.g. "560 nm"')
    p.add_argument("--demosaic", default=None, choices=["superpixel", "bilinear"])
    p.add_argument("--no-phasecorr", dest="phasecorr", action="store_false")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--jobs", type=int, default=0, help="worker threads (default: up to 4)")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--work-dir", default=None, help="folder for the temporary canvas and the cache")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="do not reuse decoded frames from earlier runs (<out>/_cache)")


def build_parser():
    ap = argparse.ArgumentParser(prog="mosaic2d", description=f"mosaic2d {__version__}: 2D mosaics "
                                 "from overlapping RGB or multispectral photographs")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build a mosaic")
    _add_common(r)
    r.add_argument("--blend", default="mosaic", choices=["mosaic", "average", "disabled"])
    r.add_argument("--seams", default="graphcut", choices=["graphcut", "voronoi"],
                   help="graphcut: move seams away from moving objects and parallax; voronoi: closest centre")
    r.add_argument("--no-deghost", dest="deghost", action="store_false",
                   help="keep moving objects (rope, fish) instead of taking those areas from consistent frames")
    r.add_argument("--no-feather", dest="feather", action="store_false",
                   help="hard seamlines (no narrow transition) in mosaic blending")
    r.add_argument("--flatfield", default="auto", choices=["auto", "none"],
                   help="estimate and remove vignetting / fixed illumination pattern from the series")
    r.add_argument("--resolution", type=float, default=1.0,
                   help="output pixel size relative to the native pixels of the median image "
                        "(0.5 = half resolution)")
    r.add_argument("--orient", default="pca", choices=["pca", "anchor"])
    r.add_argument("--name", default="mosaic")
    r.add_argument("--gsd", type=float, default=None,
                   help="metres per pixel at --resolution 1 (e.g. from a scale bar); writes GeoTIFF "
                        "scale tags and a .tfw so GIS measurements are in metres")
    r.add_argument("--no-render", action="store_true", help="align only; write a preview from thumbnails")
    r.add_argument("--keep-canvas", action="store_true")
    r.add_argument("--tiles", type=int, default=0, dest="tile_size",
                   help="split the output into tiles of this size (px) plus a .vrt that joins them")
    r.add_argument("--min-block", type=int, default=5,
                   help="also render every other group of at least this many connected images "
                        "(gaps in the coverage) as <name>_blockNN; 0 = largest group only")
    rr = sub.add_parser("render", help="render again from a saved project (<name>_project.pkl)")
    rr.add_argument("project")
    rr.add_argument("-o", "--out", default=None)
    rr.add_argument("--blend", default=None, choices=["mosaic", "average", "disabled"])
    rr.add_argument("--resolution", type=float, default=None)
    rr.add_argument("--orient", default=None, choices=["pca", "anchor"])
    rr.add_argument("--tiles", type=int, default=None, dest="tile_size")
    rr.add_argument("--seams", default=None, choices=["graphcut", "voronoi"])
    rr.add_argument("--gsd", type=float, default=None)
    rr.add_argument("--no-deghost", dest="deghost", action="store_const", const=False, default=None)
    rr.add_argument("--no-feather", dest="feather", action="store_const", const=False, default=None)
    rr.add_argument("--min-block", type=int, default=None)
    rr.add_argument("--readjust", action="store_true",
                    help="run the adjustment again from the matches stored in the project "
                         "(after an update of mosaic2d), then render")
    rr.add_argument("--model", default=None, choices=["camera", "homography", "affine", "similarity", "rigid", "translation"],
                    help="with --readjust: transform model (default camera)")
    rr.add_argument("--sequence", action="store_true",
                    help="with --readjust: bridge gaps between consecutive photos (shooting order)")
    rr.add_argument("--lens-from", default=None, dest="lens_from",
                    help="with --readjust: reuse the lens calibration of an earlier run")
    rr.add_argument("--work-dir", default=None)
    rr.add_argument("--name", default="mosaic")
    b = sub.add_parser("benchmark", help="compare feature methods on the same images")
    _add_common(b)
    sub.add_parser("methods", help="list feature methods")
    g = sub.add_parser("gui", help="open a window for choosing folders and options")
    g.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    pv = sub.add_parser("preview", help="natural-colour JPEG of finished mosaics (multispectral too)")
    pv.add_argument("files", nargs="+", help="mosaic TIFF(s)")
    pv.add_argument("--max-side", type=int, default=4000, help="longest side of the JPEG in px")
    pv.add_argument("--bands", default=None,
                    help="bands for red,green,blue: wavelengths (630,550,450) or band numbers (9,6,3); "
                         "default: nearest to the screen's red/green/blue, skipping black or clipped bands")
    pv.add_argument("-o", "--out", default=None, help="output JPEG (one input file only)")
    v = sub.add_parser("view", help="view a mosaic (any TIFF, multispectral included): pick bands, zoom, pan")
    v.add_argument("file", nargs="?", default=None)
    v.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    if argv is None and len(sys.argv) == 1 and getattr(sys, "frozen", False):
        argv = ["gui"]                       # double-clicked executable: open the window
        if os.environ.get("MOSAIC2D_GUI_SELFTEST"):
            argv.append("--selftest")        # CI: same start-up path, window closed at once
    args = build_parser().parse_args(argv)
    if args.cmd == "gui":
        from .gui import main as gui_main
        return gui_main(selftest=args.selftest)
    if args.cmd == "preview":
        from .quicklook import make_preview
        bands = args.bands.split(",") if args.bands else None
        if bands and len(bands) != 3:
            print("--bands needs three values (red,green,blue)")
            return 2
        for f in args.files:
            make_preview(f, args.out if len(args.files) == 1 else None, args.max_side, bands)
        return 0
    if args.cmd == "view":
        from .viewer import main as view_main
        return view_main(args.file, selftest=args.selftest)
    if args.cmd == "methods":
        reg = FT.build_registry()
        for m in reg.values():
            print(f"{m.name:12s} {'available ' if m.available else 'MISSING   '} {m.family:6s}  {m.note}")
        return 0
    if args.cmd == "render":
        from .pipeline import Project
        proj = Project.load(args.project, args.out, blend=args.blend, resolution=args.resolution,
                            orient=args.orient, tile_size=args.tile_size, work_dir=args.work_dir,
                            seams=args.seams, deghost=args.deghost, gsd=args.gsd,
                            feather=args.feather, min_block=args.min_block)
        if proj.result is None:              # saved after a failed adjustment: matches only
            args.readjust = True
        if args.readjust:
            proj.s.model = args.model or "camera"
            if args.sequence:
                proj.s.sequence = True
            if args.lens_from:
                proj.s.lens_from = args.lens_from
            proj.adjust()
            proj.refresh_quality()
            proj.save(Path(args.project).with_name(f"{args.name}_project.pkl"))
        proj.render(args.name)
        proj.report(args.name)
        if proj.s.min_block > 0:
            proj.render_blocks(args.name)
        return 0
    known = {f.name for f in fields(Settings)}
    if getattr(args, "quick", False):
        if args.methods == "auto":
            args.methods = "sift"
        args.quick = True
    s = Settings(**{k: v for k, v in vars(args).items() if k in known and v is not None})
    out = Path(args.out) if args.out else Path(args.inputs[0]) / "mosaic2d"
    if args.cmd == "benchmark":
        benchmark(args.inputs, out, s, methods=args.methods)
        return 0
    proj, rep = run(args.inputs, out, s, name=args.name, render=not args.no_render)
    return 0 if rep["aligned_images"] == rep["images"] else 1


if __name__ == "__main__":
    sys.exit(main())
