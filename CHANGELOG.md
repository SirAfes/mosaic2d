# Changelog

## 0.3.0

- Large series (more than 600 photos): pairs between photos far apart in the series are
  found by feature voting (every thumbnail feature votes for the photo its nearest
  neighbour comes from; each photo's best partners are verified) instead of not being
  tried at all. Neighbouring lanes of a lawn-mower survey and return passes now join
  into one mosaic when they overlap sideways (`--preselection retrieval` to force it).
- Flat field: the data area is where any band has data (a band that is black everywhere
  had emptied it), and bands that are black, clipped or brighter in the corners than in
  the centre are left uncorrected (Aran Island set: "corner/centre 1.93", a bright oval
  painted into every photo).
- `--max-gap N` (field next to "Photo series" in the window): longest run of photos
  without any match that sequence mode still places by the shooting order (default 3).
  A large value places every photo, as before, but positions inside long runs are guesses.
- Natural-colour previews of multispectral mosaics: the bands nearest the screen's red,
  green and blue (611 / 549 / 464 nm) are used, skipping bands that are black or clipped
  in the mosaic (the log names them). Wavelengths are also read from band names such as
  "560 nm" when the file has no wavelength tags (the 8-bit Aran Island exports: the
  preview had used band positions, and the reference band tie-break could not work).
- `mosaic2d preview file.tif [...]`: natural-colour JPEG of finished mosaics, read from
  the TIFF's overviews; `--bands 630,550,450` to choose, `--max-side` for the size.
- Seamline feathering is written out while rendering, as soon as the last frame that can
  touch a part of the mosaic is done, one read and one write per row of tiles. Before,
  every feathered 32-px tile (14 bands, float32) was kept in memory until the end and
  then written in random order: on the Aran Island sets the step after the last frame
  ran for more than ten hours without a log line. Same output, byte for byte.
- The window shows the elapsed time, and how long ago the last log line came.
- Rendering speed on Windows: the canvas file is created sparse. Writing into a large
  memory-mapped file far from its start made NTFS zero-fill everything before that point;
  on the Aran Island multispectral sets (57 and 160 GB canvases) rendering slowed from
  ~9 s to ~2 min per photo as it went on (12-15 h in all). Unwritten canvas costs no disk.
- Sequence mode bridges only short runs (<= 3) of photos without any match; longer runs
  (camera moving fast, turbid water) are left out instead of being placed by dead
  reckoning over dozens of steps, which had stretched the canvas to 184 000 x 62 000 px.
- Phase correlation needs texture: blank photos no longer "match" each other with zero
  shift, and zero-shift results are rejected like other matches fixed to the camera.
- Matching: after a method finds only a handful of inliers (< 8: the photos do not
  overlap) the cascade stops instead of trying up to six more methods. Extra pairs from
  the adjusted footprints only for photos placed by real matches, at most 8 per photo
  (one set had proposed 11 879 pairs, 111 matched).
- `--quick` (checkbox in the window): SIFT only plus phase correlation for neighbours and
  at most 4 extra pairs per photo.
- Reference band (multispectral): photos from three places in the series (1/4, 1/2,
  3/4) instead of six consecutive middle ones; every band is scored by verified matches
  between consecutive photos times their spread, reduced for clipped (saturated) and
  nearly black pixels; the log lists matches, saturation and black share per band. If
  no band matches, the band nearest 550 nm with the least clipping is used. Faster (ranking at 1400 px). "Ref. band" field in the window to
  set it by hand (e.g. 560 nm).
- Matches that stay at the same image position (a pattern fixed to the camera: sensor
  noise, dust, a housing part, or a band that is black or clipped) no longer count:
  such pairs are rejected and do not score in the reference band choice. On an 8-bit
  Agrowing export whose 405 nm band was black, 405 nm had been chosen and every photo
  was "matched" onto itself (adjustment error 0.00 px, photos stacked). The log warns
  about bands that are mostly black or clipped in the input.
- Viewer for large and multispectral mosaics: `mosaic2d view file.tif`, or "View a
  mosaic..." in the window. Reads only the tiles on screen from the overview level that
  fits the zoom; any three bands (or one) with a percentile stretch, presets for natural
  colour and NIR false colour, "stretch to view", band values of the clicked pixel, save
  the view as PNG. Works with any TIFF layout (tiled or strips, interleaved or planar,
  uncompressed read directly from disk); files without overviews (older mosaic2d output)
  get a small one built once when opened. LZW / JPEG TIFFs (Metashape) need the
  `imagecodecs` package, which the Windows executable includes (`pip install
  mosaic2d[view]` otherwise).
- `--sequence` (checkbox "Photo series" in the window): for photos taken as a series.
  Direct neighbours in the shooting order are matched with every method and relaxed
  thresholds (window of 6), and where two consecutive photos are still not connected by
  any matched pair (featureless sand, blur, a turn) a weak link with the typical step
  between consecutive photos joins them, so the series stays one mosaic in its order.
  Only links that join separate groups are added; photos placed only this way are listed
  and lose overlaps to matched photos. The camera adjustment now honours pair weights.
- Image quality no longer drifts along single strips: the overlap comparison is split by
  where the overlap lies in each photo, and one sharpness per photo is solved together
  with an in-frame sharpness field (quadratic in the image position) shared by all
  photos. A camera tilted forward sees its far edge through more water, so each photo
  looked sharper than the one before it; along a 281-photo ledge transect the quality
  went from 0.00 to 3000 and half the photos were wrongly called blurred.
- `mosaic2d render --readjust` re-estimates the image quality from the ingest cache.
- `--lens-from <earlier output folder | report.json | project.pkl>` (also in the window):
  reuse the focal length and distortion of an earlier run with the same camera and
  housing instead of estimating them, e.g. for a wall or ledge after a flat survey.
- Camera adjustment: per-pair residuals in the report and pairs.csv are read from the
  same thinned tie points the adjustment used (a pair kept to hold the block together
  could shift the numbers of all later pairs).
- Faster output: the TIFF writer reads the mosaic one row of tiles at a time (one
  sequential pass instead of a scattered read per tile) and compresses tiles in parallel;
  same bytes as before. Writing the 91 000 x 28 600 px RGB dive mosaic had taken 5 hours.
- Faster rendering: each frame is drawn in 512 x 512 blocks, and blocks the frame neither
  wins nor feathers are skipped after a coarse check (with 5-10x overlap most of every
  frame is covered by better views). Frames are only decoded if they contribute.
- Seam optimisation: the per-pixel median / percentile over the views is vectorised
  (NumPy's nanpercentile looped over pixels in Python).
- Preview image: read in ~256 MB chunks (a 4096-row chunk of a wide mosaic was 4.5 GB).
- A transect tape or line held above the bottom is kept: a long thin object that is
  present near the same place in several views (parallax moves it a little between
  views) is no longer removed as a moving object; it was erased over half of the RGB
  dive. A rope that moves through a single view is still removed. Such objects are also
  ignored by the veil (blur) check, which had preferred whichever views showed the tape.
- Camera adjustment: pairs that do not fit are removed even when that would cut the
  block in two -- only the best-fitting pair across each cut is kept. Before, one such
  cut cancelled all removals of the round (RGB dive: RMSE 34.8 -> 19.1 px, median 9.3 ->
  8.7 px).
- The adjustment is not repeated when the extra pair search finds no new pairs.
- Log: writing progress in 10 % steps; the seam message gives the share of the mosaic
  taken from another view because of a moving object or blur.
- New default adjustment, `--model camera`: every photo is a camera over one ground plane
  with a shared lens (position, height, heading, two tilts per photo; focal length and
  radial distortion refined from EXIF / profile), soft priors for similar height and
  near-nadir view. Photos can no longer change size along the survey without a real
  change of height, shear or fold. 476-photo RGB dive: one continuous mosaic of all 301
  connected photos with a straight transect tape (homographies: two pieces, one sheared);
  synthetic strips: scale error 0.7-1.5 % vs 6-22 %. Estimated focal length 1.31-1.34 x
  EXIF underwater (flat-port refraction). Separate groups reuse the main block's lens.
- `--height` (m): metric pixel size from camera height / focal length; also in the window.
  `--focal-px`, `--height-sigma`. Report and images.csv list focal length, heights, tilts.
- Agrowing profile carries the lens focal length (21.8 mm, 3.76 um pixels).
- Output TIFFs are tiled (512 x 512) with internal overviews (pyramids), like Metashape
  and GDAL exports: viewers read only the tiles on screen and use the small overviews
  when zoomed out, instead of decoding image-wide strips and building pyramids first.
- Main block: frames with degenerate geometry are dropped, the main mosaic keeps only
  the largest piece still connected through good pairs (pieces that hung on through
  dropped frames become separate groups instead of being drawn beside it), and that
  piece is adjusted again on its own. The affine fallback is used only if it is clearly
  better (on a 476-image RGB dive it was 3x worse and had sheared part of the mosaic).
- Projects saved on Windows can be loaded on Linux/macOS and the other way round.
- Gaps in the coverage: every other group of connected images (`--min-block`, default 5)
  is adjusted and rendered on its own as `<name>_blockNN`; `<name>_blocks.csv` lists all
  groups. Before, only the largest group was rendered.
- Multiband TIFFs (msalign output and other GDAL-style files): band names and centre
  wavelengths are read from the file (GDAL metadata or msalign JSON) instead of
  "band 1 ... band N"; preview colours chosen by wavelength. `mosaic2d render` refreshes
  the names in projects saved before this change.
- Seam maps label frames by image number. tifffile is a regular dependency.
- Windows executable (`mosaic2d.exe`, built by GitHub Actions on every version tag)
  with a small window: choose folders and options, follow the log, open the results.
  `mosaic2d gui` opens the same window from a Python installation.
- Seamline feathering: a narrow transition (about one seam cell) across every seamline
  instead of a hard cut, so the short straight seams next to removed objects no longer
  show as steps (`--no-feather` restores hard seams).
- Moving-object removal keeps thin objects: small specks are rejected by area instead
  of a morphological opening that erased thin ropes.
- Veil detection: areas where one frame shows clearly less detail than the other frames
  (out-of-focus rope or hand close to the lens) are taken from frames that show them sharply.
- Image quality is measured on log-intensity images instead of contrast-equalised (CLAHE)
  images; CLAHE had inflated the detail of hazy frames, so the hazy views were preferred.
- Gain fields (two-band blending) ignore cells where a frame is a minority outlier
  against the per-cell median of all frames: a rope seen by one frame no longer darkens
  that frame around it (dark blocks next to removed objects). Where most views disagree
  (relief, parallax) nothing is left out, so the gains stay stable.
- Feathering keeps the weight of the frame that owns a pixel outright; a lone minor band
  contribution can no longer replace it.
- `--resolution` (and `--gsd`) refer to the native pixel of the median frame, not of the
  anchor frame, so the output scale no longer depends on which frame becomes the anchor
  (1.34x difference on the RGB test series, whose camera height varies).
- Seam analysis resolution relative to the frame size (same behaviour at every `--resolution`).
- The empty temporary canvas folder is removed after rendering.

## 0.2.0

- Seamlines optimised with a graph cut inside a band around the closest-centre seams.
- Deghosting: areas where a frame disagrees with the consensus of the other frames
  (moving rope, fish) are taken from a consistent frame.
- Flat field (vignetting / fixed illumination) estimated from the whole series and removed.
- Image quality (blur) estimated from overlaps instead of per image, so scene content
  does not count as blur.
- Low-frequency gain fields defined relative to the frame size.
- Ingest cache (`<out>/_cache`) and cached reference-band selection.
- `--gsd`: metric pixel size in GeoTIFF tags, `.tfw` world files and the VRT.
- `mosaic2d render`: render a saved project again (resolution, blending, seams, scale).
- Each render uses a private temporary folder (parallel runs no longer collide).

## 0.1.0

- Sony ARW decoder, Agrowing Sextuple profile with per-frame lens alignment and
  automatic reference band.
- Eleven feature methods with per-pair cascade / union, sequential + generic +
  estimated pair preselection, MAGSAC++ verification, guided matching.
- Global bundle adjustment with radial distortion self-calibration.
- Out-of-core rendering, mosaic/average/disabled blending, (Big)TIFF and tiled VRT output.
- Method benchmark.
