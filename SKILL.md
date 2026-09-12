---
name: plate-colony-counting
description: Count colonies on agar plate photos (phone camera) using image analysis. Use when the user asks to count colonies/clones/菌落/克隆 on a petri dish or plate photo, estimate transformation efficiency or library capacity from plating, or asks "how many colonies are on this plate". Handles tilted photos, uneven illumination, glare, dense small colonies. Produces an annotated archive image for verification.
---

# Plate Colony Counting

Count colonies in plate photos with image analysis (numpy/scipy/PIL; no OpenCV required).
Photos are typically phone shots: tilted (dish is an ellipse, colonies are not circles),
unevenly lit, with glare and JPEG artifacts. The pipeline below was calibrated on such photos.

## Quick start

```bash
python3 scripts/count_colonies.py <photo.jpg> --dish-mm 90
```

Outputs to `<photo>_count/`:
- `annotated.jpg` — every counted colony outlined along its actual shape (red=1, blue=merged pair, yellow=agar boundary)
- `boundary_check.jpg` — dish outline + agar boundary overlay, to verify BEFORE trusting the count
- `count_report.json` — counts, parameters, cross-check numbers

## Mandatory verification workflow

Never report a count without steps 1-3. Image-analysis colony counts fail silently in
distinct ways; each step catches a specific failure mode.

1. **Verify the boundary** (`boundary_check.jpg`): the yellow line (agar boundary, default
   94% of the dish outer-edge ellipse) must sit on the agar/plastic-rim junction in ALL
   sectors. If it cuts into agar somewhere or overlaps the rim, adjust `--shrink` (typical
   0.92-0.96) and rerun. Tilted photos + glare can bias the ellipse fit.
2. **Verify the markers** (`annotated.jpg`, zoom 1:1 into 2-3 spots): contours must sit on
   colonies a human would count, none missed in dense areas, none on blank agar. If faint
   texture is marked, raise `--min-colony-mm` or `--diff-thr`; if visible colonies are
   missed, lower them.
3. **Cross-check** (`count_report.json`): the component count and the independent
   local-maxima count should agree within ~20%. Large disagreement means merging or
   false positives dominate - inspect visually before trusting either.
4. Report the count as "约 N (±10-15%)", never as an exact number. State the counting
   standard used (minimum colony size), because different standards change the result a lot.

## Counting rules (calibrated defaults)

Count what a human counting the plate would count: clearly visible, distinct, roughly
round white dots. Concretely:

- minimum diameter 0.22 mm (`--min-colony-mm`) - faint specks below this are agar texture,
  not colonies; ask the user for their standard if ambiguous
- candidate merged blobs (elongation >2.5 or area >2.2x median) are split by
  distance-transform watershed: each distance-transform peak above a core-radius
  threshold = one colony. Smooth elongated streaks have no two-lobe structure and
  correctly stay 1
- elongated blobs near the rim whose long axis runs parallel to the boundary (within 20°,
  elongation >2.2, >86% radius) = meniscus/edge texture, NOT colonies - exclude
- everything is counted within the agar boundary only; the glare crescent is excluded and
  its area extrapolated from neighbouring ring density

## Prior art worth knowing (surveyed on GitHub/literature)

- [OpenCFU](https://opencfu.sourceforge.net/) - classic open-source counter (C++/Qt)
- [krransby/colony-counter](https://github.com/krransby/colony-counter) - Hough circle +
  watershed (OpenCV)
- [majsylw/microbial-counting-review](https://github.com/majsylw/microbial-counting-review)
  - curated list of the whole field
- Deep learning state of the art: Mask R-CNN (dedovskaya/CFUCounter), U2-Net+density maps
  (Graczyk 2022, NeuroSys-pl/objects_counting_dmap), CentroidNetV2 for dense connected
  colonies, U2-Net/ResNet50 pipeline (Cao 2024, PMC10820204). These beat classical CV on
  dense/clumped plates but need training data/weights; adopt only if the user needs
  routinely higher accuracy and can label ~50-100 plates. Watershed + size filtering
  (this skill) tracks human counts within ~10% on moderately dense dishes.

## Common failure modes (each one caused a wrong count in practice)

| Symptom | Cause | Fix |
|---|---|---|
| Count far too high, marks on faint speckles | threshold too loose; agar texture detected | raise `--min-colony-mm` / `--diff-thr`; calibrate on zoomed crops against what the user's eye counts |
| Count too low, dense center under-marked | grey-opening background or global brightness cap removes bright dense patches | use the median-filter background (script default); never cap absolute brightness inside the plate |
| Ring of false marks along the rim, "connected into lines" | plastic ring grooves / meniscus streaks, tangentially aligned | texture filter + watershed (streaks have no two-lobe structure, stay unsplit); tighten `--shrink` if the boundary overlaps the ring |
| Colonies near edge missing | conservative boundary (rim grooves trusted more than agar) | fix the ellipse (tilt!), lower `--shrink` until it sits on the agar/rim junction |
| Marks are circles but colonies look elliptical | camera tilt | script traces actual blob contours; never draw fixed-radius circles |
| Script hangs | per-blob full-image operations | keep all per-blob work inside the find_objects bounding box |
| Handwriting/marks counted | pen marks on plate bottom, big + elongated | watershed keeps them unsplit but they still count as 1-2; exclude manually if visible, usually ±1-3, negligible |

## Interpreting results for the user

- Plate count is for the plated fraction: multiply by 1/plated fraction for total
  transformants (e.g. half of 100 µL spread -> x2), divide by vector ng x 1000 for
  transformants/µg vector. Typical good Gibson/ligation: 1e5-1e6 /µg.
- Density variation across the plate (center-heavy after spreading) is normal; mention it
  if the user will pick colonies (sample randomly across sectors).
- Density that allows picking: ~1-10 k colonies per 9-10 cm dish. Overgrown/lawn: recount
  after dilution; do not attempt to count lawns.

## Calibration notes

Defaults were tuned on 3060x4080 px phone photos of 10 cm dishes (~30 px/mm). The script
converts `--min-colony-mm` to pixels via the fitted ellipse and `--dish-mm`, so it adapts
to other resolutions. If the photo is very different (microscope scan, tiny plate), verify
`px_per_mm` in stdout and eyeball `annotated.jpg` before trusting counts.
