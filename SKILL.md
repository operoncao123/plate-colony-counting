---
name: plate-colony-counting
description: Count colonies on agar plate photos (phone camera) using image analysis. Use when the user asks to count colonies/clones/菌落/克隆 on a petri dish or plate photo, estimate transformation efficiency or library capacity from plating, or asks "how many colonies are on this plate". Handles tilted photos, uneven illumination, glare, dense small colonies. Produces annotated verification images and a JSON report.
---

# Plate Colony Counting

Count colonies in plate photos with OpenCV + numpy. Photos are typically phone shots:
tilted (the dish is an ellipse, colonies are not circles), unevenly lit, with glare,
specular rims and blotchy agar. The pipeline below was re-calibrated on such photos
(3060x4080 px, 10 cm dishes, 900-4000 colonies/plate).

## Quick start

```bash
python3 scripts/count_colonies.py <photo.jpg> --outdir <results_dir>
# several at once:
python3 scripts/count_colonies.py 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg --outdir results/
```

Outputs per image in `<results_dir>/<stem>/`:
- `annotated.jpg` — a legend is burned into the image, so it is self-describing:
  RED outline = colony counted (outline of the blob it came from);
  green ring + orange cross = one counted colony;
  BLUE outline = artefact detected but deliberately NOT counted (rim/meniscus
  streaks — these hug the agar edge and, being long and thin, draw as a long
  closed loop that can look like a chain of circles);
  yellow line = agar / counting boundary.
- `boundary_check.jpg` — cyan = fitted outer dish ellipse, yellow = counting
  boundary. Use this FIRST.
- `zoom_1..5.jpg` — 1:1 crops (centre + 4 diagonal quadrants) for eyeballing
- `count_report.json` — count, cross-check, parameters, fitted ellipse

## Mandatory verification workflow

Never report a count without steps 1-3. Image-analysis counts fail silently in
distinct ways; each step catches a specific failure mode.

1. **Boundary** (`boundary_check.jpg`): the yellow line must lie on the agar/rim
   junction all the way round, i.e. agar on the inside, plastic rim on the outside,
   and no visible colony outside it. If it cuts into agar, raise `--shrink`; if it
   overlaps the rim, lower it. Typical 0.93-0.96.
2. **Markers** (`zoom_*.jpg`, 1:1): every colony a human would count carries exactly
   one orange cross; touching colonies carry one each; nothing is marked on blank
   agar. Blue-outlined blobs should all be rim/meniscus streaks, never colonies.
   (Red is only ever drawn on counted blobs, so red = in the number.)
3. **Cross-check** (`count_report.json`): `crosscheck_peaks` is an independent
   local-maxima count that never touches the watershed. It should be **<= colonies**
   and within ~20%. Much lower → the plate is very clumped (peaks merge; the
   watershed number is the better one). Much higher → over-splitting.
4. Report as "~N (±10-15%)", never as an exact number, and state the minimum colony
   size counted. Varying `--min-diam-mm` 0.20→0.35 moves the total by ~10%, and
   `--shrink` 0.93→0.97 by ~5%; that is where the uncertainty comes from.

## Pipeline (what the script does, and why)

1. **Dish geometry**: coarse Hough circle → for each of 720 angles, walk outward and
   take the strongest radial intensity gradient → `cv2.fitEllipse` on those points.
   The ellipse absorbs camera tilt. *This replaces the old "shrink a Hough circle"
   step, which was the single biggest error source: the Hough radius landed
   sometimes on the outer rim, sometimes on the inner wall, so a fixed `--shrink`
   cut off outer colonies on one plate and overlapped the rim on the next. The
   observed agar boundary ranged from 0.85 to 0.95 of the raw Hough radius across
   five photos of the same plate type; against the refined ellipse it is a stable
   ~0.95.*
2. **Agar mask** = refined ellipse × `--shrink`.
3. **White top-hat**: `tophat = gray - median(downscale(gray, 4), 31)`. Removes
   illumination gradients and glare without capping bright dense regions (a global
   brightness cap deletes exactly the dense colonies you care about).
4. **Threshold** the top-hat: Otsu inside the mask, floored at 8.
5. **Artefact rejection** — the failure that mattered most:
   - *thin streaks* (rim highlight arcs, meniscus): reject a component whose maximum
     distance-transform radius is < 0.55 × colony radius. On one plate the agar-edge
     highlight was a single 49,578-px arc; it passed the old pure-size filter and the
     watershed then carved it into dozens of phantom colonies.
   - *low solidity*: reject components whose area / convex-hull area < 0.45 (curved
     arcs that survive the thickness test).
6. **Split merged blobs**: distance-transform watershed, peaks ≥ 0.45 × colony
   radius, minimum peak separation 1.15 × colony radius, fragments < 0.35 ×
   single-colony area discarded. Colony radius is estimated from the median area of
   the smallest 45% of components (merges inflate the rest).
7. **Independent cross-check**: local maxima of the Gaussian-smoothed top-hat,
   minimum separation 1.0 × colony radius.

## Counting rules (calibrated defaults)

Count what a human counting the plate would count: clearly visible, distinct,
roughly round white dots.

- minimum colony diameter 0.25 mm (`--min-diam-mm`) — below that is agar speckle.
  Ask the user for their standard if the answer matters: on these plates 0.20 mm
  vs 0.35 mm changes the total by ~10%.
- touching colonies are split by watershed; elongated single colonies stay 1
- only inside the agar boundary; dish rim, plastic wall and background are excluded

## Prior art worth knowing (surveyed on GitHub/literature)

- [OpenCFU](https://opencfu.sourceforge.net/) — classic open-source counter (C++/Qt)
- [krransby/colony-counter](https://github.com/krransby/colony-counter) — Hough
  circle + watershed (OpenCV)
- [majsylw/microbial-counting-review](https://github.com/majsylw/microbial-counting-review)
  — curated list of the whole field
- Deep learning state of the art: Mask R-CNN (dedovskaya/CFUCounter), U2-Net+density
  maps (Graczyk 2022, NeuroSys-pl/objects_counting_dmap), CentroidNetV2 for dense
  connected colonies, U2-Net/ResNet50 pipeline (Cao 2024, PMC10820204). These beat
  classical CV on dense/clumped plates but need training data/weights; adopt only if
  the user needs routinely higher accuracy and can label ~50-100 plates. Watershed +
  size filtering (this skill) tracks human counts within ~10% on moderately dense
  dishes.

## Common failure modes (each one caused a wrong count in practice)

| Symptom | Cause | Fix |
|---|---|---|
| Watch out: the boundary looks fine on a sparse plate and cuts off outer colonies on a dense one | shrinkage applied to a sloppy Hough radius, not to the real rim | the refined ellipse fixes it; verify per plate with `boundary_check.jpg`, don't reuse one `--shrink` blindly |
| Count far too high; marks along a ring near the rim; blobs "connected into lines" | agar-edge highlight arc / meniscus streak detected and watershed-split into many colonies | distance-transform thickness test + solidity test (built in); blue outlines in `annotated.jpg` show what was rejected |
| Count far too high, marks on faint speckles | threshold too loose, agar texture detected | raise `--min-diam-mm`; compare `zoom_*.jpg` against what your eye counts |
| Count too low, dense centre under-marked | global brightness cap or grey-opening background removes bright dense patches | keep the median-filter background (default); never cap absolute brightness inside the plate |
| Colonies near the edge missing | conservative boundary | verify against `boundary_check.jpg`, raise `--shrink` |
| Marks are circles but colonies look elliptical | camera tilt | script traces actual blob contours and fits an ellipse to the dish; never draw fixed-radius circles |
| One colony marked twice | over-splitting of a slightly elongated colony | raise `--min-diam-mm`, or check the blue/red overlay; small residual over-splitting is bounded by the ±10-15% figure |
| Script hangs | per-blob full-image operations | keep all per-blob work inside the blob's bounding box (as the script does) |
| Handwriting/marks counted | pen marks on the plate bottom | usually ±1-3, negligible; exclude manually if visible |

## Interpreting results for the user

- The plate count is for the plated fraction: multiply by 1/(plated fraction) for
  total transformants (e.g. half of 100 µL spread → ×2), then divide by vector ng
  × 1000 for transformants/µg vector. Typical good Gibson/ligation: 1e5-1e6/µg.
- `px_per_mm` in the report comes from assuming `--dish-mm` for the fitted ellipse;
  it is only used for the mm-based size filter, so a wrong `--dish-mm` shifts the
  size threshold, not the count by area.
- Density variation across the plate (centre-heavy after spreading) is normal;
  mention it if the user will pick colonies (sample randomly across sectors).
- Density that allows picking: ~1-10 k colonies per 9-10 cm dish. Overgrown/lawn:
  recount after dilution; do not attempt to count lawns.
- If two photos have very similar counts, check they are actually different plates
  (compare the spatial pattern of dense/sparse patches, not just the total).
