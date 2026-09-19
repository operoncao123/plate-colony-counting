# Plate Colony Counting

Count colonies on agar-plate photos (phone camera) with OpenCV + numpy — as an
[Agent Skill](integrations/README.md) for coding agents (Claude Code, Codex CLI,
ZCode, TRAE, OpenCode, WorkBuddy), or as a plain standalone CLI script.

Point it at a plate photo and it finds the dish, counts the colonies, and
produces annotated verification images so **you** can confirm the number before
trusting it. Handles the messiness of real bench photos: tilted hand-held shots
(the dish is an ellipse, not a circle), uneven illumination, glare, specular
rim highlights, blotchy agar, dense plates with touching colonies
(calibrated on 3060×4080 px phone shots, 10 cm dishes, 900–4000 colonies/plate).

## Requirements

- Python 3.8+
- `numpy`, `opencv-python` (no scikit-image needed)

```bash
pip install numpy opencv-python
```

## Quick start

```bash
# one photo (results land in <photo>_count/ next to the image)
python3 scripts/count_colonies.py IMG_1234.jpg

# several at once
python3 scripts/count_colonies.py 1.jpg 2.jpg 3.jpg --outdir results/

# tilted hand-held photo where boundary_check.jpg does not sit on the agar edge:
python3 scripts/count_colonies.py IMG_1234.jpg --outdir results/ \
        --robust-boundary --shrink 1.0
```

Then **always verify** (see below) before using the number.

## Outputs

Each image gets its own folder:

| File | What it is |
|---|---|
| `annotated.jpg` | Full image with the count burned in as a legend. Self-describing: RED outline = colony counted (outline of its blob); green ring + orange cross = one counted colony; BLUE outline = artefact detected but deliberately NOT counted (rim/meniscus streaks); yellow line = agar / counting boundary. |
| `boundary_check.jpg` | Cyan = fitted outer dish ellipse, yellow = counting boundary. **Look at this one FIRST.** |
| `zoom_1..5.jpg` | 1:1 crops (centre + 4 diagonal quadrants) for eyeballing markers. |
| `count_report.json` | Count, cross-check number, all parameters, fitted ellipse. |

## The mandatory verification workflow

An image-analysis count fails silently in distinct ways; each check below
catches one failure mode. Never report a count without all three.

1. **Boundary** — in `boundary_check.jpg`, the yellow line must lie on the
   agar/rim junction all the way round (agar inside, plastic rim outside, no
   colony outside it). If it cuts into agar, raise `--shrink`; if it overlaps
   the rim, lower it (typical 0.93–0.96). If the line runs off the dish in some
   sectors (common on tilted photos), the fit itself is broken — use
   `--robust-boundary --shrink 1.0` instead of nudging `--shrink`.
2. **Markers** — in the `zoom_*.jpg` crops at 1:1, every colony a human would
   count carries exactly one orange cross; touching colonies carry one each;
   nothing is marked on blank agar. Blue outlines should all be rim/meniscus
   streaks, never colonies (red is only drawn on counted blobs, so red = in
   the number).
3. **Cross-check** — in `count_report.json`, `crosscheck_peaks` is an
   independent local-maxima count that never touches the watershed. It should
   be **≤ colonies** and within ~20%. Much lower → very clumped plate (the
   watershed number is the better one). Much higher → over-splitting.

Report the result as **~N (±10–15%)**, never as an exact number, and state the
minimum colony size counted: `--min-diam-mm` 0.20 → 0.35 moves the total by
~10% and `--shrink` 0.93 → 0.97 by ~5% — that is where the uncertainty comes
from.

## CLI reference

```text
python3 scripts/count_colonies.py <photo.jpg> [more.jpg ...] [options]

images                 one or more plate photos
--outdir DIR           output root (default: <image>_count next to the image)
--shrink FLOAT         agar boundary as a fraction of the fitted dish
                       ellipse (default 0.95; use 1.0 with --robust-boundary
                       or --boundary-json)
--dish-mm FLOAT        outer plate diameter in mm (default 90) — only used to
                       derive the mm scale
--min-diam-mm FLOAT    minimum counted colony diameter in mm (default 0.25)
--robust-boundary      find the agar/rim junction from the bright-AND-speckled
                       region (RANSAC fit) instead of the strongest radial
                       gradient — for tilted hand-held photos; fits the AGAR
                       edge, so pair with --shrink 1.0
--boundary-json FILE   supply the dish ellipse (cx,cy,a,b,angle) directly
--px-per-mm FLOAT      override the mm scale (otherwise derived from the
                       ellipse)
--quiet                suppress progress output
```

**When comparing counts across plates, pin `--px-per-mm`.** Because the mm
scale is derived *from* the fitted ellipse, a boundary change silently moves
the effective `--min-diam-mm` cutoff even when the geometry barely changes.

## How it works

1. **Dish geometry** — coarse Hough circle → per-angle radial edge search
   (720 rays) → `cv2.fitEllipse`. The ellipse absorbs camera tilt. On strongly
   tilted photos the single-gradient search can jump between the agar edge,
   the rim highlight and the meniscus; `--robust-boundary` replaces it with a
   physically specific cue (agar is the only bright **and** speckled surface)
   plus a RANSAC/IRLS ellipse fit.
2. **Agar mask** = fitted ellipse × `--shrink`.
3. **White top-hat** (`gray` − median-filtered background) removes illumination
   gradients and glare **without** capping bright dense regions — a global
   brightness cap deletes exactly the dense colonies you care about.
4. **Threshold** the top-hat (Otsu inside the mask, floored at 8).
5. **Artefact rejection** — thin-streak test (rim-highlight arcs, meniscus:
   max distance-transform radius < 0.55 × colony radius) and low-solidity test
   (area / convex-hull area < 0.45).
6. **Split merged blobs** — distance-transform watershed, scaled to an
   automatically estimated single-colony radius; fragments below 0.35 ×
   single-colony area are discarded.
7. **Independent cross-check** — local maxima of the Gaussian-smoothed
   top-hat, minimum separation 1.0 × colony radius.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Yellow boundary cuts agar on one side, rim on the other (tilted photo) | `--robust-boundary --shrink 1.0`, or `--boundary-json` with a hand-measured ellipse. No `--shrink` value can repair an ellipse of the wrong shape. |
| Count far too high; marks along a ring near the rim | Rim/meniscus streak detected and split — the built-in thin-streak + solidity tests handle this; check the blue outlines in `annotated.jpg`. |
| Count far too high; marks on faint speckles | Raise `--min-diam-mm`; compare `zoom_*.jpg` against what your eye counts. |
| Count too low; dense centre under-marked | Don't cap brightness (default already avoids it); the issue is usually the boundary — re-check `boundary_check.jpg`. |
| Colonies near the edge missing | Boundary too conservative — verify with `boundary_check.jpg`, raise `--shrink`. |
| One colony marked twice | Slight over-splitting — raise `--min-diam-mm`; residual is bounded by the ±10–15% figure. |
| Two photos with near-identical counts | Check they are actually different plates (compare the dense/sparse spatial pattern, not just the total). |

## Interpreting the number

- The count is for the plated fraction. Multiply by 1/(plated fraction) for
  total transformants (e.g. half of 100 µL spread → ×2), then divide by vector
  µg for transformants/µg vector. Typical good Gibson/ligation: 1e5–1e6/µg.
- A picking-friendly density is ~1–10 k colonies per 9–10 cm dish. If the
  plate is overgrown/lawn, recount after dilution — do not count lawns.
- Centre-heavy density after spreading is normal; if you will pick colonies,
  sample randomly across sectors.

## Using it with coding agents

The repo is a standard Agent Skills folder (`SKILL.md` + `scripts/`). To
install into Claude Code, Codex CLI, ZCode, TRAE, OpenCode or WorkBuddy:

```bash
./integrations/install.sh            # install everywhere detected
./integrations/install.sh codex zcode  # …or only specific tools
```

See [integrations/README.md](integrations/README.md) for per-tool details and
invoke by just asking, e.g. `数这张平板的克隆` / `count colonies on this photo`.

## Prior art

- [OpenCFU](https://opencfu.sourceforge.net/) — classic open-source counter (C++/Qt)
- [krransby/colony-counter](https://github.com/krransby/colony-counter) — Hough circle + watershed (OpenCV)
- [majsylw/microbial-counting-review](https://github.com/majsylw/microbial-counting-review) — curated list of the field
- Deep-learning SOTA (Mask R-CNN, U2-Net + density maps, CentroidNetV2) beats
  classical CV on dense/clumped plates but needs ~50–100 labelled plates to
  train. This watershed pipeline tracks human counts within ~10% on moderately
  dense dishes with zero training data.
