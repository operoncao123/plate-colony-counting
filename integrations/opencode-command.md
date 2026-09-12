---
description: Count colonies on an agar plate photo with the Plate Colony Counting skill (数克隆/菌落计数)
---

Read the Plate Colony Counting skill instructions at `__SKILL_DIR__/SKILL.md`, then
execute its full workflow on this request:

$ARGUMENTS

Run the script (`__SKILL_DIR__/scripts/count_colonies.py`, needs only numpy/scipy/pillow)
rather than eyeballing the photo, then complete the skill's mandatory verification
workflow: check `boundary_check.jpg` (yellow line on the agar/rim edge in ALL sectors),
zoom `annotated.jpg` to confirm contours sit on colonies a human would count, sanity-check
the component count against the local-maxima cross-check, and report "约 N (±10-15%)" with
the counting standard stated. If the boundary or markers are off, recalibrate
(`--shrink` / `--ellipse` / `--min-colony-mm` / `--diff-thr`) instead of defending the number.
