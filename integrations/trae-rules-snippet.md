# Plate Colony Counting — TRAE rules snippet

Paste the block below into `.trae/rules/project_rules.md` (project level) or the
user-rules editor (global), then make sure the skill folder itself is installed
at `~/.trae/skills/plate-colony-counting/` (run `./integrations/install.sh trae`).

TRAE builds that support Agent Skills discover the folder automatically; the
rules entry below guarantees routing even on builds that only honour rules.

```markdown
## Plate Colony Counting skill (数克隆 / 菌落计数 / colony counting)

When the user asks to 数克隆, 克隆计数, 菌落计数, count colonies, count CFUs,
count colonies on a petri dish / plate photo, estimate transformation
efficiency or library capacity from plating, or shows an agar-plate photo and
asks how many colonies are on it:

1. Read `~/.trae/skills/plate-colony-counting/SKILL.md` and follow it exactly.
2. Run its script rather than eyeballing:
   `python3 ~/.trae/skills/plate-colony-counting/scripts/count_colonies.py <photo> --dish-mm 90`
   (needs only numpy/scipy/pillow; outputs annotated.jpg, boundary_check.jpg,
   count_report.json in <photo>_count/).
3. Non-negotiables of the skill: verify boundary_check.jpg (yellow line on the
   agar/rim edge in ALL sectors) and annotated.jpg (contours on colonies a
   human would count) BEFORE reporting any number; report "约 N (±10-15%)"
   with the counting standard stated; never count lawns — ask for dilution.
4. If the user questions the result, zoom the annotated image and recalibrate
   (--min-colony-mm / --diff-thr / --shrink / --ellipse) instead of defending
   the number.
```
