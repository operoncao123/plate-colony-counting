# WorkBuddy adapter — how to install Plate Colony Counting

WorkBuddy (Tencent) does not publish a skills directory the way Claude Code or
Codex do; skills are created through WorkBuddy's own **Create Skills** flow,
which generates a `skill.yml` plus implementation files from a natural-language
description. Two ways to bring Plate Colony Counting in:

## Option A — Create Skills flow (recommended)

1. Open WorkBuddy and start a new task with the Create Skills entry point.
2. When asked to describe the skill, paste the `description` block below.
3. After WorkBuddy generates the `skill.yml`, replace/merge its `instructions`
   (or system-prompt) field with the full text of [`SKILL.md`](../SKILL.md),
   and attach this folder's `scripts/count_colonies.py` as a skill resource.
4. Install, then test in a **fresh** conversation by sending a plate photo with
   `帮我数一下这张平板有多少个克隆` (or any 数克隆 topic).

## Option B — paste-in description

Use this as the Create Skills description:

```yaml
name: plate-colony-counting
description: >
  Count colonies on agar plate photos (phone camera: tilted, uneven light,
  glare, dense small colonies) with image analysis — never by eyeballing.
  Reports "约 N (±10-15%)" with the counting standard stated, plus an
  annotated verification image and a boundary-check overlay. Also converts
  plate counts to transformation efficiency (CFU/µg vector) when the user
  gives plating details. Trigger words: 数克隆, 克隆计数, 菌落计数, 平板计数,
  colony counting, count colonies, CFU count, plate count, transformation
  efficiency, library capacity.
source_instructions_file: SKILL.md   # attach the repo's SKILL.md content
resources:
  - scripts/count_colonies.py
input: an agar plate photo (+ optional dish diameter in mm; plating details for
        CFU conversion: competent-cell volume, plated fraction, vector ng)
output: colony count with uncertainty and counting standard, annotated.jpg
        (contours on every counted colony), boundary_check.jpg, count_report.json
limits: never report a number before visually verifying boundary_check.jpg and
        annotated.jpg; never report an exact integer (use 约 N ±10-15%); never
        count overgrown lawns — ask the user for a diluted photo; counting
        standard (min colony size) must be stated and, if ambiguous, confirmed
        with the user.
```

Field names may differ slightly across WorkBuddy versions — keep the *content*
of each field and map it to whatever the Create Skills dialog asks for.
