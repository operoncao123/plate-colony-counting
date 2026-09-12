#!/usr/bin/env bash
# Plate Colony Counting installer — installs the skill for every supported coding agent.
#
#   ./integrations/install.sh              install everywhere detected
#   ./integrations/install.sh claude       only Claude Code
#   ./integrations/install.sh codex zcode  only these two
#
# Tools and their native locations:
#   claude     ~/.claude/skills/<name>/            SKILL.md native (Agent Skills)
#   codex      ~/.codex/skills/<name>/             SKILL.md native (Codex >= Dec 2025)
#   zcode      ~/.zcode/skills/<name>/             SKILL.md native
#   agents     ~/.agents/skills/<name>/            cross-agent standard dir
#   trae       ~/.trae/skills/<name>/              Anthropic-style skills
#              + rules snippet to paste into .trae/rules/project_rules.md
#   opencode   ~/.config/opencode/command/plate-count.md
#              slash command that reads the skill from ~/.agents/skills/
#   workbuddy  no public skills path: use the Create Skills flow with
#              integrations/workbuddy-skill.yml.md (see integrations/README.md)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="$(cd "$HERE/.." && pwd)"          # repo root == skill folder
NAME="plate-colony-counting"
ONLY="${*:-}"
want() { [ -z "$ONLY" ] || echo " $ONLY " | grep -q " $1 "; }

ok()   { printf '  \033[32m✔\033[0m %s\n' "$1"; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

install_copy() {  # install_copy <label> <dest-dir>
  local label="$1" dest="$2"
  hdr "$label"
  if [ -e "$dest" ] && [ ! -w "$dest" ]; then
    printf '  \033[2m– %s not writable\033[0m\n' "$dest"; return
  fi
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  cp -R "$SKILL" "$dest"
  rm -rf "$dest/.git" "$dest/integrations" "$dest/demo"
  find "$dest" -name '.DS_Store' -delete 2>/dev/null || true
  ok "installed -> $dest"
}

want claude && install_copy  "Claude Code"               "$HOME/.claude/skills/$NAME"
want zcode  && install_copy  "ZCode"                     "$HOME/.zcode/skills/$NAME"
want codex  && install_copy  "OpenAI Codex CLI"          "$HOME/.codex/skills/$NAME"
want agents && install_copy  "Cross-agent (~/.agents)"   "$HOME/.agents/skills/$NAME"
want trae   && install_copy  "TRAE"                      "$HOME/.trae/skills/$NAME"

if want opencode; then
  hdr "OpenCode"
  dest="$HOME/.config/opencode/command/plate-count.md"
  mkdir -p "$(dirname "$dest")"
  sed "s|__SKILL_DIR__|$HOME/.agents/skills/$NAME|g" \
      "$HERE/opencode-command.md" > "$dest"
  ok "slash command -> $dest  (use /plate-count <photo>; reads the skill from ~/.agents/skills/$NAME)"
  [ -d "$HOME/.agents/skills/$NAME" ] || \
    ok "hint: run './integrations/install.sh agents' so the command file has a skill to read"
fi

if want workbuddy; then
  hdr "WorkBuddy"
  ok "no public skills directory — open WorkBuddy, run its Create Skills flow, and"
  ok "paste the description from integrations/workbuddy-skill.yml.md as the skill definition."
fi

if want trae && [ ! -f "$HOME/.trae/skills/$NAME/SKILL.md" ]; then
  hdr "TRAE rules hint"
  ok "if your TRAE build does not auto-discover skills, paste"
  ok "integrations/trae-rules-snippet.md into .trae/rules/project_rules.md"
fi

hdr "Done"
ok "verify: SKILL.md exists in the paths above; then just ask your agent to 数克隆 / count colonies on a plate photo."
