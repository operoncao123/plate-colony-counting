# Integrations — one skill, six agents

The skill is a plain **Agent Skills** folder (`SKILL.md` + `scripts/`), which most
current coding agents load natively. Everything here installs from this folder:

```bash
./integrations/install.sh                 # install everywhere detected
./integrations/install.sh codex zcode     # …or only specific tools
```

| Agent | How it loads Plate Colony Counting | Installed by script | Invoke with |
|---|---|---|---|
| **Claude Code** | native Agent Skills loader | `~/.claude/skills/plate-colony-counting/` | just ask: `数这张平板的克隆` / `count colonies on this photo` |
| **OpenAI Codex CLI** | native skills support (Dec 2025+) | `~/.codex/skills/plate-colony-counting/` | just ask, or mention the skill by name |
| **ZCode** | native skills loader | `~/.zcode/skills/plate-colony-counting/` | just ask |
| **TRAE** | Anthropic-style skills; rules fallback | `~/.trae/skills/plate-colony-counting/` + snippet for `.trae/rules/project_rules.md` | just ask (rules guarantee routing) |
| **OpenCode** | slash-command file | `~/.config/opencode/command/plate-count.md` | `/plate-count <photo path>` |
| **WorkBuddy** | its own Create Skills flow (`skill.yml`) | manual — see [workbuddy-skill.yml.md](workbuddy-skill.yml.md) | new task with any 数克隆 topic |
| *anything reading `~/.agents/skills`* | cross-agent standard dir | `~/.agents/skills/plate-colony-counting/` | agent-dependent |

Notes:

* The OpenCode command file points at `~/.agents/skills/plate-colony-counting/`, so run
  `./integrations/install.sh agents opencode` together (the script reminds you).
* For TRAE builds that do not auto-discover skills, paste
  [trae-rules-snippet.md](trae-rules-snippet.md) into `.trae/rules/project_rules.md`.
* The script never touches tool config beyond the paths listed above; it skips
  `integrations/` when copying (that is repo packaging, not skill).
* The counting script needs only `numpy` and `opencv-python`.

Sources for the per-tool mechanisms: [Codex skills](https://simonwillison.net/2025/Dec/12/openai-skills/) ·
[OpenCode commands](https://opencode.ai/docs/commands/) ·
[TRAE rules & agents](https://docs.trae.ai/ide/rules) ·
[WorkBuddy skills](https://www.tencentcloud.com/techpedia/145692?lang=en)
