---
name: aaml-override
description: Record a personal, persistent override to a platform skill (e.g. "always use time-series CV for my data", "never propose CatBoost") in ~/.agentic-automl/skills/overrides/<skill>.md. Use when the user wants a workflow preference to stick across trials; the only sanctioned writer of override files.
---

# aaml-override

## What it does
Captures a user preference that should change how a platform skill behaves
in **every** future trial for this user, without editing the platform skill.
Overrides are read after the platform `SKILL.md`, and the override wins.

## Inputs
- The user's stated preference and the platform skill it applies to.
- Any existing `~/.agentic-automl/skills/overrides/<skill>.md`.

## Output
- `~/.agentic-automl/skills/overrides/<skill>.md` (no artifact or gate).
- An `insight` feedback episode, so the preference can be considered for the
  platform via `aaml-feedback`.

## Key notes
- One file per platform skill, named exactly after the skill directory. Check
  `agentic-automl overrides list` and `.claude/skills/SKILLS.md`.
- Overrides adjust **how** a skill works (defaults, preferences, extra
  checks). They can't skip gates, write artifact paths, or change schemas. If
  a request needs that, explain why and record it as feedback instead.
- Keep each rule short and imperative, with the reason, the date, and the
  trial it came from.
- Confirm the exact wording with the user before writing.

## Before you start
No validator gate. Read the target skill's `SKILL.md` so the override is
phrased against what it currently says.

## How to persist
Write (or append to) `~/.agentic-automl/skills/overrides/<skill>.md`, then:
```
agentic-automl feedback-episodes insight "<one-line summary>" --skill <skill> --layer skill
```

## Steps
1. Identify the target skill. Ask if it's ambiguous.
2. Draft the rule:
   ```markdown
   ## <short title>  (added <YYYY-MM-DD>, trial <trial_id>)
   - Rule: <imperative instruction>
   - Why: <user's reason>
   ```
3. Show the draft and get approval. Then create the overrides directory if
   needed (`mkdir -p ~/.agentic-automl/skills/overrides`) and write or append
   the section.
4. Record the insight episode. Tell the user the rule applies from the next
   use of that skill, and that `agentic-automl overrides list` shows all overrides.
