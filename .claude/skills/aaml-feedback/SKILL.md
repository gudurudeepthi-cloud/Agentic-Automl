---
name: aaml-feedback
description: Turn this trial's feedback episodes (errors, insights, rationales, decision diffs) and the user's skill overrides into triaged, sanitised platform feedback, optionally filed as GitHub issues tagged by layer. Use when the user wants to share feedback or at the end of a trial when there are open episodes.
---

# aaml-feedback

## What it does
Reconciles what happened in a trial into actionable feedback for platform
maintainers. Each item is tagged with the layer it belongs to:
`[skill:<name>]`, `[schema:<artifact>]`, `[gate:<gate>]`, `[eval:<scenario>]`, `[domain:<name>]`.

## Inputs
- `agentic-automl feedback-episodes list --status open`
- `agentic-automl overrides list` (personal overrides that may deserve to be platform defaults)

## Output
- Markdown issue drafts in `work/feedback_issues/<layer>-<target>-<n>.md`.
- Optionally, GitHub issues (only with the user's explicit go-ahead).
- Triaged and resolved episodes in `work/feedback_episodes.jsonl`.

## Key notes
- **Sanitise before anything leaves the machine.** Always build the content from
  `agentic-automl feedback-episodes filter`, which redacts paths, `gs://`
  URIs, emails and long numbers, and drops decision values. Then reread the
  drafts for data values, column names or business details the user wouldn't
  want shared. Ask if unsure.
- Raw `error` episodes are only shared after triage to a layer. Many errors
  are your own mistakes, not platform issues. Resolve those with a note.
- Filing issues is **outward-facing**. The target repo comes from
  `agentic-automl config show` → `feedback_repo`. If it isn't set, stop at the
  drafts and tell the user how to configure it
  (`agentic-automl config set-feedback-repo <host/org/repo>`). Never guess a
  host or repo.
- Each issue body ends with the credit marker line, so the community
  leaderboard can attribute it:
  `<!-- aaml-feedback-credit:v1 {"trial": "<trial_id>", "layer": "<layer>", "target": "<target>"} -->`

## Before you start
No validator gate. Run `agentic-automl feedback-episodes list --status open`.
If nothing is open and there are no overrides, say so and stop.

## How to persist
- Triage: `agentic-automl feedback-episodes triage <id> --layer <layer> --target <name>`
- Dismiss: `agentic-automl feedback-episodes resolve <id> --note "<why>"`
- Drafts: write `work/feedback_issues/*.md`

## Steps
1. List the open episodes and overrides. Group them by likely layer.
2. For each episode, either **triage** it (a platform issue: wrong or missing
   skill guidance, schema too strict or loose, gate ordering, eval gap,
   domain overlay) or **resolve** it (your own mistake, or already handled).
3. For each override, ask the user whether it should become a platform default.
   If yes, draft a `[skill:<name>]` item.
4. Run `agentic-automl feedback-episodes filter` and write one draft per item:
   title `[<layer>:<target>] <summary>`, then what happened, the expected
   behaviour, the suggested change, and the credit marker.
5. Show the drafts. Only if the user explicitly approves **and** `feedback_repo`
   is configured, file them (e.g. `gh issue create --repo <feedback_repo>
   --title ... --body-file ...`, with the user approving each command). Then
   resolve the corresponding episodes with the issue URL as the note.
