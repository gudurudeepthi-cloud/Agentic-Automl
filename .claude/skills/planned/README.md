# Planned skills (not wired)

Drafts for future capabilities live here. They are **not** registered in
`SKILL_PRODUCES`, have no artifact or gate, and must not be used in trials.

To promote a draft, follow the contributor skill `contribute-capability`
(`.repo/skills/contribute-capability/SKILL.md`). That means a registry entry,
gate, requirements, schema, `SKILL_PRODUCES` entry, test fixtures, and then
moving the directory up to `.claude/skills/`.

Candidates:
- `distributed-training`: execute `workflow_plan.distributed_training_plan`
  (Dataproc/Spark or Ray). Environment-specific.
- `monitoring-plan`: drift and performance monitoring spec for a deployed model.
