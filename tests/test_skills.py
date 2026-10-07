import json
import re

import pytest

from agentic_automl.contracts import SKILL_PRODUCES, claude_template_dir

SKILLS_DIR = claude_template_dir() / "skills"
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
REQUIRED_SECTIONS = ("## What it does", "## Inputs", "## Output", "## Key notes",
                     "## Before you start", "## How to persist", "## Steps")


def frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER.match(text)
    assert match, "missing YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


@pytest.mark.parametrize("skill", sorted(SKILL_PRODUCES))
def test_registered_skill_has_wellformed_skill_md(skill):
    text = (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    meta = frontmatter(text)
    assert meta["name"] == skill
    assert len(meta["description"]) > 40
    for section in REQUIRED_SECTIONS:
        assert section in text, section
    assert f"agentic-automl artifacts begin-skill {skill}" in text
    for key in SKILL_PRODUCES[skill]:
        assert f"save-artifact {key} --payload work/{key}.json --source-skill {skill}" in text, key


def test_index_lists_every_registered_skill():
    index = (SKILLS_DIR / "SKILLS.md").read_text(encoding="utf-8")
    for skill in SKILL_PRODUCES:
        assert f"]({skill}/SKILL.md)" in index, skill


def test_index_links_resolve():
    index = (SKILLS_DIR / "SKILLS.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+\.md)\)", index):
        assert (SKILLS_DIR / target).is_file(), target


def test_domain_overlays_exist():
    registry = json.loads((SKILLS_DIR / "domains" / "registry.json").read_text(encoding="utf-8"))
    for name, entry in registry["domains"].items():
        assert (SKILLS_DIR / entry["overlay"]).is_file(), name
        assert set(entry["applies_to_skills"]) <= set(SKILL_PRODUCES), name


def test_planned_skills_not_registered():
    planned = SKILLS_DIR / "planned"
    for path in planned.glob("*/SKILL.md"):
        assert path.parent.name not in SKILL_PRODUCES
