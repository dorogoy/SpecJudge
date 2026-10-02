"""A BMAD sprint yields one recommendation per ready-for-dev story."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from specjudge.cli import app

runner = CliRunner()

_STORY = """\
# Story: Account management

As a user I want to manage my account so that I can update my profile.

## Tasks / Subtasks

- [ ] Add the account settings page in src/pages/account.tsx
- [ ] Validate the display name on the client and the server
- [ ] Add confirmation for email changes in src/api/users.ts
"""


def _project(tmp_path):
    folder = tmp_path / "_bmad-output" / "implementation-artifacts"
    folder.mkdir(parents=True)
    (folder / "sprint-status.yaml").write_text(
        "development_status:\n"
        "  epic-1: in-progress\n"
        "  1-1-done-story: done\n"
        "  1-2-account-management: ready-for-dev\n"
        "  1-3-missing-file: ready-for-dev\n"
        "  2-1-billing-export: ready-for-dev\n",
        encoding="utf-8",
    )
    (folder / "1-2-account-management.md").write_text(_STORY, encoding="utf-8")
    (folder / "2-1-billing-export.md").write_text(
        _STORY.replace("Account", "Billing"), encoding="utf-8"
    )
    return tmp_path


def test_each_ready_story_gets_a_recommendation(tmp_path, mock_ollama, test_catalog):
    project = _project(tmp_path)
    with mock_ollama(models=["llama3.1:8b"]):
        result = runner.invoke(
            app,
            [str(project), "--judge", "llama3.1:8b", "--json", "--catalog", str(test_catalog)],
        )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["story"] for item in payload["bmad"]] == [
        "1-2-account-management",
        "2-1-billing-export",
    ]
    assert payload["bmad"][0]["comparison"]["best_choice"]
    assert payload["bmad"][1]["comparison"]["best_choice"]
    assert payload["skipped"][0]["story"] == "1-3-missing-file"


def test_text_output_names_each_story(tmp_path, mock_ollama, test_catalog):
    project = _project(tmp_path)
    with mock_ollama(models=["llama3.1:8b"]):
        result = runner.invoke(
            app,
            [str(project), "--judge", "llama3.1:8b", "--no-color", "--catalog", str(test_catalog)],
        )
    assert result.exit_code == 0, result.output
    assert "BMAD story 1-2-account-management" in result.output
    assert "BMAD story 2-1-billing-export" in result.output
    assert "Skipped 1-3-missing-file" in result.output


def test_no_ready_story_exits_2(tmp_path, mock_ollama, test_catalog):
    folder = tmp_path / "_bmad-output" / "implementation-artifacts"
    folder.mkdir(parents=True)
    (folder / "sprint-status.yaml").write_text(
        "development_status:\n  1-1-user-authentication: done\n",
        encoding="utf-8",
    )
    with mock_ollama(models=["llama3.1:8b"]):
        result = runner.invoke(
            app,
            [str(tmp_path), "--judge", "llama3.1:8b", "--catalog", str(test_catalog)],
        )
    assert result.exit_code == 2
    assert "ready-for-dev" in result.output
