"""BMAD sprint-status: which stories are ready, and how one becomes a project."""

from __future__ import annotations

from specjudge.bmad import read_bmad_story, ready_for_dev
from specjudge.domain import DataState
from specjudge.rating import load_rules

_STATUS = """\
# STATUS DEFINITIONS:
# - ready-for-dev: Story file created, ready for development

generated: 2026-03-04
project: demo
story_location: "{project-root}/_bmad-output/implementation-artifacts"

development_status:
  epic-1: in-progress
  1-1-user-authentication: done
  1-2-account-management: ready-for-dev
  1-3-profile-settings: backlog
  epic-1-retrospective: optional
  2-1-content-creation: ready-for-dev
"""

_STORY = """\
# Story 1.2: Account management

As a user I want to manage my account so that I can update my profile.

## Tasks / Subtasks

- [ ] Add the account settings page in src/pages/account.tsx
- [ ] Validate the display name on the client and the server
- [ ] Add confirmation for email changes in src/api/users.ts
"""


def _sprint(tmp_path, text=_STATUS):
    path = tmp_path / "_bmad-output" / "implementation-artifacts" / "sprint-status.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.parent


def test_no_sprint_file_means_not_bmad(tmp_path):
    assert ready_for_dev(tmp_path) is None


def test_ready_stories_skip_epics_and_other_statuses(tmp_path):
    folder = _sprint(tmp_path)
    (folder / "1-2-account-management.md").write_text(_STORY, encoding="utf-8")
    (folder / "2-1-content-creation.md").write_text(_STORY, encoding="utf-8")

    stories = ready_for_dev(tmp_path)
    assert [s.key for s in stories] == ["1-2-account-management", "2-1-content-creation"]
    assert stories[0].path == folder / "1-2-account-management.md"


def test_story_location_is_used_when_the_file_is_there(tmp_path):
    folder = _sprint(
        tmp_path,
        _STATUS.replace(
            'story_location: "{project-root}/_bmad-output/implementation-artifacts"',
            'story_location: "{project-root}/stories"',
        ),
    )
    stories_dir = tmp_path / "stories"
    stories_dir.mkdir()
    (stories_dir / "1-2-account-management.md").write_text(_STORY, encoding="utf-8")
    (folder / "2-1-content-creation.md").write_text(_STORY, encoding="utf-8")

    found = {s.key: s.path for s in ready_for_dev(tmp_path)}
    assert found["1-2-account-management"] == stories_dir / "1-2-account-management.md"
    assert found["2-1-content-creation"] == folder / "2-1-content-creation.md"


def test_a_story_with_a_checklist_is_scarce_without_a_constitution(tmp_path):
    folder = _sprint(tmp_path)
    (folder / "1-2-account-management.md").write_text(_STORY, encoding="utf-8")
    story = ready_for_dev(tmp_path)[0]

    analysis = read_bmad_story(tmp_path, story, load_rules())
    assert analysis.data_state == DataState.SCARCE
    assert analysis.artifact("tasks").task_count == 3
    assert "account settings" in analysis.artifact("spec").content


def test_a_missing_story_file_is_insufficient(tmp_path):
    _sprint(tmp_path)
    story = ready_for_dev(tmp_path)[0]
    analysis = read_bmad_story(tmp_path, story, load_rules())
    assert analysis.data_state == DataState.INSUFFICIENT
