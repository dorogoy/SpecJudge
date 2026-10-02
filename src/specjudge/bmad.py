"""BMAD stories that are ready to implement.

BMAD tracks the sprint in `_bmad-output/implementation-artifacts/sprint-status.yaml`.
Story keys look like `1-2-account-management`; epics (`epic-1`) and retrospectives
are not stories. A story marked `ready-for-dev` is a markdown file named after
that key, normally beside the status file.

SpecJudge judges each of those files on its own. The story text is the spec. Its
checklist, when it has one, is the task list. The rest of the repository's context
files are still read, the same way a spec-kit run reads them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import errors
from .artifacts import (
    _classify,
    _count_detailed_tasks,
    _decide_instructions,
    _read_environment,
)
from .domain import DataState, ProjectAnalysis, RatingRules, SDDArtifact

SPRINT_STATUS = Path("_bmad-output") / "implementation-artifacts" / "sprint-status.yaml"
READY_STATUS = "ready-for-dev"
# `1-2-account-management`. Epic keys (`epic-1`) and retrospectives do not match.
_STORY_KEY = re.compile(r"^\d+-\d+-\S+$")
_CHECKLIST = re.compile(r"^\s*-\s*\[[ xX]\]\s+\S.*$", re.MULTILINE)


@dataclass(frozen=True)
class BmadStory:
    key: str
    path: Path


def sprint_status_path(project_path: Path) -> Path:
    return project_path / SPRINT_STATUS


def ready_for_dev(project_path: Path) -> list[BmadStory] | None:
    """Stories marked ready-for-dev, in file order.

    None when this project has no sprint-status file, so the caller keeps the
    spec-kit path. An empty list means the file exists and nothing is ready.
    """
    status_file = sprint_status_path(project_path)
    if not status_file.is_file():
        return None

    try:
        raw = yaml.safe_load(status_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise errors.SpecJudgeError(
            f"Could not read {status_file}.",
            hint=(
                "sprint-status.yaml must be valid YAML with a development_status map. "
                f"Parser said: {exc}"
            ),
        ) from exc

    data = raw if isinstance(raw, dict) else {}
    development = data.get("development_status")
    if not isinstance(development, dict):
        return []

    stories: list[BmadStory] = []
    for key, status in development.items():
        if not isinstance(key, str) or _STORY_KEY.match(key) is None:
            continue
        if not isinstance(status, str) or status.strip().lower() != READY_STATUS:
            continue
        stories.append(BmadStory(key=key, path=_story_file(project_path, status_file, data, key)))
    return stories


def read_bmad_story(project_path: Path, story: BmadStory, rules: RatingRules) -> ProjectAnalysis:
    """One story as the work to judge, plus the repository's context files."""
    project_path = Path(project_path)
    rel = _display(project_path, story.path)
    warnings = [f"Judging BMAD story {story.key} ({rel})."]

    text = _read_story(story.path)
    if not text or not text.strip():
        why = "is empty" if text is not None else "was not found or is unreadable"
        return ProjectAnalysis(
            artifacts=[],
            data_state=DataState.INSUFFICIENT,
            warnings=warnings + [f"Story file {rel} {why}."],
            root=str(project_path),
        )

    spec = SDDArtifact(type="spec", path=str(story.path), present=True, readable=True, content=text)
    checklist = "\n".join(_CHECKLIST.findall(text))
    if checklist:
        tasks = SDDArtifact(
            type="tasks", path=str(story.path), present=True, readable=True, content=checklist
        )
        tasks.task_count = _count_detailed_tasks(checklist) or 1
    else:
        # ponytail: no checklist still counts as one task so the story text is judged.
        # Parse numbered acceptance criteria as tasks if stories stop using checklists.
        tasks = SDDArtifact(type="tasks", path=str(story.path), present=True, readable=True)
        tasks.task_count = 1 if text.strip() else 0
        warnings.append(f"Story {story.key} has no checklist tasks; judged from the story text.")

    absent = SDDArtifact(type="constitution", path="", present=False, readable=False)
    plan = SDDArtifact(type="plan", path="", present=False, readable=False)
    instructions = _decide_instructions(project_path, plan, tasks, rules.instructions_mode)
    applicable = [project_path / i.path for i in instructions if i.included]
    environment = _read_environment(project_path, rules.max_context_files, warnings, applicable)
    excluded = [i for i in instructions if not i.included]
    if excluded:
        names = ", ".join(f"{i.path} ({i.reason})" for i in excluded)
        warnings.append(
            f"Skipped {len(excluded)} path-specific instruction file(s): {names}. "
            f"Set sources.instructions: all in the rating rules to read them anyway."
        )

    return ProjectAnalysis(
        artifacts=[absent, spec, tasks, plan, *environment],
        data_state=_classify(absent, spec, tasks, environment, rules, warnings),
        warnings=warnings,
        root=str(project_path),
        instructions=instructions,
    )


def _read_story(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _story_file(project_path: Path, status_file: Path, data: dict, key: str) -> Path:
    """Where BMAD wrote the story. Prefer a real file over the declared directory."""
    name = f"{key}.md"
    candidates: list[Path] = []
    declared = _declared_dir(project_path, data.get("story_location"))
    if declared is not None:
        candidates.append(declared / name)
    candidates.append(status_file.parent / name)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _declared_dir(project_path: Path, raw: object) -> Path | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.replace("{project-root}", str(project_path)).strip().strip("\"'")
    if not cleaned or "{" in cleaned:
        return None
    path = Path(cleaned)
    if not path.is_absolute():
        path = project_path / path
    return path


def _display(project_path: Path, path: Path) -> str:
    try:
        return path.relative_to(project_path).as_posix()
    except ValueError:
        return path.as_posix()
