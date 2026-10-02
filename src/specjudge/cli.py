"""SpecJudge command-line interface (contracts/cli.md).

Orchestrates the flow: read artifacts -> classify state -> resolve local judge ->
estimate demand -> rate the catalog -> recommend -> render. Degradation is explicit,
with distinct exit codes (Principle IV).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import typer

from . import __version__, errors
from .artifacts import read_project
from .bmad import BmadStory, read_bmad_story, ready_for_dev
from .catalog import check_freshness, load_catalog
from .config import DEFAULT_HOST, load_config, save_config
from .domain import DataState, ExecutionModel, JudgePreference, UserConfig
from .envelope import build as build_envelope
from .envelope import envelope_warnings
from .gaps import find_gaps
from .judge.client import JudgeClient, build_client, remote_notice
from .judge.evaluator import (
    budget_warnings,
    envelope_fragments,
    estimate_demand,
    evidence_warnings,
)
from .rating import assert_dimensions_match, evaluate_all, load_rules
from .recommend import build_comparison
from .render.html import open_in_browser
from .render.table import render_comparison
from .serialize import comparison_to_dict, load_schema

app = typer.Typer(
    add_completion=False,
    help="Recommend the AI model that best fits your SDD project's complexity.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"specjudge {__version__}")
        raise typer.Exit()


def _print_schema_callback(value: bool) -> None:
    """Emit the output contract so a non-Python consumer can target it (FR-022)."""
    if value:
        typer.echo(json.dumps(load_schema(), indent=2))
        raise typer.Exit()


def resolve_judge(
    config: UserConfig,
    client: JudgeClient,
    *,
    forced_judge: str | None,
    set_judge: bool,
    interactive: bool,
) -> tuple[str, UserConfig]:
    """Determine the judge model (FR-012/013). May raise JudgeUnavailableError."""
    # 1. --judge forces the model for this run (not persisted).
    if forced_judge:
        client.ensure_available(forced_judge)
        return forced_judge, config

    # 2. Valid saved preference (unless --set-judge is requested).
    if not set_judge and config.judge_preference is not None:
        wanted = config.judge_preference.judge_model
        models = client.list_models()
        if not models:
            raise errors.ollama_no_models(client.host)
        if wanted in models:
            return wanted, config
        # Preference points at an uninstalled model: re-select.
        if not interactive:
            raise errors.selected_model_missing(wanted)

    # 3. Interactive selection on first run or --set-judge.
    models = client.ensure_available()
    if not interactive:
        raise errors.JudgeUnavailableError(
            "No judge model is configured and the session is not interactive.",
            hint="Run again in an interactive terminal, or use --judge <model>.",
        )
    chosen = _prompt_judge_selection(models)
    new_config = dataclasses.replace(
        config, judge_preference=JudgePreference(judge_model=chosen, chosen_at=None)
    )
    save_config(new_config)
    return chosen, new_config


def _prompt_judge_selection(models: list[str]) -> str:
    typer.echo("Local models available to act as judge:")
    for i, m in enumerate(models, start=1):
        typer.echo(f"  {i}. {m}")
    while True:
        raw = typer.prompt("Pick the number of the judge model")
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except ValueError:
            pass
        typer.echo("Invalid selection, please try again.")


@app.command()
def main(
    project_path: Path = typer.Argument(Path("."), help="Root of the SDD project to analyze."),
    open_browser: bool = typer.Option(
        False, "--open", "-o", help="Also open the HTML matrix in your browser."
    ),
    judge: str | None = typer.Option(
        None, "--judge", help="Force the judge model for this run (not persisted)."
    ),
    judge_url: str | None = typer.Option(
        None,
        "--judge-url",
        help=(
            "Base URL of an OpenAI-compatible judge endpoint for this run (not "
            "persisted). The judge is local Ollama unless you set this. A non-local "
            "URL sends your project's text to that endpoint, and the run says so."
        ),
    ),
    set_judge: bool = typer.Option(
        False, "--set-judge", help="Re-run judge selection and save it."
    ),
    catalog: Path | None = typer.Option(
        None, "--catalog", help="Alternative model catalog (YAML)."
    ),
    execution_model: str = typer.Option(
        None,
        "--execution-model",
        help=(
            "How you will implement this: 'single' (one model for everything, ranks by "
            "the hardest part) or 'escalating' (you can switch model per task, ranks by "
            "the bulk and lists what to escalate for)."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON on stdout."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable color/highlighting."),
    print_schema: bool = typer.Option(
        False,
        "--print-schema",
        callback=_print_schema_callback,
        is_eager=True,
        help="Print the JSON Schema of the --json output and exit.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Analyze the project and show the model comparison."""
    try:
        _run(
            project_path,
            open_browser,
            judge,
            set_judge,
            catalog,
            as_json,
            no_color,
            execution_model,
            judge_url,
        )
    except errors.SpecJudgeError as exc:
        typer.echo(exc.render(), err=True)
        raise typer.Exit(code=exc.exit_code) from exc


def _resolve_execution_model(raw: str | None, rules) -> ExecutionModel:
    """The execution model for this run: the flag, else the configured default.

    An unrecognised value is refused rather than quietly falling back. The two
    readings produce different recommendations, so guessing which one the user meant
    is exactly the silent choice Principle IV rules out.
    """
    if raw is None:
        return rules.execution_model
    try:
        return ExecutionModel(raw.strip().lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in ExecutionModel)
        raise errors.SpecJudgeError(
            f"Unknown execution model '{raw}'.",
            hint=(
                f"Use one of: {valid}. `single` assumes one model implements everything "
                "and ranks on the hardest part; `escalating` assumes you can switch "
                "model per task and ranks on the bulk."
            ),
        ) from exc


def _run(
    project_path: Path,
    open_browser: bool,
    judge: str | None,
    set_judge: bool,
    catalog: Path | None,
    as_json: bool,
    no_color: bool,
    execution_model: str | None = None,
    judge_url: str | None = None,
) -> None:
    rules = load_rules()
    execution = _resolve_execution_model(execution_model, rules)
    stories = ready_for_dev(project_path)
    if stories is not None:
        _run_bmad(
            project_path,
            stories,
            rules,
            execution,
            open_browser,
            judge,
            set_judge,
            catalog,
            as_json,
            no_color,
            judge_url,
        )
        return

    # 1. Read and classify the artifacts.
    analysis = read_project(project_path, rules)
    comparison, gaps = _assess(
        analysis,
        rules,
        execution,
        *_prepare(judge, set_judge, catalog, judge_url, rules),
    )
    _emit(comparison, gaps, as_json=as_json, no_color=no_color, open_browser=open_browser)


def _run_bmad(
    project_path: Path,
    stories: list[BmadStory],
    rules,
    execution: ExecutionModel,
    open_browser: bool,
    judge: str | None,
    set_judge: bool,
    catalog: Path | None,
    as_json: bool,
    no_color: bool,
    judge_url: str | None,
) -> None:
    """One recommendation per ready-for-dev story. The judge is resolved once."""
    if not stories:
        raise errors.InsufficientInfoError(
            "No BMAD story is marked ready-for-dev.",
            hint=(
                "sprint-status.yaml was found, and none of its stories have status "
                "ready-for-dev. Create the next story, or mark one ready, and run again."
            ),
        )

    prepared = _prepare(judge, set_judge, catalog, judge_url, rules)
    judged = []
    skipped: list[tuple[str, str, str]] = []
    for story in stories:
        analysis = read_bmad_story(project_path, story, rules)
        rel = _relative(project_path, story.path)
        try:
            comparison, gaps = _assess(analysis, rules, execution, *prepared)
        except errors.InsufficientInfoError as exc:
            skipped.append((story.key, rel, exc.message))
            continue
        except errors.JudgeUnavailableError as exc:
            # A story the judge cannot score is that story's problem. The judge
            # being down, or too old, is the run's problem.
            if "usable assessment" not in exc.message:
                raise
            skipped.append((story.key, rel, exc.message))
            continue
        judged.append((story.key, rel, comparison, gaps))

    if not judged:
        detail = "\n".join(f"  - {key} ({path}): {message}" for key, path, message in skipped)
        raise errors.InsufficientInfoError(
            "No ready-for-dev BMAD story could be judged.",
            hint=detail or "The story files were missing or too thin to assess.",
        )

    if as_json:
        payload = {
            "bmad": [
                {
                    "story": key,
                    "file": path,
                    "comparison": comparison_to_dict(comparison),
                }
                for key, path, comparison, _gaps in judged
            ]
        }
        if skipped:
            payload["skipped"] = [
                {"story": key, "file": path, "error": message} for key, path, message in skipped
            ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, path, message in skipped:
            typer.echo(f"Skipped {key} ({path}): {message}", err=True)
        for key, path, comparison, gaps in judged:
            typer.echo(f"\nBMAD story {key}")
            typer.echo(path)
            _emit(
                comparison,
                gaps,
                as_json=False,
                no_color=no_color,
                open_browser=open_browser,
            )


def _prepare(
    judge: str | None,
    set_judge: bool,
    catalog: Path | None,
    judge_url: str | None,
    rules,
):
    """Catalog and judge, shared by a spec-kit run and every BMAD story in a run."""
    models, catalog_warnings = load_catalog(catalog)
    if not models:
        raise errors.catalog_empty(str(catalog) if catalog else "data/models.yaml")
    assert_dimensions_match(models, rules)
    catalog_warnings += check_freshness(models, rules.max_pricing_age_days)

    config = load_config()
    client = build_client(config, base_url=judge_url, default_host=DEFAULT_HOST)
    judge_model, _config = resolve_judge(
        config,
        client,
        forced_judge=judge,
        set_judge=set_judge,
        interactive=sys.stdin.isatty(),
    )
    return models, catalog_warnings, client, judge_model


def _assess(analysis, rules, execution, models, catalog_warnings, client, judge_model):
    """Demand, envelope, and the ranked comparison for one body of work."""
    if analysis.data_state == DataState.INSUFFICIENT:
        raise errors.insufficient_project()

    demand = estimate_demand(analysis, rules, client, judge_model)

    # No dimension the judge could ground means no demand profile to rank against.
    # That is the same "not enough information" outcome as a project without tasks.
    if not demand.scored_dimensions:
        raise errors.no_supported_dimensions(judge_model)

    fragments = envelope_fragments(analysis, rules, client, judge_model)
    envelope = build_envelope(demand, fragments, execution)
    evaluations = evaluate_all(models, demand, rules, envelope.default_demand)

    warnings = (
        list(analysis.warnings)
        + list(catalog_warnings)
        + evidence_warnings(demand)
        + envelope_warnings(demand, execution)
        + budget_warnings(analysis, rules, client, judge_model)
    )
    # Ahead of every other warning: it is the only one about where the project's
    # text went rather than about the quality of the answer (issue #4).
    notice = remote_notice(client)
    if notice:
        warnings.insert(0, notice)
    # Ungrounded dimensions make the assessment thinner than it looks, which is
    # exactly what `scarce` communicates (FR-020).
    data_state = analysis.data_state
    if demand.unsupported_dimensions and data_state == DataState.SUFFICIENT:
        data_state = DataState.SCARCE

    comparison = build_comparison(
        evaluations,
        data_state,
        judge_model,
        warnings=warnings,
        demand=demand,
        source_kinds=analysis.source_kinds,
        environment_only=analysis.environment_only,
        envelope=envelope,
    )

    # 5. Output.
    # Gaps are a presentation concern for a thin project, so they are computed here
    # and passed to the renderers rather than added to Comparison: the payload
    # carries the assessment, not editorial advice about how to improve it.
    gaps = find_gaps(analysis, rules) if data_state == DataState.SCARCE else []

    return comparison, gaps


def _emit(comparison, gaps, *, as_json: bool, no_color: bool, open_browser: bool) -> None:
    if as_json:
        typer.echo(json.dumps(comparison_to_dict(comparison), ensure_ascii=False, indent=2))
    else:
        render_comparison(comparison, no_color=no_color, gaps=gaps)

    if open_browser:
        out = open_in_browser(comparison, gaps=gaps)
        if not as_json:
            typer.echo(f"\nHTML matrix: {out}")


def _relative(project_path: Path, path: Path) -> str:
    try:
        return path.relative_to(project_path).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    app()
