"""Compare catalog prices with models.dev and, when asked, write them back.

The recommendation command never calls this. The catalog stays a reviewed file:
this only proposes a diff, and it only edits `input_per_million`,
`output_per_million` and `pricing_date` on rows that opted in with `price_source`.

models.dev publishes each lab's own list rate as USD per million tokens. The base
`cost.input` / `cost.output` is the figure the catalog stores. Context tiers and
cache rates are a different price, so they are not copied.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import typer
import yaml

from .catalog import default_catalog_path, load_catalog
from .errors import CatalogError, SpecJudgeError

MODELS_DEV_URL = "https://models.dev/api.json"

# A move this large, or a model the source marks deprecated, is shown and skipped
# until the caller passes --accept. 25% is enough to catch a real repricing and
# to ignore a rounding difference.
LARGE_MOVE = 0.25

_LOCAL_PROVIDER = "Local / Ollama"
_BLOCKING = frozenset({"unmapped", "local", "missing", "review", "current", "currency"})
_HELD = frozenset({"large", "deprecated"})


@dataclass(frozen=True)
class Proposal:
    model_id: str
    holds: frozenset[str]
    old_in: float | None = None
    old_out: float | None = None
    new_in: float | None = None
    new_out: float | None = None
    old_date: str | None = None
    new_date: str | None = None
    detail: str = ""

    @property
    def drift_label(self) -> str:
        if None in (self.old_in, self.old_out, self.new_in, self.new_out):
            return ""
        assert self.old_in is not None and self.old_out is not None
        assert self.new_in is not None and self.new_out is not None
        drift = max(_relative(self.old_in, self.new_in), _relative(self.old_out, self.new_out))
        if drift == math.inf:
            return "from 0"
        return f"{round(drift * 100)}%"


def can_write(proposal: Proposal, *, accept: bool) -> bool:
    """True when --write may change this row.

    `review` never writes: the catalog figure is a promotion or an introductory
    price, and the source's list rate is a different number on purpose.
    """
    if proposal.holds & _BLOCKING:
        return False
    if not accept and proposal.holds & _HELD:
        return False
    return proposal.new_in is not None and proposal.new_out is not None


def fetch_upstream(url: str = MODELS_DEV_URL) -> dict:
    """GET the models.dev catalog. No API key."""
    try:
        response = httpx.get(url, timeout=30, headers={"User-Agent": "specjudge"})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise SpecJudgeError(
            f"Could not read model prices from {url}.",
            hint="Pass --source with a saved models.dev JSON file, or retry.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise SpecJudgeError(
            f"The response from {url} was not JSON.",
            hint="Pass --source with a saved models.dev JSON file.",
        ) from exc
    if not isinstance(payload, dict):
        raise SpecJudgeError(f"The response from {url} was not a provider map.")
    return payload


def load_upstream(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecJudgeError(f"Could not read price source {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpecJudgeError(f"Price source {path} must be a JSON object of providers.")
    return payload


def plan(catalog: dict, upstream: dict, today: date) -> list[Proposal]:
    models = catalog.get("models")
    if not isinstance(models, list):
        raise CatalogError("Catalog must declare a 'models' list.")
    return [_propose(model, upstream, today) for model in models if isinstance(model, dict)]


def render(proposals: list[Proposal], *, accept: bool) -> str:
    lines = [
        _describe(proposal, accept=accept)
        for proposal in proposals
        if not (proposal.holds & {"unmapped", "local"})
    ]
    local = [proposal.model_id for proposal in proposals if "local" in proposal.holds]
    unmapped = [proposal.model_id for proposal in proposals if "unmapped" in proposal.holds]
    if local:
        lines.append("local, left alone: " + ", ".join(local))
    if unmapped:
        lines.append("no price_source, left alone: " + ", ".join(unmapped))
    writing = sum(1 for proposal in proposals if can_write(proposal, accept=accept))
    # A review or non-USD row can also be a large move. It is not "held":
    # --accept will not write it.
    report_only = _BLOCKING - {"current", "unmapped", "local"}
    reported = sum(1 for proposal in proposals if proposal.holds & report_only)
    held = sum(
        1
        for proposal in proposals
        if proposal.holds & _HELD
        and not (proposal.holds & (_BLOCKING - {"current"}))
        and not can_write(proposal, accept=accept)
    )
    lines.append("")
    lines.append(
        f"{writing} to write, {held} held, {reported} report only, "
        f"{len(local) + len(unmapped)} left alone."
    )
    return "\n".join(lines)


def apply_text(text: str, proposals: list[Proposal], today: date) -> str:
    """Replace price lines inside the opted-in model blocks. Comments stay put.

    ponytail: first `input_per_million` / `output_per_million` / `pricing_date`
    line in the model block. The shipped catalog is multiline. A one-line
    `price: {…}` block fails the replace and is left unwritten, rather than
    round-tripped through a YAML dump that would strip the comments.
    """
    for proposal in proposals:
        text = _replace_block(text, proposal, today)
    return text


def refresh_readme(catalog_path: Path) -> bool:
    """Regenerate the README example when the written file is the repo catalog."""
    path = catalog_path.resolve()
    root = path.parent.parent
    script = root / "scripts" / "render_example.py"
    if path.parent.name != "data" or path.name != "models.yaml" or not script.is_file():
        return False
    subprocess.run([sys.executable, str(script), "--write"], check=True, cwd=root)
    return True


def run(
    *,
    catalog: Path | None,
    source: Path | None,
    write: bool,
    accept: bool,
    today: date | None = None,
) -> str:
    today = today or date.today()
    path = Path(catalog) if catalog is not None else default_catalog_path()
    load_catalog(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CatalogError(f"Catalog {path} must be a YAML mapping.")
    upstream = load_upstream(source) if source is not None else fetch_upstream()
    proposals = plan(raw, upstream, today)
    report = render(proposals, accept=accept)
    if not write:
        return report + "\n\nNo file written. Pass --write to apply the rows marked write."

    chosen = [proposal for proposal in proposals if can_write(proposal, accept=accept)]
    original = path.read_text(encoding="utf-8")
    updated = apply_text(original, chosen, today)
    if updated == original:
        return report + "\n\nCatalog already matches the rows this command would write."
    path.write_text(updated, encoding="utf-8")
    note = f"\n\nUpdated {path}."
    try:
        if refresh_readme(path):
            note += "\nREADME example regenerated."
    except subprocess.CalledProcessError as exc:
        raise SpecJudgeError(
            f"Prices were written to {path}, and the README example was not regenerated.",
            hint="Run `uv run python scripts/render_example.py --write`.",
        ) from exc
    return report + note


def _propose(model: dict, upstream: dict, today: date) -> Proposal:
    model_id = str(model.get("id") or "?")
    if model.get("provider") == _LOCAL_PROVIDER:
        return Proposal(model_id, frozenset({"local"}))

    source = model.get("price_source")
    if not isinstance(source, dict) or not source.get("provider") or not source.get("model"):
        return Proposal(model_id, frozenset({"unmapped"}))

    provider = str(source["provider"])
    upstream_id = str(source["model"])
    found = _lookup(upstream, provider, upstream_id)
    if found is None:
        return Proposal(
            model_id,
            frozenset({"missing"}),
            detail=f"{provider}/{upstream_id} is not in the price source",
        )
    cost = found.get("cost") if isinstance(found.get("cost"), dict) else {}
    if cost.get("input") is None or cost.get("output") is None:
        return Proposal(
            model_id,
            frozenset({"missing"}),
            detail=f"{provider}/{upstream_id} has no list price",
        )

    price = model.get("price") if isinstance(model.get("price"), dict) else {}
    old_in = float(price["input_per_million"])
    old_out = float(price["output_per_million"])
    new_in = float(cost["input"])
    new_out = float(cost["output"])
    old_date = _as_date(price.get("pricing_date"))
    currency = str(price.get("currency") or "")
    status = found.get("status")
    holds: set[str] = set()
    if source.get("review") is True:
        holds.add("review")
    if isinstance(status, str) and status.lower() == "deprecated":
        holds.add("deprecated")
    if currency != "USD":
        holds.add("currency")
    if _relative(old_in, new_in) > LARGE_MOVE or _relative(old_out, new_out) > LARGE_MOVE:
        holds.add("large")
    same_price = _same(old_in, new_in) and _same(old_out, new_out)
    if same_price and old_date == today.isoformat() and not holds:
        holds.add("current")
    return Proposal(
        model_id,
        frozenset(holds),
        old_in=old_in,
        old_out=old_out,
        new_in=new_in,
        new_out=new_out,
        old_date=old_date,
        new_date=today.isoformat(),
    )


def _lookup(upstream: dict, provider: str, model_id: str) -> dict | None:
    provider_row = upstream.get(provider)
    if not isinstance(provider_row, dict):
        return None
    models = provider_row.get("models")
    if not isinstance(models, dict):
        return None
    found = models.get(model_id)
    return found if isinstance(found, dict) else None


def _describe(proposal: Proposal, *, accept: bool) -> str:
    if "missing" in proposal.holds:
        return f"{proposal.model_id}: {proposal.detail}"
    assert proposal.old_in is not None and proposal.old_out is not None
    assert proposal.new_in is not None and proposal.new_out is not None
    if _same(proposal.old_in, proposal.new_in) and _same(proposal.old_out, proposal.new_out):
        rates = f"{_fmt(proposal.old_in)}/{_fmt(proposal.old_out)} unchanged"
    else:
        rates = (
            f"{_fmt(proposal.old_in)}/{_fmt(proposal.old_out)} -> "
            f"{_fmt(proposal.new_in)}/{_fmt(proposal.new_out)}"
        )
    tags: list[str] = []
    if "large" in proposal.holds:
        tags.append(f"large ({proposal.drift_label})")
    if "deprecated" in proposal.holds:
        tags.append("deprecated")
    if "review" in proposal.holds:
        tags.append("review, not written")
    if "currency" in proposal.holds:
        tags.append("not USD, not written")
    blocked = proposal.holds & (_BLOCKING - {"current"})
    if can_write(proposal, accept=accept):
        tags.append(f"date {proposal.old_date} -> {proposal.new_date}")
        verdict = "write"
    elif proposal.holds & _HELD and not blocked:
        verdict = "hold, pass --accept"
    elif "current" in proposal.holds:
        verdict = "current"
    else:
        verdict = "report only"
    suffix = f"  {', '.join(tags)}" if tags else ""
    return f"{proposal.model_id}: {rates}{suffix}  {verdict}"


def _replace_block(text: str, proposal: Proposal, today: date) -> str:
    assert proposal.new_in is not None and proposal.new_out is not None
    pattern = re.compile(
        rf"(?ms)^(  - id: {re.escape(proposal.model_id)}\n)(.*?)(?=^  - id: |\Z)"
    )
    match = pattern.search(text)
    if match is None:
        raise CatalogError(f"Could not find model '{proposal.model_id}' in the catalog text.")
    block = match.group(2)
    block, n_in = re.subn(
        r"^([ \t]*input_per_million:[ \t]*)\S+",
        rf"\g<1>{_fmt(proposal.new_in)}",
        block,
        count=1,
        flags=re.M,
    )
    block, n_out = re.subn(
        r"^([ \t]*output_per_million:[ \t]*)\S+",
        rf"\g<1>{_fmt(proposal.new_out)}",
        block,
        count=1,
        flags=re.M,
    )
    block, n_date = re.subn(
        r"^([ \t]*pricing_date:[ \t]*)\S+",
        rf"\g<1>{today.isoformat()}",
        block,
        count=1,
        flags=re.M,
    )
    if n_in != 1 or n_out != 1 or n_date != 1:
        raise CatalogError(
            f"Model '{proposal.model_id}' has no multiline price block to update.",
            hint="Edit input_per_million, output_per_million and pricing_date by hand.",
        )
    return text[: match.start(2)] + block + text[match.end(2) :]


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        return f"{value:.2f}"
    if len(text.split(".", 1)[1]) < 2:
        return f"{value:.2f}"
    return text


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=1e-9)


def _relative(old: float, new: float) -> float:
    if _same(old, new):
        return 0.0
    if old == 0:
        return math.inf
    return abs(new - old) / abs(old)


prices_app = typer.Typer(add_completion=False, no_args_is_help=False)


@prices_app.command()
def prices(
    catalog: Path | None = typer.Option(
        None, "--catalog", help="Catalog YAML to read. Writes this file."
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="models.dev JSON file. Omit to fetch https://models.dev/api.json.",
    ),
    write: bool = typer.Option(False, "--write", help="Write the rows the report marks write."),
    accept: bool = typer.Option(
        False,
        "--accept",
        help="With --write, also apply moves over 25% and deprecated models.",
    ),
) -> None:
    """Compare catalog prices with models.dev. Writes the catalog only with --write.

    Updates input_per_million, output_per_million and pricing_date on rows that
    declare price_source. Does not change capabilities, notes, local rows, or
    rows marked review: true.
    """
    try:
        typer.echo(run(catalog=catalog, source=source, write=write, accept=accept))
    except SpecJudgeError as exc:
        typer.echo(exc.render(), err=True)
        raise typer.Exit(code=exc.exit_code) from exc


def _as_date(value: object) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)
