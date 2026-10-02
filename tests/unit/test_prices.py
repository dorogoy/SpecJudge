"""Catalog price check: models.dev in, three YAML fields out, and only when asked."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from specjudge.cli import main_entry
from specjudge.errors import CatalogError
from specjudge.prices import (
    LARGE_MOVE,
    can_write,
    plan,
    prices_app,
    refresh_readme,
    run,
)

runner = CliRunner()
TODAY = date(2026, 10, 2)
ROOT = Path(__file__).resolve().parents[2]

NO_SOURCE = {
    "gemma-4-31b",
    "gemma-4-26b-moe",
    "llama-4-scout",
    "mistral-large-3",
}
REVIEW = {"claude-sonnet-5", "qwen3-7-max"}
PROVIDERS = {"anthropic", "openai", "google", "zai", "deepseek", "moonshotai", "alibaba"}


def _catalog(*models: str) -> str:
    return "version: 1\ndimensions: [reasoning]\nmodels:\n" + "\n".join(models) + "\n"


def _row(
    model_id: str,
    inp: str = "1.00",
    out: str = "2.00",
    *,
    provider: str = "OpenAI",
    source: str | None = "    price_source:\n      provider: openai\n      model: demo\n",
    notes: str = "    notes: keep this\n",
) -> str:
    source_block = "" if source is None else source
    return (
        f"  - id: {model_id}\n"
        f"    name: {model_id}\n"
        f"    provider: {provider}\n"
        f"{source_block}"
        "    capabilities: {reasoning: low}\n"
        "    price:\n"
        f"      input_per_million: {inp}\n"
        f"      output_per_million: {out}\n"
        "      currency: USD\n"
        "      pricing_date: 2026-08-10\n"
        f"{notes}"
    )


def _upstream(provider: str, model: str, inp: float, out: float, *, status: str | None = None) -> dict:
    row: dict = {"cost": {"input": inp, "output": out, "cache_read": 0.01}}
    if status:
        row["status"] = status
    return {provider: {"models": {model: row}}}


def _write(tmp_path: Path, text: str, upstream: dict) -> tuple[Path, Path]:
    catalog = tmp_path / "models.yaml"
    catalog.write_text(text, encoding="utf-8")
    source = tmp_path / "upstream.json"
    source.write_text(json.dumps(upstream), encoding="utf-8")
    return catalog, source


def test_small_change_writes_only_the_price_lines(tmp_path):
    text = _catalog(
        "# section comment\n" + _row("priced"),
        _row("other", source=None, provider="Meta"),
    )
    catalog, source = _write(tmp_path, text, _upstream("openai", "demo", 1.10, 2.10))
    report = run(catalog=catalog, source=source, write=True, accept=False, today=TODAY)
    written = catalog.read_text(encoding="utf-8")
    assert "input_per_million: 1.10" in written
    assert "output_per_million: 2.10" in written
    assert "pricing_date: 2026-10-02" in written
    assert "notes: keep this" in written
    assert "# section comment" in written
    assert "reasoning: low" in written
    assert "id: other" in written
    assert "other" in report and "left alone" in report
    # The untouched row keeps its original date.
    assert written.count("pricing_date: 2026-08-10") == 1


def test_matching_price_bumps_the_verified_date(tmp_path):
    catalog, source = _write(
        tmp_path, _catalog(_row("priced", "4.00", "20.00")), _upstream("openai", "demo", 4, 20)
    )
    run(catalog=catalog, source=source, write=True, accept=False, today=TODAY)
    written = catalog.read_text(encoding="utf-8")
    assert "input_per_million: 4.00" in written
    assert "output_per_million: 20.00" in written
    assert "pricing_date: 2026-10-02" in written


def test_large_move_needs_accept(tmp_path):
    text = _catalog(_row("priced", "5.00", "30.00"))
    upstream = _upstream("openai", "demo", 4, 20)
    catalog, source = _write(tmp_path, text, upstream)
    report = run(catalog=catalog, source=source, write=True, accept=False, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "hold, pass --accept" in report
    assert "0 to write, 1 held, 0 report only, 0 left alone." in report

    run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    written = catalog.read_text(encoding="utf-8")
    assert "input_per_million: 4.00" in written
    assert "output_per_million: 20.00" in written


def test_exact_quarter_is_not_a_large_move():
    catalog = yaml.safe_load(
        _catalog(
            _row("priced", "1.00", "2.00").replace(
                "      output_per_million: 2.00\n", "      output_per_million: 1.00\n"
            )
        )
    )
    # 1.00 -> 1.25 is 25%, which is the threshold and not over it.
    proposals = plan(catalog, _upstream("openai", "demo", 1.25, 1.25), TODAY)
    assert "large" not in proposals[0].holds
    assert LARGE_MOVE == 0.25
    over = plan(catalog, _upstream("openai", "demo", 1.26, 1.25), TODAY)
    assert "large" in over[0].holds


def test_review_is_never_written(tmp_path):
    source_block = (
        "    price_source:\n      provider: alibaba\n      model: qwen3.7-max\n      review: true\n"
    )
    text = _catalog(_row("qwen", "1.25", "3.75", source=source_block))
    catalog, source = _write(tmp_path, text, _upstream("alibaba", "qwen3.7-max", 2.5, 7.5))
    report = run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "review, not written" in report
    assert "0 to write, 0 held, 1 report only, 0 left alone." in report
    (proposal,) = plan(yaml.safe_load(text), _upstream("alibaba", "qwen3.7-max", 2.5, 7.5), TODAY)
    assert can_write(proposal, accept=True) is False


def test_deprecated_needs_accept(tmp_path):
    text = _catalog(_row("old", "0.14", "0.28"))
    upstream = _upstream("openai", "demo", 0.15, 0.60, status="deprecated")
    catalog, source = _write(tmp_path, text, upstream)
    report = run(catalog=catalog, source=source, write=True, accept=False, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "deprecated" in report
    run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    assert "pricing_date: 2026-10-02" in catalog.read_text(encoding="utf-8")


def test_local_row_is_refused_even_if_it_has_a_source(tmp_path):
    text = _catalog(_row("gemma-local", "0.00", "0.00", provider="Local / Ollama"))
    catalog, source = _write(tmp_path, text, _upstream("openai", "demo", 0.12, 0.35))
    report = run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "local, left alone: gemma-local" in report


def test_missing_upstream_does_not_write(tmp_path):
    text = _catalog(_row("priced"))
    catalog, source = _write(tmp_path, text, {"openai": {"models": {}}})
    report = run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "not in the price source" in report


def test_from_zero_is_a_large_move():
    catalog = yaml.safe_load(_catalog(_row("free", "0.00", "0.00")))
    (proposal,) = plan(catalog, _upstream("openai", "demo", 1, 1), TODAY)
    assert "large" in proposal.holds
    assert proposal.drift_label == "from 0"


def test_non_usd_is_not_written(tmp_path):
    text = _catalog(_row("priced")).replace("currency: USD", "currency: EUR")
    catalog, source = _write(tmp_path, text, _upstream("openai", "demo", 1.1, 2.1))
    report = run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text
    assert "not USD" in report


def test_one_line_price_is_left_untouched(tmp_path):
    text = (
        "version: 1\ndimensions: [reasoning]\nmodels:\n"
        "  - id: priced\n"
        "    name: Priced\n"
        "    provider: OpenAI\n"
        "    price_source: {provider: openai, model: demo}\n"
        "    capabilities: {reasoning: low}\n"
        "    price: {input_per_million: 1, output_per_million: 2, currency: USD, "
        "pricing_date: 2026-08-10}\n"
    )
    catalog, source = _write(tmp_path, text, _upstream("openai", "demo", 1.1, 2.1))
    with pytest.raises(CatalogError, match="multiline"):
        run(catalog=catalog, source=source, write=True, accept=False, today=TODAY)
    assert catalog.read_text(encoding="utf-8") == text


def test_cache_rate_is_not_the_figure_that_is_written(tmp_path):
    """The source also carries cache and tier rates. The catalog stores the base rate."""
    upstream = {
        "openai": {
            "models": {
                "demo": {
                    "cost": {
                        "input": 4,
                        "output": 20,
                        "cache_read": 0.4,
                        "tiers": [{"input": 8, "output": 30}],
                    }
                }
            }
        }
    }
    catalog, source = _write(tmp_path, _catalog(_row("priced", "5.00", "30.00")), upstream)
    run(catalog=catalog, source=source, write=True, accept=True, today=TODAY)
    written = catalog.read_text(encoding="utf-8")
    assert "input_per_million: 4.00" in written
    assert "output_per_million: 20.00" in written
    assert "8.00" not in written
    assert "0.40" not in written


def test_command_reads_a_file_and_does_not_fetch(tmp_path, monkeypatch):
    catalog, source = _write(
        tmp_path, _catalog(_row("priced")), _upstream("openai", "demo", 1, 2)
    )

    def _boom(url: str = "") -> dict:
        raise AssertionError(f"fetched {url}")

    monkeypatch.setattr("specjudge.prices.fetch_upstream", _boom)
    result = runner.invoke(
        prices_app, ["--catalog", str(catalog), "--source", str(source), "--write"]
    )
    assert result.exit_code == 0, result.output
    assert "Updated" in result.output
    assert "README example" not in result.output


def test_report_does_not_write(tmp_path, monkeypatch):
    text = _catalog(_row("priced"))
    catalog, source = _write(tmp_path, text, _upstream("openai", "demo", 9, 9))
    monkeypatch.setattr("specjudge.prices.fetch_upstream", lambda url="": (_ for _ in ()).throw(AssertionError(url)))
    result = runner.invoke(prices_app, ["--catalog", str(catalog), "--source", str(source)])
    assert result.exit_code == 0, result.output
    assert "No file written" in result.output
    assert catalog.read_text(encoding="utf-8") == text


def test_prices_help_is_a_separate_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["specjudge", "prices", "--help"])
    with pytest.raises(SystemExit) as exc:
        main_entry()
    assert exc.value.code == 0
    assert "--write" in capsys.readouterr().out


def test_readme_regen_only_for_a_repo_catalog(tmp_path, monkeypatch):
    assert refresh_readme(tmp_path / "models.yaml") is False
    catalog = tmp_path / "data" / "models.yaml"
    catalog.parent.mkdir()
    catalog.write_text("x", encoding="utf-8")
    script = tmp_path / "scripts" / "render_example.py"
    script.parent.mkdir()
    script.write_text("# stub\n", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, check, cwd):
        calls.append((cmd, cwd))

    monkeypatch.setattr("specjudge.prices.subprocess.run", fake_run)
    assert refresh_readme(catalog) is True
    assert calls[0][1] == tmp_path
    assert calls[0][0][-1] == "--write"


def test_shipped_catalog_declares_which_rows_are_fetched():
    raw = yaml.safe_load((ROOT / "data" / "models.yaml").read_text(encoding="utf-8"))
    seen: set[str] = set()
    for model in raw["models"]:
        model_id = model["id"]
        seen.add(model_id)
        source = model.get("price_source")
        if model.get("provider") == "Local / Ollama" or model_id in NO_SOURCE:
            assert source is None, model_id
            continue
        assert isinstance(source, dict), model_id
        assert source.get("provider") in PROVIDERS
        assert isinstance(source.get("model"), str) and source["model"]
        if model_id in REVIEW:
            assert source.get("review") is True
        else:
            assert source.get("review") is not True
    assert REVIEW <= seen
    assert NO_SOURCE <= seen
