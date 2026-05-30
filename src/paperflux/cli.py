"""CLI entry point for PaperFlux."""

import asyncio
import json
import logging
import os
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import List, Optional

import typer
from pydantic import ValidationError

from .config import Config, load
from .orchestrator import batch_process

app = typer.Typer(add_completion=False)
_COMMANDS = {"run", "init"}
_INIT_TEMPLATE_FILES = (
    ("config.yaml", "config.yaml"),
    ("prompts/rag_category_prompt.j2", "prompts/rag_category_prompt.j2"),
    ("prompts/rag_category_system_prompt.txt", "prompts/rag_category_system_prompt.txt"),
    (
        "prompts/rag_category_system_prompt_anthropic.txt",
        "prompts/rag_category_system_prompt_anthropic.txt",
    ),
    ("prompts/rag_summary_prompt.j2", "prompts/rag_summary_prompt.j2"),
)


def _apply_cli_overrides(cfg: Config, *, detail: Optional[str] = None) -> Config:
    """Apply CLI overrides by re-validating the Pydantic config model."""
    if detail is None:
        return cfg

    cfg_data = cfg.model_dump()
    cfg_data["ui"]["detail_level"] = detail
    updated_cfg = Config(**cfg_data)
    updated_cfg._config_dir = getattr(cfg, "_config_dir", None)
    return updated_cfg


def _echo_section(title: str) -> None:
    typer.echo()
    typer.echo(title)


def _format_plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _entrypoint_args(args: List[str]) -> List[str]:
    if args and args[0] not in _COMMANDS and args[0] not in {"--help", "-h"}:
        return ["run", *args]
    return args


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _StageProgress:
    def __init__(self) -> None:
        self._started_at = time.monotonic()

    def __call__(self, message: str) -> None:
        elapsed = _format_elapsed(time.monotonic() - self._started_at)
        typer.echo(f"  {elapsed} {message}")


def _echo_run_context(
    *,
    config_path: Path,
    pdf_paths: List[Path],
    output_dir: Optional[Path],
    quotes_path: Optional[Path],
) -> None:
    typer.echo("PaperFlux")
    _echo_section("Input")
    typer.echo(f"- Config: {config_path}")
    typer.echo(f"- PDFs: {_format_plural(len(pdf_paths), 'file')}")
    if len(pdf_paths) == 1:
        typer.echo(f"- PDF: {pdf_paths[0]}")
    if quotes_path:
        typer.echo("- Mode: annotate from saved quotes")
        typer.echo(f"- Quotes file: {quotes_path}")
    else:
        typer.echo("- Mode: extract quotes and annotate")
    if output_dir:
        typer.echo(f"- Output directory: {output_dir}")
    elif len(pdf_paths) == 1:
        typer.echo(f"- Output directory: {pdf_paths[0].parent}")
    else:
        typer.echo("- Output directory: source PDF directories")


def _echo_output_paths(
    pdf_out: Path,
    md_out: Path,
    quotes_out: Path,
    match_report_out: Path,
) -> None:
    _echo_section("Outputs")
    typer.echo(f"- Annotated PDF: {pdf_out}")
    typer.echo(f"- Markdown summary: {md_out}")
    typer.echo(f"- Quotes JSON: {quotes_out}")
    typer.echo(f"- Quote match report: {match_report_out}")


def _echo_quote_match_report(report_path: Path, *, verbose: bool = False) -> None:
    """Print a concise quote-match report to the terminal."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Failed to read quote match report: {exc}", err=True)
        return

    matched = int(report.get("matched", 0))
    skipped = int(report.get("skipped", 0))
    total = int(report.get("total", matched + skipped))

    records = report.get("records") or []
    matched_records = [record for record in records if record.get("matched")]
    skipped_records = [record for record in records if not record.get("matched")]
    method_counts: dict[str, int] = {}
    for record in matched_records:
        method = str(record.get("method") or "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    _echo_section("Quote Matches")
    typer.echo(f"- Summary: {matched}/{total} matched, {skipped} skipped")
    if method_counts:
        methods = ", ".join(
            f"{method} {count}" for method, count in sorted(method_counts.items())
        )
        typer.echo(f"- Methods: {methods}")

    layout_gap_records = [
        record for record in matched_records if record.get("method") == "layout-gap"
    ]
    if verbose and layout_gap_records:
        typer.echo("Layout-gap matches:")
        for record in layout_gap_records:
            segments = int(record.get("segments") or 0)
            segment_text = f", {segments} segments" if segments else ""
            typer.echo(
                f"- {record['category']} #{record['quote_index']}: "
                f"p. {record['page']}, score {record['score']:.3f}{segment_text}"
            )

    if skipped_records:
        typer.echo("Skipped quotes:")
        for record in skipped_records:
            reason = record.get("skipped_reason") or "not matched"
            typer.echo(
                f"- {record['category']} #{record['quote_index']} ({reason}): "
                f"{record['text']}"
            )

    if verbose and matched_records:
        typer.echo("Matched quotes:")
        for record in matched_records:
            segments = int(record.get("segments") or 0)
            segments_suffix = (
                f", segments {segments}"
                if record.get("method") == "layout-gap" and segments > 1
                else ""
            )
            typer.echo(
                f"- {record['category']} #{record['quote_index']}: "
                f"p. {record['page']}, {record['method']}, "
                f"score {record['score']:.3f}{segments_suffix}"
            )


def _write_init_templates(target_dir: Path, *, force: bool = False) -> None:
    template_root = files("paperflux").joinpath("templates")
    planned_files = [
        (template_root.joinpath(source), target_dir / destination)
        for source, destination in _INIT_TEMPLATE_FILES
    ]
    existing_files = [destination for _, destination in planned_files if destination.exists()]
    if existing_files and not force:
        typer.echo("Refusing to overwrite existing files:", err=True)
        for path in existing_files:
            typer.echo(f"- {path}", err=True)
        typer.echo("Use --force to overwrite them.", err=True)
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in planned_files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


@app.command()
def init(
    target_dir: Path = typer.Argument(
        Path("."),
        help="Directory where config.yaml and prompt templates should be created",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """Create a starter PaperFlux config and prompt templates."""
    target_dir = target_dir.resolve()
    _write_init_templates(target_dir, force=force)
    typer.echo(f"Initialized PaperFlux project in: {target_dir}")
    typer.echo(f"- Config: {target_dir / 'config.yaml'}")
    typer.echo(f"- Prompts: {target_dir / 'prompts'}")


@app.command("run")
def main(
    pdfs: List[str] = typer.Argument(..., help="PDF files to analyze"),
    config: str = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    detail: str = typer.Option(None, "--detail", "-d", help="Detail level (low, medium, high)"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory to write outputs"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show stage-level progress updates"),
    quotes_file: Optional[str] = typer.Option(None, "--quotes-file", help="Path to JSON quotes file to annotate without rerunning extraction"),
):
    """
    Analyze one or more PDF files and produce annotated PDFs and markdown summaries.
    """
    # Configure logging before any other imports or logic.
    log_level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )

    # Convert to absolute paths
    config_path = Path(os.path.abspath(config))
    pdf_paths = [Path(os.path.abspath(pdf)) for pdf in pdfs]
    
    if not config_path.exists():
        typer.echo(f"Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)
    
    # Verify PDF files exist
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            typer.echo(f"PDF file {pdf_path} does not exist.")
            raise typer.Exit(code=1)
    
    try:
        cfg: Config = load(config_path)
    except ValueError as exc:
        typer.echo(f"Invalid configuration: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Failed to load configuration: {exc}", err=True)
        raise typer.Exit(code=1)
    try:
        cfg = _apply_cli_overrides(cfg, detail=detail)
    except ValidationError as exc:
        typer.echo(f"Invalid CLI override: {exc}", err=True)
        raise typer.Exit(code=1)

    output_dir_path: Optional[Path] = None
    if output_dir:
        output_dir_path = Path(os.path.abspath(output_dir))
        output_dir_path.mkdir(parents=True, exist_ok=True)

    if quotes_file:
        from .utils import finalize_output
        if len(pdf_paths) != 1:
            typer.echo("--quotes-file can be used with exactly one PDF.", err=True)
            raise typer.Exit(code=1)
        quotes_path = Path(os.path.abspath(quotes_file))
        if not quotes_path.exists():
            typer.echo(f"Quotes file {quotes_path} does not exist.")
            raise typer.Exit(code=1)
        _echo_run_context(
            config_path=config_path,
            pdf_paths=pdf_paths,
            output_dir=output_dir_path,
            quotes_path=quotes_path,
        )
        _echo_section("Processing")
        typer.echo(f"- Annotating {pdf_paths[0].name}")
        progress_reporter = _StageProgress() if progress else None
        try:
            if progress_reporter:
                progress_reporter(f"Loading saved quotes from {quotes_path.name}")
            quotes_payload = json.loads(quotes_path.read_text())
        except Exception as exc:
            typer.echo(f"Failed to read quotes file: {exc}", err=True)
            raise typer.Exit(code=1)
        quotes = quotes_payload.get("quotes") or quotes_payload
        md_note = quotes_payload.get("key_takeaways", "")
        pdf_path = pdf_paths[0]
        try:
            pdf_out, md_out, quotes_out, match_report_out = finalize_output(
                pdf_path,
                quotes,
                md_note,
                cfg,
                output_dir=output_dir_path,
                progress_callback=progress_reporter,
            )
        except Exception as exc:
            typer.echo(f"Error during annotation: {exc}", err=True)
            raise typer.Exit(code=1)
        _echo_output_paths(pdf_out, md_out, quotes_out, match_report_out)
        _echo_quote_match_report(match_report_out, verbose=verbose)
        return

    _echo_run_context(
        config_path=config_path,
        pdf_paths=pdf_paths,
        output_dir=output_dir_path,
        quotes_path=None,
    )
    _echo_section("Processing")
    typer.echo(f"- Processing {_format_plural(len(pdf_paths), 'PDF')}")
    progress_reporter = _StageProgress() if progress else None
    try:
        results = asyncio.run(
            batch_process(
                pdf_paths,
                cfg,
                verbose,
                output_dir=output_dir_path,
                show_progress=progress,
                progress_callback=progress_reporter,
            )
        )
        for pdf_out, md_out, quotes_out, match_report_out in results:
            _echo_output_paths(pdf_out, md_out, quotes_out, match_report_out)
            _echo_quote_match_report(match_report_out, verbose=verbose)
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(code=1)

def run() -> None:
    """Console script entry point."""
    args = _entrypoint_args(sys.argv[1:])
    if args != sys.argv[1:]:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *args]
        try:
            app(prog_name="paperflux")
        finally:
            sys.argv = original_argv
        return
    app(prog_name="paperflux")


if __name__ == "__main__":
    run()
