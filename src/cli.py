"""CLI entry point for PaperFlux."""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer

from .config import Config, load
from .orchestrator import batch_process

app = typer.Typer(add_completion=False)

@app.command(name="")  # Empty name makes this the default command
def main(
    pdfs: List[str] = typer.Argument(..., help="PDF files to analyze"),
    config: str = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    detail: str = typer.Option(None, "--detail", "-d", help="Detail level (low, medium, high)"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory to write outputs"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show per-file progress updates"),
    quotes_file: Optional[str] = typer.Option(None, "--quotes-file", help="Path to JSON quotes file to annotate without rerunning extraction"),
):
    """
    Analyze one or more PDF files and produce annotated PDFs and markdown summaries.
    """
    # Configure logging before any other imports or logic
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Convert to absolute paths
    config_path = Path(os.path.abspath(config))
    pdf_paths = [Path(os.path.abspath(pdf)) for pdf in pdfs]
    
    typer.echo(f"Load configuration from: {config_path}")
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
    if detail:
        cfg.ui.detail_level = detail

    output_dir_path: Optional[Path] = None
    if output_dir:
        output_dir_path = Path(os.path.abspath(output_dir))
        output_dir_path.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Outputs will be saved to: {output_dir_path}")

    if quotes_file:
        from .utils import finalize_output
        import json
        quotes_path = Path(os.path.abspath(quotes_file))
        if not quotes_path.exists():
            typer.echo(f"Quotes file {quotes_path} does not exist.")
            raise typer.Exit(code=1)
        try:
            quotes_payload = json.loads(quotes_path.read_text())
        except Exception as exc:
            typer.echo(f"Failed to read quotes file: {exc}", err=True)
            raise typer.Exit(code=1)
        quotes = quotes_payload.get("quotes") or quotes_payload
        md_note = quotes_payload.get("key_takeaways", "")
        pdf_path = pdf_paths[0]
        pdf_out, md_out, quotes_out = finalize_output(pdf_path, quotes, md_note, cfg, output_dir=output_dir_path)
        typer.echo(f"Annotated PDF saved to: {pdf_out}")
        typer.echo(f"Markdown summary saved to: {md_out}")
        typer.echo(f"Quotes JSON saved to: {quotes_out}")
        return

    typer.echo(f"Processing {len(pdf_paths)} file(s)...")
    try:
        results = asyncio.run(
            batch_process(
                pdf_paths,
                cfg,
                verbose,
                output_dir=output_dir_path,
                show_progress=progress,
            )
        )
        for pdf_out, md_out, quotes_out in results:
            typer.echo(f"Annotated PDF saved to: {pdf_out}")
            typer.echo(f"Markdown summary saved to: {md_out}")
            typer.echo(f"Quotes JSON saved to: {quotes_out}")
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)

if __name__ == "__main__":
    app(prog_name="PaperFlux")
