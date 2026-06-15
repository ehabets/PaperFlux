"""
Orchestrator module for PaperFlux.
Coordinates the entire pipeline from PDF extraction to annotation.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import Config
from .utils import finalize_output
from .assistants import analyze_pdf

ProgressCallback = Callable[[str], None]
"""Signature for stage-level progress notification callbacks."""


async def run_pipeline(
    pdf_path: Path,
    cfg: Config,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Path, Path, Path, Path]:
    """Run the complete pipeline on a single PDF file.

    Delegates to the LLM provider configured in *cfg* to extract key takeaways
    and quotes, then writes all output artefacts via :func:`finalize_output`.

    Args:
        pdf_path: Path to the PDF file to process.
        cfg: Application configuration, including the active LLM provider.
        output_dir: Directory where output files are written. Defaults to the
            same directory as *pdf_path* when omitted.
        progress_callback: Optional callable invoked with a short status string
            at each major pipeline stage.

    Returns:
        A four-tuple of paths: the source PDF copy, the markdown notes file,
        the extracted quotes JSON, and the quote-match report JSON.
    """
    result = await analyze_pdf(pdf_path, cfg, progress_callback=progress_callback)
    md_note = result["key_takeaways"]
    quotes = result["quotes"]
    return finalize_output(
        pdf_path,
        quotes,
        md_note,
        cfg,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )


async def batch_process(
    pdf_paths: List[Path],
    cfg: Config,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    show_progress: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Tuple[Path, Path, Path, Path]]:
    """
    Process multiple PDF files in sequence.
    
    Args:
        pdf_paths: PDF files to process, handled one at a time in order.
        cfg: Application configuration, including the active LLM provider.
        verbose: Enables DEBUG-level logging and writes a per-PDF log file
            alongside each input file.
        output_dir: Directory where output files are written. Defaults to each
            PDF's own directory when omitted.
        show_progress: When False, suppresses all progress callback calls even
            if *progress_callback* is provided.
        progress_callback: Optional callable invoked with a short status string
            at each major pipeline stage.

    Returns:
        One four-tuple per input PDF: the source PDF copy, the markdown notes
        file, the extracted quotes JSON, and the quote-match report JSON.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = []
    total = len(pdf_paths)
    for index, pdf_path in enumerate(pdf_paths, start=1):
        emit_progress = progress_callback if show_progress else None
        if emit_progress:
            emit_progress(f"[{index}/{total}] Processing {pdf_path.name}")
        handler = None
        if verbose:
            log_file = pdf_path.with_suffix('.log')
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)
        try:
            pdf_out, md_out, quotes_out, match_report_out = await run_pipeline(
                pdf_path,
                cfg,
                output_dir=output_dir,
                progress_callback=emit_progress,
            )
            results.append((pdf_out, md_out, quotes_out, match_report_out))
        finally:
            if handler:
                logging.getLogger().removeHandler(handler)
                handler.close()
    
    return results
