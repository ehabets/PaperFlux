"""
Orchestrator module for PaperFlux.
Coordinates the entire pipeline from PDF extraction to annotation.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from .config import Config
from .utils import finalize_output
from .assistants import analyze_pdf

logger = logging.getLogger(__name__)




async def run_pipeline(pdf_path: Path, cfg: Config, output_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Run the complete pipeline on a PDF file via Assistants API.
    """
    # Invoke Assistants workflow
    result = await analyze_pdf(pdf_path, cfg)
    md_note = result["key_takeaways"]
    quotes = result["quotes"]
    # Use the new utility function
    return finalize_output(pdf_path, quotes, md_note, cfg, output_dir=output_dir)


async def batch_process(
    pdf_paths: List[Path],
    cfg: Config,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> List[Tuple[Path, Path, Path]]:
    """
    Process multiple PDF files in sequence.
    
    Args:
        pdf_paths: List of paths to PDF files
        cfg: Application configuration
        verbose: Whether to enable verbose logging
        
    Returns:
        List[Tuple[Path, Path]]: List of output paths (PDF, markdown)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = []
    total = len(pdf_paths)
    for index, pdf_path in enumerate(pdf_paths, start=1):
        if show_progress:
            logger.info(f"[{index}/{total}] Processing {pdf_path.name}")
        handler = None
        if verbose:
            log_file = pdf_path.with_suffix('.log')
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)
        try:
            pdf_out, md_out, quotes_out = await run_pipeline(pdf_path, cfg, output_dir=output_dir)
            results.append((pdf_out, md_out, quotes_out))
        finally:
            if handler:
                logging.getLogger().removeHandler(handler)
                handler.close()
    
    return results
