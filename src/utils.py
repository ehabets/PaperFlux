"""
Utility functions for PaperFlux.
Handles PDF annotation and markdown generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .config import Config
from .io_pdf import annotate_pdf, save_markdown

def finalize_output(
    pdf_path: Path, 
    quotes: Dict[str, List[Any]], 
    md_note: str, 
    cfg: Config,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Annotates the PDF, builds the full markdown content, and saves both.

    Args:
        pdf_path: Path to the original PDF file.
        quotes: Dictionary of quotes by category.
        md_note: The base markdown note (e.g., key takeaways).
        cfg: Application configuration.
        output_dir: Optional directory where output files should be saved.

    Returns:
        Tuple[Path, Path]: Paths to the annotated PDF and the saved markdown file.
    """
    # Annotate PDF
    pdf_out = annotate_pdf(
        pdf_path,
        quotes,
        md_note,
        cfg=cfg,
        output_dir=output_dir,
    )
    
    # Build full markdown with key takeaways and quotes
    quote_counts_lines = "\n".join(
        f"- {category.capitalize()}: {len(items)} quote{'s' if len(items) != 1 else ''}"
        for category, items in quotes.items()
    )
    if not quote_counts_lines:
        quote_counts_lines = "- No quotes collected"

    full_md = (
        f"# Summary for {pdf_path.stem}\n\n"
        "## Key takeaways\n\n"
        f"{md_note.strip()}\n\n"
        "## Quote counts by category\n"
        f"{quote_counts_lines}\n\n"
        "## Exact quotations by category\n"
    )
    def _format_quote_entry(entry: Any) -> str:
        if isinstance(entry, dict):
            text_val = entry.get("text")
            if not isinstance(text_val, str):
                text_val = str(text_val)
            pages_val = entry.get("pages")
            page_suffix = ""
            if isinstance(pages_val, list) and pages_val:
                unique_pages = []
                for p in pages_val:
                    if isinstance(p, int) and p > 0 and p not in unique_pages:
                        unique_pages.append(p)
                if unique_pages:
                    if len(unique_pages) == 1:
                        page_suffix = f" (p. {unique_pages[0]})"
                    else:
                        page_list = ", ".join(str(p) for p in unique_pages)
                        page_suffix = f" (pp. {page_list})"
            return f"- {text_val}{page_suffix}"
        return f"- {entry}"

    for category, items in quotes.items():
        full_md += f"\n### {category.capitalize()}\n"
        for q in items:
            full_md += _format_quote_entry(q) + "\n"
            
    # Save markdown
    md_out = save_markdown(pdf_path, full_md, output_dir=output_dir)

    # Save quotes payload for reuse
    quotes_payload = {
        "key_takeaways": md_note,
        "quotes": quotes,
    }
    target_dir = output_dir if output_dir else pdf_path.parent
    quotes_path = target_dir / f"{pdf_path.stem}_quotes.json"
    quotes_path.write_text(json.dumps(quotes_payload, indent=2), encoding="utf-8")
    
    return pdf_out, md_out, quotes_path
