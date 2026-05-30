"""
PDF I/O operations for PaperFlux.
Handles text extraction and PDF annotation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import logging

import fitz  # PyMuPDF
from .config import Config
from .quote_locator import locate_quote_in_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QuotePayload:
    text: str
    pages: List[int]
    prefix: str = ""
    suffix: str = ""


def _strip_wrapping_quotes(text: str) -> str:
    """Remove a single layer of wrapping quote characters."""
    if not text:
        return text

    trimmed = text.strip()
    quote_pairs = [
        ("“", "”"),
        ("‘", "’"),
        ("'", "'"),
        ('"', '"'),
    ]
    for left, right in quote_pairs:
        if trimmed.startswith(left) and trimmed.endswith(right) and len(trimmed) > len(left) + len(right):
            trimmed = trimmed[len(left):-len(right)].strip()
            break
    return trimmed.strip('"“”‘’')


def _coerce_quote_payload(raw: Any) -> Optional[_QuotePayload]:
    """Normalize quote payloads for annotation."""
    if isinstance(raw, dict):
        text_val = raw.get("text") or raw.get("quote") or raw.get("content")
        if not isinstance(text_val, str) or not text_val.strip():
            return None
        pages_val = raw.get("pages") or raw.get("page")
        pages: List[int]
        if isinstance(pages_val, int):
            pages = [pages_val]
        elif isinstance(pages_val, list):
            pages = [p for p in pages_val if isinstance(p, int) and p > 0]
        else:
            pages = []
        prefix_val = raw.get("prefix") or raw.get("context_before") or ""
        suffix_val = raw.get("suffix") or raw.get("context_after") or ""
        prefix = prefix_val.strip() if isinstance(prefix_val, str) else ""
        suffix = suffix_val.strip() if isinstance(suffix_val, str) else ""
        return _QuotePayload(text=text_val.strip(), pages=pages, prefix=prefix, suffix=suffix)
    if isinstance(raw, str) and raw.strip():
        return _QuotePayload(text=raw.strip(), pages=[])
    return None


def annotate_pdf(
    path: Path, 
    quotes: Dict[str, List[Any]], 
    note_md: str, 
    colors: Optional[Dict[str, List[float]]] = None,
    cfg: Optional[Config] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Annotate a PDF file with highlights and a sticky note using local quote alignment.

    Args:
        path: Path to PDF file
        quotes: Dictionary of quotes by category
        note_md: Markdown note to add as a sticky note
        colors: Dictionary of RGB colors by category (optional if cfg is provided)
        cfg: Application configuration (optional)
        output_dir: Optional directory where the annotated PDF should be saved

    Returns:
        Tuple[Path, Dict[str, Any]]: Path to annotated PDF file and quote match report
    """
    # Get colors from config if not explicitly provided
    if colors is None and cfg is not None and hasattr(cfg, "ui") and hasattr(cfg.ui, "highlight_colors"):
        colors = cfg.ui.highlight_colors
    
    if not colors:
        logger.warning("No highlight colors provided or found in config")
        colors = {}  # Default to empty dict to avoid errors
    min_similarity = 0.88
    max_window_tokens = 80
    word_cache: Dict[int, List[List[Any]]] = {}
    if cfg is not None and hasattr(cfg, "matching"):
        try:
            min_similarity = getattr(cfg.matching, "min_similarity", min_similarity)
            max_window_tokens = getattr(cfg.matching, "max_window_tokens", max_window_tokens)
        except AttributeError:
            pass

    logger.debug(
        "Using local quote locator (min_similarity=%.2f, max_window_tokens=%s)",
        min_similarity,
        max_window_tokens,
    )

    logger.debug(f"Opening PDF for annotation: {path}")
    doc = fitz.open(path)
    logger.info(f"PDF has {len(doc)} pages")
    
    # Add sticky note to first page
    first_page = doc[0]
    logger.debug(f"Adding sticky note with {len(note_md)} characters")
    logger.debug(f"Note content preview: {note_md[:100]}...")
    
    try:
        annot = first_page.add_text_annot((72, 72), note_md, icon="Comment")
        logger.debug(f"Sticky note added with ID: {annot.info.get('id', 'unknown')}")
    except Exception as e:
        logger.error(f"Error adding sticky note: {str(e)}")
    
    highlight_count = 0
    quote_match_records: List[Dict[str, Any]] = []
    
    for category, category_quotes in quotes.items():
        if category not in colors:
            logger.warning(f"No color defined for category: {category}")
            for i, raw_quote in enumerate(category_quotes):
                payload = _coerce_quote_payload(raw_quote)
                quote_match_records.append({
                    "category": category,
                    "quote_index": i + 1,
                    "text": payload.text if payload else str(raw_quote),
                    "matched": False,
                    "page": None,
                    "score": None,
                    "method": None,
                    "segments": 0,
                    "matched_text": "",
                    "skipped_reason": "no highlight color defined for category",
                })
            continue
        
        rgb = colors[category]
        logger.info(f"Processing {len(category_quotes)} quotes for {category} with color {rgb}")
        
        for i, raw_quote in enumerate(category_quotes):
            report_entry: Dict[str, Any] = {
                "category": category,
                "quote_index": i + 1,
                "text": "",
                "matched": False,
                "page": None,
                "score": None,
                "method": None,
                "segments": 0,
                "matched_text": "",
                "skipped_reason": None,
            }
            payload = _coerce_quote_payload(raw_quote)
            if not payload:
                logger.warning(f"Unsupported quote payload in {category}, skipping: {raw_quote}")
                report_entry["text"] = str(raw_quote)
                report_entry["skipped_reason"] = "unsupported quote payload"
                quote_match_records.append(report_entry)
                continue

            cleaned_quote = _strip_wrapping_quotes(payload.text)
            report_entry["text"] = cleaned_quote
            if not cleaned_quote:
                logger.warning(f"Quote contains only wrapping punctuation in {category}, skipping")
                report_entry["skipped_reason"] = "quote contains only wrapping punctuation"
                quote_match_records.append(report_entry)
                continue

            logger.debug(f"{category} quote {i+1}: '{cleaned_quote[:50]}...' ({len(cleaned_quote)} chars)")
            if payload.pages:
                logger.debug(f"Page hints provided for quote: {payload.pages}")

            match = locate_quote_in_document(
                doc,
                cleaned_quote,
                page_hints=payload.pages,
                min_similarity=min_similarity,
                max_window_tokens=max_window_tokens,
                prefix=payload.prefix,
                suffix=payload.suffix,
                word_cache=word_cache,
            )
            if not match:
                logger.warning(
                    "Quote not found above similarity threshold %.2f: '%s...'",
                    min_similarity,
                    cleaned_quote[:50],
                )
                report_entry["skipped_reason"] = (
                    f"not found above similarity threshold {min_similarity:.2f}"
                )
                quote_match_records.append(report_entry)
                continue

            page = doc[match.page_index]
            report_entry.update({
                "page": match.page_index + 1,
                "score": round(match.score, 6),
                "method": match.method,
                "segments": len(match.areas),
                "matched_text": match.matched_text,
            })
            try:
                areas = match.areas if len(match.areas) > 1 else match.areas[0]
                annot = page.add_highlight_annot(areas)
                annot.set_colors(stroke=rgb)
                annot.update()
                highlight_count += 1
                report_entry["matched"] = True
                logger.debug(
                    "Highlighted quote on page %s via %s match (score=%.3f, areas=%s): '%s'",
                    match.page_index + 1,
                    match.method,
                    match.score,
                    len(match.areas),
                    match.matched_text[:80],
                )
            except Exception as e:
                logger.error(f"Error highlighting quote match: {str(e)}")
                report_entry["skipped_reason"] = f"highlight failed: {e}"
            quote_match_records.append(report_entry)
    
    logger.info(f"Added {highlight_count} highlights across all categories")
    skipped_count = sum(1 for item in quote_match_records if not item["matched"])
    match_report = {
        "total": len(quote_match_records),
        "matched": highlight_count,
        "skipped": skipped_count,
        "records": quote_match_records,
    }
    
    # Save the annotated PDF
    stem = path.stem
    target_dir = output_dir if output_dir else path.parent
    if output_dir:
        target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{stem}_annotated.pdf"
    logger.debug(f"Saving annotated PDF to: {output_path}")
    
    try:
        doc.save(output_path, incremental=False, garbage=4)
        logger.info("PDF saved successfully")
    except Exception as e:
        logger.error(f"Error saving PDF: {str(e)}")
    
    doc.close()
    return output_path, match_report

def save_markdown(path: Path, content: str, output_dir: Optional[Path] = None) -> Path:
    """
    Save markdown content to a file.
    
    Args:
    path: Path to PDF file (used to generate markdown filename)
        content: Markdown content
    output_dir: Optional directory where the markdown file should be saved
        
    Returns:
        Path: Path to markdown file
    """
    stem = path.stem
    target_dir = output_dir if output_dir else path.parent
    if output_dir:
        target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{stem}_summary.md"
    logger.debug(f"Saving markdown ({len(content)} chars) to: {output_path}")
    
    try:
        output_path.write_text(content)
        logger.debug("Markdown saved successfully")
    except Exception as e:
        logger.error(f"Error saving markdown: {str(e)}")
    
    return output_path
