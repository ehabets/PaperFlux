"""
PDF I/O operations for PaperFlux.
Handles text extraction and PDF annotation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import logging
import unicodedata
import re
import string

import fitz  # PyMuPDF
from .config import Config

# Configure logging
logger = logging.getLogger(__name__)

_TOKEN_PUNCT = string.punctuation + "“”‘’"
_HYPHEN_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


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


def _coerce_quote_payload(raw: Any) -> Optional[Tuple[str, List[int]]]:
    """Normalize quote payloads to (text, pages) tuple."""
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
        return text_val.strip(), pages
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), []
    return None

def _quote_search_variants(*candidates: str) -> List[str]:
    """Generate search strings with light normalization (ellipsis trimming, etc.)."""
    variants: List[str] = []

    def _add(text: str) -> None:
        normalized = re.sub(r"\s+", " ", text.strip())
        if len(normalized) < 10:
            return
        if normalized not in variants:
            variants.append(normalized)

    for candidate in candidates:
        if not candidate:
            continue
        base = candidate.strip()
        if not base:
            continue
        _add(base)

        for ellipsis in ("...", "…"):
            if ellipsis in base:
                before = base.split(ellipsis)[0].strip()
                if len(before) >= 10:
                    _add(before)
                removed = base.replace(ellipsis, " ").strip()
                if len(removed) >= 10:
                    _add(removed)

        if base[-1:] in ",;:" and len(base) > 11:
            _add(base[:-1])

        words = base.split()
        if len(words) > 12:
            _add(" ".join(words[:-1]))

    return variants


def _normalize_token(text: str) -> str:
    """Normalize tokens for sequence-based matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.strip(_TOKEN_PUNCT)
    text = re.sub(f"[{_HYPHEN_CHARS}]", "", text)
    return text.lower()


def _build_logical_tokens(words: List[List[Any]]) -> List[Dict[str, Any]]:
    """Convert word entries into normalized logical tokens."""
    logical: List[Dict[str, Any]] = []
    for idx, word in enumerate(words):
        if len(word) < 7:
            continue
        normalized = _normalize_token(word[4])
        if not normalized:
            continue
        logical.append(
            {
                "text": normalized,
                "indices": [idx],
                "line_key": (word[5], word[6]),
            }
        )
    return logical


def _match_logical_sequence(
    tokens: List[Dict[str, Any]],
    sequence: List[int],
    target_text: str,
) -> List[List[int]]:
    """Find token index spans whose concatenated text equals the target."""
    matches: List[List[int]] = []
    seq_len = len(sequence)
    if not target_text or seq_len == 0:
        return matches

    for start in range(seq_len):
        concat = ""
        matched_indices: List[int] = []
        pos = start
        while pos < seq_len:
            token = tokens[sequence[pos]]
            token_text: str = token["text"]
            if not token_text:
                pos += 1
                continue
            concat += token_text
            matched_indices.append(sequence[pos])
            if not target_text.startswith(concat):
                break
            if concat == target_text:
                matches.append(matched_indices.copy())
                break
            pos += 1
    return matches


def _find_token_phrase_instances(
    page: fitz.Page,
    phrase: str,
    *,
    per_line: bool = False,
    words: Optional[List[List[Any]]] = None,
) -> Tuple[List[fitz.Rect], List[str]]:
    """Find phrase instances using normalized word tokens."""
    words = words if words is not None else page.get_text("words")
    if not words:
        return [], []

    logical_tokens = _build_logical_tokens(words)
    if not logical_tokens:
        return [], []

    target_tokens = [_normalize_token(tok) for tok in phrase.split()]
    target_tokens = [tok for tok in target_tokens if tok]
    target_string = "".join(target_tokens)
    if not target_string:
        return [], []

    sequences: List[List[int]]
    if per_line:
        line_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, token in enumerate(logical_tokens):
            line_map[token["line_key"]].append(idx)
        sequences = [indices for indices in line_map.values() if indices]
    else:
        sequences = [list(range(len(logical_tokens)))]

    rects: List[fitz.Rect] = []
    matched_texts: List[str] = []
    seen_spans: set[Tuple[int, ...]] = set()
    for index_sequence in sequences:
        matches = _match_logical_sequence(logical_tokens, index_sequence, target_string)
        for logical_indices in matches:
            word_indices: List[int] = []
            for logical_idx in logical_indices:
                word_indices.extend(logical_tokens[logical_idx]["indices"])
            span_key = tuple(word_indices)
            if not word_indices or span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            xs0 = [words[i][0] for i in word_indices]
            ys0 = [words[i][1] for i in word_indices]
            xs1 = [words[i][2] for i in word_indices]
            ys1 = [words[i][3] for i in word_indices]
            rects.append(fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1)))
            matched_texts.append(" ".join(words[i][4] for i in word_indices).strip())

    return rects, matched_texts

def annotate_pdf(
    path: Path, 
    quotes: Dict[str, List[Any]], 
    note_md: str, 
    colors: Optional[Dict[str, List[float]]] = None,
    cfg: Optional[Config] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Annotate a PDF file with highlights and a sticky note using token-based matching.

    Args:
        path: Path to PDF file
        quotes: Dictionary of quotes by category
        note_md: Markdown note to add as a sticky note
        colors: Dictionary of RGB colors by category (optional if cfg is provided)
        cfg: Application configuration (optional)
        output_dir: Optional directory where the annotated PDF should be saved

    Returns:
        Path: Path to annotated PDF file
    """
    # Get colors from config if not explicitly provided
    if colors is None and cfg is not None and hasattr(cfg, "ui") and hasattr(cfg.ui, "highlight_colors"):
        colors = cfg.ui.highlight_colors
    
    if not colors:
        logger.warning("No highlight colors provided or found in config")
        colors = {}  # Default to empty dict to avoid errors
    token_per_line = False
    word_cache: Dict[int, List[List[Any]]] = {}
    if cfg is not None and hasattr(cfg, "matching"):
        try:
            token_per_line = getattr(getattr(cfg.matching, "token", None), "per_line", token_per_line)
        except AttributeError:
            pass

    logger.debug(
        "Using token-based matching (per_line=%s)",
        token_per_line,
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
    
    for category, category_quotes in quotes.items():
        if category not in colors:
            logger.warning(f"No color defined for category: {category}")
            continue
        
        rgb = colors[category]
        logger.info(f"Processing {len(category_quotes)} quotes for {category} with color {rgb}")
        
        for i, raw_quote in enumerate(category_quotes):
            payload = _coerce_quote_payload(raw_quote)
            if not payload:
                logger.warning(f"Unsupported quote payload in {category}, skipping: {raw_quote}")
                continue

            quote_text, page_hints = payload
            cleaned_quote = _strip_wrapping_quotes(quote_text)
            if not cleaned_quote:
                logger.warning(f"Quote contains only wrapping punctuation in {category}, skipping")
                continue

            logger.debug(f"{category} quote {i+1}: '{cleaned_quote[:50]}...' ({len(cleaned_quote)} chars)")
            if page_hints:
                logger.debug(f"Page hints provided for quote: {page_hints}")

            quote_variants = _quote_search_variants(cleaned_quote) or [cleaned_quote]
            found_instances = 0
            best_match_info: List[Tuple[int, float, str]] = []
            checked_pages: set[int] = set()

            def process_pages(page_indices: List[int]) -> None:
                nonlocal found_instances, highlight_count
                for page_idx in page_indices:
                    if page_idx in checked_pages:
                        continue
                    checked_pages.add(page_idx)
                    if page_idx < 0 or page_idx >= len(doc):
                        logger.debug(f"Ignoring out-of-range page index {page_idx}")
                        continue
                    page = doc[page_idx]
                    page_num_display = page_idx + 1

                    def _filter_instances(instances):
                        filtered = []
                        for rect in instances or []:
                            snippet = page.get_text("text", clip=rect)
                            if snippet.strip():
                                filtered.append(rect)
                        return filtered

                    direct_instances = []
                    for search_text in quote_variants:
                        direct_instances = _filter_instances(page.search_for(search_text))
                        if direct_instances:
                            logger.debug(
                                f"Direct match for quote on page {page_num_display} "
                                f"({len(direct_instances)} instances) using '{search_text[:40]}...'"
                            )
                            for rect in direct_instances:
                                try:
                                    annot = page.add_highlight_annot(rect)
                                    annot.set_colors(stroke=rgb)
                                    annot.update()
                                    highlight_count += 1
                                    found_instances += 1
                                except Exception as e:
                                    logger.error(f"Error highlighting direct match: {str(e)}")
                            best_match_info.append((page_num_display, 100.0, search_text[:60]))
                            break
                    if direct_instances:
                        continue

                    words = word_cache.get(page_idx)
                    if words is None:
                        words = page.get_text("words")
                        word_cache[page_idx] = words

                    token_instances, phrase_variants = _find_token_phrase_instances(
                        page,
                        cleaned_quote,
                        per_line=token_per_line,
                        words=words,
                    )
                    if token_instances:
                        logger.debug(
                            f"Token match for quote on page {page_num_display} "
                            f"({len(token_instances)} instances, per_line={token_per_line})"
                        )
                        for rect in token_instances:
                            try:
                                annot = page.add_highlight_annot(rect)
                                annot.set_colors(stroke=rgb)
                                annot.update()
                                highlight_count += 1
                                found_instances += 1
                            except Exception as e:
                                logger.error(f"Error highlighting token match: {str(e)}")
                        best_phrase = phrase_variants[0] if phrase_variants else cleaned_quote
                        best_match_info.append((page_num_display, 100.0, best_phrase[:60]))
                    else:
                        best_match_info.append((page_num_display, 0.0, cleaned_quote[:60]))
                    continue

            hint_indices: List[int] = []
            if page_hints:
                for hinted_page in page_hints:
                    page_idx = hinted_page - 1
                    if 0 <= page_idx < len(doc):
                        hint_indices.append(page_idx)
                    else:
                        logger.debug(f"Ignoring out-of-range page hint {hinted_page}")
                if hint_indices:
                    process_pages(hint_indices)
                else:
                    logger.debug("No valid page hints after filtering; falling back to full scan")

            if found_instances == 0:
                remaining_indices = [idx for idx in range(len(doc)) if idx not in checked_pages]
                if remaining_indices:
                    process_pages(remaining_indices)

            if found_instances == 0:
                logger.warning(
                    f"Quote not found in any page (token search, per_line={token_per_line}): '{cleaned_quote[:50]}...'"
                )
                if best_match_info:
                    logger.info(f"Closest matches for quote: {best_match_info}")
                if len(cleaned_quote) > 40:
                    shorter_quote = cleaned_quote[:40]
                    logger.debug(f"Trying shorter version: '{shorter_quote}'")
                    fallback_pages = hint_indices if hint_indices else range(len(doc))
                    for page_idx in fallback_pages:
                        if page_idx < 0 or page_idx >= len(doc):
                            continue
                        page = doc[page_idx]
                        highlight_instances = page.search_for(shorter_quote)
                        if highlight_instances:
                            logger.debug(
                                f"Found {len(highlight_instances)} instances of shorter quote on page {page_idx+1}"
                            )
                            for rect in highlight_instances:
                                try:
                                    annot = page.add_highlight_annot(rect)
                                    annot.set_colors(stroke=rgb)
                                    annot.update()
                                    highlight_count += 1
                                except Exception as e:
                                    logger.error(f"Error highlighting shorter quote: {str(e)}")
    
    logger.info(f"Added {highlight_count} highlights across all categories")
    
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
    return output_path

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
