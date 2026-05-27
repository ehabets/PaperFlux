"""
Local quote-to-PDF span alignment.

The model may return evidence text that differs slightly from PyMuPDF's text
extraction. This module aligns that text back to page words and returns
line-level rectangles that can be highlighted without guessing broad regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import string
import unicodedata

import fitz


_TOKEN_PUNCT = string.punctuation + "“”‘’"
_HYPHEN_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


@dataclass(frozen=True)
class QuoteMatch:
    """Best local match for a model-provided quote."""

    page_index: int
    score: float
    method: str
    areas: List[fitz.Rect]
    matched_text: str


@dataclass(frozen=True)
class _WordToken:
    text: str
    word_index: int
    line_key: Tuple[int, int]


def normalize_token(text: str) -> str:
    """Normalize a word token for robust comparison."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.strip(_TOKEN_PUNCT)
    for hyphen in _HYPHEN_CHARS:
        text = text.replace(hyphen, "")
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _quote_tokens(text: str) -> List[str]:
    return [token for token in (normalize_token(part) for part in text.split()) if token]


def _word_tokens(words: Sequence[Sequence[Any]]) -> List[_WordToken]:
    tokens: List[_WordToken] = []
    for index, word in enumerate(words):
        if len(word) < 7:
            continue
        normalized = normalize_token(str(word[4]))
        if not normalized:
            continue
        tokens.append(
            _WordToken(
                text=normalized,
                word_index=index,
                line_key=(int(word[5]), int(word[6])),
            )
        )
    return tokens


def _joined(tokens: Iterable[str]) -> str:
    return "".join(tokens)


def _line_rects_for_words(
    words: Sequence[Sequence[Any]],
    word_indices: Sequence[int],
) -> List[fitz.Rect]:
    grouped: Dict[Tuple[int, int], List[int]] = {}
    ordered_keys: List[Tuple[int, int]] = []
    for index in word_indices:
        word = words[index]
        key = (int(word[5]), int(word[6]))
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(index)

    rects: List[fitz.Rect] = []
    for key in ordered_keys:
        indices = grouped[key]
        rects.append(
            fitz.Rect(
                min(float(words[i][0]) for i in indices),
                min(float(words[i][1]) for i in indices),
                max(float(words[i][2]) for i in indices),
                max(float(words[i][3]) for i in indices),
            )
        )
    return rects


def _matched_text(words: Sequence[Sequence[Any]], word_indices: Sequence[int]) -> str:
    return " ".join(str(words[index][4]) for index in word_indices).strip()


def _score_context(
    base_score: float,
    page_tokens: Sequence[_WordToken],
    start: int,
    end: int,
    target_tokens: Sequence[str],
    prefix_tokens: Sequence[str],
    suffix_tokens: Sequence[str],
) -> float:
    if not prefix_tokens and not suffix_tokens:
        return base_score

    context_start = max(0, start - len(prefix_tokens) - 4)
    context_end = min(len(page_tokens), end + len(suffix_tokens) + 4)
    target_context = _joined([*prefix_tokens, *target_tokens, *suffix_tokens])
    page_context = _joined(token.text for token in page_tokens[context_start:context_end])
    if not target_context or not page_context:
        return base_score

    context_score = SequenceMatcher(None, target_context, page_context).ratio()
    return (base_score * 0.75) + (context_score * 0.25)


def locate_quote_in_words(
    words: Sequence[Sequence[Any]],
    quote_text: str,
    *,
    page_index: int = 0,
    min_similarity: float = 0.88,
    max_window_tokens: int = 80,
    prefix: str = "",
    suffix: str = "",
) -> Optional[QuoteMatch]:
    """
    Locate one quote on a page represented by PyMuPDF words.

    Exact normalized matches win. If exact matching fails, a bounded fuzzy
    search finds the closest local word span and accepts it only above
    ``min_similarity``.
    """
    page_tokens = _word_tokens(words)
    target_tokens = _quote_tokens(quote_text)
    if not page_tokens or not target_tokens:
        return None

    target_key = _joined(target_tokens)
    if not target_key:
        return None

    window_limit = max(max_window_tokens, len(target_tokens) + 12)
    prefix_tokens = _quote_tokens(prefix)
    suffix_tokens = _quote_tokens(suffix)

    # Exact normalized span. Joining tokens lets hyphenation and punctuation
    # differ while still requiring the same underlying characters.
    for start in range(len(page_tokens)):
        candidate_key = ""
        matched_word_indices: List[int] = []
        for end in range(start, min(len(page_tokens), start + window_limit)):
            candidate_key += page_tokens[end].text
            matched_word_indices.append(page_tokens[end].word_index)
            if not target_key.startswith(candidate_key):
                break
            if candidate_key == target_key:
                score = _score_context(
                    1.0,
                    page_tokens,
                    start,
                    end + 1,
                    target_tokens,
                    prefix_tokens,
                    suffix_tokens,
                )
                return QuoteMatch(
                    page_index=page_index,
                    score=score,
                    method="exact",
                    areas=_line_rects_for_words(words, matched_word_indices),
                    matched_text=_matched_text(words, matched_word_indices),
                )

    target_len = len(target_key)
    min_chars = max(8, int(target_len * 0.55))
    max_chars = max(target_len + 12, int(target_len * 1.45))
    target_set = set(target_tokens)
    min_overlap = 0.45 if len(target_set) >= 6 else 0.35
    min_quick_score = max(0.55, min_similarity - 0.12)

    best_score = 0.0
    best_span: List[int] = []

    for start in range(len(page_tokens)):
        candidate_key = ""
        candidate_tokens: List[str] = []
        matched_word_indices = []
        for end in range(start, min(len(page_tokens), start + window_limit)):
            token = page_tokens[end]
            candidate_key += token.text
            candidate_tokens.append(token.text)
            matched_word_indices.append(token.word_index)

            candidate_len = len(candidate_key)
            if candidate_len > max_chars:
                break
            if candidate_len < min_chars:
                continue

            overlap = len(target_set.intersection(candidate_tokens)) / max(1, len(target_set))
            if overlap < min_overlap:
                continue

            matcher = SequenceMatcher(None, target_key, candidate_key)
            if matcher.quick_ratio() < min_quick_score:
                continue
            char_score = matcher.ratio()
            score = (char_score * 0.82) + (overlap * 0.18)
            score = _score_context(
                score,
                page_tokens,
                start,
                end + 1,
                target_tokens,
                prefix_tokens,
                suffix_tokens,
            )
            if score > best_score:
                best_score = score
                best_span = matched_word_indices.copy()

    if best_score < min_similarity or not best_span:
        return None

    return QuoteMatch(
        page_index=page_index,
        score=best_score,
        method="fuzzy",
        areas=_line_rects_for_words(words, best_span),
        matched_text=_matched_text(words, best_span),
    )


def locate_quote_in_document(
    doc: fitz.Document,
    quote_text: str,
    *,
    page_hints: Optional[Sequence[int]] = None,
    min_similarity: float = 0.88,
    max_window_tokens: int = 80,
    prefix: str = "",
    suffix: str = "",
    word_cache: Optional[Dict[int, Sequence[Sequence[Any]]]] = None,
) -> Optional[QuoteMatch]:
    """Locate the best quote match across a document."""
    valid_hints: List[int] = []
    for page_number in page_hints or []:
        page_index = page_number - 1
        if 0 <= page_index < len(doc) and page_index not in valid_hints:
            valid_hints.append(page_index)

    ordered_pages = valid_hints + [idx for idx in range(len(doc)) if idx not in valid_hints]
    best: Optional[QuoteMatch] = None

    for page_index in ordered_pages:
        if word_cache is not None and page_index in word_cache:
            words = word_cache[page_index]
        else:
            words = doc[page_index].get_text("words")
            if word_cache is not None:
                word_cache[page_index] = words

        match = locate_quote_in_words(
            words,
            quote_text,
            page_index=page_index,
            min_similarity=min_similarity,
            max_window_tokens=max_window_tokens,
            prefix=prefix,
            suffix=suffix,
        )
        if not match:
            continue
        if best is None or match.score > best.score:
            best = match
            if match.score >= 1.0 and not prefix and not suffix:
                break

    return best
