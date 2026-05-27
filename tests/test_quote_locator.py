from src.quote_locator import locate_quote_in_words, normalize_token


def _word(x0, y0, x1, y1, text, block=0, line=0, word_no=0):
    return [x0, y0, x1, y1, text, block, line, word_no]


def test_normalize_token_handles_ligatures_and_hyphens():
    assert normalize_token("ﬁne-grained") == "finegrained"


def test_exact_match_returns_line_level_rectangles():
    words = [
        _word(0, 0, 20, 10, "This", line=0, word_no=0),
        _word(22, 0, 42, 10, "quote", line=0, word_no=1),
        _word(0, 12, 24, 22, "spans", line=1, word_no=0),
        _word(26, 12, 44, 22, "lines", line=1, word_no=1),
    ]

    match = locate_quote_in_words(words, "This quote spans lines", min_similarity=0.95)

    assert match is not None
    assert match.method == "exact"
    assert match.score == 1.0
    assert len(match.areas) == 2
    assert match.matched_text == "This quote spans lines"


def test_exact_match_handles_pdf_line_hyphenation():
    words = [
        _word(0, 0, 50, 10, "frequency-", line=0, word_no=0),
        _word(0, 12, 45, 22, "dependent", line=1, word_no=0),
        _word(48, 12, 75, 22, "decay", line=1, word_no=1),
    ]

    match = locate_quote_in_words(words, "frequency-dependent decay", min_similarity=0.95)

    assert match is not None
    assert match.method == "exact"
    assert len(match.areas) == 2


def test_fuzzy_match_accepts_small_inflection_difference():
    words = [
        _word(0, 0, 20, 10, "The", word_no=0),
        _word(22, 0, 58, 10, "proposed", word_no=1),
        _word(60, 0, 102, 10, "approach", word_no=2),
        _word(104, 0, 154, 10, "outperforms", word_no=3),
        _word(156, 0, 176, 10, "the", word_no=4),
        _word(178, 0, 220, 10, "baseline", word_no=5),
    ]

    match = locate_quote_in_words(
        words,
        "the proposed approach outperformed the baseline",
        min_similarity=0.86,
    )

    assert match is not None
    assert match.method == "fuzzy"
    assert match.score >= 0.86
    assert match.matched_text == "The proposed approach outperforms the baseline"


def test_fuzzy_match_rejects_unrelated_text():
    words = [
        _word(0, 0, 20, 10, "The", word_no=0),
        _word(22, 0, 60, 10, "method", word_no=1),
        _word(62, 0, 90, 10, "uses", word_no=2),
        _word(92, 0, 130, 10, "audio", word_no=3),
    ]

    match = locate_quote_in_words(
        words,
        "a formal listening experiment would be required",
        min_similarity=0.88,
    )

    assert match is None
