from paperflux.quote_locator import locate_quote_in_words, normalize_token


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


def test_layout_gap_match_skips_intervening_table_or_figure_words():
    words = [
        _word(0, 0, 10, 10, "A", block=0, line=0, word_no=0),
        _word(12, 0, 44, 10, "general", block=0, line=0, word_no=1),
        _word(46, 0, 70, 10, "trend", block=0, line=0, word_no=2),
        _word(0, 12, 46, 22, "observed", block=0, line=1, word_no=0),
        _word(48, 12, 64, 22, "for", block=0, line=1, word_no=1),
        _word(66, 12, 88, 22, "both", block=0, line=1, word_no=2),
        _word(90, 12, 130, 22, "methods", block=0, line=1, word_no=3),
        _word(132, 12, 140, 22, "is", block=0, line=1, word_no=4),
        _word(142, 12, 160, 22, "that", block=0, line=1, word_no=5),
        _word(0, 24, 48, 34, "estimating", block=0, line=2, word_no=0),
        _word(50, 24, 90, 34, "acoustic", block=0, line=2, word_no=1),
        _word(92, 24, 140, 34, "parameters", block=0, line=2, word_no=2),
        _word(0, 36, 8, 46, "is", block=0, line=3, word_no=0),
        _word(10, 36, 34, 46, "most", block=0, line=3, word_no=1),
        _word(36, 36, 88, 46, "challenging", block=0, line=3, word_no=2),
        _word(90, 36, 104, 46, "in", block=0, line=3, word_no=3),
        _word(106, 36, 122, 46, "the", block=0, line=3, word_no=4),
        _word(124, 36, 154, 46, "lowest", block=0, line=3, word_no=5),
        _word(156, 36, 172, 46, "and", block=0, line=3, word_no=6),
        _word(174, 36, 210, 46, "highest", block=0, line=3, word_no=7),
        _word(212, 36, 244, 46, "octave", block=0, line=3, word_no=8),
        _word(246, 36, 284, 46, "bands,", block=0, line=3, word_no=9),
        _word(286, 36, 318, 46, "where", block=0, line=3, word_no=10),
        _word(0, 72, 26, 82, "Table", block=1, line=0, word_no=0),
        _word(28, 72, 36, 82, "1.", block=1, line=0, word_no=1),
        _word(38, 72, 92, 82, "Comparison", block=1, line=0, word_no=2),
        _word(94, 72, 122, 82, "speech", block=1, line=0, word_no=3),
        _word(124, 72, 168, 82, "generated", block=1, line=0, word_no=4),
        _word(320, 0, 350, 10, "speech", block=2, line=0, word_no=0),
        _word(352, 0, 370, 10, "does", block=2, line=0, word_no=1),
        _word(372, 0, 386, 10, "not", block=2, line=0, word_no=2),
        _word(388, 0, 440, 10, "sufficiently", block=2, line=0, word_no=3),
        _word(442, 0, 474, 10, "excite", block=2, line=0, word_no=4),
        _word(476, 0, 490, 10, "the", block=2, line=0, word_no=5),
        _word(492, 0, 532, 10, "acoustic", block=2, line=0, word_no=6),
        _word(534, 0, 582, 10, "response.", block=2, line=0, word_no=7),
    ]
    words = (
        words[:28]
        + [
            _word(0, 84 + n * 2, 20, 94 + n * 2, f"junk{n}", block=1, line=n + 1, word_no=0)
            for n in range(80)
        ]
        + words[28:]
    )

    quote = (
        "A general trend observed for both methods is that estimating acoustic "
        "parameters is most challenging in the lowest and highest octave bands, "
        "where speech does not sufficiently excite the acoustic response."
    )

    match = locate_quote_in_words(words, quote, min_similarity=0.88)

    assert match is not None
    assert match.method == "layout-gap"
    assert match.score >= 0.88
    assert "Table" not in match.matched_text
    assert "speech generated" not in match.matched_text
    assert match.matched_text.endswith("speech does not sufficiently excite the acoustic response.")
    assert len(match.areas) == 5
