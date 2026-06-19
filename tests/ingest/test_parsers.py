"""
tests/ingest/test_parsers.py

Regression tests for the parser, specifically targeting the bug where
_looks_like_heading() matched reference-list entries and digit-led
sentences as section headings, exploding ~15-page papers into 700+
chunks instead of the expected ~20-40.
"""

from research_navigator.ingest.parsers import _looks_like_heading, _REFERENCES_RE


class TestLooksLikeHeading:
    """
    Direct tests on the heuristic heading detector.

    Each case documents WHY a line should or shouldn't be treated as a
    heading. If you ever touch this function again, these are the
    behaviours that must keep holding.
    """

    # ── Things that ARE real headings ──────────────────────────────────────
    def test_numbered_section_heading(self) -> None:
        assert _looks_like_heading("1 Introduction") is True

    def test_numbered_subsection_heading(self) -> None:
        assert _looks_like_heading("3.1 Scaled Dot-Product Attention") is True

    def test_deep_numbered_heading(self) -> None:
        assert _looks_like_heading("3.2.1 Multi-Head Attention Details") is True

    def test_all_caps_heading(self) -> None:
        assert _looks_like_heading("ABSTRACT") is True
        assert _looks_like_heading("REFERENCES") is True
        assert _looks_like_heading("CONCLUSION") is True

    def test_heading_with_glue_words(self) -> None:
        # "Is", "All", "You" are lowercase glue words but should still count
        # as part of a Title Case heading
        assert _looks_like_heading("6 Attention Is All You Need") is True

    # ── Things that are NOT headings (the regression cases) ────────────────
    def test_reference_entry_not_a_heading(self) -> None:
        """
        This is THE bug: reference list entries start with a number
        (the citation index) followed by a capitalised surname, which
        used to match the old (too-loose) regex.
        """
        line = "12 Vaswani, A., Shazeer, N. Parmar, N. Attention is all you need."
        assert _looks_like_heading(line) is False

    def test_multiple_reference_entries_not_headings(self) -> None:
        refs = [
            "1 Devlin, J., Chang, M., Lee, K. BERT: Pre-training of deep bidirectional transformers.",
            "2 Brown, T., Mann, B., Ryder, N. Language models are few-shot learners.",
            "30 Touvron, H., Martin, L., Stone, K. Llama 2: Open foundation models.",
        ]
        for ref in refs:
            assert _looks_like_heading(ref) is False, f"False positive on: {ref}"

    def test_digit_led_sentence_not_a_heading(self) -> None:
        """
        Body text that happens to start with a digit (a footnote marker,
        a stray OCR artefact, a sentence that begins with a number) must
        not be misread as a new section.
        """
        assert (
            _looks_like_heading(
                "2017 The Transformer follows this overall architecture"
            )
            is False
        )
        assert (
            _looks_like_heading(
                "2 We propose a new architecture for sequence transduction"
            )
            is False
        )

    def test_short_digit_led_sentence_not_a_heading(self) -> None:
        """Even a SHORT digit-led sentence shouldn't pass as Title Case."""
        assert _looks_like_heading("1 We use 8 attention heads") is False

    def test_long_line_never_a_heading(self) -> None:
        """Anything longer than 6 words is never treated as a heading."""
        long_line = "This is a very long sentence that goes on and on past the heading word limit"
        assert _looks_like_heading(long_line) is False

    def test_ordinary_lowercase_sentence_not_a_heading(self) -> None:
        assert _looks_like_heading("the model was trained for ten epochs") is False


class TestReferencesRegex:
    """
    Tests for the references-section detector, which must catch common
    real-world headings ('7 References', 'Bibliography.') so the chunker
    correctly excludes them from retrieval.
    """

    def test_plain_references(self) -> None:
        assert _REFERENCES_RE.match("References")

    def test_numbered_references(self) -> None:
        assert _REFERENCES_RE.match("7 References")
        assert _REFERENCES_RE.match("8. Bibliography")

    def test_case_insensitive(self) -> None:
        assert _REFERENCES_RE.match("REFERENCES")
        assert _REFERENCES_RE.match("references")

    def test_trailing_period(self) -> None:
        assert _REFERENCES_RE.match("References.")

    def test_works_cited(self) -> None:
        assert _REFERENCES_RE.match("Works Cited")

    def test_does_not_match_body_text(self) -> None:
        """Should not accidentally match a sentence that merely mentions references."""
        assert _REFERENCES_RE.match("See references for more details") is None


class TestChunkExplosionRegression:
    """
    End-to-end regression test: simulate a realistic paper's line stream
    (heading + body with digit-led sentences + a references section) and
    assert the section count stays sane instead of exploding.

    This directly reproduces the production bug: 741 chunks/paper average
    when it should be roughly 20-40.
    """

    def test_section_count_stays_sane_on_realistic_paper(self) -> None:
        # Build ~9 numbered sections + abstract + references, each with
        # digit-led body sentences (the exact shape that broke production).
        lines: list[str] = ["ABSTRACT", "We propose a new architecture."]
        section_headings = [
            "1 Introduction",
            "2 Background",
            "3 Model Architecture",
            "3.1 Scaled Dot-Product Attention",
            "4 Why Self-Attention",
            "5 Training",
            "6 Results",
            "7 Conclusion",
        ]
        for heading in section_headings:
            lines.append(heading)
            for i in range(15):
                # Digit-led sentence — the exact pattern that used to
                # trigger a false section break on every single line.
                lines.append(
                    f"{i} We evaluate this approach across several benchmarks."
                )

        lines.append("REFERENCES")
        for i in range(1, 41):
            lines.append(f"{i} Vaswani, A., Shazeer, N. Some paper title. NeurIPS, 2017.")

        # Run the same section-splitting loop parse_pdf uses internally,
        # relying purely on _looks_like_heading (worst case: no font-size signal).
        detected_sections: list[str] = []
        current_parts: list[str] = []
        for line_text in lines:
            if _looks_like_heading(line_text) and len(line_text) < 120:
                if current_parts:
                    detected_sections.append("section")
                current_parts = []
            else:
                current_parts.append(line_text)
        if current_parts:
            detected_sections.append("section")

        # Expected: abstract + 8 numbered headings + references = 9 sections
        # (the loop above doesn't track the abstract as separate from the
        # very first heading-free preamble; what matters is staying in a
        # sane single-digit-to-low-double-digit range, NOT 100+).
        assert len(detected_sections) <= 15, (
            f"Expected a sane number of sections (<=15) for a 9-heading "
            f"paper, got {len(detected_sections)}. This indicates the "
            f"heading detector is firing on body text or references again."
        )