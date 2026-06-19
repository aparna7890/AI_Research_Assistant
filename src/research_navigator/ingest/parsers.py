"""PDF and Markdown parsers."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Matches a references/bibliography heading, optionally prefixed with a
# section number ("7 References", "References", "8. Bibliography") and
# optionally followed by trailing punctuation.
_REFERENCES_RE = re.compile(
    r"^(\d{1,2}(\.\d{1,2})*\.?\s+)?(references|bibliography|works cited)\.?\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    title: str
    index: int
    text: str
    is_abstract: bool = False
    is_references: bool = False


def parse_pdf(path: Path) -> list[Section]:
    try:
        import pymupdf  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install pymupdf")

    doc = pymupdf.open(str(path))
    sections: list[Section] = []
    current_title = "preamble"
    current_parts: list[str] = []
    section_index = 0

    # Collect font sizes to detect headings
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:  # type: ignore[union-attr]
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(round(span["size"], 1))

    body_size = _most_common(sizes) if sizes else 10.0
    heading_threshold = body_size * 1.15

    for page in doc:
        for block in page.get_text("dict")["blocks"]:  # type: ignore[union-attr]
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = " ".join(s["text"] for s in line.get("spans", [])).strip()
                if not line_text:
                    continue
                max_size = max((s["size"] for s in line.get("spans", [])), default=body_size)
                is_heading = max_size >= heading_threshold or _looks_like_heading(line_text)
                if is_heading and len(line_text) < 120:
                    if current_parts:
                        body = "\n".join(current_parts).strip()
                        if body:
                            sections.append(_make_section(current_title, section_index, body))
                            section_index += 1
                    current_title = line_text
                    current_parts = []
                else:
                    current_parts.append(line_text)

    if current_parts:
        body = "\n".join(current_parts).strip()
        if body:
            sections.append(_make_section(current_title, section_index, body))

    doc.close()
    logger.info("pdf_parsed", path=str(path), sections=len(sections))
    return sections


def parse_markdown(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8")
    sections: list[Section] = []
    current_title = "introduction"
    current_lines: list[str] = []
    section_index = 0
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$")

    for line in text.splitlines():
        m = heading_re.match(line)
        if m:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(_make_section(current_title, section_index, body))
                    section_index += 1
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(_make_section(current_title, section_index, body))

    logger.info("markdown_parsed", path=str(path), sections=len(sections))
    return sections


def _make_section(title: str, index: int, text: str) -> Section:
    return Section(
        title=title,
        index=index,
        text=text,
        is_abstract=title.strip().lower() == "abstract",
        is_references=bool(_REFERENCES_RE.match(title.strip())),
    )


# Matches numbered headings like "3.1 Scaled Dot-Product Attention".
# Capped at 2 numbering levels deep ("1", "1.2", "1.2.3" — not "1.2.3.4.5").
# This regex alone is still loose — the heavy filtering (word count, comma
# check, title-case ratio) happens in _looks_like_heading below. Requiring a
# Title Case word right after the number already excludes most body
# sentences and reference-list entries, which usually start lowercase or
# with a surname followed by a comma (e.g. "12 Vaswani, A., Shazeer, N. ...").
_NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}(\.\d{1,2}){0,2}\s+[A-Z][a-z]")

# Short "glue" words that are allowed to stay lowercase inside an otherwise
# Title Case heading, e.g. "Attention Is All You Need", "What Does BERT Learn".
_GLUE_WORDS = {
    "a", "an", "the", "is", "are", "of", "in", "on", "to", "and", "or",
    "for", "with", "via", "vs", "all", "you", "we", "do", "does",
}


def _looks_like_title_case(words: list[str]) -> bool:
    """
    Decide whether a sequence of words reads like a heading ("Title Case")
    rather than a sentence ("Sentence case").

    Headings: most words start with a capital letter, except small glue words.
    Sentences: only the first word is capitalised; the rest are lowercase.

    We require at least 70% of words to be "capitalised or glue" so that
    "Scaled Dot-Product Attention" passes but "We propose a new architecture"
    (only "We" is capitalised) fails.
    """
    if not words:
        return False
    capitalised = 0
    for w in words:
        clean = w.strip(",.;:()")
        if not clean:
            continue
        if clean[0].isupper() or clean.lower() in _GLUE_WORDS:
            capitalised += 1
    return (capitalised / len(words)) >= 0.7


def _looks_like_heading(text: str) -> bool:
    """
    Heuristically decide whether a line of text is a section heading,
    as a FALLBACK for when font-size detection (the primary signal in
    parse_pdf) doesn't clearly mark it as one.

    This function used to match ANY line starting with "<number> <Capital>",
    which is also the shape of:
      - reference list entries:  "12 Vaswani, A., Shazeer, N. ..."
      - ordinary sentences that happen to start with a digit:
        "2017 The Transformer follows this overall architecture"
        "2 We propose a new architecture for sequence transduction"
    That caused hundreds of false "section breaks" per paper (one per
    numbered reference + one per digit-led sentence), which in turn made
    the chunker treat each false section as its own tiny chunk — exploding
    a ~15-page paper into 700+ chunks instead of the expected ~20-40.

    Fixed by requiring ALL of:
      1. Short line (<= 6 words) — real headings are short, sentences and
         reference entries are long.
      2. No comma in the first 25 characters — reference entries
         ("Surname, Initial., ...") have one; headings don't.
      3. Title Case — most words capitalised, not just the first
         ("Scaled Dot-Product Attention" vs "We propose a new...").
    """
    text = text.strip()
    words = text.split()

    # Real section titles are short, e.g. "3.1 Scaled Dot-Product Attention"
    # is 4 words. Sentences and reference entries run much longer.
    if len(words) > 6:
        return False

    # ALL-CAPS short lines: "ABSTRACT", "REFERENCES", "CONCLUSION"
    if text.isupper() and 2 <= len(text) < 60:
        return True

    # Numbered section headings: "1 Introduction", "4.2 Why Self-Attention"
    if _NUMBERED_HEADING_RE.match(text):
        # Reference entries look like "12 Vaswani, A., Shazeer, N. ..." —
        # a comma shows up almost immediately after the author's surname.
        if "," in text[:25]:
            return False
        # Strip the leading "1.2 " and check the remainder is Title Case,
        # not a sentence like "1 We use 8 attention heads".
        rest = re.sub(r"^\d{1,2}(\.\d{1,2}){0,2}\s+", "", text)
        return _looks_like_title_case(rest.split())

    return False


def _most_common(values: list[float]) -> float:
    from collections import Counter
    return Counter(values).most_common(1)[0][0]
