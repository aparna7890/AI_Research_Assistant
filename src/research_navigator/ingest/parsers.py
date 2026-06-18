"""PDF and Markdown parsers."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

_REFERENCES_RE = re.compile(r"^(references|bibliography|works cited)\s*$", re.IGNORECASE)


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


def _looks_like_heading(text: str) -> bool:
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
        return True
    if text.isupper() and len(text) < 60:
        return True
    return False


def _most_common(values: list[float]) -> float:
    from collections import Counter
    return Counter(values).most_common(1)[0][0]
