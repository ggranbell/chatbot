"""Layout-aware PDF document loader.

Wraps the existing PyMuPDF extraction logic (header/footer detection,
column detection, table extraction, OCR fallback) as a LangChain-compatible
document loader so it can be dropped into any LangChain pipeline.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterator

import fitz
from langchain_core.documents import Document

from rag.config import OCR_DPI, OCR_LANG, TESSERACT_CMD

# ---------------------------------------------------------------------------
# Optional OCR — gracefully degrade if Tesseract is missing
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image
    from pytesseract import TesseractNotFoundError

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    def _is_ocr_available() -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except TesseractNotFoundError:
            return False

    OCR_AVAILABLE = _is_ocr_available()
except ImportError:
    OCR_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Low-level extraction helpers (ported from pdf_to_vectordb.py)
# ═══════════════════════════════════════════════════════════════════════════

# Fonts where glyph codes don't map to normal Unicode text.
# Characters like 'o', 'l', 'n', '·' in these fonts are really bullets/symbols.
_SYMBOL_FONT_RE = re.compile(
    r"symbol|wingding|dingbat|webding|zapfdingbats", re.IGNORECASE
)
_SYMBOL_BULLET_CHARS = re.compile(r"[o·lnuüvŸ\x6F\xB7]")


def _extract_blocks_with_meta(page: fitz.Page) -> list[dict]:
    """Extract text blocks with bounding-box and font metadata."""
    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    page_width = page.rect.width
    page_height = page.rect.height
    blocks: list[dict] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_text_parts: list[str] = []
        font_sizes: list[float] = []
        is_bold = False

        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                font_name = span.get("font", "")
                # Symbol fonts encode bullets as ordinary ASCII letters
                if text.strip() and _SYMBOL_FONT_RE.search(font_name):
                    text = _SYMBOL_BULLET_CHARS.sub("•", text)
                if text.strip():
                    line_text_parts.append(text)
                    font_sizes.append(span.get("size", 0))
                    if span.get("flags", 0) & (1 << 4):
                        is_bold = True
            if line_text_parts:
                block_text_parts.append("".join(line_text_parts))

        full_text = "\n".join(block_text_parts).strip()
        if not full_text:
            continue

        bbox = block.get("bbox", (0, 0, 0, 0))
        blocks.append({
            "text": full_text,
            "bbox": bbox,
            "x0": bbox[0],
            "y0": bbox[1],
            "x1": bbox[2],
            "y1": bbox[3],
            "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 0,
            "max_font_size": max(font_sizes) if font_sizes else 0,
            "is_bold": is_bold,
            "page_width": page_width,
            "page_height": page_height,
        })

    return blocks


def _detect_header_footer_zones(
    pages_blocks: list[list[dict]],
    page_height: float,
    margin_ratio: float = 0.08,
    repeat_threshold: int = 2,
) -> tuple[float, float]:
    """Detect repeated header/footer zones across pages."""
    if not pages_blocks or len(pages_blocks) < 2:
        return 0.0, page_height

    header_zone = page_height * margin_ratio
    footer_zone = page_height * (1 - margin_ratio)

    header_texts: Counter[str] = Counter()
    footer_texts: Counter[str] = Counter()

    for blocks in pages_blocks:
        seen_h: set[str] = set()
        seen_f: set[str] = set()
        for b in blocks:
            norm = re.sub(r"\d+", "#", b["text"].strip().lower())
            if len(norm) > 120:
                continue
            if b["y1"] <= header_zone and norm not in seen_h:
                header_texts[norm] += 1
                seen_h.add(norm)
            if b["y0"] >= footer_zone and norm not in seen_f:
                footer_texts[norm] += 1
                seen_f.add(norm)

    repeated_header = {t for t, c in header_texts.items() if c >= repeat_threshold}
    repeated_footer = {t for t, c in footer_texts.items() if c >= repeat_threshold}

    header_cutoff = 0.0
    footer_cutoff = page_height
    for blocks in pages_blocks:
        for b in blocks:
            norm = re.sub(r"\d+", "#", b["text"].strip().lower())
            if norm in repeated_header:
                header_cutoff = max(header_cutoff, b["y1"])
            if norm in repeated_footer:
                footer_cutoff = min(footer_cutoff, b["y0"])

    return header_cutoff, footer_cutoff


def _filter_header_footer(
    blocks: list[dict], header_cutoff: float, footer_cutoff: float
) -> list[dict]:
    return [b for b in blocks if b["y0"] >= header_cutoff and b["y1"] <= footer_cutoff]


def _detect_columns(blocks: list[dict], page_width: float) -> list[list[dict]]:
    """Split blocks into reading-order columns."""
    if not blocks:
        return []

    page_center = page_width / 2
    gap_tolerance = page_width * 0.10

    left, right, center = [], [], []
    for b in blocks:
        mid_x = (b["x0"] + b["x1"]) / 2
        block_width = b["x1"] - b["x0"]
        if block_width > page_width * 0.60:
            center.append(b)
        elif mid_x < page_center - gap_tolerance:
            left.append(b)
        elif mid_x > page_center + gap_tolerance:
            right.append(b)
        elif b["x1"] < page_center + gap_tolerance:
            left.append(b)
        else:
            right.append(b)

    if len(left) <= 1 or len(right) <= 1:
        return [sorted(blocks, key=lambda b: b["y0"])]

    columns: list[list[dict]] = []
    if left:
        columns.append(sorted(left, key=lambda b: b["y0"]))
    if right:
        columns.append(sorted(right, key=lambda b: b["y0"]))
    if center:
        for cb in sorted(center, key=lambda b: b["y0"]):
            columns[0].append(cb)
        columns[0].sort(key=lambda b: b["y0"])

    return columns


def _classify_block(block: dict, body_font_size: float) -> str:
    size_ratio = block["avg_font_size"] / body_font_size if body_font_size else 1.0
    if size_ratio >= 1.25 or (block["is_bold"] and size_ratio >= 1.10):
        return "heading"
    if block["text"].count("\t") >= 2 or len(re.findall(r" {3,}", block["text"])) >= 2:
        return "table_row"
    return "body"


def _estimate_body_font_size(blocks: list[dict]) -> float:
    sizes = [b["avg_font_size"] for b in blocks if b["avg_font_size"] > 0]
    return median(sizes) if sizes else 12.0


def _extract_tables_from_page(page: fitz.Page) -> list[str]:
    tables: list[str] = []
    try:
        for table in page.find_tables().tables:
            rows = table.extract()
            if not rows:
                continue
            md = []
            header = rows[0]
            md.append("| " + " | ".join(str(c or "").strip() for c in header) + " |")
            md.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                md.append("| " + " | ".join(str(c or "").strip() for c in row) + " |")
            text = "\n".join(md)
            if text.strip():
                tables.append(text)
    except Exception:
        pass
    return tables


def _blocks_to_structured_text(blocks: list[dict], body_font_size: float) -> str:
    parts: list[str] = []
    prev_type: str | None = None
    import re
    for block in blocks:
        block_type = _classify_block(block, body_font_size)
        text = block["text"].strip()
        if not text:
            continue
        # Insert a space between lowercase-uppercase or number-uppercase (Notion PDF issue)
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        if block_type == "heading":
            # Add extra newline after heading for Notion PDFs
            parts.append(f"\n\n## {text}\n\n")
            prev_type = "heading"
        elif block_type == "table_row":
            if prev_type != "table_row":
                parts.append("\n\n[TABLE]\n")
            parts.append(text)
            prev_type = "table_row"
        else:
            if prev_type == "table_row":
                parts.append("\n[/TABLE]\n\n")
            parts.append(f"\n\n{text}")
            prev_type = "body"
    if prev_type == "table_row":
        parts.append("\n[/TABLE]\n")
    return "".join(parts)


def _layout_aware_page_text(
    page: fitz.Page,
    page_tables: list[str],
    header_cutoff: float,
    footer_cutoff: float,
) -> str:
    """Full layout-aware extraction for a single page."""
    blocks = _extract_blocks_with_meta(page)
    if not blocks:
        return ""

    blocks = _filter_header_footer(blocks, header_cutoff, footer_cutoff)
    if not blocks:
        return ""

    body_font_size = _estimate_body_font_size(blocks)
    page_width = blocks[0]["page_width"] if blocks else page.rect.width

    columns = _detect_columns(blocks, page_width)
    column_texts = []
    for col_blocks in columns:
        structured = _blocks_to_structured_text(col_blocks, body_font_size)
        if structured.strip():
            column_texts.append(structured.strip())

    full_text = "\n\n".join(column_texts)
    if page_tables:
        full_text = f"{full_text}\n\n" + "\n\n".join(page_tables) if full_text else "\n\n".join(page_tables)
    return full_text


def _ocr_page_text(page: fitz.Page, matrix: fitz.Matrix, lang: str) -> tuple[str, bool]:
    """OCR fallback when native extraction yields nothing."""
    native = page.get_text("text") or ""
    if not OCR_AVAILABLE:
        return native, False
    try:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(image, lang=lang) or ""
        if ocr_text.strip():
            return ocr_text, True
        return native, False
    except Exception:
        return native, False


# ═══════════════════════════════════════════════════════════════════════════
# Public loader
# ═══════════════════════════════════════════════════════════════════════════

def load_pdf(
    pdf_path: str | Path,
    *,
    ocr_dpi: int = OCR_DPI,
    ocr_lang: str = OCR_LANG,
    label: str = "",
) -> list[Document]:
    """Load a PDF and return one LangChain ``Document`` per page.

    Each document's ``page_content`` is the layout-aware extracted text and
    ``metadata`` carries source info for downstream processing.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = max(ocr_dpi, 72) / 72
    matrix = fitz.Matrix(zoom, zoom)
    documents: list[Document] = []

    with fitz.open(pdf_path) as doc:
        # Pass 1 — collect blocks for header/footer detection
        all_page_blocks: list[list[dict]] = []
        page_height = 0.0
        for page in doc:
            blocks = _extract_blocks_with_meta(page)
            all_page_blocks.append(blocks)
            if blocks:
                page_height = max(page_height, blocks[0]["page_height"])

        header_cutoff, footer_cutoff = _detect_header_footer_zones(all_page_blocks, page_height)

        # Pass 2 — extract text per page
        for page_num, page in enumerate(doc, start=1):
            page_tables = _extract_tables_from_page(page)
            text = _layout_aware_page_text(page, page_tables, header_cutoff, footer_cutoff)

            used_ocr = False
            if not text.strip() and OCR_AVAILABLE:
                text, used_ocr = _ocr_page_text(page, matrix, ocr_lang)

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "extractor": "ocr" if used_ocr else "layout-aware",
                        "label": label.strip() if label.strip() else pdf_path.stem,
                    },
                )
            )

    return documents
