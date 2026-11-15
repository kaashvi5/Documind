"""Document validation, text extraction (PDF / DOCX / TXT / Markdown) and cleaning."""
from __future__ import annotations

import hashlib
import io
import re
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from config import ALLOWED_EXTENSIONS, MAX_FILE_MB


class DocumentError(Exception):
    """A file cannot be used (unsupported, empty, corrupted, too large, ...)."""


@dataclass
class Page:
    number: int | None  # None when the format has no page concept (DOCX, TXT, MD)
    text: str


@dataclass
class LoadedDocument:
    filename: str
    file_type: str
    pages: list[Page]
    size_bytes: int
    sha256: str
    total_pages: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


# --------------------------------------------------------------------------- cleaning
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_WRAP = re.compile(r"(?<!\n)\n(?!\n)(?!\s*(?:[-*\u2022#>]|\d+[.)]\s))")


def clean_text(text: str) -> str:
    """Normalise whitespace and PDF artefacts while keeping paragraph breaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t\u00a0\u200b]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _SOFT_WRAP.sub(" ", text)                         # join soft-wrapped lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^```.*$", "", text, flags=re.M)
    text = re.sub(r"^[ ]{0,3}#{1,6}[ \t]*(.+?)[ \t]*#*[ \t]*$", r"\1\n", text, flags=re.M)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*([-*_][ \t]*){3,}$", "", text, flags=re.M)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


# ------------------------------------------------------------------------- validation
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: str | Path, max_mb: float = MAX_FILE_MB) -> str:
    """Raise DocumentError for unusable files; return the lower-case extension otherwise."""
    p = Path(path)
    if not p.is_file():
        raise DocumentError(f"File not found: {p.name}")
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise DocumentError(f"Unsupported file type '{ext or 'none'}'. Allowed types: {allowed}.")
    size = p.stat().st_size
    if size == 0:
        raise DocumentError("The file is empty (0 bytes).")
    if size > max_mb * 1024 * 1024:
        raise DocumentError(f"The file is {size / 1048576:.1f} MB; the limit is {max_mb:g} MB.")
    with open(p, "rb") as fh:
        head = fh.read(1024)
    if ext == ".pdf" and b"%PDF" not in head:
        raise DocumentError("Not a valid PDF (missing PDF header).")
    if ext == ".docx" and not zipfile.is_zipfile(p):
        raise DocumentError("Not a valid DOCX file (the Word container is damaged).")
    return ext


# --------------------------------------------------------------------------------- OCR
def ocr_available() -> bool:
    """True when pytesseract, Pillow and the tesseract binary are all present."""
    try:
        import PIL  # noqa: F401
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None


def _ocr_pdf_page(page) -> str:
    import pytesseract
    from PIL import Image

    texts = []
    for image in page.images:  # embedded scans; no poppler needed
        try:
            texts.append(pytesseract.image_to_string(Image.open(io.BytesIO(image.data))))
        except Exception:
            continue
    return "\n".join(texts)


# -------------------------------------------------------------------------- extractors
def _load_pdf(path: Path, use_ocr: bool, warnings: list[str]) -> tuple[list[Page], int]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise DocumentError("The PDF is password-protected.")
        total = len(reader.pages)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Corrupted or unreadable PDF ({type(exc).__name__}).") from exc

    ocr_ok = use_ocr and ocr_available()
    if use_ocr and not ocr_ok:
        warnings.append("OCR was requested but Tesseract is not installed; scanned pages are skipped.")
    pages: list[Page] = []
    failed = ocr_pages = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text, failed = "", failed + 1
        if not text and ocr_ok:
            try:
                text = clean_text(_ocr_pdf_page(page))
                ocr_pages += bool(text)
            except Exception:
                text = ""
        if text:
            pages.append(Page(number, text))
    if failed:
        warnings.append(f"{failed} page(s) could not be read and were skipped.")
    if ocr_pages:
        warnings.append(f"OCR was used for {ocr_pages} page(s).")
    if not pages:
        hint = "" if ocr_ok else " If this is a scanned PDF, enable OCR in Settings."
        raise DocumentError("No extractable text found in the PDF." + hint)
    if len(pages) < total:
        warnings.append(f"{total - len(pages)} page(s) had no text.")
    return pages, total


def _load_docx(path: Path) -> list[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise DocumentError(f"Corrupted or unreadable DOCX ({type(exc).__name__}).") from exc

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if text:
                style = (para.style.name or "") if para.style is not None else ""
                blocks.append(text + ("\n" if style.lower().startswith(("heading", "title")) else ""))
        elif tag == "tbl":
            rows = [[c.text.strip() for c in row.cells] for row in Table(child, document).rows]
            rows = [r for r in rows if any(r)]
            if len(rows) >= 2:  # header + data rows -> one self-contained sentence per row
                header = rows[0]
                for row in rows[1:]:
                    pairs = [f"{h or 'Column'}: {v}" for h, v in zip(header, row) if v]
                    blocks.append("; ".join(pairs) + ".")
            elif rows:
                blocks.append(" | ".join(rows[0]))
    text = clean_text("\n\n".join(blocks))
    if not text:
        raise DocumentError("The DOCX file contains no text.")
    return [Page(None, text)]


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:4096]:
        raise DocumentError("The file looks binary, not plain text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _load_text(path: Path, markdown: bool) -> list[Page]:
    try:
        text = _decode_text(path.read_bytes())
    except DocumentError:
        raise
    except OSError as exc:
        raise DocumentError(f"The file could not be read ({exc.strerror or type(exc).__name__}).") from exc
    if markdown:
        text = _strip_markdown(text)
    text = clean_text(text)
    if not text:
        raise DocumentError("The file contains no readable text.")
    return [Page(None, text)]


def load_document(
    path: str | Path,
    display_name: str | None = None,
    use_ocr: bool = False,
    max_mb: float = MAX_FILE_MB,
) -> LoadedDocument:
    """Validate and extract a document. Raises DocumentError with a user-friendly message."""
    p = Path(path)
    ext = validate_file(p, max_mb)
    warnings: list[str] = []
    try:
        total = 0
        if ext == ".pdf":
            pages, total = _load_pdf(p, use_ocr, warnings)
        elif ext == ".docx":
            pages = _load_docx(p)
        else:
            pages = _load_text(p, markdown=ext in (".md", ".markdown"))
    except DocumentError:
        raise
    except Exception as exc:  # last line of defence: never crash the batch
        raise DocumentError(f"Could not read the file ({type(exc).__name__}: {exc}).") from exc
    return LoadedDocument(
        filename=display_name or p.name,
        file_type=ext.lstrip("."),
        pages=pages,
        size_bytes=p.stat().st_size,
        sha256=sha256_file(p),
        total_pages=total,
        warnings=warnings,
    )
