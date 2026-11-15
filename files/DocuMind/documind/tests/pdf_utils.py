"""Minimal dependency-free PDF writer (text only) used by the tests and the sample generator."""
from __future__ import annotations

import textwrap
from pathlib import Path


def _escape(line: str) -> bytes:
    return line.encode("latin-1", "replace").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def make_pdf(pages: list[str], path: str | Path) -> Path:
    """Write one PDF page per string. Paragraphs are separated by blank lines."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines: list[str] = []
        for paragraph in text.strip().split("\n\n"):
            lines.extend(textwrap.wrap(" ".join(paragraph.split()), 88) or [""])
            lines.append("")
        stream = b"BT /F1 11 Tf 15 TL 50 740 Td\n" + b"".join(b"(" + _escape(l) + b") Tj T*\n" for l in lines) + b"ET"
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                        f"/Contents {5 + 2 * i} 0 R >>").encode())
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path = Path(path)
    path.write_bytes(bytes(out))
    return path
