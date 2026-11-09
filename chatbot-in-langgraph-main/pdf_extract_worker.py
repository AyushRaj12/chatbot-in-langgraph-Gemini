"""Isolated PDF text extractor used by the Streamlit process."""

import json
import resource
import sys
from io import BytesIO

from pypdf import PdfReader

MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_PDF_TEXT_CHARS = 100_000


def _apply_limits() -> None:
    limits = (
        (resource.RLIMIT_CPU, (8, 10)),
        (resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024)),
        (resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024)),
    )
    for kind, value in limits:
        try:
            resource.setrlimit(kind, value)
        except (OSError, ValueError):
            pass


def extract(pdf_bytes: bytes) -> dict:
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The PDF exceeds the 5 MB upload limit.")
    if not pdf_bytes.lstrip().startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    reader = PdfReader(BytesIO(pdf_bytes), strict=True)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDFs are limited to {MAX_PDF_PAGES} pages.")

    pages = []
    total_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        remaining = MAX_PDF_TEXT_CHARS - total_chars
        if remaining <= 0:
            break
        text = (page.extract_text() or "")[:remaining].strip()
        if text:
            pages.append({"page": page_number, "text": text})
            total_chars += len(text)

    if not pages:
        raise ValueError("No readable text was found in the PDF.")

    return {
        "pages": pages,
        "total_chars": total_chars,
        "truncated": total_chars >= MAX_PDF_TEXT_CHARS,
    }


def main() -> int:
    _apply_limits()
    try:
        result = extract(sys.stdin.buffer.read(MAX_PDF_BYTES + 1))
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
        return 0
    except ValueError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    except Exception:
        sys.stdout.write(
            json.dumps({"ok": False, "error": "The PDF could not be read safely."})
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
