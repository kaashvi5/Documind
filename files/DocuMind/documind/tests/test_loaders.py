import pytest
from pypdf import PdfWriter

from rag.loaders import DocumentError, clean_text, load_document, ocr_available, validate_file
from tests.pdf_utils import make_pdf


def test_txt_extraction(sample_files):
    doc = load_document(sample_files["03_renewable_energy_overview.txt"])
    assert doc.file_type == "txt" and doc.pages[0].number is None
    assert "capacity factor" in doc.pages[0].text and doc.size_bytes > 0 and len(doc.sha256) == 64


def test_markdown_syntax_is_stripped(sample_files):
    text = load_document(sample_files["01_intro_to_machine_learning.md"]).pages[0].text
    assert "Supervised learning uses labeled examples" in text
    assert "**" not in text and "# " not in text


def test_docx_extracts_paragraphs_and_tables(sample_files):
    text = load_document(sample_files["02_remote_work_policy.docx"]).pages[0].text
    assert "90-day probation" in text
    assert "Expense: Home-office stipend; Amount: $500; Frequency: One time." in text


def test_pdf_extraction_keeps_page_numbers(sample_files):
    doc = load_document(sample_files["04_photosynthesis_and_plant_biology.pdf"])
    assert [p.number for p in doc.pages] == [1, 2, 3] and doc.total_pages == 3
    assert "stroma" in doc.pages[1].text and "limiting factor" in doc.pages[2].text


def test_clean_text_rules():
    assert clean_text("exam-\nple of  a   hyphen") == "example of a hyphen"
    assert clean_text("line one\nline two\n\n\n\nnew paragraph") == "line one line two\n\nnew paragraph"
    assert clean_text("a\x00b\x07c\u00a0d") == "a b c d"
    assert clean_text("- item one\n- item two") == "- item one\n- item two"
    assert clean_text("\ufb01ne") == "fine"  # ligature normalised
    assert clean_text("") == "" and clean_text("   \n ") == ""


def test_cp1252_text_is_decoded(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("Caf\xe9 au lait is a popular coffee drink in many countries.".encode("cp1252"))
    assert "Café au lait" in load_document(path).pages[0].text


@pytest.mark.parametrize("name,content,message", [
    ("empty.txt", b"", "empty"),
    ("blank.txt", b"   \n\n  ", "no readable text"),
    ("data.exe", b"MZ....", "Unsupported"),
    ("noext", b"hello", "Unsupported"),
    ("fake.pdf", b"this is not a pdf", "PDF"),
    ("fake.docx", b"this is not a zip", "DOCX"),
    ("binary.txt", b"abc\x00\x01\x02\x00" * 50, "binary"),
])
def test_invalid_files_are_rejected_gracefully(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(DocumentError) as err:
        load_document(path)
    assert message.lower() in str(err.value).lower()


def test_corrupted_pdf_with_valid_header(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\n1 0 obj << /Broken >> endobj\ngarbage garbage")
    with pytest.raises(DocumentError):
        load_document(path)


def test_corrupted_docx_zip(tmp_path):
    import zipfile
    path = tmp_path / "broken.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "not a word file")
    with pytest.raises(DocumentError):
        load_document(path)


def test_pdf_without_text_and_encrypted_pdf(tmp_path):
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    with open(blank, "wb") as fh:
        writer.write(fh)
    with pytest.raises(DocumentError, match="No extractable text"):
        load_document(blank)
    locked = tmp_path / "locked.pdf"
    writer.encrypt("secret")
    with open(locked, "wb") as fh:
        writer.write(fh)
    with pytest.raises(DocumentError, match="password"):
        load_document(locked)


def test_file_size_limit(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("word " * 5000)
    with pytest.raises(DocumentError, match="limit"):
        load_document(path, max_mb=0.001)
    assert validate_file(path) == ".txt"


def test_missing_file():
    with pytest.raises(DocumentError, match="not found"):
        load_document("/nonexistent/file.txt")


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_ocr_reads_scanned_pdf(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1000, 220), "white")
    ImageDraw.Draw(image).text((30, 60), "Quarterly revenue grew strongly", fill="black", font=ImageFont.load_default(size=60))
    path = tmp_path / "scan.pdf"
    image.save(path, "PDF")
    with pytest.raises(DocumentError, match="No extractable text"):
        load_document(path, use_ocr=False)
    doc = load_document(path, use_ocr=True)
    assert "revenue" in doc.pages[0].text.lower() and any("OCR" in w for w in doc.warnings)
