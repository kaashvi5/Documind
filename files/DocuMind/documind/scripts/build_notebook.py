"""Generate DocuMind_Colab.ipynb: a self-contained notebook that writes every source file and launches the app."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["config.py", "rag/__init__.py", "rag/textutils.py", "rag/loaders.py", "rag/chunking.py", "rag/embeddings.py",
         "rag/vectorstore.py", "rag/retrieval.py", "rag/generator.py", "rag/pipeline.py", "ui/__init__.py",
         "ui/components.py", "ui/dashboard.py", "ui/chat.py", "app.py", "requirements.txt", "pytest.ini",
         "tests/conftest.py", "tests/pdf_utils.py", "tests/test_loaders.py", "tests/test_chunking.py",
         "tests/test_embeddings.py", "tests/test_vectorstore.py", "tests/test_retrieval.py", "tests/test_generator.py",
         "tests/test_pipeline.py", "scripts/make_samples.py", "scripts/generate_example_outputs.py",
         "docs/example_questions.md", "README.md"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


def main() -> None:
    cells = [
        md("# DocuMind: RAG document assistant (Google Colab)\n\n"
           "Run all cells from top to bottom (*Runtime > Run all*). A GPU is optional: *Runtime > Change runtime type > T4 GPU* "
           "makes answers faster, otherwise the CPU is used. No API keys are needed.\n\n"
           "**What happens:** install packages, write the project files, create sample documents, run the tests, load the models "
           "(about 1 GB download the first time), and launch the Gradio app with a public link."),
        md("## 1. Install dependencies\nPyTorch is already installed on Colab, so it is not reinstalled."),
        code("!pip install -q \"gradio>=6.0,<7\" faiss-cpu sentence-transformers transformers sentencepiece pypdf python-docx pillow pytesseract pytest"),
        md("## 2. Optional: OCR for scanned PDFs\nSet `INSTALL_OCR = False` to skip (about 30 seconds)."),
        code("INSTALL_OCR = True\nif INSTALL_OCR:\n    !apt-get -qq install -y tesseract-ocr > /dev/null 2>&1\n    !which tesseract"),
        md("## 3. Create the project folder"),
        code("import os, sys\nBASE = '/content/documind' if os.path.isdir('/content') else os.path.abspath('documind')\n"
             "for d in ['rag', 'ui', 'data/uploads', 'data/vector_db', 'sample_docs', 'tests', 'scripts', 'docs']:\n"
             "    os.makedirs(os.path.join(BASE, d), exist_ok=True)\nos.chdir(BASE)\nsys.path.insert(0, BASE)\nprint('Working in', os.getcwd())"),
        md("## 4. Source code\nEach cell below writes one file of the project (`%%writefile`)."),
    ]
    for name in FILES:
        cells.append(code(f"%%writefile {name}\n" + (ROOT / name).read_text(encoding="utf-8")))
    cells += [
        md("## 5. Create the sample documents"),
        code("!python scripts/make_samples.py\n!ls sample_docs"),
        md("## 6. Run the automated tests\nThey use a lightweight offline embedder, so no model download is needed."),
        code("!python -m pytest -q"),
        md("## 7. Load the models and initialise the pipeline\nUses the GPU automatically when available."),
        code("import torch\nprint('GPU available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
             "from rag.pipeline import DocuMindPipeline\npipeline = DocuMindPipeline()\nfor warning in pipeline.load_models():\n    print('WARNING:', warning)\n"
             "print(pipeline.status())"),
        md("## 8. Launch DocuMind\nOpen the public `gradio.live` link below (or use the inline view). Choose **Load samples** to try it immediately."),
        code("from app import build_app, launch_app\ndemo = build_app(pipeline)\nlaunch_app(demo, share=True, debug=True)"),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "name": "DocuMind_Colab.ipynb"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "accelerator": "GPU"}}
    out = ROOT / "DocuMind_Colab.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
