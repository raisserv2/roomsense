"""Export the Markdown documents to PDF:  python docs/build_pdfs.py

Needs `pandoc` (Markdown -> HTML) and Chrome or Chromium (HTML -> PDF, headless).
Output: docs/pdf/<name>.pdf
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "pdf"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.5 Inter, "Helvetica Neue", Arial, sans-serif; color: #1f2933; max-width: none; }
h1 { font-size: 21pt; margin: 0 0 8pt; } h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #d5dee6; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } h4 { font-size: 10.5pt; margin: 10pt 0 3pt; }
h2, h3, h4 { break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; }
th, td { border: 1px solid #d5dee6; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f2f6f9; } tr { break-inside: avoid; }
code { font: 8.6pt "DejaVu Sans Mono", Menlo, Consolas, monospace; background: #f2f4f6; padding: 0 2pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: #f2f4f6; padding: 7pt; border-radius: 4pt; white-space: pre-wrap; break-inside: avoid; } pre code { background: none; padding: 0; }
img { max-width: 100%; break-inside: avoid; } blockquote { margin: 6pt 0; padding: 2pt 10pt; border-left: 3px solid #9aa5b1; color: #4b5563; }
a { color: #2a63c9; text-decoration: none; }
"""


def main() -> int:
    pandoc = shutil.which("pandoc")
    chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser", "chrome") if shutil.which(c)),
                  None)
    if not pandoc or not chrome:
        print("Need pandoc and Chrome/Chromium on PATH (sudo apt install pandoc chromium).")
        return 1
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        css = Path(tmp) / "doc.css"
        css.write_text(CSS)
        for md in sorted(DOCS.glob("0*.md")):
            # The HTML is written next to the Markdown so relative image paths keep working.
            html = DOCS / f".{md.stem}.html"
            subprocess.run([pandoc, str(md), "-f", "gfm", "-t", "html5", "-s", "--embed-resources", "-c", str(css),
                            "--metadata", f"title={md.stem}", "--metadata", "pagetitle=RoomSense", "-o", str(html)],
                           check=True, cwd=DOCS)
            text = html.read_text().replace('<h1 class="title">' + md.stem + "</h1>", "")
            html.write_text(text)
            subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={OUT / (md.stem + '.pdf')}", html.as_uri()],
                           check=True, capture_output=True)
            html.unlink()
            print("wrote", OUT / f"{md.stem}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
