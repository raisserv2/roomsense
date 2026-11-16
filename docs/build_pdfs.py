"""Export the Markdown documents to PDF with LaTeX:  python docs/build_pdfs.py [name ...]

Needs `pandoc` and a TeX distribution with XeLaTeX and latexmk
(Ubuntu: sudo apt install pandoc texlive-xetex texlive-latex-extra texlive-fonts-extra latexmk).
Fonts: TeX Gyre Pagella (body), Inter (headings), DejaVu Sans Mono (code).
Style: docs/latex/header.tex. Markdown handling: docs/latex/roomsense.lua.
Output: docs/pdf/<name>.pdf
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "pdf"
STYLE = DOCS / "latex"

# file stem -> (document type, one-line description, table of contents?)
META = {
    "00-project-report": ("Project Report",
                          "A self-hosted machine learning lifecycle for room-state recognition from a phone camera", True),
    "01-architecture": ("Architecture", "System blocks, data flow and deployment", False),
    "02-hld": ("High-Level Design", "Design choices and the reasons behind them", True),
    "03-lld": ("Low-Level Design", "Endpoints with input and output specifications, modules, file formats and metrics",
               True),
    "04-test-plan": ("Test Plan and Test Cases", "Scope, strategy, test cases and acceptance criteria", False),
    "05-test-report": ("Test Report", "Results of the automated suite and the acceptance criteria", False),
    "06-user-manual": ("User Manual", "Using RoomSense day to day, written for a non-technical user", False),
}


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def short_tables(tex: str) -> str:
    """Typeset tables of up to 10 rows as tabular, so a short table never splits across pages."""
    out, pos, opener = [], 0, "\\begin{longtable}[]{"
    while (i := tex.find(opener, pos)) != -1:
        j, depth = i + len(opener), 1
        while depth:  # find the brace that closes the column spec
            depth += {"{": 1, "}": -1}.get(tex[j], 0)
            j += 1
        k = tex.index("\\end{longtable}", j)
        spec, body = tex[i + len(opener):j - 1], tex[j:k]
        out.append(tex[pos:i])
        if body.count("\\\\") > 11:
            out.append(tex[i:k + len("\\end{longtable}")])
        else:
            body = re.sub(r"\\endfirsthead\n.*?\\endhead\n", "", body, flags=re.S)
            body = body.replace("\\endhead\n", "").replace("\\bottomrule\\noalign{}\n\\endlastfoot\n", "")
            out.append(f"\\begin{{rstable}}\\begin{{tabular}}{{{spec}}}{body}\\bottomrule\n\\end{{tabular}}\\end{{rstable}}")
        pos = k + len("\\end{longtable}")
    out.append(tex[pos:])
    return "".join(out)


def build(md: Path, tmp: Path) -> Path:
    doctype, blurb, toc = META[md.stem]
    per_doc = tmp / f"{md.stem}-meta.tex"
    per_doc.write_text(
        f"\\newcommand{{\\doctype}}{{{tex_escape(doctype)}}}\n"
        f"\\newcommand{{\\docblurb}}{{{tex_escape(blurb)}}}\n"
        "\\usepackage{graphicx}\n"
        f"\\graphicspath{{{{{DOCS.as_posix()}/}}}}\n")
    tex = tmp / f"{md.stem}.tex"
    cmd = ["pandoc", str(md), "-f", "gfm", "-t", "latex", "-s", "-o", str(tex),
           "--lua-filter", str(STYLE / "roomsense.lua"), "--resource-path", str(DOCS),
           "--pdf-engine=xelatex", "--highlight-style=tango",
           "-H", str(per_doc), "-H", str(STYLE / "header.tex"),
           "-M", f"title=RoomSense: {doctype}", "-M", "author=Khaja Mohammed",
           "-V", "documentclass=article", "-V", "fontsize=11pt", "-V", "papersize=a4",
           "-V", "geometry=a4paper,left=24mm,right=24mm,top=24mm,bottom=22mm,headsep=8mm",
           "-V", "mainfont=TeX Gyre Pagella", "-V", "sansfont=Inter",
           "-V", "monofont=DejaVu Sans Mono", "-V", "monofontoptions=Scale=0.84",
           "-V", "colorlinks=true", "-V", "linkcolor=accent", "-V", "urlcolor=accent2",
           "-V", "toccolor=ink", "-V", "toc-depth=1"]
    if toc:
        cmd += ["-M", "rstoc=true", "-V", "toc-title=Contents"]
    subprocess.run(cmd, check=True, cwd=DOCS)
    tex.write_text(short_tables(tex.read_text()))
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-quiet",
                    f"-outdir={tmp}", str(tex)], check=True, cwd=tmp, capture_output=True)
    pdf = OUT / f"{md.stem}.pdf"
    shutil.copy(tmp / f"{md.stem}.pdf", pdf)
    return pdf


def main(names: list[str]) -> int:
    if not shutil.which("pandoc") or not shutil.which("latexmk") or not shutil.which("xelatex"):
        print(__doc__)
        return 1
    OUT.mkdir(exist_ok=True)
    docs = [DOCS / f"{n}.md" for n in names] if names else sorted(DOCS.glob("0*.md"))
    with tempfile.TemporaryDirectory() as tmp:
        for md in docs:
            try:
                print("wrote", build(md, Path(tmp)).relative_to(DOCS.parent))
            except subprocess.CalledProcessError as e:
                log = Path(tmp) / f"{md.stem}.log"
                if log.exists():
                    lines = log.read_text(errors="replace").splitlines()
                    errs = [i for i, line in enumerate(lines) if line.startswith("!")]
                    for i in errs[:3]:
                        print("\n".join(lines[i:i + 6]))
                print(f"failed: {md.name} ({e})")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
