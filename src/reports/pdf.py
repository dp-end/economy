"""Markdown raporlarını tek PDF'e çevirir: markdown -> HTML (yerel CSS) -> headless Chrome --print-to-pdf.

  python -m src.reports.pdf                      # reports/research/*.md -> reports/research/research_reports.pdf
"""
import argparse
import subprocess
from pathlib import Path

import markdown

from src.reports.common import OUT

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 16mm 14mm; }
html, body { max-width: 100%; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #0b0b0b; background: #fff; }
h1 { font-size: 17pt; margin: 0 0 6pt; border-bottom: 2px solid #2a78d6; padding-bottom: 4pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; color: #184f95; break-after: avoid; }
h3 { font-size: 11pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 7.6pt; width: 100%; max-width: 100%;
        table-layout: auto; break-inside: auto; }
th { background: #f0efec; text-align: left; }
th, td { border: 1px solid #d8d7d2; padding: 2.5pt 4pt; overflow-wrap: anywhere; }
td code, th code { white-space: normal; overflow-wrap: anywhere; }
td { font-variant-numeric: tabular-nums; }
tr { break-inside: avoid; }
img { max-width: 100%; display: block; margin: 8pt auto; break-inside: avoid; }
blockquote { color: #52514e; border-left: 3px solid #d8d7d2; margin: 6pt 0; padding-left: 8pt; }
code { font-size: 8.5pt; background: #f4f3f0; padding: 0 2pt; border-radius: 2pt; }
em:first-child { color: #52514e; }
.report { break-before: page; }
.report:first-of-type { break-before: auto; }
"""


def to_html(md_files: list) -> str:
    parts = []
    for f in md_files:
        body = markdown.markdown(Path(f).read_text(encoding="utf-8"), extensions=["tables", "sane_lists"])
        parts.append(f'<section class="report">{body}</section>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CSE-481 Research Reports</title>'
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")


def build_pdf(md_files: list = None, out: Path = OUT / "research_reports.pdf") -> Path:
    md_files = md_files or sorted(OUT.glob("0*.md"))
    html = OUT / (out.stem + ".html")                     # resimler fig/ altında: HTML aynı klasörde olmalı
    html.write_text(to_html(md_files), encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out.resolve()}", html.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Raporları PDF'e çevir")
    ap.add_argument("files", nargs="*", help="varsayılan: reports/research/0*.md")
    a = ap.parse_args()
    print("yazıldı:", build_pdf(a.files or None))
