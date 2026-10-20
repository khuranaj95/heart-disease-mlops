"""Build report/report.pdf from report/report.md.

Markdown -> styled HTML -> PDF, printed by headless Firefox (needs Firefox installed;
selenium downloads the matching geckodriver automatically).

Usage (from the repo root):
    python scripts/build_report.py
"""

import base64
from pathlib import Path

import markdown
from selenium import webdriver
from selenium.webdriver.common.print_page_options import PrintOptions
from selenium.webdriver.firefox.options import Options

SRC = Path("report/report.md")
HTML = Path("report/report.html")
PDF = Path("report/report.pdf")

CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1f2937; max-width: 100%; }
h1 { font-size: 20pt; color: #1e3a8a; border-bottom: 3px solid #1e3a8a; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #1e3a8a; border-bottom: 1px solid #cbd5e1; padding-bottom: 2px;
     margin-top: 22px; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #e2e8f0; }
code { background: #f1f5f9; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #0f172a; color: #e2e8f0; padding: 8px 10px; border-radius: 5px; font-size: 8.5pt;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; color: inherit; }
img { max-width: 100%; display: inline-block; margin: 6px 0; break-inside: avoid;
      border: 1px solid #e2e8f0; }
p:has(> img) { text-align: center; break-inside: avoid; }
blockquote { border-left: 4px solid #f59e0b; background: #fffbeb; margin: 8px 0; padding: 4px 10px; }
hr { border: none; border-top: 1px solid #cbd5e1; }
"""


def main():
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "fenced_code"])
    HTML.write_text(
        f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    )

    opts = Options()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
    try:
        # Open the HTML from report/ so relative image paths (../screenshots/...) resolve
        driver.get(HTML.resolve().as_uri())
        print_opts = PrintOptions()
        print_opts.page_width, print_opts.page_height = 21.0, 29.7  # A4 in cm
        print_opts.margin_top = print_opts.margin_bottom = 1.5
        print_opts.margin_left = print_opts.margin_right = 1.6
        PDF.write_bytes(base64.b64decode(driver.print_page(print_opts)))
    finally:
        driver.quit()
    HTML.unlink()  # the HTML is only an intermediate file
    print(f"Saved {PDF}")


if __name__ == "__main__":
    main()
