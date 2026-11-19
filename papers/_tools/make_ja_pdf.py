"""和訳の HTML（papers/キー_ja.html）から和訳の PDF（papers/キー_ja.pdf）を作る。

使い方（リポジトリのルートで）:
    python -I papers/_tools/make_ja_pdf.py papers/2026_tjokrovonco_artificial_ja.html

- HTML の中の図は <img data-page="7" data-rect="x0,y0,x1,y1"> と書く。
  原文 PDF（papers/キー.pdf）の該当ページの該当範囲を PyMuPDF で切り出し、画像として埋め込む。
  （data-rect は PDF の座標。省略するとページ全体）
- PDF への変換は Microsoft Edge のヘッドレス印刷を使う（インストール不要）。
- 必要なもの: PyMuPDF（pip install pymupdf）
"""
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
DPI = 200


def crop_png(doc, page_no, rect):
    page = doc[page_no - 1]
    clip = pymupdf.Rect(*rect) if rect else page.rect
    return page.get_pixmap(dpi=DPI, clip=clip).tobytes("png")


def main():
    src = Path(sys.argv[1]).resolve()
    key = src.name.removesuffix("_ja.html")
    orig = src.with_name(f"{key}.pdf")
    out = src.with_name(f"{key}_ja.pdf")
    doc = pymupdf.open(orig)
    html = src.read_text(encoding="utf-8")

    def embed(m):
        tag = m.group(0)
        page = int(re.search(r'data-page="(\d+)"', tag).group(1))
        r = re.search(r'data-rect="([^"]+)"', tag)
        rect = [float(v) for v in r.group(1).split(",")] if r else None
        b64 = base64.b64encode(crop_png(doc, page, rect)).decode()
        return re.sub(r"<img", f'<img src="data:image/png;base64,{b64}"', tag, count=1)

    html = re.sub(r"<img[^>]*data-page=[^>]*>", embed, html)
    edge = next(p for p in EDGE_CANDIDATES if Path(p).exists())
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "page.html"
        tmp.write_text(html, encoding="utf-8")
        subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={out}", tmp.as_uri()],
            check=True, capture_output=True,
        )
    print(f"{out}  ({pymupdf.open(out).page_count} pages)")


if __name__ == "__main__":
    main()
