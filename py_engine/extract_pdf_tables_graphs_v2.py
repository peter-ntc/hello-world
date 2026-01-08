#!/usr/bin/env python3
"""
extract_pdf_tables_graphs_v2.py

PDF-only extractor that saves **ONLY**:
  1) Tables (ruled/grid tables)
  2) Graphs/Charts (plots, bar charts, line charts, etc.)

V2 changes (vs v1):
- Output visuals are written into: <output_root>/<MANAGER_ID>_visuals/
- Each extracted visual file is named: <DOCUMENT_ID>_<fileNumber>.png (fileNumber is 4-digit, 0001, 0002, ...)
- Raster extraction is NOT included (keeps output clean; preserves v1 speed profile)

Detection approach (same as v1, optimized):
- VECTOR candidates:
  - Uses PyMuPDF page.get_drawings() to locate clusters of vector strokes.
  - Classifies clusters as TABLE (dense horizontal+vertical strokes) or GRAPH (stroke-rich, non-table).
- Renders each page once, crops detected regions, saves PNGs.

Dependencies:
  pip install pymupdf numpy opencv-python

Usage (Windows CMD):
  python extract_pdf_tables_graphs_v2.py ^
    --input "C:/path/to/file.pdf" ^
    --manager "Asana" ^
    --manager_id 10067 ^
    --document_id 55555 ^
    --document_date 2024-11-01 ^
    --document_source "Manager" ^
    --pages "7-14"

If --pages is omitted/blank, ALL pages are processed.

Output root selection:
- If C:/Users/<user>/Downloads exists, uses that.
- Else uses the input PDF directory.

Output:
  <output_root>/<manager_id>_visuals/
    <document_id>_0001.png
    <document_id>_0002.png
    ...
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2


@dataclass
class Config:
    visuals_dir: Path
    dpi: int = 220

    # Minimum crop size (in rendered pixels)
    min_w: int = 520
    min_h: int = 320

    # Merge distance for vector clusters (in PDF points)
    merge_pad: float = 6.0

    # Table classification thresholds (vector)
    v_table_min_segments: int = 60
    v_table_min_hv_ratio: float = 0.65  # share of segments that are horiz/vert
    v_table_min_balance: float = 0.25   # min(h,v)/max(h,v)

    # Graph classification thresholds (vector)
    v_graph_min_segments: int = 45
    v_graph_min_area_pts2: float = 120 * 120


# -----------------------------
# Utility: rectangles & merging
# -----------------------------
def rect_union(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def rect_expand(r: fitz.Rect, pad: float) -> fitz.Rect:
    return fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)


def rect_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)


def merge_rects(rects: List[fitz.Rect], pad: float) -> List[List[int]]:
    """
    Return clusters of indices where rects overlap after expansion by pad.
    Union-Find; O(n^2) over drawings on a page (usually manageable).
    """
    n = len(rects)
    parents = list(range(n))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parents[rj] = ri

    expanded = [rect_expand(r, pad) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            if rect_intersects(expanded[i], expanded[j]):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


# -----------------------------
# Vector analysis (tables/graphs)
# -----------------------------
def segment_counts_from_drawing(d: dict) -> Tuple[int, int, int, int]:
    """
    Count line-like segments and their orientation from a PyMuPDF drawing dict.
    Returns (total_segments, horiz_vert_segments, horiz_count, vert_count).
    """
    total = 0
    hv = 0
    h = 0
    v = 0

    for item in d.get("items", []):
        if not item:
            continue
        t = item[0]

        # Lines
        if t == "l" and len(item) >= 3:
            p1 = item[1]
            p2 = item[2]
            dx = float(p2.x - p1.x)
            dy = float(p2.y - p1.y)
            length = math.hypot(dx, dy)
            if length < 2.0:
                continue
            total += 1
            # near horizontal / vertical
            if abs(dy) <= 1.2 and abs(dx) > 3.0:
                hv += 1
                h += 1
            elif abs(dx) <= 1.2 and abs(dy) > 3.0:
                hv += 1
                v += 1

        # Rectangles / quads approximate many ruled-line visuals
        elif t in ("re", "qu"):
            total += 4
            hv += 4
            h += 2
            v += 2

        # Curves count as "segments" but not hv
        else:
            if t in ("c", "v", "y"):
                total += 1

    return total, hv, h, v


def classify_vector_cluster(drawings: List[dict], bbox: fitz.Rect, cfg: Config) -> str | None:
    """
    Return 'table' or 'graph' or None.
    """
    total = hv = h = v = 0
    for d in drawings:
        t, hv_i, h_i, v_i = segment_counts_from_drawing(d)
        total += t
        hv += hv_i
        h += h_i
        v += v_i

    area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
    if total <= 0:
        return None

    hv_ratio = hv / max(total, 1)
    balance = min(h, v) / max(max(h, v), 1)

    # Table: lots of segments, mostly horiz/vert, and both h & v present.
    if (total >= cfg.v_table_min_segments and hv_ratio >= cfg.v_table_min_hv_ratio and balance >= cfg.v_table_min_balance):
        return "table"

    # Graph: sizable area and enough segments (including curves), but not a table.
    if (total >= cfg.v_graph_min_segments and area >= cfg.v_graph_min_area_pts2):
        return "graph"

    return None


# -----------------------------
# Rendering and saving crops
# -----------------------------
def render_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def crop_rendered(page_img: np.ndarray, bbox: fitz.Rect, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    x0 = int(max(0, math.floor(bbox.x0 * zoom)))
    y0 = int(max(0, math.floor(bbox.y0 * zoom)))
    x1 = int(min(page_img.shape[1], math.ceil(bbox.x1 * zoom)))
    y1 = int(min(page_img.shape[0], math.ceil(bbox.y1 * zoom)))
    return page_img[y0:y1, x0:x1]


def save_png(bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


# -----------------------------
# Page parsing
# -----------------------------
def parse_pages(pages: str, page_count: int) -> List[int]:
    """
    Parse a 1-indexed pages string like '1-5,8,10-12' into 0-indexed page indices.
    Blank -> all pages.
    """
    if not pages:
        return list(range(page_count))
    out: List[int] = []
    parts = [p.strip() for p in pages.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = max(1, int(a.strip()))
            b_i = min(page_count, int(b.strip()))
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            for p in range(a_i, b_i + 1):
                out.append(p - 1)
        else:
            p = int(part.strip())
            if 1 <= p <= page_count:
                out.append(p - 1)

    # dedupe preserving order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


# -----------------------------
# Output root selection
# -----------------------------
def choose_output_root(input_path: Path) -> Path:
    """
    Use <HOME>\\Downloads if it exists; else input file directory.
    """
    home = Path.home()
    downloads = home / "Downloads"
    if downloads.exists() and downloads.is_dir():
        return downloads
    return input_path.parent


# -----------------------------
# Main extraction
# -----------------------------
def extract_vector_tables_graphs(doc: fitz.Document, cfg: Config, page_indices: List[int], document_id: int) -> int:
    """
    Saves crops into cfg.visuals_dir as <document_id>_<fileNumber>.png
    Returns number saved.
    """
    saved = 0
    file_no = 0

    for page_index in page_indices:
        page = doc.load_page(page_index)

        drawings = page.get_drawings()
        if not drawings:
            continue

        rects: List[fitz.Rect] = []
        for d in drawings:
            r = d.get("rect", None)
            if r is None:
                continue
            rects.append(fitz.Rect(r))

        if not rects:
            continue

        clusters = merge_rects(rects, pad=cfg.merge_pad)

        # Render page once (keeps v1 efficiency)
        page_img = render_page_to_bgr(page, cfg.dpi)

        for cluster in clusters:
            cluster_drawings = [drawings[i] for i in cluster]
            bbox = rects[cluster[0]]
            for idx in cluster[1:]:
                bbox = rect_union(bbox, rects[idx])

            bbox = rect_expand(bbox, 2.0)

            kind = classify_vector_cluster(cluster_drawings, bbox, cfg)
            if kind is None:
                continue

            crop = crop_rendered(page_img, bbox, cfg.dpi)
            if crop.shape[1] < cfg.min_w or crop.shape[0] < cfg.min_h:
                continue

            file_no += 1
            out_name = f"{document_id}_{file_no:04d}.png"
            out_path = cfg.visuals_dir / out_name

            save_png(crop, out_path)
            saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser(description="Extract ONLY tables and graphs from a PDF (vector-based), with manager/document naming.")
    ap.add_argument("--input", type=Path, required=True, help="Input PDF file")
    ap.add_argument("--manager", type=str, required=True, help="Manager name")
    ap.add_argument("--manager_id", type=int, required=True, help="Manager ID (used for visuals directory)")
    ap.add_argument("--document_id", type=int, required=True, help="Document ID (used for visual filenames)")
    ap.add_argument("--document_date", type=str, required=True, help="Document date YYYY-MM-DD")
    ap.add_argument("--document_source", type=str, required=True, help="Document source: Manager | Townsend | 3rd party")

    ap.add_argument("--pages", type=str, default="", help='Optional pages (1-indexed): "2,4,7" or "7-14" or "2,4,7-14". Default: all.')
    ap.add_argument("--dpi", type=int, default=220, help="Render DPI for vector crops")
    ap.add_argument("--min-w", type=int, default=520, help="Minimum crop width (pixels at given DPI)")
    ap.add_argument("--min-h", type=int, default=320, help="Minimum crop height (pixels at given DPI)")
    ap.add_argument("--merge-pad", type=float, default=6.0, help="Merge padding for vector drawing clusters (PDF points)")
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise ValueError("This v2 extractor is PDF-only. Provide a .pdf input.")

    out_root = choose_output_root(in_path)
    visuals_dir = out_root / f"{args.manager_id}_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        visuals_dir=visuals_dir,
        dpi=args.dpi,
        min_w=args.min_w,
        min_h=args.min_h,
        merge_pad=args.merge_pad,
    )

    doc = fitz.open(str(in_path))
    try:
        page_indices = parse_pages(args.pages, doc.page_count)
        saved = extract_vector_tables_graphs(doc, cfg, page_indices=page_indices, document_id=args.document_id)
    finally:
        doc.close()

    print(f"Saved {saved} visuals to: {visuals_dir.resolve()}")
    print(f"Naming: {args.document_id}_####.png")
    print("Raster extraction: DISABLED")
    if args.pages:
        print(f"Pages: {args.pages}")
    else:
        print("Pages: ALL")


if __name__ == "__main__":
    main()
