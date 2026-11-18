"""Layout audit of compiled PDFs (TikZ diagrams and the manuscript).

For every page:
  * overlap  - two words from different text lines whose boxes intersect;
  * struck   - a stroked line or curve segment of the drawing that passes
               through the interior of a word (arrows or lines crossing text);
  * outside  - text or drawing extending beyond the page (or, with --margins,
               beyond the text block of the page).
Word boxes are shrunk slightly (0.35 pt) so that glyphs touching a frame
are not reported.  Exit status 1 if any problem is found.

usage: python tools/audit_pdf.py FILE.pdf [--margins L,R,T,B] [--no-struck] [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import pymupdf as fitz  # PyMuPDF
import numpy as np

SHRINK = 0.35


def _words(page):
    # ink-accurate glyph boxes (not font ascender/descender boxes), so stacked
    # sub- and superscripts in formulas are not reported as overlapping
    flags = fitz.TEXTFLAGS_WORDS | getattr(fitz, "TEXT_ACCURATE_BBOXES", 0)
    ws = page.get_text("words", flags=flags)   # x0, y0, x1, y1, word, block, line, wordno
    out = []
    for w in ws:
        x0, y0, x1, y1, t, b, l, n = w
        if not t.strip():
            continue
        out.append(dict(r=fitz.Rect(x0 + SHRINK, y0 + SHRINK, x1 - SHRINK, y1 - SHRINK), t=t, key=(b, l)))
    return out


def _chars(page):
    """Characters with ink-accurate boxes and the (block, line) they belong to."""
    flags = getattr(fitz, "TEXT_ACCURATE_BBOXES", 0)
    d = page.get_text("rawdict", flags=flags)
    raw = []
    for bi, b in enumerate(d["blocks"]):
        for li, l in enumerate(b.get("lines", [])):
            for sp in l["spans"]:
                for ch in sp["chars"]:
                    if not ch["c"].strip():
                        continue
                    x0, y0, x1, y1 = ch["bbox"]
                    r = fitz.Rect(x0 + SHRINK, y0 + SHRINK, x1 - SHRINK, y1 - SHRINK)
                    if r.width > 0 and r.height > 0:
                        raw.append((r, ch["c"], (bi, li), sp["size"]))
    # a glyph is a script if it is smaller than the text around it (within 15 pt)
    if not raw:
        return []
    cx = np.array([0.5 * (r.x0 + r.x1) for r, *_ in raw])
    cy = np.array([0.5 * (r.y0 + r.y1) for r, *_ in raw])
    sz = np.array([q[3] for q in raw])
    out = []
    for k, (r, c, key, size) in enumerate(raw):
        near = (np.abs(cx - cx[k]) < 15) & (np.abs(cy - cy[k]) < 15)
        out.append((r, c, key, bool(size < 0.85 * np.median(sz[near]))))
    return out


def _char_overlaps(chars, min_side=0.6):
    """Pairs of characters from different lines whose ink boxes intersect
    (spatial hashing on a 12 pt grid)."""
    grid = {}
    for k, (r, c, key, sc) in enumerate(chars):
        for gx in range(int(r.x0 // 12), int(r.x1 // 12) + 1):
            for gy in range(int(r.y0 // 12), int(r.y1 // 12) + 1):
                grid.setdefault((gx, gy), []).append(k)
    seen = set()
    hits = []
    for cell in grid.values():
        for i, j in itertools.combinations(cell, 2):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            (ra, ca, ka, sa), (rb, cb, kb, sb) = chars[i], chars[j]
            if ka == kb:
                continue
            # a TeX math stack: sub- and superscript of one symbol, left-aligned,
            # vertically offset and only partly overlapping
            hmin = min(ra.height, rb.height)
            dcy = abs(0.5 * (ra.y0 + ra.y1) - 0.5 * (rb.y0 + rb.y1))
            ov = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
            if abs(ra.x0 - rb.x0) < 1.5 and dcy > 0.35 * hmin and ov < 0.6 * hmin:
                continue
            inter = ra & rb
            if not inter.is_empty and inter.width > min_side and inter.height > min_side:
                hits.append((ca, cb, inter))
    return hits


def _segments(page):
    segs = []
    for d in page.get_drawings():
        if d.get("type") not in ("s", "fs"):     # stroked paths only
            continue
        if (d.get("width") or 0) <= 0:
            continue
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1], it[2]))
            elif it[0] == "c":
                p0, p1, p2, p3 = it[1], it[2], it[3], it[4]
                ts = np.linspace(0, 1, 9)
                pts = [(1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) +
                       3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3) for t in ts]
                for a, b in zip(pts[:-1], pts[1:]):
                    segs.append((fitz.Point(*a), fitz.Point(*b)))
            elif it[0] == "re":
                r = it[1]
                segs += [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
            elif it[0] == "qu":
                q = it[1]
                segs += [(q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)]
    return segs


def _seg_hits(a, b, r: fitz.Rect):
    """Liang-Barsky: does segment ab enter the open rectangle r?"""
    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = b.x - a.x, b.y - a.y
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a.x - x0), (dx, x1 - a.x), (-dy, a.y - y0), (dy, y1 - a.y)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return t1 - t0 > 1e-6


def audit(path, margins=None, struck=True):
    doc = fitz.open(path)
    problems = []
    for pno, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        words = _words(page)
        # text-text overlaps between different lines, judged on the glyphs
        for ca, cb, inter in _char_overlaps(_chars(page)):
            problems.append(dict(page=pno + 1, type="overlap", a=ca, b=cb,
                                 at=[round(inter.x0, 1), round(inter.y0, 1)]))
        # strokes through glyphs (glyph boxes shrunk to their central part)
        if struck:
            segs = _segments(page)
            reported = set()
            for r, c, key, sc in _chars(page):
                inner = fitz.Rect(r.x0 + 0.25 * r.width, r.y0 + 0.30 * r.height,
                                  r.x1 - 0.25 * r.width, r.y1 - 0.25 * r.height)
                for a, b in segs:
                    if _seg_hits(a, b, inner):
                        k = (key, round(r.x0))
                        if k not in reported:
                            reported.add(k)
                            problems.append(dict(page=pno + 1, type="struck", word=c,
                                                 at=[round(r.x0, 1), round(r.y0, 1)]))
                        break
        # outside the page / text block
        box = fitz.Rect(0, 0, W, H)
        if margins:
            l, r, t, b = margins
            box = fitz.Rect(l, t, W - r, H - b)
        for w in words:
            # thin glyphs (dashes, rules) shrink to empty boxes: test their centre instead
            r = w["r"] if not w["r"].is_empty else fitz.Rect(w["r"].tl, w["r"].tl)
            if r.is_empty and not box.contains(r.tl):
                problems.append(dict(page=pno + 1, type="outside", word=w["t"],
                                     at=[round(r.x0, 1), round(r.y0, 1)]))
                continue
            if r.is_empty:
                continue
            if not box.contains(w["r"]):
                problems.append(dict(page=pno + 1, type="outside", word=w["t"],
                                     at=[round(w["r"].x0, 1), round(w["r"].y0, 1)]))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--margins", default="")
    ap.add_argument("--no-struck", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    margins = [float(x) for x in a.margins.split(",")] if a.margins else None
    probs = audit(a.pdf, margins, not a.no_struck)
    for p in probs[:60]:
        print(json.dumps(p))
    print(f"{a.pdf}: {len(probs)} problem(s)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(probs, f, indent=1)
    sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
