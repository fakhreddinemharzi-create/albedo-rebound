"""Figure styling (SciencePlots; Garrett, 2021) at the exact column widths
of the Elsevier ``elsarticle`` 5p layout, fonts matched to the manuscript
(Times via newtx), perceptually uniform colour maps (Crameri et al., 2020),
and an automatic audit of overlapping text, clipped labels and legends or
annotations covering data in every Matplotlib figure before it is saved."""
from __future__ import annotations

import itertools
import json
import os

import matplotlib.collections
import matplotlib.patches
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MM = 1.0 / 25.4
PT = 1.0 / 72.27
# elsarticle 5p: \columnwidth = 252 pt, \textwidth = 522 pt (measured)
SINGLE = 252.0 * PT
DOUBLE = 522.0 * PT
ONEHALF = 0.68 * DOUBLE
AUDIT_LOG = Path(__file__).resolve().parents[1] / "figures" / "audit_log.json"


def setup(usetex=True, fontsize=7.5):
    """Apply the SciencePlots 'science' style (github.com/garrettj403/SciencePlots)."""
    import scienceplots  # noqa: F401  (registers the styles)
    styles = ["science"] if usetex else ["science", "no-latex"]
    plt.style.use(styles)
    matplotlib.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize - 0.5,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "standard",
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "lines.linewidth": 1.0,
        "text.latex.preamble": (r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}"
                                r"\usepackage{siunitx}\sisetup{detect-all}"),
        "font.family": "serif",
        "grid.linewidth": 0.3,
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "axes.titlepad": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _hidden_tick_labels(fig):
    """Tick-label artists whose tick lies outside the axis view interval
    (Matplotlib creates but does not draw them)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            logscale = axis.get_scale() == "log"
            if logscale:
                lo, hi = np.log10(max(lo, 1e-300)), np.log10(max(hi, 1e-300))
            eps = 1e-6 * abs(hi - lo)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tick.get_loc()
                if loc is not None and logscale:
                    loc = np.log10(loc) if loc > 0 else -np.inf
                if loc is None or not (lo - eps <= loc <= hi + eps):
                    hidden.add(id(tick.label1)); hidden.add(id(tick.label2))
    return hidden


def _text_artists(fig):
    hidden = _hidden_tick_labels(fig)
    # tick and axis labels of axes switched off with ax.axis("off") are not drawn
    for ax in fig.axes:
        if not ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                for t in axis.get_ticklabels(which="both") + [axis.label]:
                    hidden.add(id(t))
    arts = []
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in hidden:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_alpha() == 0:
            continue
        # skip artists belonging to invisible axes (e.g. colour-bar helpers)
        ax = t.axes
        if ax is not None and not ax.get_visible():
            continue
        arts.append(t)
    return arts


def _bbox_overlap(b1, b2):
    ow = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    oh = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    return max(ow, 0.0) * max(oh, 0.0), ow, oh


def _text_box(t, renderer):
    patch = t.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer)
    return matplotlib.text.Text.get_window_extent(t, renderer)


def _segments_hit_box(P, box, shrink=1.0):
    """True if the polyline ``P`` (display coords, n x 2) crosses the box
    shrunk by ``shrink`` pixels (Liang-Barsky clipping of every segment)."""
    x0, y0, x1, y1 = box.x0 + shrink, box.y0 + shrink, box.x1 - shrink, box.y1 - shrink
    if x1 <= x0 or y1 <= y0 or len(P) == 0:
        return False
    P = np.asarray(P, float)
    if len(P) == 1:
        return bool(x0 < P[0, 0] < x1 and y0 < P[0, 1] < y1)
    A, B = P[:-1], P[1:]
    ok = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    A, B = A[ok], B[ok]
    if not len(A):
        return False
    d = B - A
    t0 = np.zeros(len(A)); t1 = np.ones(len(A)); alive = np.ones(len(A), bool)
    for pk, qk in ((-d[:, 0], A[:, 0] - x0), (d[:, 0], x1 - A[:, 0]), (-d[:, 1], A[:, 1] - y0),
                   (d[:, 1], y1 - A[:, 1])):
        par = np.abs(pk) < 1e-12
        alive &= ~(par & (qk < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = qk / pk
        neg = (pk < 0) & ~par
        pos = (pk > 0) & ~par
        t0 = np.where(neg, np.maximum(t0, r), t0)
        t1 = np.where(pos, np.minimum(t1, r), t1)
    alive &= t0 <= t1
    return bool(alive.any())


def _axes_paths(a):
    """Display-coordinate polylines of the visible lines, line collections,
    contour sets and bar/rectangle patches of an axes."""
    out = []
    for ln in a.get_lines():
        if not ln.get_visible() or ln.get_linestyle() in ("None", "", " ", None) or ln.get_linewidth() <= 0:
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        out.append(("line", xy))
    for col in a.collections:
        if not col.get_visible() or isinstance(col, matplotlib.collections.PathCollection):
            continue
        if isinstance(col, matplotlib.collections.QuadMesh) or type(col).__name__ in ("QuadMesh", "PolyQuadMesh"):
            continue
        lw = np.atleast_1d(col.get_linewidth())
        if isinstance(col, (matplotlib.collections.LineCollection,)) or type(col).__name__ == "ContourSet":
            if lw.size and np.all(lw <= 0):
                continue
            tr = col.get_transform()
            for path in col.get_paths():
                for poly in path.to_polygons(closed_only=False):
                    if len(poly):
                        out.append(("line", tr.transform(poly)))
    for pt in a.patches:
        if pt.get_gid() == "span":        # background shading, labelled on purpose
            continue
        if isinstance(pt, matplotlib.patches.Rectangle) and pt.get_visible() and pt.get_width() != 0:
            bb = pt.get_window_extent()
            out.append(("bar", np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1],
                                         [bb.x0, bb.y0]])))
            out.append(("barfill", bb))
    return out


def audit_overlaps(fig, name, tol_px=0.5, ignore=(), allow_outside_legend=False, check_data=True,
                   check_lines=True, min_gap_pt=0.8):
    """Audit a laid-out figure for layout defects.

    Checks (all in display pixels):
      * text-text overlaps (titles, labels, tick labels, annotations, legend
        entries), ignoring annotation arrows;
      * text clipped by the canvas edge;
      * text extending over a *different* axes (e.g. a title running under a
        colour bar);
      * legends that spill outside their own axes (unless allowed);
      * legend boxes and in-axes text covering plotted markers/points;
      * legend boxes and in-axes text crossing lines, contours or bars;
      * texts closer than ``min_gap_pt`` (e.g. tick labels running together).
    Returns a list of problems and appends a record to ``AUDIT_LOG``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    texts = _text_artists(fig)
    boxes = []
    for t in texts:
        try:
            bb = _text_box(t, renderer)
        except Exception:
            continue
        if not np.all(np.isfinite(bb.extents)) or bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))
    problems = []
    for t, bb in boxes:
        if bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px:
            problems.append({"type": "clipped", "text": t.get_text()[:40],
                             "bbox": [round(v, 1) for v in bb.extents]})
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if t1.get_text() in ignore or t2.get_text() in ignore:
            continue
        area, ow, oh = _bbox_overlap(b1, b2)
        if ow > tol_px and oh > tol_px:
            problems.append({"type": "overlap", "a": t1.get_text()[:40], "b": t2.get_text()[:40],
                             "area_px": round(area, 1)})
    axes = [a for a in fig.axes if a.get_visible()]
    ax_boxes = {id(a): a.get_window_extent(renderer) for a in axes}
    legend_texts = set()
    for a in axes:
        lg = a.get_legend()
        if lg is not None:
            for tt in lg.get_texts():
                legend_texts.add(id(tt))
    # text over a different axes
    for t, bb in boxes:
        own = t.axes
        if t.get_text() in ignore:
            continue
        for a in axes:
            if own is not None and a is own:
                continue
            if own is None:
                continue
            abb = ax_boxes[id(a)]
            obb = ax_boxes.get(id(own))
            if obb is None:          # inset axes are children of an axes, not in fig.axes
                obb = own.get_window_extent(renderer)
            # inset relationship: the text's own axes lies inside axes a -> not a defect
            if obb is not None and obb.x0 >= abb.x0 - 1 and obb.x1 <= abb.x1 + 1 and \
                    obb.y0 >= abb.y0 - 1 and obb.y1 <= abb.y1 + 1:
                continue
            area, ow, oh = _bbox_overlap(bb, abb)
            if ow > 1.5 and oh > 1.5 and id(t) not in legend_texts:
                # tick labels of shared/adjacent axes routinely touch neighbours' frames only
                problems.append({"type": "text_over_axes", "text": t.get_text()[:40], "area_px": round(area, 1)})
                break
    # legends outside their axes and legends/texts covering data points
    for a in axes:
        abb = ax_boxes[id(a)]
        lg = a.get_legend()
        covers = []
        if lg is not None and lg.get_visible():
            lbb = lg.get_window_extent(renderer)
            if not allow_outside_legend and lg.get_bbox_to_anchor() is not None:
                inside = (lbb.x0 >= abb.x0 - 1 and lbb.x1 <= abb.x1 + 1 and lbb.y0 >= abb.y0 - 1 and
                          lbb.y1 <= abb.y1 + 1)
                if not inside:
                    problems.append({"type": "legend_outside_axes", "axes": a.get_title(loc="left")[:40]})
            covers.append(("legend", lbb))
        for tt in a.texts:
            if tt.get_visible() and tt.get_text().strip() and id(tt) not in legend_texts:
                try:
                    covers.append((tt.get_text()[:30], _text_box(tt, renderer)))
                except Exception:
                    pass
        if not check_data or not covers:
            continue
        pts = []
        for ln in a.get_lines():
            if not ln.get_visible() or ln.get_marker() in (None, "None", "", " "):
                continue
            xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
            if xy.size:
                pts.append(a.transData.transform(xy))
        for col in a.collections:
            if isinstance(col, matplotlib.collections.PathCollection) and col.get_visible():
                off = col.get_offsets()
                if len(off):
                    pts.append(col.get_offset_transform().transform(np.asarray(off, float)))
        if not pts:
            continue
        P = np.vstack(pts)
        P = P[np.all(np.isfinite(P), axis=1)]
        for lab, cb in covers:
            k = int(np.sum((P[:, 0] > cb.x0 + 1) & (P[:, 0] < cb.x1 - 1) & (P[:, 1] > cb.y0 + 1) &
                           (P[:, 1] < cb.y1 - 1)))
            if k > 0:
                problems.append({"type": "covers_data", "what": lab, "n_points": k})
    # minimum clearance between texts (tick labels running together)
    gap = min_gap_pt * fig.dpi / 72.0
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if t1.get_text() in ignore or t2.get_text() in ignore:
            continue
        area, ow, oh = _bbox_overlap(b1, b2)
        if ow > tol_px and oh > tol_px:
            continue  # already reported as an overlap
        if (-gap < ow <= tol_px and oh > 0.5 * min(b1.height, b2.height)) or \
                (-gap < oh <= tol_px and ow > 0.5 * min(b1.width, b2.width)):
            problems.append({"type": "too_close", "a": t1.get_text()[:40], "b": t2.get_text()[:40],
                             "gap_px": round(float(-min(ow, oh)), 2)})
    # in-axes texts anchored inside their axes must not run over the axes frame
    for a in axes:
        abb = ax_boxes[id(a)]
        for tt in a.texts:
            if not tt.get_visible() or not tt.get_text().strip() or tt.get_text() in ignore:
                continue
            try:
                px, py = tt.get_transform().transform(tt.get_unitless_position())
                bb = _text_box(tt, renderer)
            except Exception:
                continue
            inside = abb.x0 < px < abb.x1 and abb.y0 < py < abb.y1
            if inside and (bb.x0 < abb.x0 - 0.5 or bb.x1 > abb.x1 + 0.5 or bb.y0 < abb.y0 - 0.5 or
                           bb.y1 > abb.y1 + 0.5):
                problems.append({"type": "text_exceeds_axes", "text": tt.get_text()[:40],
                                 "axes": a.get_title(loc="left")[:30]})
    # in-axes texts and legends crossing lines, contours or bars of their axes
    if check_lines:
        for a in axes:
            paths = _axes_paths(a)
            if not paths:
                continue
            # twin axes drawn on top of the same frame share their lines
            frame = ax_boxes[id(a)]
            for b in axes:
                if b is not a and np.allclose(ax_boxes[id(b)].extents, frame.extents, atol=1.0):
                    paths = paths + [pp for pp in _axes_paths(b) if pp[0] == "line"]
            items = []
            lg = a.get_legend()
            if lg is not None and lg.get_visible() and not getattr(lg, "_audit_allow_cover", False):
                items.append(("legend", lg.get_window_extent(renderer)))
            for tt in a.texts:
                if not tt.get_visible() or not tt.get_text().strip() or tt.get_text() in ignore:
                    continue
                if getattr(tt, "_audit_skip", False):
                    continue
                try:
                    items.append((tt.get_text()[:30], _text_box(tt, renderer)))
                except Exception:
                    pass
            for lab, bb in items:
                for kind, P in paths:
                    if kind == "barfill":
                        area, ow, oh = _bbox_overlap(bb, P)
                        hit = ow > 1.0 and oh > 1.0
                    else:
                        hit = _segments_hit_box(P, bb, shrink=1.0)
                    if hit:
                        problems.append({"type": "crosses_" + ("bar" if kind.startswith("bar") else "line"),
                                         "what": lab, "axes": a.get_title(loc="left")[:30]})
                        break
    rec = {"figure": name, "n_text": len(boxes), "n_problems": len(problems), "problems": problems[:50]}
    try:
        log = json.loads(AUDIT_LOG.read_text()) if AUDIT_LOG.exists() else {}
    except Exception:
        log = {}
    log[name] = rec
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_LOG.write_text(json.dumps(log, indent=1))
    return problems


def save(fig, path, audit=True, ignore=(), strict=True, **kw):
    """Audit, then save as PDF (vector) and PNG (preview)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_dpi = kw.pop("pdf_dpi", 300)     # raster layers inside the vector PDF (print quality)
    probs = audit_overlaps(fig, path.stem, ignore=ignore, **kw) if audit else []
    if probs:
        print(f"[audit] {path.stem}: {len(probs)} problem(s)")
        for p_ in probs[:30]:
            print("   ", json.dumps(p_))
    if os.environ.get("ALRB_AUDIT_REPORT_ONLY"):
        strict = False
    if probs and strict:
        msg = "\n".join(json.dumps(p) for p in probs[:20])
        raise RuntimeError(f"overlap audit failed for {path.stem}:\n{msg}")
    fig.savefig(path.with_suffix(".pdf"), dpi=pdf_dpi)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    return probs


def hspan(ax, y0, y1, **kw):
    """Horizontal background band (not audited as a bar: labels may sit on it)."""
    r = ax.axhspan(y0, y1, **kw)
    r.set_gid("span")
    return r


def vspan(ax, x0, x1, **kw):
    r = ax.axvspan(x0, x1, **kw)
    r.set_gid("span")
    return r


def panel_label(ax, label, x=0.0, y=1.0, dx=-2.0, dy=4.0, **kw):
    """Bold panel letter above the top-left corner of the axes, offset in
    points so that it clears the top tick label whatever the axes size."""
    txt = rf"\textbf{{({label})}}" if matplotlib.rcParams["text.usetex"] else f"({label})"
    kw.setdefault("ha", "right")
    kw.setdefault("va", "bottom")
    ax.annotate(txt, xy=(x, y), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=matplotlib.rcParams["font.size"] + 0.5, annotation_clip=False, **kw)


def hillshade(dem, res, azdeg=315, altdeg=40, vert_exag=1.5):
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(dem, vert_exag=vert_exag, dx=res, dy=res)


def scalebar(ax, length_m, x0, y0, unit="km", lw=2.0, color="k", fs=None):
    fs = fs or matplotlib.rcParams["font.size"] - 0.5
    ax.plot([x0, x0 + length_m], [y0, y0], color=color, lw=lw, solid_capstyle="butt")
    lab = f"{length_m / 1000:g} {unit}" if unit == "km" else f"{length_m:g} m"
    ax.text(x0 + length_m / 2, y0, lab, ha="center", va="bottom", fontsize=fs, color=color)


def north_arrow(ax, x, y, size, color="k", fs=None):
    fs = fs or matplotlib.rcParams["font.size"]
    ax.annotate("N", xy=(x, y + size), xytext=(x, y), ha="center", va="top", fontsize=fs,
                color=color, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))


def sci_tex(x, digits=1):
    """Format a number as LaTeX scientific notation, e.g. 3.2\\times10^{-15}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{digits}f}" + r"\times10^{" + str(e) + "}"
