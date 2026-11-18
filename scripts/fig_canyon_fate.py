"""Figure: where the additional reflection of a reflective street goes, as
a function of the canyon aspect ratio (Durham, east-west street), and the
effect of the wall albedo on the urban-form factor (all sites).
Writes results/canyon_fatecurve_v3.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import canyon, plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, RES  # noqa: E402

TAG = "v3"
TP = np.radians(np.linspace(0.0, 89.75, 360))
HS = np.round(np.concatenate([np.arange(0.1, 1.0, 0.1), np.arange(1.0, 3.01, 0.25)]), 3)


def main():
    q = RES / f"canyon_fatecurve_{TAG}.json"
    if q.exists():
        fc = json.loads(q.read_text())
    else:
        inp = canyon.site_inputs("durham")
        fc = dict(h=HS.tolist(), floor=[], walls=[], escape=[])
        for h in HS:
            c = canyon.build(float(h), 60)
            dd, db = canyon.absorbed_change(c, 0.14, 0.45, 0.30, TP)
            seg = canyon.weighted_segments(inp, dd, db, TP, 90.0)
            surf = c.kind != 2
            L, k = c.length[surf], c.kind[surf]
            f = float(np.sum((seg * L)[k == 0]))
            w = float(np.sum((seg * L)[k != 0]))
            fc["floor"].append(-f)
            fc["walls"].append(w)
            fc["escape"].append(-(f + w))
        q.write_text(json.dumps(fc, indent=1))
    cy = json.loads((RES / f"canyon_{TAG}.json").read_text())
    P.setup()
    fig = plt.figure(figsize=(P.SINGLE, 4.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    # (a) fate of the additional reflection
    ax = fig.add_subplot(gs[0, 0])
    h = np.concatenate([[0.0], fc["h"]])
    esc = np.concatenate([[1.0], fc["escape"]])
    wal = np.concatenate([[0.0], fc["walls"]])
    flo = np.concatenate([[1.0], fc["floor"]])
    ax.fill_between(h, 0, esc, color=C.CREDIT, alpha=0.75, lw=0, label="leaves the canyon")
    ax.fill_between(h, esc, esc + wal, color=C.WARM, alpha=0.75, lw=0, label="absorbed by the walls")
    ax.plot(h, flo, color=C.INK, lw=1.0, label="removed from the street")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"Aspect ratio $H/W$")
    ax.set_ylabel(r"Per unit $\Delta\alpha\,S^{\downarrow}$ of an open street")
    ax.legend(loc="upper right", fontsize=5.6, borderaxespad=0.3, ncol=1)
    P.panel_label(ax, "a")
    # (b) wall albedo
    ax = fig.add_subplot(gs[1, 0])
    hh = np.array(cy["h"])
    cmap = matplotlib.colormaps["cividis"]
    aws = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60"]
    for i, aw in enumerate(aws):
        arr = np.array([cy["sites"][k]["EW_by_aw"][aw] for k in C.SITE_ORDER])
        col = cmap(i / (len(aws) - 1))
        ax.fill_between(hh, arr.min(0), arr.max(0), color=col, alpha=0.25, lw=0)
        ax.plot(hh, np.median(arr, 0), color=col, lw=0.9, label=rf"$\alpha_w={float(aw):.1f}$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Aspect ratio $H/W$")
    ax.set_ylabel(r"Urban-form factor $\psi$ (east--west)")
    ax.legend(loc="upper right", fontsize=5.6, borderaxespad=0.3, ncol=2, columnspacing=0.8)
    P.panel_label(ax, "b")
    P.save(fig, FIG / "fig_canyon_fate")


if __name__ == "__main__":
    main()
