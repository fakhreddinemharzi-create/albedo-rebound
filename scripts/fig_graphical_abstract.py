"""Graphical abstract (13.3 x 5.3 cm aspect): the rendered street, the
fifty-year balance at the ten sites and the key numbers."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, MS, RES  # noqa: E402

TAG = "v3"


def values():
    txt = (MS / "generated" / "numbers.tex").read_text()
    return {m.group(1): m.group(2).replace("\\vminus ", "-") for m in re.finditer(r"\\defval\{([^}]*)\}\{([^}]*)\}", txt)}


def main():
    V = values()
    lg = json.loads((RES / f"ledger_{TAG}.json").read_text())
    img = np.load(sorted((FIG / "_render").glob("street_*_320.npy"))[0])
    a = img[..., 3:4]
    rgb = (img[..., :3] * a + (1 - a))[250:1760, 470:2310]
    P.setup(fontsize=7.5)
    fig = plt.figure(figsize=(7.2, 2.88))
    # ---------------------------------------------------------------- render
    ax = fig.add_axes([0.0, 0.0, 0.36, 0.90])
    ax.imshow(rgb, interpolation="lanczos")
    ax.axis("off")
    fig.text(0.012, 0.965, r"\textbf{Reflective coating on a street}", fontsize=8.5, ha="left", va="top")
    fig.text(0.012, 0.885, r"$\alpha$ 0.14$\rightarrow$0.45, decays in months, renewed every 3 yr", fontsize=6.6,
             ha="left", va="top", color=C.GREY)
    # ---------------------------------------------------------------- balance
    ax = fig.add_axes([0.445, 0.16, 0.25, 0.70])
    keys = C.SITE_ORDER[::-1]
    y = np.arange(len(keys))
    rf = np.array([lg[k]["reference"]["dynamic"]["G_rf"] for k in keys])
    co = np.array([lg[k]["reference"]["dynamic"]["G_co"] for k in keys])
    stt = np.array([lg[k]["reference"]["dynamic"]["G_st"] for k in keys])
    net = rf + co + stt
    ax.barh(y, rf + stt, height=0.62, color=C.CREDIT, label="radiative + structural credit")
    ax.barh(y, co, height=0.62, color=C.BURDEN, label="coating emissions")
    ax.plot(net, y, "D", ms=3.0, color=C.INK, mew=0, label="net")
    ax.plot([0, 0], [-0.6, len(keys) - 0.4], color=C.INK, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([C.SITE_SHORT[k] for k in keys], fontsize=6.3)
    ax.set_xlim(-20, 45)
    ax.set_xticks([-20, 0, 20, 40])
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlabel(r"kg CO$_2$e m$^{-2}$ over 50 yr (dynamic GWP$_{100}$)", fontsize=6.4)
    ax.set_ylim(-0.6, len(keys) - 0.4 + 3.1)
    ax.grid(False)
    ax.legend(loc="upper center", fontsize=5.8, ncol=1, borderaxespad=0.2, handlelength=1.2, frameon=False)
    fig.text(0.445, 0.965, r"\textbf{Fifty-year balance, ten climates}", fontsize=8.5, ha="left", va="top")
    # ---------------------------------------------------------------- key numbers
    x0 = 0.735
    items = [
        (rf"{V['lg.phipct']}\,\%", "of the nominal albedo gain is delivered"),
        (rf"{float(V['lg.dynamic.R.min']):.1f}--{float(V['lg.dynamic.R.max']):.1f}$\times$",
         "emissions exceed the credit"),
        (rf"$\approx$4$\times$ more", r"in street canyons with $H/W=1$"),
        (rf"{float(V['ds.gstar.dynamic.min']):.1f}--{float(V['ds.gstar.dynamic.max']):.1f} kg",
         r"CO$_2$e m$^{-2}$ per coat at most, to break even"),
    ]
    fig.text(x0, 0.965, r"\textbf{Carbon rebound}", fontsize=8.5, ha="left", va="top")
    yy = 0.80
    for big, small in items:
        fig.text(x0, yy, big, fontsize=10.5, ha="left", va="top", color=C.BURDEN)
        fig.text(x0, yy - 0.085, small, fontsize=6.6, ha="left", va="top", color=C.INK)
        yy -= 0.195
    P.save(fig, FIG / "graphical_abstract", pdf_dpi=400)
    fig.savefig(FIG / "graphical_abstract.png", dpi=300)


if __name__ == "__main__":
    main()
