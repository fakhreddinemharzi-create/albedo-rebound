"""One palette for every figure (Matplotlib, PyVista and the TikZ diagrams,
see manuscript/tikz/styles.tex).  Colour-blind-safe pairs; sequential and
diverging maps from Crameri's Scientific Colour Maps."""
CREDIT = "#2166AC"      # radiative credit, cooling, treated surface
BURDEN = "#B2182B"      # embodied burden, warming
STRUCT = "#1B7837"      # structural term
WARM = "#E08214"        # untreated / warm
COOL = "#3A8FC4"        # treated / cool
FROST = "#6A51A3"       # frost, ice
INK = "#22262A"
GREY = "#7F868E"
PALE = "#E9ECEF"
GOLD = "#C9A227"

METRIC_COLORS = {"static": "#1F4E79", "dynamic": "#2E86AB", "gtp": "#C73E1D", "stock": "#8E7DBE"}
METRIC_LABELS = {"static": "Static GWP$_{100}$", "dynamic": "Dynamic GWP$_{100}$",
                 "gtp": "Dynamic GTP$_{100}$", "stock": "Stock equivalence"}

SITE_ORDER = ["goodridge", "wolfpoint", "ithaca", "lajunta", "wooster", "bowlinggreen",
              "monahans", "durham", "merced", "tucson"]
SITE_LABEL = {"goodridge": "Goodridge, MN", "wolfpoint": "Wolf Point, MT", "ithaca": "Ithaca, NY",
              "lajunta": "La Junta, CO", "wooster": "Wooster, OH", "bowlinggreen": "Bowling Green, KY",
              "monahans": "Monahans, TX", "durham": "Durham, NC", "merced": "Merced, CA",
              "tucson": "Tucson, AZ"}
SITE_SHORT = {"goodridge": "GDR", "wolfpoint": "WPT", "ithaca": "ITH", "lajunta": "LJT", "wooster": "WST",
              "bowlinggreen": "BWG", "monahans": "MNH", "durham": "DRM", "merced": "MCD", "tucson": "TUS"}


def site_colors():
    """Colour per site, ordered from cold (dark blue) to hot (orange), batlow."""
    from cmcrameri import cm
    import numpy as np
    cols = cm.batlow(np.linspace(0.05, 0.85, len(SITE_ORDER)))
    return {k: cols[i] for i, k in enumerate(SITE_ORDER)}
