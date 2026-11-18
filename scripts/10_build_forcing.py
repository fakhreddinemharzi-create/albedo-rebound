"""Build the hourly forcing series of the ten sites (QC, gap filling,
derived long-wave, clear-sky and cloud fraction) and a QC summary."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, PROC, RES, SITES  # noqa: E402
from alrb import forcing  # noqa: E402


def one(site):
    m2 = pd.read_parquet(RAW / "power" / f"{site.key}_merra2_hourly.parquet")
    ce = pd.read_parquet(RAW / "power" / f"{site.key}_syn1deg_hourly.parquet")
    gh = forcing.ghcn_station(f"USW000{site.wban}")
    df, qc = forcing.build(site, m2, ce, gh)
    df.to_parquet(PROC / f"forcing_{site.key}.parquet", compression="zstd")
    # radiation cross-checks (hours with both values)
    both = df[["sw", "sw_ceres"]].dropna()
    qc["sw_vs_ceres_bias"] = float((both.sw - both.sw_ceres).mean())
    qc["sw_vs_ceres_r"] = float(np.corrcoef(both.sw, both.sw_ceres)[0, 1])
    lw = df[["lw", "lw_ceres"]].dropna()
    qc["lw_vs_ceres_bias"] = float((lw.lw - lw.lw_ceres).mean())
    qc["lw_vs_ceres_rmse"] = float(np.sqrt(((lw.lw - lw.lw_ceres) ** 2).mean()))
    qc["lw_vs_ceres_r"] = float(np.corrcoef(lw.lw, lw.lw_ceres)[0, 1])
    g = df[["g0", "g0_ceres"]].dropna()
    qc["toa_vs_ceres_maxabs"] = float((g.g0 - g.g0_ceres).abs().max())
    return site.key, qc


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(4) as ex:
        for key, qc in ex.map(one, SITES):
            out[key] = qc
            print(key, json.dumps(qc), flush=True)
    (RES / "forcing_qc.json").write_text(json.dumps(out, indent=1))
