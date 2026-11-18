"""Build the critical-strain table and measure its interpolation error
against direct layered-elastic solutions at random states."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import strain_table as st  # noqa: E402
from alrb.config import RES, PROC  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    tab = st.build(4)
    print("table built in %.0fs" % (time.time() - t0))
    np.savez_compressed(PROC / "strain_table.npz", tab=tab, X1=st.X1, X2=st.X2, X3=st.X3, X4=st.X4)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(400):
        E5 = 10 ** rng.uniform(np.log10(35), np.log10(6895))
        x1 = rng.uniform(st.X1[0], st.X1[-1]); x2 = rng.uniform(st.X2[0], st.X2[-1])
        x3 = rng.uniform(st.X3[0], st.X3[-1]); x4 = rng.uniform(st.X4[0], st.X4[-1])
        E = np.array([E5 * 10 ** x1, E5 * 10 ** (x1 + x2), E5 * 10 ** x3, E5 * 10 ** x4, E5])
        direct = st.critical(E * 1e6)
        itp = st.interp(tab, st.X1, st.X2, st.X3, st.X4, E[:1], E[1:2], E[2:3], E[3:4], E[4:5])[0]
        scale = np.abs(direct)
        errs.append(np.abs(itp - direct) / np.maximum(scale, 1e-9))
    errs = np.array(errs)
    # relative errors, ignoring strains near zero (sign change of the tensile strain)
    res = {}
    for q, name in enumerate(["eps_t", "eps_v", "eps_r25", "eps_r75"]):
        e = errs[:, q]
        res[name] = dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(e.max()))
    print(json.dumps(res, indent=1))
    (RES / "strain_table_accuracy.json").write_text(json.dumps(res, indent=1))
