"""Bounds on four simplifications questioned in review.

1. Seasonal alignment of the urban-form factor.  The canyon factor psi is an
   annual mean weighted by the radiative kernel, whereas the credit of a coat
   laid on 1 May is concentrated in the months after it is laid.  The
   effective psi weights the monthly psi by the kernel and by the albedo
   gain of the coat (no residual gain, as in the reference).
2. Loss of the coat at a structural renewal.  The coat schedule is not tied
   to the structural renewals; a renewal that removes the coat on the street
   loses the credit that the coat would still have delivered before the next
   scheduled coat (phase of the renewal uniform within the interval).
3. Leave-one-out residuals of the calibration of the grid kernel (median of
   the ten station-to-grid ratios).
4. Accuracy of the critical-strain tables of the 50 and 200 mm asphalt layers
   against direct layered-elastic solutions (60 random states, as for the
   100 mm table).

Writes results/bounds_v3.json.
"""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import strain_table as stt  # noqa: E402
from alrb.config import RAW, REF, RES, SITES  # noqa: E402

H_YR = 50.0


def seasonal_psi():
    c = json.loads((RES / "canyon_v3.json").read_text())
    k = json.loads((RES / "kernels.json").read_text())["sites"]
    out = {}
    for tau in (0.5, REF.tau, 3.0):
        days = np.arange(int(REF.Tr * 365.25))
        gain = np.exp(-days / 365.25 / tau)
        d0 = dt.date(2017, 5, 1)
        mon = np.array([(d0 + dt.timedelta(days=int(d))).month - 1 for d in days])
        res = {}
        for o in ("EW", "NS", "diag"):
            for h in ("0.5", "1.0"):
                r = []
                for s in SITES:
                    K = np.abs(np.array(k[s.key]["monthly_cycle"]))
                    pm = np.array(c["sites"][s.key][o]["monthly"][h])
                    annual = np.sum(pm * K) / np.sum(K)
                    w = K[mon] * gain
                    r.append(float(np.sum(pm[mon] * w) / np.sum(w) / annual))
                res[f"{o}|{h}"] = dict(min=min(r), max=max(r))
        out[f"{tau:g}"] = res
    return out


def coat_loss():
    """Credit lost per structural renewal, as a fraction of the fifty-year credit."""
    tau, Tr = REF.tau, REF.Tr
    e = np.exp(-Tr / tau)
    phi = tau / Tr * (1.0 - e)                         # mean fraction of the gain delivered over a cycle
    lost = tau * (phi - e)                             # yr of initial gain, averaged over the phase of the renewal
    frac = lost / (H_YR * phi)
    led = json.loads((RES / "ledger_v3.json").read_text())
    n = np.array([H_YR / led[s.key]["reference"]["L_eff"] for s in SITES])
    return dict(phi=phi, lost_yr=lost, frac_per_renewal=frac, renewals_min=float(n.min()), renewals_max=float(n.max()),
                frac_total_min=float(frac * n.min()), frac_total_max=float(frac * n.max()),
                extra_coat_frac=Tr / H_YR)


def kernel_loo():
    cj = json.loads((RES / "conus_kernel_v3.json").read_text())["sites"]
    r = np.array([cj[s.key]["station_Km"] / cj[s.key]["grid_Km"] for s in SITES])
    med = np.median(r)
    ins = r / med - 1.0                                # station kernel against the calibrated grid kernel
    loo = np.array([r[i] / np.median(np.delete(r, i)) - 1.0 for i in range(len(r))])
    return dict(insample_min=float(ins.min()), insample_max=float(ins.max()), loo_min=float(loo.min()),
                loo_max=float(loo.max()), loo_rms=float(np.sqrt(np.mean(loo ** 2))))


def strain_accuracy():
    spec = importlib.util.spec_from_file_location("sv87", Path(__file__).with_name("87_structure_variants.py"))
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)
    out = {}
    for h_ac in (0.05, 0.20):
        z = np.load(RAW / "strain_tables" / f"strain_table_ac{int(round(h_ac * 1000)):03d}.npz")
        stt.H = np.array([h_ac / 2, h_ac / 2, 0.25, 0.30])
        stt._R, stt._Z, stt._L = sv._points(stt.H)
        rng = np.random.default_rng(3)
        err = []
        for _ in range(60):
            E5 = 10 ** rng.uniform(np.log10(35), np.log10(500))
            E = np.array([E5 * 10 ** rng.uniform(0.5, 2.5), 0, E5 * 10 ** rng.uniform(0.0, 1.0),
                          E5 * 10 ** rng.uniform(0.0, 0.8), E5])
            E[1] = E[0] * 10 ** rng.uniform(-0.6, 0.0)
            d = stt.critical(E * 1e6)
            i = stt.interp_cubic(z["tab"], z["X1"], z["X2"], z["X3"], z["X4"], E[:1], E[1:2], E[2:3], E[3:4], E[4:5])[0]
            err.append(np.abs(i[:2] - d[:2]) / np.abs(d[:2]))
        m = np.max(err, axis=0)
        out[f"{h_ac:g}"] = dict(eps_t_max=float(m[0]), eps_v_max=float(m[1]))
        print(f"strain table {h_ac} m: max rel. error eps_t {m[0]:.4f}, eps_v {m[1]:.4f}", flush=True)
    return out


def main():
    out = dict(seasonal_psi=seasonal_psi(), coat_loss=coat_loss(), kernel_loo=kernel_loo())
    print(json.dumps(out, indent=1))
    out["strain_tables"] = strain_accuracy()
    (RES / "bounds_v3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
