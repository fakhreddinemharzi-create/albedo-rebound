"""Snow-masked radiative kernel over the contiguous United States (Sect. 6.8).

The kernel of Sect. 4.5 is evaluated hour by hour for every 1-degree land
cell of the lower 48 states over the seven ledger years (1 May 2017 to
1 May 2024), from the all-sky surface and top-of-atmosphere irradiance and
the solar zenith angle of CERES SYN1deg (NASA POWER, hourly), with the
first-pass transmission from the MERRA-2 daily precipitable water, ozone
and surface pressure and the SYN1deg daily aerosol optical depth, and the
street masked on days with more than 1 mm of MERRA-2 snowfall and the day
after.  Differences from the station kernel: satellite instead of
pyranometer irradiance, a 1-degree cell instead of a point, daily instead
of hourly absorbers, and UTC instead of local days for the snow mask; the
ten station cells test the combined effect.

Writes results/conus_kernel_v3.npz and results/conus_kernel_v3.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import io_power as ip, kernel  # noqa: E402
from alrb.config import RAW, RES, SITES  # noqa: E402

T0, T1 = "2017-05-01", "2024-05-01"
LAT = (24.0, 50.0)
LON = (-125.0, -66.0)
EXCLUDE = {"Alaska", "Hawaii"}


def conus_polygons():
    gj = json.loads((RAW / "ne" / "ne_50m_admin_1_states_provinces_lakes.geojson").read_text())
    polys = []
    for f in gj["features"]:
        pr = f["properties"]
        if pr.get("iso_a2") != "US" or pr.get("name") in EXCLUDE:
            continue
        g = f["geometry"]
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        else:
            rings = [poly[0] for poly in g["coordinates"]]
        polys += [MPath(np.asarray(r)) for r in rings]
    return polys


def land_mask(lats, lons, polys):
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    m = np.zeros(len(pts), bool)
    for p in polys:
        m |= p.contains_points(pts)
    return m.reshape(LA.shape)


def block(product, freq, var, ci, cj, lo, hi):
    """Array (hi-lo, cy, cx) of chunk column (ci, cj) over time indices [lo, hi)."""
    m = ip.meta(product, freq)
    za = m[f"{var}/.zarray"]
    sep = za.get("dimension_separator", ".")
    nt = za["shape"][0]
    ct, cy, cx = za["chunks"]
    out = np.full((hi - lo, cy, cx), np.nan, np.float32)
    for c in range(lo // ct, (hi - 1) // ct + 1):
        buf = ip._get(f"{ip.BASE}/{ip.STORES[(product, freq)]}/{var}/{sep.join([str(c), str(ci), str(cj)])}")
        if buf is None:
            continue
        vt = min(ct, nt - c * ct)
        vy = min(cy, za["shape"][1] - ci * cy)
        vx = min(cx, za["shape"][2] - cj * cx)
        blk = ip._decode(buf, za, (vt, vy, vx))
        a = c * ct
        s0, s1 = max(a, lo), min(a + vt, hi)
        out[s0 - lo:s1 - lo, :vy, :vx] = blk[s0 - a:s1 - a]
    out[out < -900] = np.nan
    return out


def tindex(product, freq):
    t = ip.times(product, freq)
    return int(np.searchsorted(t, pd.Timestamp(T0, tz="UTC"))), int(np.searchsorted(t, pd.Timestamp(T1, tz="UTC"))), t


def main():
    lat = ip._coord("syn1deg", "hourly", "lat")
    lon = ip._coord("syn1deg", "hourly", "lon")
    iy = np.where((lat > LAT[0]) & (lat < LAT[1]))[0]
    ix = np.where((lon > LON[0]) & (lon < LON[1]))[0]
    polys = conus_polygons()
    land = land_mask(lat[iy], lon[ix], polys)
    print("land cells", int(land.sum()), flush=True)
    lo_h, hi_h, th = tindex("syn1deg", "hourly")
    lo_d, hi_d, td = tindex("merra2", "daily")
    lo_a, hi_a, ta = tindex("syn1deg", "daily")
    hours = th[lo_h:hi_h]
    day_of_hour = ((hours - pd.Timestamp(T0, tz="UTC")).total_seconds() // 86400).astype(int).to_numpy()
    ndays = int(day_of_hour.max()) + 1
    mlat = ip._coord("merra2", "daily", "lat")
    mlon = ip._coord("merra2", "daily", "lon")
    mz = ip.meta("merra2", "daily")["PS/.zarray"]
    mcy, mcx = mz["chunks"][1], mz["chunks"][2]
    az = ip.meta("syn1deg", "daily")["AOD_55/.zarray"]
    acy, acx = az["chunks"][1], az["chunks"][2]
    cy, cx = 4, 4
    chunks = sorted({(i // cy, j // cx) for i in iy for j in ix})
    shape = (len(iy), len(ix))
    K = np.full(shape, np.nan)
    Km = np.full(shape, np.nan)
    KT = np.full(shape, np.nan)
    FS = np.full(shape, np.nan)
    MU = np.full(shape, np.nan)
    Kmon = np.full((12,) + shape, np.nan)
    months = hours.month.to_numpy()
    cache = {}

    def mblock(var, ci, cj):
        key = ("m", var, ci, cj)
        if key not in cache:
            cache[key] = block("merra2", "daily", var, ci, cj, lo_d, hi_d)
        return cache[key]

    def ablock(ci, cj):
        key = ("a", ci, cj)
        if key not in cache:
            cache[key] = block("syn1deg", "daily", "AOD_55", ci, cj, lo_a, hi_a)
        return cache[key]

    def work(ch):
        ci, cj = ch
        cells = [(a, b) for a in range(cy) for b in range(cx)
                 if ci * cy + a in iy and cj * cx + b in ix
                 and land[np.searchsorted(iy, ci * cy + a), np.searchsorted(ix, cj * cx + b)]]
        if not cells:
            return ch, []
        sw = block("syn1deg", "hourly", "ALLSKY_SFC_SW_DWN", ci, cj, lo_h, hi_h)
        toa = block("syn1deg", "hourly", "TOA_SW_DWN", ci, cj, lo_h, hi_h)
        sza = block("syn1deg", "hourly", "SZA", ci, cj, lo_h, hi_h)
        res = []
        for a, b in cells:
            gi, gj = ci * cy + a, cj * cx + b
            la, lo_ = lat[gi], lon[gj]
            mi_ = int(np.argmin(np.abs(mlat - la)))
            mj_ = int(np.argmin(np.abs(mlon - lo_)))
            daily = {v: mblock(v, mi_ // mcy, mj_ // mcx)[:, mi_ % mcy, mj_ % mcx] for v in ("TQV", "TO3", "PS",
                                                                                             "PRECSNO")}
            ai = int(np.argmin(np.abs(lat - la)))
            aj = int(np.argmin(np.abs(lon - lo_)))
            aod = ablock(ai // acy, aj // acx)[:, ai % acy, aj % acx]
            dd = np.clip(day_of_hour, 0, len(daily["TQV"]) - 1)

            def fill(x, default):
                x = np.asarray(x, float)
                return np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(np.nanmean(x)) else default)
            pw = fill(daily["TQV"], 15.0)[dd]
            o3 = fill(daily["TO3"], 300.0)[dd]
            ps = fill(daily["PS"], 1013.0)[dd]
            ad = fill(aod, 0.1)[np.clip(day_of_hour, 0, len(aod) - 1)]
            snow = fill(daily["PRECSNO"], 0.0)
            sday = snow > 1.0
            mday = sday | np.concatenate([[False], sday[:-1]])
            mask = mday[np.clip(day_of_hour, 0, len(mday) - 1)]
            g0 = np.nan_to_num(toa[:, a, b].astype(float))
            s = sw[:, a, b].astype(float)
            ok = np.isfinite(s) & (g0 > 5.0)
            muz = np.cos(np.radians(np.nan_to_num(sza[:, a, b].astype(float), nan=90.0)))
            kt = np.where(ok, np.clip(np.nan_to_num(s) / np.maximum(g0, 1e-9), 0, 1), 0.0)
            mua = kernel.first_pass_transmission(np.where(muz > 0, muz, 0.05), pw, o3, ad, ps)
            q = np.minimum(kt, mua)
            k = np.where(ok, -g0 * q ** 2 / mua, 0.0) / 100.0          # W m-2 per 0.01 of albedo
            valid = np.isfinite(s) | (g0 <= 5.0)
            km = k * (1.0 - mask)
            mon = [float(np.mean(km[valid & (months == mm)])) for mm in range(1, 13)]
            res.append((gi, gj, float(np.mean(k[valid])), float(np.mean(km[valid])),
                        float(np.nansum(np.where(ok, s, 0)) / np.sum(np.where(ok, g0, 0))),
                        float(np.mean(mday)), float(np.sum(np.where(ok, mua * g0, 0)) / np.sum(np.where(ok, g0, 0))),
                        mon, float(np.mean(valid))))
        return ch, res

    n = 0
    with ThreadPoolExecutor(6) as ex:
        for ch, res in ex.map(work, chunks):
            n += 1
            for gi, gj, k, km, kt, fs, mu, mon, fv in res:
                a = int(np.searchsorted(iy, gi))
                b = int(np.searchsorted(ix, gj))
                K[a, b], Km[a, b], KT[a, b], FS[a, b], MU[a, b] = k, km, kt, fs, mu
                Kmon[:, a, b] = mon
            if n % 10 == 0:
                print(f"{n}/{len(chunks)} chunks", flush=True)
    np.savez_compressed(RES / "conus_kernel_v3.npz", lat=lat[iy], lon=lon[ix], land=land, K=K, Km=Km, kt=KT,
                        fsnow=FS, mu=MU, Kmon=Kmon)
    # station cells
    kj = json.loads((RES / "kernels.json").read_text())["sites"]
    out = {}
    for s in SITES:
        a = int(np.argmin(np.abs(lat[iy] - s.lat)))
        b = int(np.argmin(np.abs(lon[ix] - s.lon)))
        out[s.key] = dict(grid_Km=float(Km[a, b]), grid_K=float(K[a, b]), station_Km=kj[s.key]["K_aprp_snowmasked"],
                          station_K=kj[s.key]["K_aprp_hourly"], grid_kt=float(KT[a, b]))
        print(f"{s.key:13s} grid {Km[a, b]:.3f} station {kj[s.key]['K_aprp_snowmasked']:.3f} (all-sky "
              f"{K[a, b]:.3f} vs {kj[s.key]['K_aprp_hourly']:.3f})", flush=True)
    (RES / "conus_kernel_v3.json").write_text(json.dumps(dict(sites=out, ncells=int(land.sum())), indent=1))


if __name__ == "__main__":
    main()
