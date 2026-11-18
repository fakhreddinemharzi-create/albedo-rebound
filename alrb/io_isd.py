"""U.S. Climate Reference Network records from NOAA's Integrated Surface
Database (Global-Hourly CSV, AWS open-data mirror ``noaa-global-hourly-pds``).

USCRN stations enter ISD as report type CRN05: one record every five
minutes carrying the five-minute air temperature, relative humidity, wind
speed and precipitation, and, on the record at the top of each hour, the
network's hourly fields (global solar radiation GH1, derived hourly air
temperature KF1, hourly precipitation AA1).  Field layouts follow the ISD
format document (NOAA/NCEI, 2018): values are scaled integers with a
quality code (1 = passed all checks) and a network flag (0 = passed).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BUCKET = "https://noaa-global-hourly-pds.s3.amazonaws.com"
USECOLS = ["DATE", "REPORT_TYPE", "TMP", "WND", "CH1", "CB1", "AA1", "GH1", "KF1", "OB1"]


def download(year: int, wban: str, dest: Path, tries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = f"{BUCKET}/{year}/999999{wban}.csv"
    for k in range(tries):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            return dest
        except Exception:  # network hiccup: back off and retry
            time.sleep(2 ** k)
    raise RuntimeError(f"could not download {url}")


def _num(s: pd.Series, idx: int, scale: float, missing: tuple = ()) -> np.ndarray:
    """Element ``idx`` of a comma-separated ISD field as float, NaN if absent/missing."""
    parts = s.fillna("").astype(str).str.split(",", expand=True)
    if parts.shape[1] <= idx:
        return np.full(len(s), np.nan)
    col = parts[idx].replace("", np.nan)
    v = pd.to_numeric(col, errors="coerce").to_numpy(float)
    for m in missing:
        v[v == m] = np.nan
    return v / scale


def _qc_ok(s: pd.Series, idx: int) -> np.ndarray:
    parts = s.fillna("").astype(str).str.split(",", expand=True)
    if parts.shape[1] <= idx:
        return np.zeros(len(s), bool)
    return (parts[idx] == "1").to_numpy()


def parse(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (five-minute frame, hourly frame) indexed by UTC time stamp as
    written in ISD (no shift applied here)."""
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, dtype=str, low_memory=False)
    df = df[df["REPORT_TYPE"].str.strip() == "CRN05"].copy()
    t = pd.to_datetime(df["DATE"], utc=True)
    five = pd.DataFrame(index=t)
    # five-minute air temperature (TMP: value, qc)
    ta = _num(df["TMP"], 0, 10.0, (9999.0, 999.9 * 10))
    ta[~_qc_ok(df["TMP"], 1)] = np.nan
    five["ta"] = ta
    # five-minute relative humidity (CH1: period, T, qc, flag, RH, qc, flag)
    if "CH1" in df:
        rh = _num(df["CH1"], 4, 10.0, (9999.0,))
        rh[~_qc_ok(df["CH1"], 5)] = np.nan
        five["rh"] = rh
    # five-minute wind speed at 1.5 m (WND: dir, qc, type, speed, qc)
    ws = _num(df["WND"], 3, 10.0, (9999.0,))
    ws[~_qc_ok(df["WND"], 4)] = np.nan
    five["ws"] = ws
    # five-minute precipitation (CB1: period, depth, qc, flag)
    if "CB1" in df:
        pr = _num(df["CB1"], 1, 10.0, (99999.0,))
        pr[~_qc_ok(df["CB1"], 2)] = np.nan
        five["pr"] = pr
    top = df["GH1"].notna() if "GH1" in df else pd.Series(False, index=df.index)
    h = df[top.to_numpy()]
    hourly = pd.DataFrame(index=pd.to_datetime(h["DATE"], utc=True))
    sw = _num(h["GH1"], 0, 10.0, (99999.0,))
    sw_ok = _qc_ok(h["GH1"], 1) & (h["GH1"].astype(str).str.split(",").str[2] == "0").to_numpy()
    sw[~sw_ok] = np.nan
    hourly["sw"] = sw
    if "KF1" in h:
        kf = _num(h["KF1"], 0, 10.0, (9999.0,))
        kf[~_qc_ok(h["KF1"], 1)] = np.nan
        hourly["ta_h"] = kf
    if "AA1" in h:
        aa = _num(h["AA1"], 1, 10.0, (9999.0,))
        per = _num(h["AA1"], 0, 1.0, (99.0,))
        aa[(per != 1) | ~_qc_ok(h["AA1"], 3)] = np.nan
        hourly["pr_h"] = aa
    return five, hourly
