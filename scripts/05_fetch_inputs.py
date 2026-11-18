"""Retrieve the public inputs of the maps and of the analyses over the
contiguous United States that the other fetch scripts do not cover.

Writes to RAW (alrb/config.py):
  ne/      Natural Earth 1:50m country and state outlines, release v5.1.2
           (public domain; https://www.naturalearthdata.com)
  census/  county centres of population 2020, U.S. Census Bureau
  lcz/     map of local climate zones of the contiguous United States,
           Demuzere et al. (2020), doi:10.6084/m9.figshare.11416950 (CC BY 4.0)

Files already present are kept. The SHA-256 of each file is compared with
that of the copy used for the paper (first 16 hexadecimal digits below).
"""
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW  # noqa: E402

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
FILES = [
    ("ne", "ne_50m_admin_0_countries_lakes.geojson", NE + "ne_50m_admin_0_countries_lakes.geojson",
     "b4c14911c7eec4a5"),
    ("ne", "ne_50m_admin_1_states_provinces_lakes.geojson", NE + "ne_50m_admin_1_states_provinces_lakes.geojson",
     "b92c3f709b691240"),
    ("census", "CenPop2020_Mean_CO.txt",
     "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt", "4f754a3bda9714e5"),
    ("lcz", "readme.txt", "https://ndownloader.figshare.com/files/28373826", "6a8fb1b270f8e552"),
    ("lcz", "CONUS_LCZ_map_NLCD_v1.0_epsg4326.tif", "https://ndownloader.figshare.com/files/28373844",
     "6ba00f531fde23e1"),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for sub, name, url, ref in FILES:
        out = RAW / sub / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            print(f"downloading {url}")
            tmp = out.with_suffix(out.suffix + ".part")
            with requests.get(url, stream=True, timeout=900) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            tmp.rename(out)
        h = sha(out)
        note = "identical to the copy used" if h.startswith(ref) else "DIFFERS from the copy used"
        print(f"{sub}/{name}: {out.stat().st_size / 1e6:.1f} MB, sha256 {h[:16]} ({note})")


if __name__ == "__main__":
    main()
