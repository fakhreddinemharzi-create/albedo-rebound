"""Download the USCRN station-years from the ISD Global-Hourly mirror and
convert each to two parquet files (five-minute and hourly fields)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, SITES, YEARS  # noqa: E402
from alrb import io_isd  # noqa: E402

OUT = RAW / "isd"


def job(args):
    s, y = args
    csv = OUT / "csv" / f"{s.wban}_{y}.csv"
    f5 = OUT / "parquet" / f"{s.key}_{y}_5min.parquet"
    fh = OUT / "parquet" / f"{s.key}_{y}_hourly.parquet"
    if f5.exists() and fh.exists():
        return f"{s.key} {y} cached"
    io_isd.download(y, s.wban, csv)
    five, hourly = io_isd.parse(csv)
    f5.parent.mkdir(parents=True, exist_ok=True)
    five.to_parquet(f5)
    hourly.to_parquet(fh)
    return f"{s.key} {y}: {len(five)} five-minute, {len(hourly)} hourly records"


if __name__ == "__main__":
    jobs = [(s, y) for s in SITES for y in YEARS]
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(job, jobs):
            print(msg, flush=True)
    print("ISD_DONE")
