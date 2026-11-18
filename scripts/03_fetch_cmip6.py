"""Monthly CMIP6 series at the ten sites: historical (1995-2014) and
SSP2-4.5 / SSP5-8.5 (2015-2100), first ensemble member, nine models."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, SITES  # noqa: E402
from alrb import io_cmip6  # noqa: E402

MODELS = ["ACCESS-CM2", "CanESM5", "EC-Earth3", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
          "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
VARS = ["tas", "tasmax", "tasmin", "pr", "rsds", "hurs"]
EXPS = {"historical": (1995, 2014), "ssp245": (2015, 2100), "ssp585": (2015, 2100)}
OUT = RAW / "cmip6"
OUT.mkdir(parents=True, exist_ok=True)


def job(args):
    model, exp, var, zstore = args
    dest = OUT / f"{model}_{exp}_{var}.parquet"
    if dest.exists():
        return f"cached {dest.name}"
    try:
        df = io_cmip6.points(zstore, var, [s.lat for s in SITES], [s.lon for s in SITES], EXPS[exp])
    except Exception as e:
        return f"FAILED {model} {exp} {var}: {e}"
    df.columns = [s.key for s in SITES]
    df.index = df.index.astype(str)
    df.to_parquet(dest)
    return f"{model} {exp} {var}: {df.shape}"


if __name__ == "__main__":
    cat = pd.read_csv(io_cmip6.CATALOG)
    cat = cat[(cat.table_id == "Amon") & (cat.member_id == "r1i1p1f1") & cat.source_id.isin(MODELS)
              & cat.experiment_id.isin(EXPS) & cat.variable_id.isin(VARS)]
    cat = cat.sort_values("version").groupby(["source_id", "experiment_id", "variable_id"]).tail(1)
    jobs = [(r.source_id, r.experiment_id, r.variable_id, r.zstore) for r in cat.itertuples()]
    print(len(jobs), "stores", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(job, jobs):
            print(msg, flush=True)
    print("CMIP6_DONE")
