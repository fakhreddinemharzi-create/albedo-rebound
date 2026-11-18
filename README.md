# albedo-rebound

Models and data for the life-cycle carbon accounting of reflective
(high-albedo) coatings on asphalt streets. A coating earns a radiative
credit only while it stays bright, whereas every re-application emits CO2
that persists, and a brighter surface changes the temperature, stiffness and
fatigue life of the pavement beneath. The code couples

* an hourly surface energy balance with an enthalpy-based freezing column of
  the pavement and its subgrade, driven by station meteorology and satellite
  radiation at ten U.S. Climate Reference Network sites;
* a layered-elastic pavement model with fatigue and rutting damage (Miner's
  rule) that turns the temperature record into the service life of the
  structure;
* an observation-driven radiative kernel, a two-dimensional radiosity model of
  street canyons and the canyon forcing of the pavement column;
* a fifty-year carbon ledger under four metric conventions (static and
  dynamic GWP100, GTP100 and a stock metric), a variance-based sensitivity
  analysis, projected climates (CMIP6), every one-degree cell of the
  contiguous United States, and a Monte Carlo over streets of measured urban
  form (local climate zones), sampled products and pavement structures.

The paper that describes the models and the results is under review; its
reference and an archived release with a DOI will be added here on
publication.

## Contents

| Path | Content |
|---|---|
| `alrb/` | model package: forcing and data readers, surface energy balance and freezing column (`column.py`, `driver.py`; forcing of a street in a canyon, `urban.py`), stiffness, layered-elastic solution and damage (`stiffness.py`, `let.py`, `fe_axi.py`, `damage.py`, `structural.py`), radiative kernel and metrics (`kernel.py`, `metrics.py`), ledger (`ledger.py`, `ledger_batch.py`), street-canyon radiosity (`canyon.py`), path tracing and plotting helpers; `config.py` holds the sites, the periods, the paths and the reference parameters |
| `scripts/` | numbered pipeline (`00`-`05` data retrieval, `10` forcing, `20`-`22` verification, `30`-`36` sweeps and benchmarks, `40`-`41` structure, `50`-`51` projected climates, `60`-`68` sensitivity, ledger, design space, canyons and the contiguous United States, `70` and `83` every number and table row, `80`-`88` discrete renewals, population and local-climate-zone grids, Monte Carlo over streets, streets in canyons and warmer cities, pavement-structure variants, bounds on the seasonal urban-form factor, coat removal at renewals, the kernel calibration and the strain tables) and one script per figure (`fig_*.py`) |
| `data/processed/` | hourly forcing of the ten sites (2017-2024) and the strain table of the reference structure |
| `results/` | every result (JSON and NPZ) read by the figures and by the number scripts |
| `figures/` | the figures (PDF and PNG) |
| `tools/` | PDF layout audit used by the figure scripts |

## Reproducing

Python 3.11 with the packages of `requirements.txt`. Raw downloads and model
runs (about 1 GB) go to `data/raw/`, or to the folder named by the
environment variable `ALRB_RAW`.

```bash
pip install -r requirements.txt
# figures and every number and table row from the stored results
python scripts/05_fetch_inputs.py              # outlines, census and local-climate-zone map (130 MB)
for f in scripts/fig_*.py; do python "$f"; done
python scripts/70_numbers.py && python scripts/83_numbers_revision.py   # -> manuscript/generated/
# the whole chain from the raw data
python scripts/00_fetch_isd.py && python scripts/01_fetch_goes.py && python scripts/02_fetch_power.py
python scripts/03_fetch_cmip6.py && python scripts/04_fetch_cmip6_rad.py && python scripts/10_build_forcing.py
python scripts/20_verify_thermal.py && python scripts/32_natural_validation.py && python scripts/33_strain_table.py
python scripts/30_pavement_sweep.py v3 --z0m 0.0021 --lw lw --deep dirichlet --zbot 8
bash scripts/run_v3.sh && bash scripts/run_v3b.sh && bash scripts/run_v3c.sh
python scripts/80_discrete_structure.py && python scripts/81_population_grid.py && python scripts/84_lcz_grid.py
python scripts/85_urban_column.py && python scripts/86_urban_ledger.py && python scripts/87_structure_variants.py
python scripts/88_bounds_revision.py && python scripts/82_conus_mc.py && python scripts/83_numbers_revision.py
```

Most figures and numbers need only the stored results. `70_numbers.py`,
`fig_thermal.py` and `fig_frostmech.py` also read the column runs that
`30_pavement_sweep.py` writes to `data/raw/sweep/` (about 0.7 GB),
`fig_evaluation.py` reads the satellite pairs of `32_natural_validation.py`,
and the three-dimensional figures (`fig_*3d.py`, `fig_street3d.py`,
`fig_canyon.py`) need PyVista and Mitsuba 3. Every figure script audits its
own layout (overlapping text, labels crossing data, clipped text) and fails
on any problem.

## Data sources

All inputs are public: NOAA Integrated Surface Database and U.S. Climate
Reference Network (station records), NASA POWER (MERRA-2 and CERES SYN1deg),
NOAA GOES-R land-surface temperature, CMIP6 through the Pangeo archive,
Natural Earth outlines (public domain), U.S. Census Bureau county centres of
population 2020, and the map of local climate zones of the contiguous United
States of Demuzere et al. (2020), doi:10.6084/m9.figshare.11416950 (CC BY
4.0). `scripts/05_fetch_inputs.py` checks the downloaded files against the
copies used.

Repository: https://github.com/fakhreddinemharzi-create/albedo-rebound

## Licence

Code: MIT (`LICENSE`). Processed data, results and figures: CC BY 4.0.
