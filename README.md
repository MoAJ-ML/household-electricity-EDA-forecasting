# Household electricity: 2008 EDA and backtest

This project studies one household's minute-level electricity measurements and backtests a 30-day forecast of **daily mean active power (kW)**. It does not estimate total energy consumed (kWh), represent a utility load, or establish causes of usage patterns.

## Setup and clean run

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python electricity_analysis.py --download
pytest -q
```

`--download` fetches the archive only if `data/household_power_consumption.txt` is absent. Or supply your own file with `--data PATH`. The script writes `results/validation.json` with the SHA-256 of the exact input and a plot. A failed download, missing 2008 data, or whole missing days exits clearly. Generated data and results are ignored by Git.

## Data and method

The [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption) covers one home in Sceaux, France, from December 2006 to November 2010. UCI licenses the dataset under CC BY 4.0. This repository analyzes only 2008, drops minute rows without a valid timestamp or target, and averages the remaining kW readings by day. The final 30 days are held out chronologically. SARIMA(1,1,1)(1,1,1,7) is fitted on training days only and compared with a seven-day seasonal naive forecast. Both are evaluated on the same untouched days with MAE and RMSE in kW. This is one holdout, not a rolling-origin study; it does not establish generalization to other households or years.

The original `electricity_analysis.ipynb` and `summary.txt` are historical exploratory artifacts. Their unverified seasonal, clustering, and model-ranking claims are not current results. Use `results/validation.json` after a fresh run for current metrics.

On the locally retrieved input with SHA-256 `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7` (526,905 valid 2008 minute rows), the 2–31 December 2008 holdout gave SARIMA MAE 0.349 kW and RMSE 0.446 kW; seasonal naive MAE 0.379 kW and RMSE 0.460 kW. Re-downloads can change the input, so compare hashes before reusing these figures.

**Repository description recommendation:** “Single-household 2008 electricity EDA with a chronological SARIMA vs seasonal-naive backtest.”
