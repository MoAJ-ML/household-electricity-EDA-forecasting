"""Reproducible 2008 single-household EDA and held-out daily forecasting."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

SOURCE = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def get_data(path: Path, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}. Pass --download to fetch the UCI archive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(".zip")
    with urlopen(SOURCE, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("UCI returned an empty archive")
    archive.write_bytes(payload)
    with ZipFile(archive) as zf:
        with zf.open("household_power_consumption.txt") as source:
            path.write_bytes(source.read())
    return path


def load_daily(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    required = {"Date", "Time", "Global_active_power"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    frame["Datetime"] = pd.to_datetime(frame["Date"] + " " + frame["Time"], dayfirst=True, errors="coerce")
    frame["Global_active_power"] = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    frame = frame.loc[frame["Datetime"].dt.year.eq(2008)].dropna(subset=["Datetime", "Global_active_power"])
    frame = frame.sort_values("Datetime")
    if frame.empty:
        raise ValueError("No valid 2008 observations")
    daily = frame.set_index("Datetime")["Global_active_power"].resample("D").mean()
    if daily.isna().any():
        raise ValueError("Whole missing days require an explicit imputation policy")
    return daily, frame


def split_daily(daily: pd.Series, holdout: int = 30) -> tuple[pd.Series, pd.Series]:
    if not daily.index.is_monotonic_increasing or daily.index.has_duplicates:
        raise ValueError("Daily series must have unique chronological dates")
    if len(daily) <= holdout + 14:
        raise ValueError("Need at least 14 training days beyond the holdout")
    return daily.iloc[:-holdout], daily.iloc[-holdout:]


def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    return np.resize(train.iloc[-7:].to_numpy(), horizon)


def evaluate(daily: pd.Series, holdout: int = 30) -> dict:
    train, test = split_daily(daily, holdout)
    predictions = {"seasonal_naive_7d": seasonal_naive(train, len(test))}
    fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    predictions["sarima_111_1117"] = fit.forecast(len(test)).to_numpy()
    return {
        "train_end": str(train.index[-1].date()), "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()), "holdout_days": len(test),
        "target": "daily mean minute-averaged active power (kW)",
        "metrics": {name: {"mae_kw": float(mean_absolute_error(test, values)),
                            "rmse_kw": float(root_mean_squared_error(test, values))}
                    for name, values in predictions.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/household_power_consumption.txt"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    path = get_data(args.data, args.download)
    daily, frame = load_daily(path)
    results = evaluate(daily)
    results["source"] = SOURCE
    results["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    results["valid_2008_minute_rows"] = len(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    plt.figure(figsize=(11, 4))
    daily.plot()
    plt.ylabel("Daily mean active power (kW)")
    plt.title("One household, 2008")
    plt.tight_layout()
    plt.savefig(args.output / "daily_power.png")
    plt.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
