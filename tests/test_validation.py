import numpy as np
import pandas as pd
import pytest

from electricity_analysis import evaluate, seasonal_naive, split_daily


def test_holdout_and_naive_use_training_only():
    dates = pd.date_range("2008-01-01", periods=90, freq="D")
    series = pd.Series(np.arange(90, dtype=float), index=dates)
    train, test = split_daily(series, 30)
    assert train.index.max() < test.index.min()
    original = seasonal_naive(train, 30)
    altered = series.copy()
    altered.iloc[-30:] = 10000
    assert np.array_equal(original, seasonal_naive(split_daily(altered, 30)[0], 30))


def test_evaluation_rejects_unsorted_dates():
    series = pd.Series(np.arange(50), index=pd.date_range("2008-01-01", periods=50))
    with pytest.raises(ValueError, match="chronological"):
        evaluate(series.iloc[::-1])
