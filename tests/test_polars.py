import pytest
import numpy as np

pl = pytest.importorskip("polars")
import pandas as pd  # noqa: E402
from percentify import (  # noqa: E402
    change, vif, missing, cv, outliers, r_squared,
    pca_variance, imbalance, difference, split, display,
)


# ===== DataFrame in -> polars DataFrame out =====

def test_vif_polars():
    np.random.seed(0)
    base = np.random.randn(80)
    df = pl.DataFrame({
        "a": base,
        "b": base * 2 + np.random.randn(80) * 0.1,
        "c": np.random.randn(80),
    })
    result = vif(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "VIF"]
    assert set(result["feature"].to_list()) == {"a", "b", "c"}


def test_missing_polars_values():
    df = pl.DataFrame({"salary": [1.0, None, 3.0, None], "age": [1.0, 2.0, None, 4.0]})
    result = missing(df)
    assert isinstance(result, pl.DataFrame)
    d = dict(zip(result["column"].to_list(), result["missing_pct"].to_list()))
    assert d["salary"] == 50.0
    assert d["age"] == 25.0


def test_cv_polars_dataframe():
    result = cv(pl.DataFrame({"a": [10.0, 20, 30], "b": [1.0, 2, 3]}))
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "cv"]


def test_outliers_polars_dataframe():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 100], "b": [1.0, 2, 3, 4, 5, 6, 7]})
    result = outliers(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "outlier_pct"]


def test_pca_variance_polars():
    np.random.seed(0)
    base = np.random.randn(80)
    df = pl.DataFrame({
        "a": base + np.random.randn(80) * 0.3,
        "b": base + np.random.randn(80) * 0.3,
        "c": np.random.randn(80),
    })
    result = pca_variance(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["component", "variance_explained", "cumulative"]


# ===== Series in -> scalar out =====

def test_cv_polars_series_returns_float():
    assert isinstance(cv(pl.Series([10.0, 20, 30, 40, 50])), float)


def test_outliers_polars_series_returns_float():
    assert isinstance(outliers(pl.Series([1.0, 2, 3, 4, 5, 6, 100])), float)


def test_r_squared_polars_series():
    result = r_squared(pl.Series([1.0, 2, 3, 4, 5]), pl.Series([1.1, 1.9, 3.2, 3.8, 5.1]))
    assert isinstance(result, float)
    assert 90 < result < 100


# ===== Series in -> polars Series out =====

def test_change_polars_period_over_period():
    result = change(pl.Series([100.0, 150, 90, 135]))
    assert isinstance(result, pl.Series)
    vals = result.to_list()
    assert vals[1] == 50.0
    assert vals[2] == -40.0


def test_change_polars_two_columns():
    result = change(pl.Series([100.0, 200, 50]), pl.Series([150.0, 150, 100]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [50.0, -25.0, 100.0]


def test_difference_polars_two_columns():
    result = difference(pl.Series([10.0, 50]), pl.Series([20.0, 50]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [66.67, 0.0]


def test_split_polars_series():
    result = split(200, pl.Series([1, 3]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [50.0, 150.0]


def test_display_polars_series():
    result = display(pl.Series([0.25, 0.5]), multiply=True)
    assert isinstance(result, pl.Series)
    assert result.to_list() == ["25.0%", "50.0%"]


# ===== imbalance: polars DataFrame + summary preserved =====

def test_imbalance_polars_with_summary():
    result = imbalance(pl.Series(["No"] * 850 + ["Yes"] * 150))
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["class", "count", "pct"]
    summary = result.attrs["summary"]
    assert summary["majority_class"] == "No"
    assert summary["minority_class"] == "Yes"
    assert summary["imbalance_ratio"] == 5.67


# ===== regression: pandas in still returns pandas =====

def test_pandas_input_unchanged():
    result = missing(pd.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]}))
    assert isinstance(result, pd.DataFrame)


def test_scalar_input_unchanged():
    assert change(100, 150) == 50.0
    assert display(0.45, multiply=True) == "45.0%"
