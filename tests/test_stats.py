import pytest
import numpy as np
import pandas as pd
from percentify import (
    vif, missing, cv, outliers, r_squared, variance_explained, PercentifyWarning
)


# ===== Fixtures =====

@pytest.fixture
def independent_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })


@pytest.fixture
def collinear_df():
    np.random.seed(42)
    x = np.random.randn(100)
    return pd.DataFrame({
        "x": x,
        "y": x * 2 + np.random.randn(100) * 0.01,
        "z": np.random.randn(100),
    })


def _to_map(df, key, val):
    return dict(zip(df[key], df[val]))


# ===== vif =====

def test_vif_returns_dataframe(independent_df):
    result = vif(independent_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "VIF"]


def test_vif_returns_all_columns(independent_df):
    result = vif(independent_df)
    assert set(result["feature"]) == {"a", "b", "c"}


def test_vif_independent_features_low(independent_df):
    result = vif(independent_df)
    assert (result["VIF"] < 5.0).all()


def test_vif_collinear_features_high(collinear_df):
    vals = _to_map(vif(collinear_df), "feature", "VIF")
    assert vals["x"] > 10
    assert vals["y"] > 10


def test_vif_sorted_descending(collinear_df):
    vifs = vif(collinear_df)["VIF"].tolist()
    assert vifs == sorted(vifs, reverse=True)


def test_vif_flag_filters(independent_df):
    result = vif(independent_df, flag=5.0)
    assert result.empty


def test_vif_flag_returns_collinear(collinear_df):
    features = set(vif(collinear_df, flag=5.0)["feature"])
    assert "x" in features
    assert "y" in features
    assert "z" not in features


def test_vif_ignores_non_numeric():
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    assert set(vif(df)["feature"]) == {"a", "b"}


def test_vif_all_categorical_warns():
    df = pd.DataFrame({"city": ["NY", "LA"], "team": ["A", "B"]})
    with pytest.warns(PercentifyWarning, match="[Nn]umeric"):
        result = vif(df)
    assert result.empty
    assert list(result.columns) == ["feature", "VIF"]


def test_vif_single_numeric_warns():
    df = pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    with pytest.warns(PercentifyWarning):
        result = vif(df)
    assert result.empty


def test_vif_custom_decimals(independent_df):
    for val in vif(independent_df, decimals=4)["VIF"]:
        parts = str(float(val)).split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 4


# ===== missing =====

def test_missing_returns_dataframe():
    df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
    result = missing(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["column", "missing_pct"]


def test_missing_basic():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["b"] == 40.0
    assert vals["a"] == 20.0
    assert vals["c"] == 0.0


def test_missing_sorted_descending():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    assert missing(df)["column"].tolist() == ["b", "a", "c"]


def test_missing_no_nulls():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert (missing(df)["missing_pct"] == 0.0).all()


def test_missing_all_null():
    df = pd.DataFrame({"a": [None, None, None]})
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["a"] == 100.0


def test_missing_empty_df():
    df = pd.DataFrame({"a": [], "b": []})
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["a"] == 0.0


def test_missing_includes_non_numeric():
    df = pd.DataFrame({
        "name": ["Alice", None, "Charlie"],
        "age": [25, None, 35],
    })
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["name"] == 33.33
    assert vals["age"] == 33.33


# ===== cv =====

def test_cv_series_returns_float():
    result = cv(pd.Series([10, 20, 30, 40, 50]))
    assert isinstance(result, float)
    assert result > 0


def test_cv_dataframe_returns_dataframe():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    result = cv(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "cv"]


def test_cv_dataframe_values():
    df = pd.DataFrame({
        "a": [10, 20, 30, 40, 50],
        "b": [100, 100, 100, 100, 100],
    })
    vals = _to_map(cv(df), "feature", "cv")
    assert vals["a"] > 0
    assert vals["b"] == 0.0


def test_cv_zero_mean_series_warns():
    with pytest.warns(PercentifyWarning):
        result = cv(pd.Series([-1, 0, 1]))
    assert result == float("inf")


def test_cv_zero_mean_dataframe():
    df = pd.DataFrame({"a": [-1, 0, 1], "b": [10, 20, 30]})
    vals = _to_map(cv(df), "feature", "cv")
    assert vals["a"] == float("inf")
    assert vals["b"] > 0


def test_cv_non_numeric_series_warns():
    with pytest.warns(PercentifyWarning):
        result = cv(pd.Series(["cat", "dog", "bird"]))
    assert np.isnan(result)


def test_cv_ignores_non_numeric():
    df = pd.DataFrame({"a": [10, 20, 30], "name": ["x", "y", "z"]})
    assert set(cv(df)["feature"]) == {"a"}


def test_cv_custom_decimals():
    result = cv(pd.Series([10, 20, 30, 40, 50]), decimals=4)
    parts = str(result).split(".")
    if len(parts) == 2:
        assert len(parts[1]) <= 4


# ===== outliers =====

def test_outliers_series_returns_float():
    result = outliers(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
    assert isinstance(result, float)
    assert result > 0


def test_outliers_no_outliers():
    assert outliers(pd.Series([1, 2, 3, 4, 5])) == 0.0


def test_outliers_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    result = outliers(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "outlier_pct"]
    vals = _to_map(result, "feature", "outlier_pct")
    assert vals["a"] > 0
    assert vals["b"] == 0.0


def test_outliers_all_nan():
    assert outliers(pd.Series([np.nan, np.nan, np.nan])) == 0.0


def test_outliers_non_numeric_series_warns():
    with pytest.warns(PercentifyWarning):
        result = outliers(pd.Series(["cat", "dog"]))
    assert np.isnan(result)


def test_outliers_ignores_non_numeric():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "name": ["x", "y", "z", "w", "v"]})
    assert set(outliers(df)["feature"]) == {"a"}


def test_outliers_custom_multiplier():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    assert outliers(s, multiplier=1.0) >= outliers(s, multiplier=3.0)


# ===== r_squared =====

def test_r_squared_perfect():
    y = [1, 2, 3, 4, 5]
    assert r_squared(y, y) == 100.0


def test_r_squared_good_fit():
    result = r_squared([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1])
    assert 90 < result < 100


def test_r_squared_bad_fit():
    assert r_squared([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0


def test_r_squared_with_numpy():
    assert r_squared(np.array([1, 2, 3]), np.array([1, 2, 3])) == 100.0


def test_r_squared_with_series():
    result = r_squared(pd.Series([1, 2, 3, 4, 5]), pd.Series([1.1, 2.1, 2.9, 4.0, 5.1]))
    assert result > 0


def test_r_squared_mismatched_length():
    with pytest.raises(ValueError):
        r_squared([1, 2, 3], [1, 2])


def test_r_squared_too_few():
    with pytest.raises(ValueError):
        r_squared([1], [1])


def test_r_squared_non_numeric():
    with pytest.raises(ValueError):
        r_squared(["a", "b", "c"], ["a", "b", "c"])


def test_r_squared_custom_decimals():
    result = r_squared([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1], decimals=4)
    parts = str(result).split(".")
    if len(parts) == 2:
        assert len(parts[1]) <= 4


# ===== variance_explained =====

def test_variance_explained_returns_dataframe(independent_df):
    result = variance_explained(independent_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["component", "variance_explained", "cumulative"]


def test_variance_explained_all_components(independent_df):
    result = variance_explained(independent_df)
    assert result["component"].tolist() == ["PC1", "PC2", "PC3"]


def test_variance_explained_sums_to_100(independent_df):
    result = variance_explained(independent_df, decimals=None)
    assert pytest.approx(result["variance_explained"].sum(), abs=0.01) == 100.0


def test_variance_explained_cumulative_ends_at_100(independent_df):
    result = variance_explained(independent_df, decimals=None)
    assert pytest.approx(result["cumulative"].iloc[-1], abs=0.01) == 100.0


def test_variance_explained_sorted_descending(independent_df):
    values = variance_explained(independent_df)["variance_explained"].tolist()
    assert values == sorted(values, reverse=True)


def test_variance_explained_correlated_features():
    np.random.seed(42)
    x = np.random.randn(100)
    df = pd.DataFrame({
        "a": x,
        "b": x + np.random.randn(100) * 0.01,
        "c": np.random.randn(100),
    })
    vals = _to_map(variance_explained(df), "component", "variance_explained")
    assert vals["PC1"] > 50


def test_variance_explained_n_components(independent_df):
    result = variance_explained(independent_df, n_components=2)
    assert result["component"].tolist() == ["PC1", "PC2"]


def test_variance_explained_too_few_columns_warns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.warns(PercentifyWarning):
        result = variance_explained(df)
    assert result.empty


def test_variance_explained_ignores_non_numeric():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    assert len(variance_explained(df)) == 2
