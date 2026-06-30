import pytest
import numpy as np
import pandas as pd
from percentify import vif, missing, cv, outliers, r_squared, variance_explained


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


# ===== vif =====

def test_vif_returns_all_columns(independent_df):
    result = vif(independent_df)
    assert set(result.keys()) == {"a", "b", "c"}


def test_vif_independent_features_low(independent_df):
    result = vif(independent_df)
    for val in result.values():
        assert val < 5.0


def test_vif_collinear_features_high(collinear_df):
    result = vif(collinear_df)
    assert result["x"] > 10
    assert result["y"] > 10


def test_vif_flag_filters(independent_df):
    result = vif(independent_df, flag=5.0)
    assert len(result) == 0


def test_vif_flag_returns_collinear(collinear_df):
    result = vif(collinear_df, flag=5.0)
    assert "x" in result
    assert "y" in result
    assert "z" not in result


def test_vif_ignores_non_numeric():
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    result = vif(df)
    assert "name" not in result
    assert set(result.keys()) == {"a", "b"}


def test_vif_too_few_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        vif(df)


def test_vif_custom_decimals(independent_df):
    result = vif(independent_df, decimals=4)
    for val in result.values():
        str_val = str(val)
        if "." in str_val:
            assert len(str_val.split(".")[1]) <= 4


# ===== missing =====

def test_missing_basic():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    result = missing(df)
    assert result["b"] == 40.0
    assert result["a"] == 20.0
    assert result["c"] == 0.0


def test_missing_sorted_descending():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    result = missing(df)
    keys = list(result.keys())
    assert keys == ["b", "a", "c"]


def test_missing_no_nulls():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = missing(df)
    assert result["a"] == 0.0
    assert result["b"] == 0.0


def test_missing_all_null():
    df = pd.DataFrame({"a": [None, None, None]})
    result = missing(df)
    assert result["a"] == 100.0


def test_missing_empty_df():
    df = pd.DataFrame({"a": [], "b": []})
    result = missing(df)
    assert result["a"] == 0.0


def test_missing_includes_non_numeric():
    df = pd.DataFrame({
        "name": ["Alice", None, "Charlie"],
        "age": [25, None, 35],
    })
    result = missing(df)
    assert "name" in result
    assert "age" in result
    assert result["name"] == 33.33
    assert result["age"] == 33.33


# ===== cv =====

def test_cv_series():
    s = pd.Series([10, 20, 30, 40, 50])
    result = cv(s)
    assert result > 0
    assert isinstance(result, float)


def test_cv_dataframe():
    df = pd.DataFrame({
        "a": [10, 20, 30, 40, 50],
        "b": [100, 100, 100, 100, 100],
    })
    result = cv(df)
    assert isinstance(result, dict)
    assert result["a"] > 0
    assert result["b"] == 0.0


def test_cv_zero_mean_series():
    s = pd.Series([-1, 0, 1])
    with pytest.raises(ValueError):
        cv(s)


def test_cv_zero_mean_dataframe():
    df = pd.DataFrame({
        "a": [-1, 0, 1],
        "b": [10, 20, 30],
    })
    result = cv(df)
    assert result["a"] == float("inf")
    assert result["b"] > 0


def test_cv_ignores_non_numeric():
    df = pd.DataFrame({
        "a": [10, 20, 30],
        "name": ["x", "y", "z"],
    })
    result = cv(df)
    assert "name" not in result
    assert "a" in result


def test_cv_custom_decimals():
    s = pd.Series([10, 20, 30, 40, 50])
    result = cv(s, decimals=4)
    str_val = str(result)
    if "." in str_val:
        assert len(str_val.split(".")[1]) <= 4


# ===== outliers =====

def test_outliers_series():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    result = outliers(s)
    assert isinstance(result, float)
    assert result > 0


def test_outliers_no_outliers():
    s = pd.Series([1, 2, 3, 4, 5])
    result = outliers(s)
    assert result == 0.0


def test_outliers_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    result = outliers(df)
    assert isinstance(result, dict)
    assert result["a"] > 0
    assert result["b"] == 0.0


def test_outliers_all_nan():
    s = pd.Series([None, None, None])
    result = outliers(s)
    assert result == 0.0


def test_outliers_ignores_non_numeric():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 100],
        "name": ["x", "y", "z", "w", "v"],
    })
    result = outliers(df)
    assert "name" not in result
    assert "a" in result


def test_outliers_custom_multiplier():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    strict = outliers(s, multiplier=1.0)
    loose = outliers(s, multiplier=3.0)
    assert strict >= loose


# ===== r_squared =====

def test_r_squared_perfect():
    y = [1, 2, 3, 4, 5]
    assert r_squared(y, y) == 100.0


def test_r_squared_good_fit():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
    result = r_squared(y_true, y_pred)
    assert 90 < result < 100


def test_r_squared_bad_fit():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [5, 4, 3, 2, 1]
    result = r_squared(y_true, y_pred)
    assert result < 0


def test_r_squared_with_numpy():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert r_squared(y_true, y_pred) == 100.0


def test_r_squared_with_series():
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.0, 5.1])
    result = r_squared(y_true, y_pred)
    assert result > 0


def test_r_squared_mismatched_length():
    with pytest.raises(ValueError):
        r_squared([1, 2, 3], [1, 2])


def test_r_squared_too_few():
    with pytest.raises(ValueError):
        r_squared([1], [1])


def test_r_squared_custom_decimals():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
    result = r_squared(y_true, y_pred, decimals=4)
    str_val = str(result)
    if "." in str_val:
        assert len(str_val.split(".")[1]) <= 4


# ===== variance_explained =====

def test_variance_explained_returns_all_components():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })
    result = variance_explained(df)
    assert len(result) == 3
    assert "PC1" in result
    assert "PC2" in result
    assert "PC3" in result


def test_variance_explained_sums_to_100():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })
    result = variance_explained(df, decimals=None)
    assert pytest.approx(sum(result.values()), abs=0.01) == 100.0


def test_variance_explained_sorted_descending():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })
    result = variance_explained(df)
    values = list(result.values())
    assert values == sorted(values, reverse=True)


def test_variance_explained_correlated_features():
    np.random.seed(42)
    x = np.random.randn(100)
    df = pd.DataFrame({
        "a": x,
        "b": x + np.random.randn(100) * 0.01,
        "c": np.random.randn(100),
    })
    result = variance_explained(df)
    assert result["PC1"] > 50


def test_variance_explained_n_components():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })
    result = variance_explained(df, n_components=2)
    assert len(result) == 2
    assert "PC1" in result
    assert "PC2" in result
    assert "PC3" not in result


def test_variance_explained_too_few_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        variance_explained(df)


def test_variance_explained_ignores_non_numeric():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    result = variance_explained(df)
    assert len(result) == 2
